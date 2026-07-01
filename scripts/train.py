import os
import multiprocessing
from typing import Callable
import numpy as np
import torch
import gymnasium as gym

from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback

# Импорты наших локальных модулей
from envs.quoridor.engine import QuoridorEngine
from envs.quoridor.quoridor_env import QuoridorEnv
from envs.wrappers.self_play_wrapper import SelfPlayWrapper
from models.feature_extractor import QuoridorCNN


def mask_fn(env: gym.Env) -> np.ndarray:
    """
    Адаптер для извлечения Action Mask.
    Пробивается сквозь все обертки (Monitor, SelfPlayWrapper) до оригинальной среды.
    """
    return env.unwrapped._get_info()["action_mask"]


def make_env(rank: int, seed: int = 0) -> Callable:
    """
    Фабрика для создания изолированных экземпляров среды.
    """
    def _init() -> gym.Env:
        engine = QuoridorEngine()
        env = QuoridorEnv(engine=engine)
        env = SelfPlayWrapper(env)              # Изолируем игрока от второго игрока
        env = Monitor(env)                      # Сбор статистики
        env = ActionMasker(env, mask_fn)        # Маскирование недопустимых действий
        env.action_space.seed(seed + rank)
        return env
    return _init


class SelfPlayUpdateCallback(BaseCallback):
    """
    Коллбэк, который периодически сохраняет текущую модель и заставляет 
    все среды (тренировочные и валидационные) загрузить ее как оппонента.
    """
    def __init__(self, update_freq: int, save_path: str, eval_env, verbose=0):
        super().__init__(verbose)
        self.update_freq = update_freq
        self.save_path = save_path
        self.eval_env = eval_env
        self.latest_model_path = os.path.join(save_path, "latest_opponent.zip")

    def _init_callback(self) -> None:
        # Сохраняем случайного агента в самом начале, чтобы было против кого играть на старте
        os.makedirs(self.save_path, exist_ok=True)
        self.model.save(self.latest_model_path)
        self.training_env.env_method("load_opponent", self.latest_model_path)
        self.eval_env.env_method("load_opponent", self.latest_model_path)
        if self.verbose > 0:
            print("[Self-Play] Инициализирован базовый оппонент.")

    def _on_step(self) -> bool:
        if self.n_calls % self.update_freq == 0:
            self.model.save(self.latest_model_path)
            
            # Обновляем оппонентов во всех параллельных воркерах
            self.training_env.env_method("load_opponent", self.latest_model_path)
            self.eval_env.env_method("load_opponent", self.latest_model_path)
            
            if self.verbose > 0:
                print(f"\n[Self-Play] Модели оппонентов обновлены (Шаг: {self.num_timesteps})")
        return True


if __name__ == "__main__":
    # 1. Настройка оборудования (Apple MPS)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Запуск обучения на устройстве: {device.upper()}")

    # 2. Параллелизация
    num_cpu = multiprocessing.cpu_count()
    num_envs = max(1, num_cpu - 2) # Оставляем 2 ядра ОС для стабильности
    print(f"Инициализация {num_envs} параллельных сред...")
    
    vec_env = SubprocVecEnv([make_env(i) for i in range(num_envs)])
    eval_env = DummyVecEnv([make_env(999)]) # Среда для тестов (в одном потоке)

    # 3. Настройка архитектуры нейросети
    policy_kwargs = dict(
        features_extractor_class=QuoridorCNN,
        features_extractor_kwargs=dict(features_dim=256),
        net_arch=[dict(pi=[128, 128], vf=[128, 128])]
    )

    # 4. Инициализация Maskable PPO
    model = MaskablePPO(
        "CnnPolicy",
        vec_env,
        policy_kwargs=policy_kwargs,
        learning_rate=3e-4,
        n_steps=2048,           
        batch_size=256,         
        n_epochs=10,            
        gamma=0.99,             
        clip_range=0.2,         
        ent_coef=0.01,          # Важно для настольных игр: заставляет агента пробовать разные ходы
        tensorboard_log="./logs/quoridor_tensorboard/",
        device=device,
        verbose=1
    )

    # 5. Настройка системы Callbacks
    os.makedirs("./models/checkpoints", exist_ok=True)
    os.makedirs("./models/opponents", exist_ok=True)
    
    # 5.1 Сохранение чекпоинтов
    checkpoint_callback = CheckpointCallback(
        save_freq=max(200_000 // num_envs, 1),
        save_path="./models/checkpoints/",
        name_prefix="quoridor_ppo"
    )

    # 5.2 Валидация лучшей модели
    eval_callback = MaskableEvalCallback(
        eval_env,
        best_model_save_path="./models/best_model/",
        log_path="./logs/eval/",
        eval_freq=max(100_000 // num_envs, 1),
        deterministic=True, 
        render=False
    )

    # 5.3 Коллбэк Self-Play (обновляем оппонента каждые ~50к шагов)
    self_play_callback = SelfPlayUpdateCallback(
        update_freq=max(50_000 // num_envs, 1),
        save_path="./models/opponents/",
        eval_env=eval_env,
        verbose=1
    )

    # 6. Запуск тренировочного цикла
    TOTAL_TIMESTEPS = 10_000_000 # 10 миллионов шагов - хорошая цель для первого прогона
    print(f"Начало обучения на {TOTAL_TIMESTEPS} шагов...")
    
    try:
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=[self_play_callback, checkpoint_callback, eval_callback],
            progress_bar=True
        )
        model.save("./models/quoridor_ppo_final")
        print("Обучение успешно завершено!")
        
    except KeyboardInterrupt:
        print("\nОбучение прервано пользователем. Сохранение текущего состояния...")
        model.save("./models/quoridor_ppo_interrupted")
        print("Модель сохранена.")