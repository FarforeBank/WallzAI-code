import os
import sys
import sys
sys.path.append(os.getcwd())
import gymnasium as gym
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import EvalCallback
from sb3_contrib import MaskablePPO
from envs.quoridor.quoridor_env import QuoridorEnv

# Исправление пути для macOS
BASE_DIR = os.getcwd()
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

def make_env(rank, seed=0):
    def _init():
        env = QuoridorEnv()
        env.reset(seed=seed + rank)
        return env
    return _init

def main():
    num_envs = 24  # Используем 24 ядра для параллельного обучения
    print(f"Инициализация {num_envs} параллельных сред...")
    
    vec_env = SubprocVecEnv([make_env(i) for i in range(num_envs)])
    eval_env = QuoridorEnv()

    # Путь для сохранения
    save_path = os.path.join(BASE_DIR, "models", "best_model")
    os.makedirs(save_path, exist_ok=True)
    model_path = os.path.join(save_path, "best_model.zip")

    # Callbacks для оценки
    eval_callback = EvalCallback(
        eval_env, 
        best_model_save_path=save_path,
        log_path=os.path.join(BASE_DIR, "logs", "eval"), 
        eval_freq=500, # Тесты каждые 12 000 шагов (500*24)
        deterministic=False, 
        render=False
    )

    # Загрузка модели или старт с нуля
    if os.path.exists(model_path):
        print("Найдена существующая модель. Продолжаем обучение...")
        model = MaskablePPO.load(model_path, env=vec_env, device="cpu")
    else:
        print("Начинаем обучение с нуля...")
        model = MaskablePPO(
            "MlpPolicy", 
            vec_env, 
            verbose=1, 
            device="cpu",
            learning_rate=5e-5,
            n_steps=1024,
            batch_size=512
        )

    print("Запуск обучения (останови через Ctrl+C)...")
    try:
        model.learn(total_timesteps=3_000_000, callback=eval_callback, progress_bar=True)
    except KeyboardInterrupt:
        print("\nОбучение прервано пользователем. Сохраняем прогресс...")
        model.save(model_path)
    finally:
        vec_env.close()

if __name__ == "__main__":
    main()