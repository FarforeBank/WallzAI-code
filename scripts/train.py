# scripts/train.py
import os
import sys

# Добавляем корневую папку WallzAi в пути поиска модулей, чтобы Python видел пакет envs
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed

from envs.quoridor.quoridor_env import QuoridorEnv

def make_env(rank, seed=0):
    """
    Утилита для создания многопроцессорной среды.
    """
    def _init():
        env = QuoridorEnv()
        env.reset(seed=seed + rank)
        return env
    set_random_seed(seed)
    return _init

def main():
    os.makedirs("./models", exist_ok=True)
    os.makedirs("./logs", exist_ok=True)

    num_envs = 10
    print(f"Инициализация {num_envs} параллельных сред...")
    
    env = SubprocVecEnv([make_env(i) for i in range(num_envs)])
    
    raw_eval_env = QuoridorEnv()
    monitored_eval_env = Monitor(raw_eval_env)
    eval_env = DummyVecEnv([lambda: monitored_eval_env])

    print("Инициализация MaskablePPO...")
    model = MaskablePPO(
        "MlpPolicy",
        env,
        learning_rate=5e-5,       
        n_steps=2048,             
        batch_size=512,           
        clip_range=0.1,           
        ent_coef=0.02,            
        gamma=0.99,
        verbose=1,
        tensorboard_log="./logs/",
        device="mps"  # <--- ПРИНУДИТЕЛЬНО ВКЛЮЧАЕМ GPU APPLE SILICON
    )

    eval_callback = EvalCallback(
        eval_env, 
        best_model_save_path='./models/best_model',
        log_path='./logs/eval', 
        eval_freq=10000,
        deterministic=False, 
        render=False
    )

    print("Запуск обучения (останови через Ctrl+C)...")
    try:
        model.learn(total_timesteps=10_000_000, callback=eval_callback)
    except KeyboardInterrupt:
        print("\nОбучение прервано пользователем. Сохраняем прогресс...")
    finally:
        model.save("models/quoridor_latest")
        print("Модель сохранена.")
        env.close()

if __name__ == "__main__":
    main()