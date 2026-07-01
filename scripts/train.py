import os
import sys
from pathlib import Path

from stable_baselines3.common.vec_env import SubprocVecEnv
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from envs.quoridor.quoridor_env import QuoridorEnv


def make_env(rank: int, seed: int = 0):
    def _init():
        env = QuoridorEnv()
        env.reset(seed=seed + rank)
        return env

    return _init


def load_maskable_model(model_path: Path, env):
    """Continue old checkpoints even if only Box bounds changed.

    Older checkpoints were saved with Box(-1, 2, (9, 9, 3), int8). The env shape
    stayed the same, but the bounds were corrected to Box(0, 10, ...). SB3 checks
    the full space object on load, so we override the saved metadata.
    """
    return MaskablePPO.load(
        str(model_path),
        env=env,
        device="cpu",
        custom_objects={
            "observation_space": env.observation_space,
            "action_space": env.action_space,
        },
    )


def main():
    num_envs = max(1, min(8, os.cpu_count() or 1))
    print(f"Инициализация {num_envs} параллельных сред...")

    vec_env = SubprocVecEnv([make_env(i) for i in range(num_envs)], start_method="spawn")
    eval_env = QuoridorEnv()

    save_path = ROOT_DIR / "models" / "best_model"
    log_path = ROOT_DIR / "logs" / "eval"
    save_path.mkdir(parents=True, exist_ok=True)
    log_path.mkdir(parents=True, exist_ok=True)
    model_path = save_path / "best_model.zip"

    eval_callback = MaskableEvalCallback(
        eval_env,
        best_model_save_path=str(save_path),
        log_path=str(log_path),
        eval_freq=max(1, 12_000 // num_envs),
        deterministic=True,
        render=False,
    )

    if model_path.exists():
        print("Найдена существующая модель. Продолжаем обучение...")
        model = load_maskable_model(model_path, vec_env)
    else:
        print("Начинаем обучение с нуля...")
        model = MaskablePPO(
            "MlpPolicy",
            vec_env,
            verbose=1,
            device="cpu",
            learning_rate=5e-5,
            n_steps=1024,
            batch_size=512,
        )

    print("Запуск обучения (останови через Ctrl+C)...")
    try:
        model.learn(total_timesteps=3_000_000, callback=eval_callback, progress_bar=True)
    except KeyboardInterrupt:
        print("\nОбучение прервано пользователем. Сохраняем прогресс...")
        model.save(str(model_path))
    finally:
        vec_env.close()
        eval_env.close()


if __name__ == "__main__":
    main()
