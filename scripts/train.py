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

# Stage 1 curriculum: teach pathing around random walls before enabling wall placement.
RANDOM_WALLS_RANGE = (0, 8)
MOVE_ONLY = True
REPEAT_PENALTY = True


def make_env(rank: int, seed: int = 0):
    def _init():
        env = QuoridorEnv(
            random_walls_range=RANDOM_WALLS_RANGE,
            move_only=MOVE_ONLY,
            repeat_penalty=REPEAT_PENALTY,
        )
        env.reset(seed=seed + rank)
        return env

    return _init


def make_eval_env():
    return QuoridorEnv(
        random_walls_range=RANDOM_WALLS_RANGE,
        move_only=MOVE_ONLY,
        repeat_penalty=REPEAT_PENALTY,
    )


def load_maskable_model(model_path: Path, env):
    """Continue old checkpoints even if only Box bounds changed."""
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
    print(
        f"Curriculum: random_walls={RANDOM_WALLS_RANGE}, "
        f"move_only={MOVE_ONLY}, repeat_penalty={REPEAT_PENALTY}"
    )

    vec_env = SubprocVecEnv([make_env(i) for i in range(num_envs)], start_method="spawn")
    eval_env = make_eval_env()

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
        backup_path = save_path / "base_before_random_walls.zip"
        if not backup_path.exists():
            backup_path.write_bytes(model_path.read_bytes())
            print(f"Сохранил backup старой модели: {backup_path}")
        print("Найдена существующая модель. Продолжаем обучение на random walls...")
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
