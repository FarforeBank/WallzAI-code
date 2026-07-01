import os
import sys
from pathlib import Path

import torch
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback

# Torch 2.x can be overly strict when validating categorical probability simplexes
# after action masking. MaskablePPO already receives finite logits; this avoids
# rare false-positive crashes during long runs.
torch.distributions.Distribution.set_default_validate_args(False)

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from envs.quoridor.quoridor_env import QuoridorEnv

# Stage 4 curriculum: smart model.
# Observation is now (9, 9, 5): board, H walls, V walls, our BFS map, opponent BFS map.
# The policy can use real moves, jumps/diagonals, and wall placement.
RANDOM_WALLS_RANGE = (0, 4)
MOVE_ONLY = False
REPEAT_PENALTY = True
OPPONENT_POLICY = "greedy"
OPPONENT_RANDOMNESS = 0.15
SMART_OBSERVATION = True
WALL_REWARD = True
SHOW_PROGRESS_BAR = True
FORCE_NEW_SMART_MODEL = True


def make_quoridor_env():
    return QuoridorEnv(
        random_walls_range=RANDOM_WALLS_RANGE,
        move_only=MOVE_ONLY,
        repeat_penalty=REPEAT_PENALTY,
        opponent_policy=OPPONENT_POLICY,
        opponent_randomness=OPPONENT_RANDOMNESS,
        smart_observation=SMART_OBSERVATION,
        wall_reward=WALL_REWARD,
    )


def make_env(rank: int, seed: int = 0):
    def _init():
        env = make_quoridor_env()
        env.reset(seed=seed + rank)
        return env

    return _init


def make_eval_env():
    return Monitor(make_quoridor_env())


def load_maskable_model(model_path: Path, env):
    """Continue compatible checkpoints only."""
    return MaskablePPO.load(
        str(model_path),
        env=env,
        device="cpu",
        custom_objects={
            "observation_space": env.observation_space,
            "action_space": env.action_space,
        },
    )


def save_model_safely(model, model_path: Path, reason: str):
    print(f"\n{reason} Сохраняем прогресс в {model_path}...")
    model.save(str(model_path))


def create_new_model(vec_env):
    return MaskablePPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        device="cpu",
        learning_rate=5e-5,
        n_steps=1024,
        batch_size=512,
        clip_range=0.1,
        ent_coef=0.01,
    )


def main():
    num_envs = max(1, min(16, os.cpu_count() or 1))
    print(f"Инициализация {num_envs} параллельных сред...")
    print(
        "Curriculum: "
        f"random_walls={RANDOM_WALLS_RANGE}, "
        f"move_only={MOVE_ONLY}, repeat_penalty={REPEAT_PENALTY}, "
        f"opponent={OPPONENT_POLICY}, opponent_randomness={OPPONENT_RANDOMNESS}, "
        f"smart_observation={SMART_OBSERVATION}, wall_reward={WALL_REWARD}, "
        f"progress_bar={SHOW_PROGRESS_BAR}"
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
        eval_freq=max(1, 50_000 // num_envs),
        n_eval_episodes=40,
        deterministic=True,
        render=False,
    )

    if model_path.exists():
        backup_path = save_path / "base_before_smart_wall_model.zip"
        if not backup_path.exists():
            backup_path.write_bytes(model_path.read_bytes())
            print(f"Сохранил backup старой модели: {backup_path}")

    if FORCE_NEW_SMART_MODEL or not model_path.exists():
        print("Stage 4: новый smart observation. Начинаем новую модель с нуля...")
        model = create_new_model(vec_env)
    else:
        print("Найдена совместимая модель. Продолжаем обучение...")
        try:
            model = load_maskable_model(model_path, vec_env)
        except Exception as exc:
            print(f"Не удалось загрузить старую модель ({type(exc).__name__}). Стартуем с нуля.")
            model = create_new_model(vec_env)

    print("Запуск обучения (останови через Ctrl+C)...")
    try:
        model.learn(
            total_timesteps=5_000_000,
            callback=eval_callback,
            progress_bar=SHOW_PROGRESS_BAR,
        )
    except KeyboardInterrupt:
        save_model_safely(model, model_path, "Обучение прервано пользователем.")
    except Exception as exc:
        save_model_safely(model, model_path, f"Обучение упало с ошибкой {type(exc).__name__}:")
        raise
    finally:
        try:
            vec_env.close()
        except EOFError:
            pass
        eval_env.close()


if __name__ == "__main__":
    main()
