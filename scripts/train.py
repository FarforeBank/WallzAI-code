import argparse
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

# One smart architecture for all stages:
# observation = (9, 9, 5): board, H walls, V walls, our BFS map, opponent BFS map.
# action space = 140: real movement/jumps/diagonals + wall placements.
STAGES = {
    "1": {
        "name": "smart-path-empty",
        "model_dir": "best_model",
        "random_walls_range": (0, 0),
        "move_only": True,
        "repeat_penalty": True,
        "opponent_policy": "none",
        "opponent_randomness": 0.0,
        "wall_reward": False,
        "wall_candidate_limit": 0,
        "opponent_start_advantage_range": (0, 0),
        "defensive_wall_reward": False,
        "opponent_wall_probability": 0.0,
        "timesteps": 500_000,
        "n_eval_episodes": 30,
        "description": "learn clean pathing to the finish with no moving opponent and no maze noise",
    },
    "2": {
        "name": "smart-path-maze",
        "model_dir": "best_model",
        "random_walls_range": (0, 8),
        "move_only": True,
        "repeat_penalty": True,
        "opponent_policy": "none",
        "opponent_randomness": 0.0,
        "wall_reward": False,
        "wall_candidate_limit": 0,
        "opponent_start_advantage_range": (0, 0),
        "defensive_wall_reward": False,
        "opponent_wall_probability": 0.0,
        "timesteps": 1_200_000,
        "n_eval_episodes": 40,
        "description": "learn BFS-map navigation through random wall mazes without opponent pressure",
    },
    "3": {
        "name": "smart-race-opponent",
        "model_dir": "best_model",
        "random_walls_range": (0, 6),
        "move_only": True,
        "repeat_penalty": True,
        "opponent_policy": "greedy",
        "opponent_randomness": 0.12,
        "wall_reward": False,
        "wall_candidate_limit": 0,
        "opponent_start_advantage_range": (0, 0),
        "defensive_wall_reward": False,
        "opponent_wall_probability": 0.0,
        "timesteps": 1_500_000,
        "n_eval_episodes": 50,
        "description": "add a moving opponent and refine racing, jumps and diagonals",
    },
    "4": {
        "name": "smart-wall-soft-fast",
        "model_dir": "best_model",
        "random_walls_range": (0, 2),
        "move_only": False,
        "repeat_penalty": True,
        "opponent_policy": "greedy",
        "opponent_randomness": 0.12,
        "wall_reward": True,
        "wall_candidate_limit": 24,
        "opponent_start_advantage_range": (0, 0),
        "defensive_wall_reward": False,
        "opponent_wall_probability": 0.05,
        "timesteps": 1_200_000,
        "n_eval_episodes": 40,
        "description": "start learning useful wall placement with a compact candidate mask",
    },
    "5": {
        "name": "smart-wall-hard",
        "model_dir": "best_model",
        "random_walls_range": (0, 8),
        "move_only": False,
        "repeat_penalty": True,
        "opponent_policy": "greedy",
        "opponent_randomness": 0.25,
        "wall_reward": True,
        "wall_candidate_limit": 36,
        "opponent_start_advantage_range": (0, 0),
        "defensive_wall_reward": False,
        "opponent_wall_probability": 0.15,
        "timesteps": 3_000_000,
        "n_eval_episodes": 60,
        "description": "full smart model: harder mazes, both players can wall, traps and noisy opponent",
    },
    "6": {
        "name": "empty-board-wall-specialist",
        "model_dir": "empty_model",
        "random_walls_range": (0, 0),
        "move_only": False,
        "repeat_penalty": True,
        "opponent_policy": "greedy",
        "opponent_randomness": 0.10,
        "wall_reward": True,
        "wall_candidate_limit": 32,
        "opponent_start_advantage_range": (0, 0),
        "defensive_wall_reward": False,
        "opponent_wall_probability": 0.20,
        "timesteps": 2_500_000,
        "n_eval_episodes": 60,
        "description": "separate specialist model: starts from an empty board and learns its own wall strategy",
    },
    "7": {
        "name": "defensive-wall-finetune",
        "model_dir": "best_model",
        "random_walls_range": (0, 2),
        "move_only": False,
        "repeat_penalty": True,
        "opponent_policy": "greedy",
        "opponent_randomness": 0.10,
        "wall_reward": True,
        "wall_candidate_limit": 40,
        "opponent_start_advantage_range": (1, 3),
        "defensive_wall_reward": True,
        "opponent_wall_probability": 0.30,
        "timesteps": 1_500_000,
        "n_eval_episodes": 70,
        "description": "fine-tune current model to defend when the opponent has tempo, walls, and is close to winning",
    },
}

CURRENT_STAGE = None
SHOW_PROGRESS_BAR = True
SMART_OBSERVATION = True
MODEL_DEVICE = "cpu"


def resolve_device(requested: str) -> str:
    requested = requested.lower()
    if requested == "auto":
        # Env/BFS is the bottleneck, so CPU is often still faster for this small MLP.
        # But on Apple Silicon we allow MPS when explicitly requested.
        return "cpu"
    if requested == "mps":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        print("[Предупреждение] MPS недоступен, использую CPU.")
        return "cpu"
    return requested


def parse_args():
    parser = argparse.ArgumentParser(description="Train Wallz AI with staged smart curriculum.")
    parser.add_argument(
        "--stage",
        choices=sorted(STAGES.keys()),
        default="1",
        help=(
            "Curriculum stage: 1 empty pathing, 2 maze pathing, 3 opponent race, "
            "4 soft walls, 5 hard walls, 6 empty-board specialist, 7 defensive fine-tune."
        ),
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Start a fresh smart model for this stage after backing up the current model.",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=None,
        help="Override stage default timesteps.",
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=None,
        help="Number of parallel env processes. Try 12, 16, 20, 24 on M4 Pro.",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "mps", "auto"],
        default="cpu",
        help="Torch device. CPU is usually faster here because env/BFS dominates; test mps manually.",
    )
    parser.add_argument(
        "--torch-threads",
        type=int,
        default=1,
        help="Torch CPU threads. Keep low with many SubprocVecEnv workers to avoid oversubscription.",
    )
    return parser.parse_args()


def make_quoridor_env():
    cfg = CURRENT_STAGE
    return QuoridorEnv(
        random_walls_range=cfg["random_walls_range"],
        move_only=cfg["move_only"],
        repeat_penalty=cfg["repeat_penalty"],
        opponent_policy=cfg["opponent_policy"],
        opponent_randomness=cfg["opponent_randomness"],
        smart_observation=SMART_OBSERVATION,
        wall_reward=cfg["wall_reward"],
        wall_candidate_limit=cfg["wall_candidate_limit"],
        opponent_start_advantage_range=cfg["opponent_start_advantage_range"],
        defensive_wall_reward=cfg["defensive_wall_reward"],
        opponent_wall_probability=cfg["opponent_wall_probability"],
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
        device=MODEL_DEVICE,
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
        device=MODEL_DEVICE,
        learning_rate=5e-5,
        n_steps=1024,
        batch_size=512,
        clip_range=0.1,
        ent_coef=0.01,
    )


def backup_existing_model(model_path: Path, save_path: Path, label: str):
    if not model_path.exists():
        return
    backup_path = save_path / f"backup_before_{label}.zip"
    counter = 2
    while backup_path.exists():
        backup_path = save_path / f"backup_before_{label}_{counter}.zip"
        counter += 1
    backup_path.write_bytes(model_path.read_bytes())
    print(f"Сохранил backup: {backup_path}")


def main():
    global CURRENT_STAGE, MODEL_DEVICE

    args = parse_args()
    CURRENT_STAGE = STAGES[args.stage]
    timesteps = args.timesteps or CURRENT_STAGE["timesteps"]
    MODEL_DEVICE = resolve_device(args.device)
    torch.set_num_threads(max(1, int(args.torch_threads)))

    detected_cpus = os.cpu_count() or 1
    if args.num_envs is None:
        num_envs = max(1, min(20, detected_cpus))
    else:
        num_envs = max(1, int(args.num_envs))

    print(f"Инициализация {num_envs} параллельных сред...")
    print(f"Torch device={MODEL_DEVICE}, torch_threads={torch.get_num_threads()}, detected_cpus={detected_cpus}")
    print(f"Stage {args.stage}: {CURRENT_STAGE['name']}")
    print(CURRENT_STAGE["description"])
    print(
        "Curriculum: "
        f"model_dir={CURRENT_STAGE['model_dir']}, "
        f"random_walls={CURRENT_STAGE['random_walls_range']}, "
        f"move_only={CURRENT_STAGE['move_only']}, repeat_penalty={CURRENT_STAGE['repeat_penalty']}, "
        f"opponent={CURRENT_STAGE['opponent_policy']}, opponent_randomness={CURRENT_STAGE['opponent_randomness']}, "
        f"opponent_start_advantage={CURRENT_STAGE['opponent_start_advantage_range']}, "
        f"opponent_wall_probability={CURRENT_STAGE['opponent_wall_probability']}, "
        f"wall_candidate_limit={CURRENT_STAGE['wall_candidate_limit']}, "
        f"smart_observation={SMART_OBSERVATION}, wall_reward={CURRENT_STAGE['wall_reward']}, "
        f"defensive_wall_reward={CURRENT_STAGE['defensive_wall_reward']}, "
        f"timesteps={timesteps}, progress_bar={SHOW_PROGRESS_BAR}"
    )

    vec_env = SubprocVecEnv([make_env(i) for i in range(num_envs)], start_method="spawn")
    eval_env = make_eval_env()

    save_path = ROOT_DIR / "models" / CURRENT_STAGE["model_dir"]
    log_path = ROOT_DIR / "logs" / "eval" / f"stage_{args.stage}_{CURRENT_STAGE['name']}"
    save_path.mkdir(parents=True, exist_ok=True)
    log_path.mkdir(parents=True, exist_ok=True)
    model_path = save_path / "best_model.zip"

    eval_callback = MaskableEvalCallback(
        eval_env,
        best_model_save_path=str(save_path),
        log_path=str(log_path),
        eval_freq=max(1, 50_000 // num_envs),
        n_eval_episodes=CURRENT_STAGE["n_eval_episodes"],
        deterministic=True,
        render=False,
    )

    if args.reset:
        backup_existing_model(model_path, save_path, f"reset_stage_{args.stage}")
        print("--reset указан. Начинаем новую smart-модель.")
        model = create_new_model(vec_env)
    elif model_path.exists():
        backup_existing_model(model_path, save_path, f"stage_{args.stage}")
        print("Пробуем загрузить совместимую smart-модель...")
        try:
            model = load_maskable_model(model_path, vec_env)
            print("Совместимая smart-модель найдена. Продолжаем обучение.")
        except Exception as exc:
            print(f"Старая модель несовместима ({type(exc).__name__}). Стартуем smart-модель с нуля.")
            model = create_new_model(vec_env)
    else:
        print("Модель не найдена. Стартуем smart-модель с нуля.")
        model = create_new_model(vec_env)

    print("Запуск обучения (останови через Ctrl+C)...")
    try:
        model.learn(
            total_timesteps=timesteps,
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
