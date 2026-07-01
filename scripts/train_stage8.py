import argparse
import os
import sys
from pathlib import Path

import torch
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback

# Keep the same categorical validation workaround as scripts/train.py.
torch.distributions.Distribution.set_default_validate_args(False)

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from envs.quoridor.quoridor_env import MOVE_ACTIONS, QuoridorEnv as BaseQuoridorEnv


STAGE8 = {
    "name": "race-finish-wall-timing-finetune",
    "model_dir": "best_model",
    "random_walls_range": (0, 2),
    "move_only": False,
    "repeat_penalty": True,
    "opponent_policy": "greedy",
    "opponent_randomness": 0.08,
    "wall_reward": True,
    "wall_candidate_limit": 40,
    "opponent_start_advantage_range": (0, 2),
    "defensive_wall_reward": True,
    "opponent_wall_probability": 0.25,
    "self_trap_penalty": True,
    "timesteps": 1_500_000,
    "n_eval_episodes": 80,
    "description": "fine-tune without browser guards: finish winning races and use walls only when they really slow opponent",
}


class RaceFinishQuoridorEnv(BaseQuoridorEnv):
    """Reward wrapper for late-game race discipline.

    This does not add any browser/runtime guard. It changes training feedback so the
    policy learns to finish races itself:
    - reward direct progress more when close to the finish;
    - punish sideways/backward moves near the finish;
    - punish non-urgent walls when our path is already short;
    - reward emergency walls only when they actually increase opponent BFS distance.
    """

    def step(self, action):
        action = int(action)
        prev_p1_dist = self.engine.get_bfs_distance(self.engine.p1_pos, 0)
        prev_p2_dist = self.engine.get_bfs_distance(self.engine.p2_pos, 8)
        was_wall = action >= MOVE_ACTIONS

        obs, reward, terminated, truncated, info = super().step(action)

        new_p1_dist = self.engine.get_bfs_distance(self.engine.p1_pos, 0)
        new_p2_dist = self.engine.get_bfs_distance(self.engine.p2_pos, 8)
        p1_progress = prev_p1_dist - new_p1_dist
        p2_slowdown = new_p2_dist - prev_p2_dist

        # Endgame: if we can finish soon, teach the policy to actually run.
        if prev_p1_dist <= 5:
            if not was_wall:
                if p1_progress > 0:
                    reward += 0.28 + 0.45 * p1_progress
                else:
                    reward -= 0.45
            else:
                # Near finish, walls are only worth it if opponent is also dangerous
                # and the wall measurably slows them.
                if prev_p2_dist > 2:
                    reward -= 0.55
                if p2_slowdown <= 0:
                    reward -= 0.35
                if new_p1_dist > prev_p1_dist:
                    reward -= 0.25

        # Emergency defense: reward walls only when they really delay an opponent
        # who is close to winning. This should reduce random wall spam.
        if was_wall and prev_p2_dist <= 3:
            if p2_slowdown > 0:
                reward += 0.45 + 0.35 * p2_slowdown
            else:
                reward -= 0.35

        # Race pressure: do not let the opponent walk freely into the finish.
        if not was_wall and prev_p2_dist <= 2 and new_p2_dist <= prev_p2_dist:
            reward -= 0.20

        return obs, reward, terminated, truncated, info


def resolve_device(requested: str) -> str:
    requested = requested.lower()
    if requested == "auto":
        return "cpu"
    if requested == "mps":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        print("[Предупреждение] MPS недоступен, использую CPU.")
        return "cpu"
    return requested


def parse_args():
    parser = argparse.ArgumentParser(description="Train Stage 8 race/finish fine-tune without browser guards.")
    parser.add_argument("--timesteps", type=int, default=None, help="Override stage default timesteps.")
    parser.add_argument("--num-envs", type=int, default=None, help="Number of parallel env processes.")
    parser.add_argument("--device", choices=["cpu", "mps", "auto"], default="cpu", help="Torch device.")
    parser.add_argument("--torch-threads", type=int, default=1, help="Torch CPU threads.")
    parser.add_argument("--reset", action="store_true", help="Start a fresh model after backing up the current one.")
    return parser.parse_args()


def make_quoridor_env():
    cfg = STAGE8
    return RaceFinishQuoridorEnv(
        random_walls_range=cfg["random_walls_range"],
        move_only=cfg["move_only"],
        repeat_penalty=cfg["repeat_penalty"],
        opponent_policy=cfg["opponent_policy"],
        opponent_randomness=cfg["opponent_randomness"],
        smart_observation=True,
        wall_reward=cfg["wall_reward"],
        wall_candidate_limit=cfg["wall_candidate_limit"],
        opponent_start_advantage_range=cfg["opponent_start_advantage_range"],
        defensive_wall_reward=cfg["defensive_wall_reward"],
        opponent_wall_probability=cfg["opponent_wall_probability"],
        self_trap_penalty=cfg["self_trap_penalty"],
    )


def make_env(rank: int, seed: int = 0):
    def _init():
        env = make_quoridor_env()
        env.reset(seed=seed + rank)
        return env

    return _init


def make_eval_env():
    return Monitor(make_quoridor_env())


def create_new_model(vec_env, device):
    return MaskablePPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        device=device,
        learning_rate=5e-5,
        n_steps=1024,
        batch_size=512,
        clip_range=0.1,
        ent_coef=0.01,
    )


def load_maskable_model(model_path: Path, env, device):
    return MaskablePPO.load(
        str(model_path),
        env=env,
        device=device,
        custom_objects={
            "observation_space": env.observation_space,
            "action_space": env.action_space,
        },
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


def save_model_safely(model, model_path: Path, reason: str):
    print(f"\n{reason} Сохраняем прогресс в {model_path}...")
    model.save(str(model_path))


def main():
    args = parse_args()
    timesteps = args.timesteps or STAGE8["timesteps"]
    device = resolve_device(args.device)
    torch.set_num_threads(max(1, int(args.torch_threads)))

    detected_cpus = os.cpu_count() or 1
    if args.num_envs is None:
        num_envs = max(1, min(20, detected_cpus))
    else:
        num_envs = max(1, int(args.num_envs))

    print(f"Инициализация {num_envs} параллельных сред...")
    print(f"Torch device={device}, torch_threads={torch.get_num_threads()}, detected_cpus={detected_cpus}")
    print(f"Stage 8: {STAGE8['name']}")
    print(STAGE8["description"])
    print(
        "Curriculum: "
        f"model_dir={STAGE8['model_dir']}, "
        f"random_walls={STAGE8['random_walls_range']}, "
        f"move_only={STAGE8['move_only']}, repeat_penalty={STAGE8['repeat_penalty']}, "
        f"opponent={STAGE8['opponent_policy']}, opponent_randomness={STAGE8['opponent_randomness']}, "
        f"opponent_start_advantage={STAGE8['opponent_start_advantage_range']}, "
        f"opponent_wall_probability={STAGE8['opponent_wall_probability']}, "
        f"wall_candidate_limit={STAGE8['wall_candidate_limit']}, "
        f"wall_reward={STAGE8['wall_reward']}, defensive_wall_reward={STAGE8['defensive_wall_reward']}, "
        f"self_trap_penalty={STAGE8['self_trap_penalty']}, timesteps={timesteps}"
    )

    vec_env = SubprocVecEnv([make_env(i) for i in range(num_envs)], start_method="spawn")
    eval_env = make_eval_env()

    save_path = ROOT_DIR / "models" / STAGE8["model_dir"]
    log_path = ROOT_DIR / "logs" / "eval" / f"stage_8_{STAGE8['name']}"
    save_path.mkdir(parents=True, exist_ok=True)
    log_path.mkdir(parents=True, exist_ok=True)
    model_path = save_path / "best_model.zip"

    eval_callback = MaskableEvalCallback(
        eval_env,
        best_model_save_path=str(save_path),
        log_path=str(log_path),
        eval_freq=max(1, 50_000 // num_envs),
        n_eval_episodes=STAGE8["n_eval_episodes"],
        deterministic=True,
        render=False,
    )

    if args.reset:
        backup_existing_model(model_path, save_path, "reset_stage_8")
        print("--reset указан. Начинаем новую smart-модель.")
        model = create_new_model(vec_env, device)
    elif model_path.exists():
        backup_existing_model(model_path, save_path, "stage_8")
        print("Пробуем загрузить совместимую smart-модель...")
        try:
            model = load_maskable_model(model_path, vec_env, device)
            print("Совместимая smart-модель найдена. Продолжаем обучение.")
        except Exception as exc:
            print(f"Старая модель несовместима ({type(exc).__name__}). Стартуем smart-модель с нуля.")
            model = create_new_model(vec_env, device)
    else:
        print("Модель не найдена. Стартуем smart-модель с нуля.")
        model = create_new_model(vec_env, device)

    print("Запуск обучения (останови через Ctrl+C)...")
    try:
        model.learn(total_timesteps=timesteps, callback=eval_callback, progress_bar=True)
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
