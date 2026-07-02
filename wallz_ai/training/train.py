from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from wallz_ai.agents.model import WallzPolicyValueNet, select_device
from wallz_ai.agents.ppo import PPOBatch, ppo_update
from wallz_ai.training.evaluate import NeuralAgent, evaluate_against_baselines, load_checkpoint
from wallz_ai.training.self_play import play_self_play_games


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_checkpoint(path: Path, model, optimizer, config: dict, step: int, metrics: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save({"model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(), "config": config, "step": step, "metrics": metrics}, tmp)
    tmp.replace(path)


def append_csv(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Wallz AI with masked PPO self-play.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--device", default=None, choices=[None, "auto", "mps", "cuda", "cpu"])
    parser.add_argument("--steps", type=int, default=None, help="Override total self-play positions.")
    parser.add_argument("--games-per-iter", type=int, default=None)
    parser.add_argument("--max-moves", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.steps is not None:
        cfg["training"]["total_positions"] = args.steps
    set_seed(int(cfg.get("seed", 42)))
    device = select_device(args.device or cfg.get("device", "auto"))
    torch.set_num_threads(int(cfg.get("torch_threads", 1)))
    model = WallzPolicyValueNet(**cfg.get("model", {})).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["training"].get("learning_rate", 3e-4)), weight_decay=float(cfg["training"].get("weight_decay", 1e-4)))
    start_step = 0
    if args.resume:
        old_model, checkpoint = load_checkpoint(args.resume, device)
        model.load_state_dict(old_model.state_dict())
        if "optimizer_state" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_step = int(checkpoint.get("step", 0))

    out_dir = Path(cfg["paths"].get("run_dir", "runs/wallz_ppo"))
    ckpt_dir = out_dir / "checkpoints"
    log_csv = out_dir / "metrics.csv"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "config.resolved.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)

    total_positions = int(cfg["training"].get("total_positions", 100_000))
    games_per_iter = int(args.games_per_iter or cfg["training"].get("games_per_iter", 16))
    max_moves = int(args.max_moves or cfg["env"].get("max_moves", 300))
    gamma = float(cfg["training"].get("gamma", 0.997))
    eval_every = int(cfg["training"].get("eval_every_positions", 10_000))
    save_every = int(cfg["training"].get("save_every_positions", 10_000))
    next_eval = start_step + eval_every
    next_save = start_step + save_every
    step = start_step
    print(f"Training on device={device}; output={out_dir}; total_positions={total_positions}")
    while step < total_positions:
        buffer, sp_stats = play_self_play_games(model, device, games=games_per_iter, max_moves=max_moves, dense_shaping=bool(cfg["training"].get("dense_shaping", False)), shaping_coef=float(cfg["training"].get("shaping_coef", 0.03)))
        if len(buffer) == 0:
            raise RuntimeError("Self-play produced no data")
        arrays = buffer.to_arrays(gamma=gamma)
        update_metrics = ppo_update(model, optimizer, PPOBatch(**arrays), device=device, epochs=int(cfg["training"].get("ppo_epochs", 4)), minibatch_size=int(cfg["training"].get("minibatch_size", 256)), clip_range=float(cfg["training"].get("clip_range", 0.2)), value_coef=float(cfg["training"].get("value_coef", 0.5)), entropy_coef=float(cfg["training"].get("entropy_coef", 0.01)), max_grad_norm=float(cfg["training"].get("max_grad_norm", 1.0)))
        step += len(buffer)
        row = {"step": step, "games": sp_stats.games, "positions": len(buffer), "win_rate_p0": sp_stats.wins_p0 / max(1, sp_stats.games), "avg_game_length": sp_stats.avg_game_length, "avg_walls_used": sp_stats.avg_walls_used, "invalid_actions": sp_stats.invalid_actions, "policy_entropy": sp_stats.policy_entropy, **update_metrics}
        if step >= next_eval:
            agent = NeuralAgent(model, device, deterministic=True)
            for result in evaluate_against_baselines(agent, games=int(cfg["evaluation"].get("games", 20)), max_moves=max_moves, seed=int(cfg.get("seed", 42))):
                row[f"win_rate_vs_{result.opponent}"] = result.win_rate
                row[f"invalid_vs_{result.opponent}"] = result.invalid_actions
            next_eval += eval_every
        append_csv(log_csv, row)
        print(row)
        if step >= next_save:
            save_checkpoint(ckpt_dir / f"step_{step}.pt", model, optimizer, cfg, step, row)
            save_checkpoint(ckpt_dir / "latest.pt", model, optimizer, cfg, step, row)
            next_save += save_every
    save_checkpoint(ckpt_dir / "final.pt", model, optimizer, cfg, step, {"done": True})


if __name__ == "__main__":
    main()
