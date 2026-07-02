from __future__ import annotations

import argparse

from wallz_ai.agents.model import select_device
from wallz_ai.training.evaluate import NeuralAgent, evaluate_against_baselines, load_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a Wallz checkpoint against baselines.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--games", type=int, default=50)
    parser.add_argument("--max-moves", type=int, default=300)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    device = select_device(args.device)
    model, _ = load_checkpoint(args.checkpoint, device)
    agent = NeuralAgent(model, device, deterministic=True)
    for result in evaluate_against_baselines(agent, games=args.games, max_moves=args.max_moves):
        print(f"{result.opponent}: win_rate={result.win_rate:.3f} wins={result.wins} losses={result.losses} draws={result.draws} avg_len={result.avg_game_length:.1f} invalid={result.invalid_actions}")


if __name__ == "__main__":
    main()
