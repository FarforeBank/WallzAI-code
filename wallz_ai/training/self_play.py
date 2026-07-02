from __future__ import annotations

from dataclasses import dataclass

import torch

from wallz_ai.agents.model import WallzPolicyValueNet
from wallz_ai.agents.ppo import sample_action
from wallz_ai.env.wallz_env import WallzEnv
from .replay_buffer import RolloutBuffer


@dataclass
class SelfPlayStats:
    games: int = 0
    steps: int = 0
    wins_p0: int = 0
    wins_p1: int = 0
    draws: int = 0
    invalid_actions: int = 0
    walls_used: int = 0
    entropy_sum: float = 0.0

    @property
    def avg_game_length(self) -> float:
        return self.steps / max(1, self.games)

    @property
    def avg_walls_used(self) -> float:
        return self.walls_used / max(1, self.games)

    @property
    def policy_entropy(self) -> float:
        return self.entropy_sum / max(1, self.steps)


def play_self_play_games(model: WallzPolicyValueNet, device: torch.device, games: int, max_moves: int, dense_shaping: bool = False, shaping_coef: float = 0.03) -> tuple[RolloutBuffer, SelfPlayStats]:
    buffer = RolloutBuffer()
    stats = SelfPlayStats(games=games)
    for _ in range(games):
        env = WallzEnv(max_moves=max_moves, dense_shaping=dense_shaping, shaping_coef=shaping_coef)
        obs, _ = env.reset()
        start_walls = sum(env.state.walls_remaining)
        while not env.state.terminal:
            mask = env.legal_action_mask()
            action, log_prob, value, entropy = sample_action(model, obs, mask, device, deterministic=False)
            next_obs, reward, terminal, _, _ = env.step(action)
            buffer.add(obs, mask, action, log_prob, value, reward)
            stats.steps += 1
            stats.entropy_sum += entropy
            obs = next_obs
            if terminal:
                break
        if env.state.winner == 0:
            stats.wins_p0 += 1
        elif env.state.winner == 1:
            stats.wins_p1 += 1
        else:
            stats.draws += 1
        stats.invalid_actions += env.invalid_action_count
        stats.walls_used += start_walls - sum(env.state.walls_remaining)
    return buffer, stats
