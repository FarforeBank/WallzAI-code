from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import torch

from wallz_ai.agents.heuristic_bot import GreedyShortestPathBot, WallHeuristicBot
from wallz_ai.agents.model import WallzPolicyValueNet, select_device
from wallz_ai.agents.ppo import sample_action
from wallz_ai.agents.random_bot import RandomLegalBot
from wallz_ai.env.wallz_env import WallzEnv


class Agent(Protocol):
    def act(self, state): ...


class NeuralAgent:
    def __init__(self, model: WallzPolicyValueNet, device: torch.device, deterministic: bool = True):
        self.model = model
        self.device = device
        self.deterministic = deterministic

    def act(self, state) -> int:
        env = WallzEnv(max_moves=state.max_moves)
        env.state = state.clone()
        action, _, _, _ = sample_action(self.model, env.observation(), env.legal_action_mask(), self.device, self.deterministic)
        return action


@dataclass
class EvalResult:
    opponent: str
    games: int
    wins: int
    losses: int
    draws: int
    avg_game_length: float
    invalid_actions: int

    @property
    def win_rate(self) -> float:
        return self.wins / max(1, self.games)


def play_match(agent0: Agent, agent1: Agent, max_moves: int = 300) -> tuple[int | None, int, int]:
    env = WallzEnv(max_moves=max_moves)
    env.reset()
    invalid = 0
    agents = [agent0, agent1]
    while not env.state.terminal:
        actor = env.state.current_player
        try:
            env.step(agents[actor].act(env.state.clone()))
        except Exception:
            invalid += 1
            env.state.winner = 1 - actor
            break
    return env.state.winner, env.state.move_count, invalid


def evaluate_agent(agent: Agent, opponent_agent: Agent, games: int = 50, max_moves: int = 300, opponent_name: str = "opponent") -> EvalResult:
    wins = losses = draws = invalid = total_len = 0
    for game in range(games):
        if game % 2 == 0:
            winner, length, bad = play_match(agent, opponent_agent, max_moves)
            agent_player = 0
        else:
            winner, length, bad = play_match(opponent_agent, agent, max_moves)
            agent_player = 1
        total_len += length
        invalid += bad
        if winner is None:
            draws += 1
        elif winner == agent_player:
            wins += 1
        else:
            losses += 1
    return EvalResult(opponent_name, games, wins, losses, draws, total_len / max(1, games), invalid)


def evaluate_against_baselines(agent: Agent, games: int = 50, max_moves: int = 300, seed: int = 0) -> list[EvalResult]:
    return [
        evaluate_agent(agent, RandomLegalBot(seed), games, max_moves, "RandomLegalBot"),
        evaluate_agent(agent, GreedyShortestPathBot(), games, max_moves, "GreedyShortestPathBot"),
        evaluate_agent(agent, WallHeuristicBot(), games, max_moves, "WallHeuristicBot"),
    ]


def load_checkpoint(path: str | Path, device: torch.device | None = None) -> tuple[WallzPolicyValueNet, dict]:
    device = select_device("auto") if device is None else device
    checkpoint = torch.load(path, map_location=device)
    cfg = checkpoint.get("config", {})
    model_cfg = cfg.get("model", {}) if isinstance(cfg, dict) else {}
    model = WallzPolicyValueNet(channels=int(model_cfg.get("channels", 64)), residual_blocks=int(model_cfg.get("residual_blocks", 3))).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint
