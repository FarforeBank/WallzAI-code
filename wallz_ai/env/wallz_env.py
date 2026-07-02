"""Local Wallz environment with action masks."""

from __future__ import annotations

from typing import Optional

import numpy as np

from .action_space import ACTION_SIZE, BOARD_SIZE, WALL_GRID_SIZE
from .rules import (
    WallzState,
    apply_action,
    distance_map_to_goal,
    goal_row,
    legal_action_mask,
    legal_actions,
    opponent,
    shortest_path_distance,
)

OBS_CHANNELS = 11


class WallzEnv:
    action_size = ACTION_SIZE
    observation_shape = (OBS_CHANNELS, BOARD_SIZE, BOARD_SIZE)

    def __init__(self, max_moves: int = 300, dense_shaping: bool = False, shaping_coef: float = 0.03):
        self.max_moves = int(max_moves)
        self.dense_shaping = bool(dense_shaping)
        self.shaping_coef = float(shaping_coef)
        self.state = WallzState(max_moves=self.max_moves)
        self.invalid_action_count = 0

    def reset(self, seed: Optional[int] = None) -> tuple[np.ndarray, dict]:
        if seed is not None:
            np.random.seed(seed)
        self.state = WallzState(max_moves=self.max_moves)
        self.invalid_action_count = 0
        return self.observation(), self.info()

    def clone(self) -> "WallzEnv":
        env = WallzEnv(self.max_moves, self.dense_shaping, self.shaping_coef)
        env.state = self.state.clone()
        env.invalid_action_count = self.invalid_action_count
        return env

    def legal_action_mask(self) -> np.ndarray:
        return legal_action_mask(self.state)

    def legal_actions(self) -> list[int]:
        return legal_actions(self.state)

    def observation(self) -> np.ndarray:
        state = self.state
        player = state.current_player
        enemy = opponent(player)
        obs = np.zeros(self.observation_shape, dtype=np.float32)
        pr, pc = state.pawn_positions[player]
        er, ec = state.pawn_positions[enemy]
        obs[0, pr, pc] = 1.0
        obs[1, er, ec] = 1.0
        obs[2, :WALL_GRID_SIZE, :WALL_GRID_SIZE] = state.horizontal_walls.astype(np.float32)
        obs[3, :WALL_GRID_SIZE, :WALL_GRID_SIZE] = state.vertical_walls.astype(np.float32)
        obs[4, goal_row(player), :] = 1.0
        obs[5, goal_row(enemy), :] = 1.0
        obs[6, :, :] = state.walls_remaining[player] / 10.0
        obs[7, :, :] = state.walls_remaining[enemy] / 10.0
        obs[8, :, :] = float(player)
        obs[9, :, :] = np.minimum(distance_map_to_goal(state, player), 32) / 32.0
        obs[10, :, :] = np.minimum(distance_map_to_goal(state, enemy), 32) / 32.0
        return obs

    def info(self) -> dict:
        state = self.state
        player = state.current_player
        return {
            "current_player": player,
            "terminal": state.terminal,
            "winner": state.winner,
            "legal_action_mask": self.legal_action_mask(),
            "walls_remaining": tuple(state.walls_remaining),
            "move_count": state.move_count,
            "shortest_path_current": shortest_path_distance(state, player) if not state.terminal else None,
            "shortest_path_opponent": shortest_path_distance(state, opponent(player)) if not state.terminal else None,
        }

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        actor = self.state.current_player
        before_actor = shortest_path_distance(self.state, actor)
        before_enemy = shortest_path_distance(self.state, opponent(actor))
        mask = self.legal_action_mask()
        if action < 0 or action >= ACTION_SIZE or not mask[int(action)]:
            self.invalid_action_count += 1
            raise ValueError(f"Illegal Wallz action selected: {action}")
        self.state = apply_action(self.state, int(action))
        reward = 0.0
        if self.state.winner is not None:
            reward = 1.0 if self.state.winner == actor else -1.0
        elif self.dense_shaping:
            after_actor = shortest_path_distance(self.state, actor)
            after_enemy = shortest_path_distance(self.state, opponent(actor))
            reward += self.shaping_coef * float((before_actor - after_actor) + (after_enemy - before_enemy))
        terminal = self.state.terminal
        truncated = self.state.winner is None and self.state.move_count >= self.state.max_moves
        return self.observation(), float(reward), terminal, truncated, self.info()

    def render_ascii(self) -> str:
        rows = []
        for row in range(BOARD_SIZE):
            cells = []
            for col in range(BOARD_SIZE):
                pos = (row, col)
                if pos == self.state.pawn_positions[0]:
                    cells.append("A")
                elif pos == self.state.pawn_positions[1]:
                    cells.append("B")
                else:
                    cells.append(".")
            rows.append(" ".join(cells))
        return "\n".join(rows)
