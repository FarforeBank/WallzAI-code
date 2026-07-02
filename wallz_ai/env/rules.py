"""Pure Wallz/Quoridor-like rules independent from the browser UI."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np

from .action_space import (
    ACTION_SIZE,
    BOARD_SIZE,
    WALL_GRID_SIZE,
    WallOrientation,
    action_to_square,
    action_to_wall,
    square_to_action,
    wall_to_action,
)

Position = Tuple[int, int]
ORTHOGONAL_DIRS: tuple[Position, ...] = ((-1, 0), (1, 0), (0, -1), (0, 1))


@dataclass
class WallzState:
    """Serializable board state.

    player 0 starts at row 8 and wins by reaching row 0.
    player 1 starts at row 0 and wins by reaching row 8.
    """

    pawn_positions: list[Position] = field(default_factory=lambda: [(8, 4), (0, 4)])
    horizontal_walls: np.ndarray = field(default_factory=lambda: np.zeros((WALL_GRID_SIZE, WALL_GRID_SIZE), dtype=bool))
    vertical_walls: np.ndarray = field(default_factory=lambda: np.zeros((WALL_GRID_SIZE, WALL_GRID_SIZE), dtype=bool))
    walls_remaining: list[int] = field(default_factory=lambda: [10, 10])
    current_player: int = 0
    move_count: int = 0
    max_moves: int = 300
    winner: Optional[int] = None
    history: list[int] = field(default_factory=list)

    def clone(self) -> "WallzState":
        return WallzState(
            pawn_positions=list(self.pawn_positions),
            horizontal_walls=self.horizontal_walls.copy(),
            vertical_walls=self.vertical_walls.copy(),
            walls_remaining=list(self.walls_remaining),
            current_player=int(self.current_player),
            move_count=int(self.move_count),
            max_moves=int(self.max_moves),
            winner=self.winner,
            history=list(self.history),
        )

    @property
    def terminal(self) -> bool:
        return self.winner is not None or self.move_count >= self.max_moves


def goal_row(player: int) -> int:
    return 0 if player == 0 else BOARD_SIZE - 1


def opponent(player: int) -> int:
    return 1 - player


def in_bounds(pos: Position) -> bool:
    row, col = pos
    return 0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE


def _wall_grid_bounds(row: int, col: int) -> bool:
    return 0 <= row < WALL_GRID_SIZE and 0 <= col < WALL_GRID_SIZE


def wall_blocks_edge(state: WallzState, a: Position, b: Position) -> bool:
    ar, ac = a
    br, bc = b
    if not in_bounds(a) or not in_bounds(b):
        return True
    if abs(ar - br) + abs(ac - bc) != 1:
        return True

    if ac == bc:
        top_row = min(ar, br)
        col = ac
        if top_row >= WALL_GRID_SIZE:
            return False
        if col < WALL_GRID_SIZE and state.horizontal_walls[top_row, col]:
            return True
        if col > 0 and state.horizontal_walls[top_row, col - 1]:
            return True
        return False

    row = ar
    left_col = min(ac, bc)
    if left_col >= WALL_GRID_SIZE:
        return False
    if row < WALL_GRID_SIZE and state.vertical_walls[row, left_col]:
        return True
    if row > 0 and state.vertical_walls[row - 1, left_col]:
        return True
    return False


def legal_pawn_targets(state: WallzState, player: Optional[int] = None) -> list[Position]:
    player = state.current_player if player is None else player
    current = state.pawn_positions[player]
    enemy = state.pawn_positions[opponent(player)]
    moves: list[Position] = []

    for dr, dc in ORTHOGONAL_DIRS:
        adjacent = (current[0] + dr, current[1] + dc)
        if not in_bounds(adjacent) or wall_blocks_edge(state, current, adjacent):
            continue

        if adjacent != enemy:
            moves.append(adjacent)
            continue

        jump = (adjacent[0] + dr, adjacent[1] + dc)
        if in_bounds(jump) and not wall_blocks_edge(state, adjacent, jump):
            moves.append(jump)
            continue

        perpendicular = ((0, -1), (0, 1)) if dr != 0 else ((-1, 0), (1, 0))
        for pdr, pdc in perpendicular:
            diagonal = (adjacent[0] + pdr, adjacent[1] + pdc)
            if in_bounds(diagonal) and not wall_blocks_edge(state, adjacent, diagonal):
                moves.append(diagonal)

    unique: list[Position] = []
    seen: set[Position] = set()
    for move in moves:
        if move not in seen and move != enemy:
            unique.append(move)
            seen.add(move)
    return unique


def _wall_slot_is_free(state: WallzState, row: int, col: int, orientation: WallOrientation | str) -> bool:
    orientation = WallOrientation(orientation)
    if not _wall_grid_bounds(row, col):
        return False

    if orientation == WallOrientation.HORIZONTAL:
        if state.horizontal_walls[row, col]:
            return False
        if col > 0 and state.horizontal_walls[row, col - 1]:
            return False
        if col < WALL_GRID_SIZE - 1 and state.horizontal_walls[row, col + 1]:
            return False
        if state.vertical_walls[row, col]:
            return False
        return True

    if state.vertical_walls[row, col]:
        return False
    if row > 0 and state.vertical_walls[row - 1, col]:
        return False
    if row < WALL_GRID_SIZE - 1 and state.vertical_walls[row + 1, col]:
        return False
    if state.horizontal_walls[row, col]:
        return False
    return True


def shortest_path(state: WallzState, player: int) -> Optional[list[Position]]:
    start = state.pawn_positions[player]
    target = goal_row(player)
    queue: deque[Position] = deque([start])
    parent: dict[Position, Optional[Position]] = {start: None}

    while queue:
        pos = queue.popleft()
        if pos[0] == target:
            path: list[Position] = []
            cur: Optional[Position] = pos
            while cur is not None:
                path.append(cur)
                cur = parent[cur]
            return list(reversed(path))
        for dr, dc in ORTHOGONAL_DIRS:
            nxt = (pos[0] + dr, pos[1] + dc)
            if not in_bounds(nxt) or nxt in parent or wall_blocks_edge(state, pos, nxt):
                continue
            parent[nxt] = pos
            queue.append(nxt)
    return None


def shortest_path_distance(state: WallzState, player: int) -> int:
    path = shortest_path(state, player)
    return 999 if path is None else len(path) - 1


def distance_map_to_goal(state: WallzState, player: int, cap: int = 32) -> np.ndarray:
    target = goal_row(player)
    distances = np.full((BOARD_SIZE, BOARD_SIZE), cap, dtype=np.float32)
    queue: deque[Position] = deque()
    for col in range(BOARD_SIZE):
        distances[target, col] = 0
        queue.append((target, col))

    while queue:
        pos = queue.popleft()
        next_distance = distances[pos] + 1
        for dr, dc in ORTHOGONAL_DIRS:
            nxt = (pos[0] + dr, pos[1] + dc)
            if not in_bounds(nxt) or wall_blocks_edge(state, pos, nxt):
                continue
            if next_distance < distances[nxt]:
                distances[nxt] = next_distance
                queue.append(nxt)
    return distances


def has_path(state: WallzState, player: int) -> bool:
    return shortest_path(state, player) is not None


def can_place_wall(state: WallzState, player: int, row: int, col: int, orientation: WallOrientation | str) -> bool:
    orientation = WallOrientation(orientation)
    if state.terminal or state.walls_remaining[player] <= 0:
        return False
    if not _wall_slot_is_free(state, row, col, orientation):
        return False

    trial = state.clone()
    wall_map = trial.horizontal_walls if orientation == WallOrientation.HORIZONTAL else trial.vertical_walls
    wall_map[row, col] = True
    return has_path(trial, 0) and has_path(trial, 1)


def legal_action_mask(state: WallzState) -> np.ndarray:
    mask = np.zeros(ACTION_SIZE, dtype=bool)
    if state.terminal:
        return mask

    player = state.current_player
    for target in legal_pawn_targets(state, player):
        mask[square_to_action(target)] = True

    if state.walls_remaining[player] > 0:
        for row in range(WALL_GRID_SIZE):
            for col in range(WALL_GRID_SIZE):
                if can_place_wall(state, player, row, col, WallOrientation.HORIZONTAL):
                    mask[wall_to_action(row, col, WallOrientation.HORIZONTAL)] = True
                if can_place_wall(state, player, row, col, WallOrientation.VERTICAL):
                    mask[wall_to_action(row, col, WallOrientation.VERTICAL)] = True
    return mask


def is_legal_action(state: WallzState, action: int) -> bool:
    if not (0 <= int(action) < ACTION_SIZE):
        return False
    return bool(legal_action_mask(state)[int(action)])


def apply_action(state: WallzState, action: int) -> WallzState:
    action = int(action)
    if not is_legal_action(state, action):
        raise ValueError(f"Illegal Wallz action {action}")

    next_state = state.clone()
    player = next_state.current_player

    if action < BOARD_SIZE * BOARD_SIZE:
        next_state.pawn_positions[player] = action_to_square(action)
    else:
        row, col, orientation = action_to_wall(action)
        wall_map = next_state.horizontal_walls if orientation == WallOrientation.HORIZONTAL else next_state.vertical_walls
        wall_map[row, col] = True
        next_state.walls_remaining[player] -= 1

    next_state.history.append(action)
    next_state.move_count += 1
    if next_state.pawn_positions[player][0] == goal_row(player):
        next_state.winner = player
    else:
        next_state.current_player = opponent(player)
    return next_state


def legal_actions(state: WallzState) -> list[int]:
    return np.flatnonzero(legal_action_mask(state)).astype(int).tolist()
