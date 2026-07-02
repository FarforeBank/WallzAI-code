from __future__ import annotations

from wallz_ai.env.action_space import WALL_GRID_SIZE, WallOrientation, square_to_action, wall_to_action
from wallz_ai.env.rules import (
    WallzState,
    can_place_wall,
    legal_pawn_targets,
    opponent,
    shortest_path,
    shortest_path_distance,
)


class GreedyShortestPathBot:
    def act(self, state: WallzState) -> int:
        player = state.current_player
        targets = set(legal_pawn_targets(state, player))
        path = shortest_path(state, player)
        if path is not None:
            for pos in path[1:]:
                if pos in targets:
                    return square_to_action(pos)
        if targets:
            return square_to_action(min(targets, key=lambda pos: _distance_after_move(state, player, pos)))
        raise RuntimeError("No legal pawn move available")


class WallHeuristicBot:
    def __init__(self, min_gain: float = 1.0, max_candidates: int = 128):
        self.greedy = GreedyShortestPathBot()
        self.min_gain = float(min_gain)
        self.max_candidates = int(max_candidates)

    def act(self, state: WallzState) -> int:
        player = state.current_player
        if state.walls_remaining[player] <= 0:
            return self.greedy.act(state)

        base_own = shortest_path_distance(state, player)
        base_enemy = shortest_path_distance(state, opponent(player))
        best_score = self.min_gain
        best_action = None
        for row, col, orientation in _wall_candidates_near_paths(state)[: self.max_candidates]:
            if not can_place_wall(state, player, row, col, orientation):
                continue
            trial = state.clone()
            wall_map = trial.horizontal_walls if orientation == WallOrientation.HORIZONTAL else trial.vertical_walls
            wall_map[row, col] = True
            own_delta = shortest_path_distance(trial, player) - base_own
            enemy_delta = shortest_path_distance(trial, opponent(player)) - base_enemy
            score = enemy_delta - 0.65 * max(0, own_delta)
            if score > best_score:
                best_score = score
                best_action = wall_to_action(row, col, orientation)
        return best_action if best_action is not None else self.greedy.act(state)


def _distance_after_move(state: WallzState, player: int, pos: tuple[int, int]) -> int:
    trial = state.clone()
    trial.pawn_positions[player] = pos
    return shortest_path_distance(trial, player)


def _wall_candidates_near_paths(state: WallzState) -> list[tuple[int, int, WallOrientation]]:
    points = []
    for player in (state.current_player, opponent(state.current_player)):
        points.extend((shortest_path(state, player) or [])[:8])
    points.append((4, 4))
    seen = set()
    candidates = []
    for row, col in points:
        for dr in range(-2, 3):
            for dc in range(-2, 3):
                wr = min(max(row + dr, 0), WALL_GRID_SIZE - 1)
                wc = min(max(col + dc, 0), WALL_GRID_SIZE - 1)
                for orientation in (WallOrientation.HORIZONTAL, WallOrientation.VERTICAL):
                    key = (wr, wc, orientation)
                    if key not in seen:
                        seen.add(key)
                        candidates.append(key)
    candidates.sort(key=lambda item: abs(item[0] - 3.5) + abs(item[1] - 3.5))
    return candidates
