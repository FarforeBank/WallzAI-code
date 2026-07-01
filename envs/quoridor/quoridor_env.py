from collections import deque

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from envs.quoridor.engine import QuoridorEngine


# 0-3: normal moves, 4-7: straight jumps, 8-11: diagonals around opponent.
MOVES = [
    (0, -1),   # 0 UP
    (0, 1),    # 1 DOWN
    (-1, 0),   # 2 LEFT
    (1, 0),    # 3 RIGHT
    (0, -2),   # 4 JUMP_UP
    (0, 2),    # 5 JUMP_DOWN
    (-2, 0),   # 6 JUMP_LEFT
    (2, 0),    # 7 JUMP_RIGHT
    (-1, -1),  # 8 UP_LEFT
    (1, -1),   # 9 UP_RIGHT
    (-1, 1),   # 10 DOWN_LEFT
    (1, 1),    # 11 DOWN_RIGHT
]
MOVE_ACTIONS = len(MOVES)
H_WALL_OFFSET = MOVE_ACTIONS
V_WALL_OFFSET = H_WALL_OFFSET + 64
TOTAL_ACTIONS = V_WALL_OFFSET + 64
ORTHOGONAL_DIRS = [(0, -1), (0, 1), (-1, 0), (1, 0)]


class QuoridorEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(
        self,
        render_mode=None,
        random_walls_range=(0, 0),
        move_only=False,
        repeat_penalty=False,
        opponent_policy="none",
        opponent_randomness=0.0,
        smart_observation=True,
        wall_reward=True,
        wall_candidate_limit=32,
        opponent_start_advantage_range=(0, 0),
        defensive_wall_reward=False,
        opponent_wall_probability=0.0,
        self_trap_penalty=False,
    ):
        super().__init__()
        self.render_mode = render_mode
        self.engine = QuoridorEngine()
        self.random_walls_range = random_walls_range
        self.move_only = move_only
        self.repeat_penalty = repeat_penalty
        self.opponent_policy = opponent_policy
        self.opponent_randomness = float(opponent_randomness)
        self.smart_observation = smart_observation
        self.wall_reward = wall_reward
        self.wall_candidate_limit = wall_candidate_limit
        self.opponent_start_advantage_range = opponent_start_advantage_range
        self.defensive_wall_reward = defensive_wall_reward
        self.opponent_wall_probability = float(opponent_wall_probability)
        self.self_trap_penalty = self_trap_penalty
        self.position_history = []

        self.action_space = spaces.Discrete(TOTAL_ACTIONS)
        self.observation_space = spaces.Box(
            low=0,
            high=10,
            shape=(9, 9, 5),
            dtype=np.int8,
        )

        self.max_steps = 200
        self.current_step = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.engine.reset()
        self.current_step = 0
        self._place_random_walls()
        self._apply_opponent_start_advantage()
        self.position_history = [self.engine.p1_pos]
        return self._get_obs(), {}

    def _random_wall_count(self):
        low, high = self.random_walls_range
        low = max(0, int(low))
        high = max(low, int(high))
        if high == 0:
            return 0
        return int(self.np_random.integers(low, high + 1))

    def _place_random_walls(self):
        target_count = self._random_wall_count()
        if target_count <= 0:
            return

        placed = 0
        attempts = 0
        max_attempts = target_count * 80

        while placed < target_count and attempts < max_attempts:
            attempts += 1
            r = int(self.np_random.integers(0, 8))
            c = int(self.np_random.integers(0, 8))
            orientation = "H" if int(self.np_random.integers(0, 2)) == 0 else "V"
            if self.engine.place_wall(2, r, c, orientation):
                placed += 1

    def _apply_opponent_start_advantage(self):
        low, high = self.opponent_start_advantage_range
        low = max(0, int(low))
        high = max(low, int(high))
        if high <= 0:
            return

        steps = int(self.np_random.integers(low, high + 1))
        for _ in range(steps):
            move = self._choose_greedy_opponent_move()
            if move is None:
                break
            self._move_player_to(2, move)
            if self._is_p2_win():
                break

    def _distance_map_to_row(self, target_row):
        distances = np.full((9, 9), 10, dtype=np.int8)
        queue = deque()

        for x in range(9):
            distances[target_row, x] = 0
            queue.append((x, target_row))

        while queue:
            x, y = queue.popleft()
            next_dist = int(distances[y, x]) + 1
            if next_dist > 10:
                continue
            for dx, dy in ORTHOGONAL_DIRS:
                nx, ny = x + dx, y + dy
                if not self.engine._in_bounds(nx, ny):
                    continue
                if self.engine.has_wall_between(x, y, nx, ny):
                    continue
                if next_dist < distances[ny, nx]:
                    distances[ny, nx] = next_dist
                    queue.append((nx, ny))

        return distances

    def _get_obs(self):
        obs = np.zeros((9, 9, 5), dtype=np.int8)
        obs[:, :, 0] = self.engine.board
        obs[:8, :8, 1] = self.engine.horizontal_walls.astype(np.int8)
        obs[:8, :8, 2] = self.engine.vertical_walls.astype(np.int8)
        obs[:, :, 3] = self._distance_map_to_row(0)
        obs[:, :, 4] = self._distance_map_to_row(8)
        obs[8, 0, 1] = self.engine.walls_left[1]
        obs[8, 1, 1] = self.engine.walls_left[2]
        return obs

    def _wall_action_to_parts(self, action):
        if H_WALL_OFFSET <= action < V_WALL_OFFSET:
            idx = action - H_WALL_OFFSET
            return idx // 8, idx % 8, "H"
        if V_WALL_OFFSET <= action < TOTAL_ACTIONS:
            idx = action - V_WALL_OFFSET
            return idx // 8, idx % 8, "V"
        return None

    def _nearby_wall_candidates(self):
        limit = self.wall_candidate_limit
        if limit is None or limit >= 128:
            return list(range(H_WALL_OFFSET, TOTAL_ACTIONS))
        if self.engine.walls_left[1] <= 0:
            return []

        p1x, p1y = self.engine.p1_pos
        p2x, p2y = self.engine.p2_pos
        anchors = [
            (p2x, p2y),
            (p2x, min(8, p2y + 1)),
            (p2x, min(8, p2y + 2)),
            (p2x, max(0, p2y - 1)),
            (p1x, p1y),
            (4, 4),
        ]

        actions = set()
        for ax, ay in anchors:
            for dy in range(-3, 4):
                for dx in range(-3, 4):
                    r = max(0, min(7, ay + dy))
                    c = max(0, min(7, ax + dx))
                    idx = r * 8 + c
                    actions.add(H_WALL_OFFSET + idx)
                    actions.add(V_WALL_OFFSET + idx)

        def score(action):
            r, c, orientation = self._wall_action_to_parts(action)
            wx = c + 0.5
            wy = r + 0.5
            near_opponent = abs(wx - p2x) + abs(wy - p2y)
            near_us = abs(wx - p1x) + abs(wy - p1y)
            center_bias = abs(wx - 4) * 0.15
            danger_bias = 0.0
            if self.defensive_wall_reward:
                danger_bias = max(0, wy - p2y) * 0.05
            orientation_bias = 0.0 if orientation == "H" else 0.08
            return near_opponent + 0.25 * near_us + center_bias + danger_bias + orientation_bias

        return sorted(actions, key=score)[: int(limit)]

    def action_masks(self):
        mask = np.zeros(self.action_space.n, dtype=bool)

        valid_moves = self.engine.get_valid_moves(1)
        cx, cy = self.engine.p1_pos
        for i, (dx, dy) in enumerate(MOVES):
            if (cx + dx, cy + dy) in valid_moves:
                mask[i] = True

        if self.move_only:
            return mask

        for action in self._nearby_wall_candidates():
            r, c, orientation = self._wall_action_to_parts(action)
            if self.engine.can_place_wall(1, r, c, orientation):
                mask[action] = True

        return mask

    def _is_action_valid(self, action):
        if action < MOVE_ACTIONS:
            cx, cy = self.engine.p1_pos
            dx, dy = MOVES[action]
            return (cx + dx, cy + dy) in self.engine.get_valid_moves(1)

        if self.move_only:
            return False

        parts = self._wall_action_to_parts(action)
        if parts is None:
            return False
        r, c, orientation = parts
        return self.engine.can_place_wall(1, r, c, orientation)

    def _move_player_to(self, player_id, target_pos):
        if player_id == 1:
            old_x, old_y = self.engine.p1_pos
            self.engine.board[old_y, old_x] = 0
            self.engine.p1_pos = target_pos
            self.engine.board[target_pos[1], target_pos[0]] = 1
        else:
            old_x, old_y = self.engine.p2_pos
            self.engine.board[old_y, old_x] = 0
            self.engine.p2_pos = target_pos
            self.engine.board[target_pos[1], target_pos[0]] = 2

    def _repeat_penalty(self, new_pos):
        if not self.repeat_penalty:
            return 0.0

        penalty = 0.0
        if len(self.position_history) >= 2 and new_pos == self.position_history[-2]:
            penalty -= 0.45
        if new_pos in self.position_history[-8:]:
            penalty -= 0.12
        return penalty

    def _choose_greedy_opponent_move(self):
        valid_moves = self.engine.get_valid_moves(2)
        if not valid_moves:
            return None

        if self.opponent_randomness > 0 and self.np_random.random() < self.opponent_randomness:
            return valid_moves[int(self.np_random.integers(0, len(valid_moves)))]

        cx, cy = self.engine.p2_pos

        def score(pos):
            dist = self.engine.get_bfs_distance(pos, 8)
            dx = abs(pos[0] - 4) * 0.05
            progress_bonus = -0.15 if pos[1] > cy else 0.0
            return dist + dx + progress_bonus

        return min(valid_moves, key=score)

    def _opponent_wall_candidates(self):
        if self.engine.walls_left[2] <= 0:
            return []

        p1x, p1y = self.engine.p1_pos
        anchors = [
            (p1x, p1y),
            (p1x, max(0, p1y - 1)),
            (p1x, max(0, p1y - 2)),
            (4, p1y),
        ]
        actions = set()
        for ax, ay in anchors:
            for dy in range(-2, 3):
                for dx in range(-3, 4):
                    r = max(0, min(7, ay + dy))
                    c = max(0, min(7, ax + dx))
                    idx = r * 8 + c
                    actions.add(H_WALL_OFFSET + idx)
                    actions.add(V_WALL_OFFSET + idx)
        return actions

    def _evaluate_opponent_wall_action(self, action, prev_p1_dist=None, prev_p2_dist=None):
        parts = self._wall_action_to_parts(action)
        if parts is None:
            return None
        r, c, orientation = parts
        if not self.engine._wall_slot_is_free(r, c, orientation):
            return None

        prev_p1_dist = self.engine.get_bfs_distance(self.engine.p1_pos, 0) if prev_p1_dist is None else prev_p1_dist
        prev_p2_dist = self.engine.get_bfs_distance(self.engine.p2_pos, 8) if prev_p2_dist is None else prev_p2_dist

        if orientation == "H":
            self.engine.horizontal_walls[r, c] = True
        else:
            self.engine.vertical_walls[r, c] = True
        new_p1_dist = self.engine.get_bfs_distance(self.engine.p1_pos, 0)
        new_p2_dist = self.engine.get_bfs_distance(self.engine.p2_pos, 8)
        if orientation == "H":
            self.engine.horizontal_walls[r, c] = False
        else:
            self.engine.vertical_walls[r, c] = False

        if new_p1_dist == 999 or new_p2_dist == 999:
            return None

        p1_delta = new_p1_dist - prev_p1_dist
        p2_delta = new_p2_dist - prev_p2_dist
        score = p1_delta * 0.95 - max(0, p2_delta) * 0.55
        if prev_p1_dist <= 4:
            score += p1_delta * 0.30
        if p1_delta <= 0:
            score -= 0.15
        return score, p1_delta, p2_delta, (r, c, orientation)

    def _choose_greedy_opponent_wall(self):
        if self.engine.walls_left[2] <= 0:
            return None

        prev_p1_dist = self.engine.get_bfs_distance(self.engine.p1_pos, 0)
        prev_p2_dist = self.engine.get_bfs_distance(self.engine.p2_pos, 8)
        best = None
        best_score = 0.05

        for action in self._opponent_wall_candidates():
            evaluated = self._evaluate_opponent_wall_action(action, prev_p1_dist, prev_p2_dist)
            if evaluated is None:
                continue
            score, _, _, wall = evaluated
            if score > best_score:
                best_score = score
                best = wall

        return best

    def _self_trap_penalty_value(self, current_p1_dist):
        if not self.self_trap_penalty or self.engine.walls_left[2] <= 0:
            return 0.0

        prev_p2_dist = self.engine.get_bfs_distance(self.engine.p2_pos, 8)
        best_delta = 0
        best_score = 0.0
        for action in self._opponent_wall_candidates():
            evaluated = self._evaluate_opponent_wall_action(action, current_p1_dist, prev_p2_dist)
            if evaluated is None:
                continue
            score, p1_delta, _, _ = evaluated
            if p1_delta > best_delta or score > best_score:
                best_delta = max(best_delta, p1_delta)
                best_score = max(best_score, score)

        if best_delta <= 0:
            return 0.0

        # Penalize moves that allow the opponent to close a corridor/trap next turn.
        # The penalty is strongest when we are already near the finish, where one bad
        # self-trap often loses the game in Wallz.
        penalty = min(1.35, best_delta * 0.24 + max(0.0, best_score) * 0.08)
        if current_p1_dist <= 5:
            penalty *= 1.25
        return -penalty

    def _opponent_step(self):
        if self.opponent_policy == "none":
            return False
        if self.opponent_policy != "greedy":
            raise ValueError(f"Unknown opponent_policy={self.opponent_policy!r}")

        wall_probability = self.opponent_wall_probability
        if self.engine.get_bfs_distance(self.engine.p1_pos, 0) <= 4:
            wall_probability = max(wall_probability, min(0.75, wall_probability + 0.25))

        if wall_probability > 0 and self.np_random.random() < wall_probability:
            wall = self._choose_greedy_opponent_wall()
            if wall is not None:
                r, c, orientation = wall
                if self.engine.place_wall(2, r, c, orientation):
                    return True

        move = self._choose_greedy_opponent_move()
        if move is None:
            return False
        self._move_player_to(2, move)
        return True

    def _is_p1_win(self):
        return self.engine.p1_pos[1] == 0

    def _is_p2_win(self):
        return self.engine.p2_pos[1] == 8

    def _wall_action_reward(self, prev_p1_dist, prev_p2_dist):
        if not self.wall_reward:
            return -0.02

        new_p1_dist = self.engine.get_bfs_distance(self.engine.p1_pos, 0)
        new_p2_dist = self.engine.get_bfs_distance(self.engine.p2_pos, 8)
        opponent_delta = new_p2_dist - prev_p2_dist
        own_delta = new_p1_dist - prev_p1_dist

        if self.defensive_wall_reward:
            reward = -0.02
            reward += max(0, opponent_delta) * 0.95
            reward -= max(0, own_delta) * 0.30

            if prev_p2_dist <= 4:
                if opponent_delta > 0:
                    reward += 0.45 + (4 - prev_p2_dist) * 0.12
                else:
                    reward -= 0.35
            if own_delta > opponent_delta:
                reward -= 0.30
            return reward

        reward = -0.08
        reward += max(0, opponent_delta) * 0.45
        reward -= max(0, own_delta) * 0.35

        if opponent_delta <= 0 and own_delta <= 0:
            reward -= 0.22
        if own_delta > opponent_delta:
            reward -= 0.25

        return reward

    def step(self, action):
        action = int(action)
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action}; expected 0..{self.action_space.n - 1}")

        self.current_step += 1
        p1_target_row = 0
        p2_target_row = 8
        prev_p1_pos = self.engine.p1_pos
        prev_p1_dist = self.engine.get_bfs_distance(prev_p1_pos, p1_target_row)
        prev_p2_dist = self.engine.get_bfs_distance(self.engine.p2_pos, p2_target_row)
        valid_action = self._is_action_valid(action)

        reward = -0.02

        if not valid_action:
            reward -= 1.0
        elif action < MOVE_ACTIONS:
            dx, dy = MOVES[action]
            cx, cy = self.engine.p1_pos
            new_pos = (cx + dx, cy + dy)
            self._move_player_to(1, new_pos)
            new_p1_dist = self.engine.get_bfs_distance(new_pos, p1_target_row)

            reward += (prev_p1_dist - new_p1_dist) * 0.35
            if abs(dx) + abs(dy) >= 2:
                reward += 0.08
            reward += self._repeat_penalty(new_pos)
            reward += self._self_trap_penalty_value(new_p1_dist)

            if self.defensive_wall_reward and prev_p2_dist <= 3 and self.engine.walls_left[1] > 0 and new_p1_dist > 0:
                reward -= 0.55 + (3 - prev_p2_dist) * 0.20

            self.position_history.append(new_pos)
        elif action < V_WALL_OFFSET:
            idx = action - H_WALL_OFFSET
            if self.engine.place_wall(1, idx // 8, idx % 8, "H"):
                reward += self._wall_action_reward(prev_p1_dist, prev_p2_dist)
            else:
                reward -= 1.0
        else:
            idx = action - V_WALL_OFFSET
            if self.engine.place_wall(1, idx // 8, idx % 8, "V"):
                reward += self._wall_action_reward(prev_p1_dist, prev_p2_dist)
            else:
                reward -= 1.0

        terminated = False
        if self._is_p1_win():
            reward += 12.0
            terminated = True

        if not terminated:
            moved = self._opponent_step()
            if moved:
                new_p2_dist = self.engine.get_bfs_distance(self.engine.p2_pos, p2_target_row)
                reward -= max(0, prev_p2_dist - new_p2_dist) * 0.10

                if self.defensive_wall_reward and new_p2_dist <= 2 and self.engine.walls_left[1] > 0:
                    reward -= 0.30

            if self._is_p2_win():
                reward -= 12.0
                terminated = True

        truncated = False
        if self.current_step >= self.max_steps and not terminated:
            truncated = True
            reward -= 6.0

        return self._get_obs(), reward, terminated, truncated, {}
