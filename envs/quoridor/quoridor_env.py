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
    ):
        super().__init__()
        self.render_mode = render_mode
        self.engine = QuoridorEngine()
        self.random_walls_range = random_walls_range
        self.move_only = move_only
        self.repeat_penalty = repeat_penalty
        self.opponent_policy = opponent_policy
        self.opponent_randomness = float(opponent_randomness)
        self.position_history = []

        # 0-11: movement actions, including jumps and diagonals
        # 12-75: horizontal walls
        # 76-139: vertical walls
        self.action_space = spaces.Discrete(TOTAL_ACTIONS)

        # Channel 0 stores board values 0/1/2.
        # Channels 1/2 store walls and a walls-left hint in [0, 10].
        self.observation_space = spaces.Box(
            low=0,
            high=10,
            shape=(9, 9, 3),
            dtype=np.int8,
        )

        self.max_steps = 200
        self.current_step = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.engine.reset()
        self.current_step = 0
        self.position_history = [self.engine.p1_pos]
        self._place_random_walls()
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

    def _get_obs(self):
        obs = np.zeros((9, 9, 3), dtype=np.int8)
        obs[:, :, 0] = self.engine.board
        obs[:8, :8, 1] = self.engine.horizontal_walls.astype(np.int8)
        obs[:8, :8, 2] = self.engine.vertical_walls.astype(np.int8)

        # Give the policy simple hints about remaining walls.
        obs[8, 0, 1] = self.engine.walls_left[1]
        obs[8, 1, 1] = self.engine.walls_left[2]
        return obs

    def action_masks(self):
        mask = np.zeros(self.action_space.n, dtype=bool)

        # 1. Movement actions
        valid_moves = self.engine.get_valid_moves(1)
        cx, cy = self.engine.p1_pos
        for i, (dx, dy) in enumerate(MOVES):
            if (cx + dx, cy + dy) in valid_moves:
                mask[i] = True

        if self.move_only:
            return mask

        # 2. Horizontal wall actions
        for i in range(64):
            r, c = divmod(i, 8)
            if self.engine.can_place_wall(1, r, c, "H"):
                mask[H_WALL_OFFSET + i] = True

        # 3. Vertical wall actions
        for i in range(64):
            r, c = divmod(i, 8)
            if self.engine.can_place_wall(1, r, c, "V"):
                mask[V_WALL_OFFSET + i] = True

        return mask

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
            penalty -= 0.45  # immediate backtracking
        if new_pos in self.position_history[-8:]:
            penalty -= 0.12  # short loop
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

    def _opponent_step(self):
        if self.opponent_policy == "none":
            return False
        if self.opponent_policy != "greedy":
            raise ValueError(f"Unknown opponent_policy={self.opponent_policy!r}")

        move = self._choose_greedy_opponent_move()
        if move is None:
            return False
        self._move_player_to(2, move)
        return True

    def _is_p1_win(self):
        return self.engine.p1_pos[1] == 0

    def _is_p2_win(self):
        return self.engine.p2_pos[1] == 8

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
        valid_action = bool(self.action_masks()[action])

        reward = -0.02  # time pressure: win quickly

        if not valid_action:
            reward -= 1.0
        elif action < MOVE_ACTIONS:
            dx, dy = MOVES[action]
            cx, cy = self.engine.p1_pos
            new_pos = (cx + dx, cy + dy)
            self._move_player_to(1, new_pos)
            new_p1_dist = self.engine.get_bfs_distance(new_pos, p1_target_row)

            reward += (prev_p1_dist - new_p1_dist) * 0.30
            if abs(dx) + abs(dy) >= 2:
                reward += 0.05  # tiny bonus for using real jump/diagonal rules when useful
            reward += self._repeat_penalty(new_pos)
            self.position_history.append(new_pos)
        elif action < V_WALL_OFFSET:
            idx = action - H_WALL_OFFSET
            if self.engine.place_wall(1, idx // 8, idx % 8, "H"):
                reward -= 0.02
            else:
                reward -= 1.0
        else:
            idx = action - V_WALL_OFFSET
            if self.engine.place_wall(1, idx // 8, idx % 8, "V"):
                reward -= 0.02
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

            if self._is_p2_win():
                reward -= 12.0
                terminated = True

        truncated = False
        if self.current_step >= self.max_steps and not terminated:
            truncated = True
            reward -= 6.0

        return self._get_obs(), reward, terminated, truncated, {}
