import gymnasium as gym
from gymnasium import spaces
import numpy as np

from envs.quoridor.engine import QuoridorEngine


class QuoridorEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode
        self.engine = QuoridorEngine()

        # 0-3: movement actions
        # 4-67: horizontal walls
        # 68-131: vertical walls
        self.action_space = spaces.Discrete(132)

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
        return self._get_obs(), {}

    def _get_obs(self):
        obs = np.zeros((9, 9, 3), dtype=np.int8)
        obs[:, :, 0] = self.engine.board
        obs[:8, :8, 1] = self.engine.horizontal_walls.astype(np.int8)
        obs[:8, :8, 2] = self.engine.vertical_walls.astype(np.int8)

        # Give the policy a simple hint about remaining walls.
        obs[8, 0, 1] = self.engine.walls_left[1]
        return obs

    def action_masks(self):
        mask = np.zeros(self.action_space.n, dtype=bool)

        # 1. Movement actions
        valid_moves = self.engine.get_valid_moves(1)
        moves = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # Up, Down, Left, Right
        cx, cy = self.engine.p1_pos
        for i, (dx, dy) in enumerate(moves):
            if (cx + dx, cy + dy) in valid_moves:
                mask[i] = True

        # 2. Horizontal wall actions
        for i in range(64):
            r, c = divmod(i, 8)
            if self.engine.can_place_wall(1, r, c, "H"):
                mask[i + 4] = True

        # 3. Vertical wall actions
        for i in range(64):
            r, c = divmod(i, 8)
            if self.engine.can_place_wall(1, r, c, "V"):
                mask[i + 68] = True

        return mask

    def step(self, action):
        action = int(action)
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action}; expected 0..{self.action_space.n - 1}")

        self.current_step += 1
        target_row = 0
        prev_dist = self.engine.get_bfs_distance(self.engine.p1_pos, target_row)
        valid_action = bool(self.action_masks()[action])

        if valid_action:
            if action < 4:
                moves = [(0, -1), (0, 1), (-1, 0), (1, 0)]
                dx, dy = moves[action]
                cx, cy = self.engine.p1_pos
                self.engine.board[cy, cx] = 0
                self.engine.p1_pos = (cx + dx, cy + dy)
                self.engine.board[cy + dy, cx + dx] = 1
            elif action < 68:
                idx = action - 4
                self.engine.place_wall(1, idx // 8, idx % 8, "H")
            else:
                idx = action - 68
                self.engine.place_wall(1, idx // 8, idx % 8, "V")

        new_dist = self.engine.get_bfs_distance(self.engine.p1_pos, target_row)

        if not valid_action:
            reward = -0.2
        elif action < 4:
            reward = (prev_dist - new_dist) * 0.1
        else:
            reward = -0.01

        terminated = False
        if new_dist == 0 or self.engine.p1_pos[1] == target_row:
            reward += 10.0
            terminated = True

        truncated = False
        if self.current_step >= self.max_steps and not terminated:
            truncated = True
            reward -= 5.0

        return self._get_obs(), reward, terminated, truncated, {}
