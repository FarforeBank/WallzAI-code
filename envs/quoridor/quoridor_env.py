import gymnasium as gym
from gymnasium import spaces
import numpy as np
from envs.quoridor.engine import QuoridorEngine

class QuoridorEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, render_mode=None):
        super().__init__()
        self.engine = QuoridorEngine()
        
        # 0-3: Вверх, Вниз, Влево, Вправо
        # 4-67: Горизонтальные стены
        # 68-131: Вертикальные стены
        self.action_space = spaces.Discrete(132) 
        self.observation_space = spaces.Box(low=-1, high=2, shape=(9, 9, 3), dtype=np.int8)
        
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
        obs[:8, :8, 1] = self.engine.horizontal_walls
        obs[:8, :8, 2] = self.engine.vertical_walls
        
        # Подсказка сети о том, сколько стен у нее осталось
        obs[8, 0, 1] = self.engine.walls_left[1] 
        return obs

    def action_masks(self):
        mask = np.zeros(self.action_space.n, dtype=np.int8)
        
        # 1. Доступные шаги
        valid_moves = self.engine.get_valid_moves(1)
        moves = [(0, -1), (0, 1), (-1, 0), (1, 0)] # Up, Down, Left, Right
        for i, move in enumerate(moves):
            cx, cy = self.engine.p1_pos
            if (cx + move[0], cy + move[1]) in valid_moves:
                mask[i] = 1
                
        # 2. Доступные горизонтальные стены
        for i in range(64):
            r, c = i // 8, i % 8
            if self.engine.can_place_wall(1, r, c, 'H'):
                mask[i + 4] = 1
                
        # 3. Доступные вертикальные стены
        for i in range(64):
            r, c = i // 8, i % 8
            if self.engine.can_place_wall(1, r, c, 'V'):
                mask[i + 68] = 1
                
        return mask

    def step(self, action):
        self.current_step += 1
        target_row = 0 
        prev_dist = self.engine.get_bfs_distance(self.engine.p1_pos, target_row)
        
        # Выполнение действия
        if self.action_masks()[action] == 1:
            if action < 4:
                # Ходьба
                moves = [(0, -1), (0, 1), (-1, 0), (1, 0)]
                dx, dy = moves[action]
                cx, cy = self.engine.p1_pos
                self.engine.board[cy, cx] = 0 # Стираем старую позицию
                self.engine.p1_pos = (cx + dx, cy + dy)
                self.engine.board[cy + dy, cx + dx] = 1 # Отмечаем новую
            elif action < 68:
                # Горизонтальная стена
                idx = action - 4
                self.engine.place_wall(1, idx // 8, idx % 8, 'H')
            else:
                # Вертикальная стена
                idx = action - 68
                self.engine.place_wall(1, idx // 8, idx % 8, 'V')
        
        new_dist = self.engine.get_bfs_distance(self.engine.p1_pos, target_row)
        
        # Награда
        reward = 0.0
        if action < 4:
            reward = (prev_dist - new_dist) * 0.1 # За движение к цели
        else:
            reward = -0.01 # Микро-штраф за трату стены, чтобы не ставил просто так
        
        terminated = False
        if new_dist == 0 or self.engine.p1_pos[1] == target_row:
            reward += 10.0 # Огромная награда за победу
            terminated = True
            
        truncated = False
        if self.current_step >= self.max_steps and not terminated:
            truncated = True
            reward -= 5.0
            
        return self._get_obs(), reward, terminated, truncated, {}