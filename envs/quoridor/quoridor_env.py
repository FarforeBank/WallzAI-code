# envs/quoridor/quoridor_env.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from envs.quoridor.engine import QuoridorEngine

class QuoridorEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, render_mode=None):
        super().__init__()
        # Создаем движок внутри среды
        self.engine = QuoridorEngine()
        
        # Пространство действий: 4 направления шага + установка стен (упрощенный пример, 132 действия)
        self.action_space = spaces.Discrete(132) 
        
        # Пространство состояний: матрица 9x9 (позиции) + матрицы стен 8x8 х2
        self.observation_space = spaces.Box(low=-1, high=2, shape=(9, 9, 3), dtype=np.int8)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.engine.reset()
        return self._get_obs(), {}

    def _get_obs(self):
        # Формируем 3D тензор состояния для нейросети
        obs = np.zeros((9, 9, 3), dtype=np.int8)
        obs[:, :, 0] = self.engine.board
        obs[:8, :8, 1] = self.engine.horizontal_walls
        obs[:8, :8, 2] = self.engine.vertical_walls
        return obs

    def action_masks(self):
        """Метод для MaskablePPO: 1 - действие разрешено, 0 - запрещено"""
        mask = np.zeros(self.action_space.n, dtype=np.int8)
        valid_moves = self.engine.get_valid_moves(1)
        for i, move in enumerate([(0,-1), (0,1), (-1,0), (1,0)]):
            cx, cy = self.engine.p1_pos
            nx, ny = cx + move[0], cy + move[1]
            if (nx, ny) in valid_moves:
                mask[i] = 1
        return mask

    def step(self, action):
        target_row = 0 # Игрок 1 идет на нулевую строку
        prev_dist = self.engine.get_bfs_distance(self.engine.p1_pos, target_row)
        
        # Временная заглушка логики действия
        if action == 0 and self.action_masks()[0] == 1:
             self.engine.p1_pos = (self.engine.p1_pos[0], self.engine.p1_pos[1] - 1)
        
        new_dist = self.engine.get_bfs_distance(self.engine.p1_pos, target_row)
        
        # Reward Shaping: даем награду, если агент стал ближе к финишу
        reward = (prev_dist - new_dist) * 0.1
        
        terminated = False
        if new_dist == 0:
            reward += 10.0 # Награда за победу
            terminated = True
            
        truncated = False
        info = {}
        
        return self._get_obs(), reward, terminated, truncated, info