import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Tuple, Dict, Any

class QuoridorEnv(gym.Env):
    """
    Gymnasium-совместимая среда для игры Quoridor.
    Реализована полная логика перемещений, включая прямые и диагональные прыжки.
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, engine, render_mode: str = None):
        super().__init__()
        self.engine = engine
        self.render_mode = render_mode
        self.max_steps = 200
        self.current_step = 0

        self.action_space = spaces.Discrete(136)

        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(7, 9, 9), dtype=np.float32
        )

    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        self.engine.reset()
        self.current_step = 0
        return self._get_obs(), self._get_info()

    def _get_movement_destinations(self, player_idx: int) -> Dict[int, Tuple[int, int]]:
        state = self.engine.get_state()
        y, x = state['p1_pos'] if player_idx == 0 else state['p2_pos']
        opp_y, opp_x = state['p2_pos'] if player_idx == 0 else state['p1_pos']
        
        dests = {}
        moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        
        for act_idx, (dy, dx) in enumerate(moves):
            ny, nx = y + dy, x + dx
            
            if 0 <= ny < 9 and 0 <= nx < 9 and self.engine._can_move(y, x, ny, nx):
                if ny == opp_y and nx == opp_x:
                    jy, jx = ny + dy, nx + dx
                    can_jump_straight = (0 <= jy < 9 and 0 <= jx < 9 and self.engine._can_move(ny, nx, jy, jx))
                    
                    if can_jump_straight:
                        dests[act_idx] = (jy, jx)
                    else:
                        if dy != 0: 
                            if 0 <= nx - 1 < 9 and self.engine._can_move(ny, nx, ny, nx - 1):
                                dests[4 if dy == -1 else 6] = (ny, nx - 1) 
                            if 0 <= nx + 1 < 9 and self.engine._can_move(ny, nx, ny, nx + 1):
                                dests[5 if dy == -1 else 7] = (ny, nx + 1) 
                        else: 
                            if 0 <= ny - 1 < 9 and self.engine._can_move(ny, nx, ny - 1, nx):
                                dests[4 if dx == -1 else 5] = (ny - 1, nx) 
                            if 0 <= ny + 1 < 9 and self.engine._can_move(ny, nx, ny + 1, nx):
                                dests[6 if dx == -1 else 7] = (ny + 1, nx) 
                else:
                    dests[act_idx] = (ny, nx)
                    
        return dests

    def step(self, action: Any) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        # ФИКС: Очистка типа от NumPy, переводим в чистый int
        if isinstance(action, np.ndarray):
            action = action.item()
        action = int(action)

        self.current_step += 1
        curr_player = self.engine.current_player
        
        start_dist = self.engine.get_shortest_path_length(
            self.engine.p1_pos if curr_player == 0 else self.engine.p2_pos,
            8 if curr_player == 0 else 0
        )

        is_valid = self._apply_action(action)
        state = self.engine.get_state()
        
        terminated, truncated = False, False
        reward = 0.0

        new_dist = self.engine.get_shortest_path_length(
            self.engine.p1_pos if curr_player == 0 else self.engine.p2_pos,
            8 if curr_player == 0 else 0
        )

        if is_valid:
            reward += (start_dist - new_dist) * 0.05 

        if state['p1_pos'][0] == 8:
            terminated = True
            reward += 1.0 if curr_player == 0 else -1.0 
        elif state['p2_pos'][0] == 0:
            terminated = True
            reward += 1.0 if curr_player == 1 else -1.0

        if self.current_step >= self.max_steps and not terminated:
            truncated = True

        if not terminated:
            reward -= 0.01 

        if not is_valid:
            reward = -1.0 
            terminated = True 

        if not terminated and is_valid:
            self.engine.current_player = 1 - self.engine.current_player

        return self._get_obs(), reward, terminated, truncated, self._get_info()

    def _apply_action(self, action: int) -> bool:
        player_idx = self.engine.current_player

        if action < 8:
            valid_moves = self._get_movement_destinations(player_idx)
            if action in valid_moves:
                dest_y, dest_x = valid_moves[action]
                if player_idx == 0: self.engine.p1_pos = np.array([dest_y, dest_x])
                else: self.engine.p2_pos = np.array([dest_y, dest_x])
                return True
            return False

        wall_action = action - 8
        orientation = 'H' if wall_action < 64 else 'V'
        coord_idx = wall_action % 64
        wy, wx = coord_idx // 8, coord_idx % 8

        walls_left = self.engine.p1_walls if player_idx == 0 else self.engine.p2_walls
        if walls_left > 0 and self.engine.is_valid_wall(wy, wx, orientation):
            if player_idx == 0: self.engine.p1_walls -= 1
            else: self.engine.p2_walls -= 1
            
            if orientation == 'H': self.engine.walls_h[wy, wx] = True
            else: self.engine.walls_v[wy, wx] = True
                
            return True
            
        return False

    def _get_obs(self) -> np.ndarray:
        obs = np.zeros((7, 9, 9), dtype=np.float32)
        state = self.engine.get_state()
        curr_player = state['current_player']
        
        my_pos = state['p1_pos'] if curr_player == 0 else state['p2_pos']
        opp_pos = state['p2_pos'] if curr_player == 0 else state['p1_pos']
        
        obs[0, my_pos[0], my_pos[1]] = 1.0
        obs[1, opp_pos[0], opp_pos[1]] = 1.0
        obs[2, :8, :8] = state['walls_h'].astype(np.float32)
        obs[3, :8, :8] = state['walls_v'].astype(np.float32)
        
        my_walls = state['p1_walls'] if curr_player == 0 else state['p2_walls']
        opp_walls = state['p2_walls'] if curr_player == 0 else state['p1_walls']
        obs[4, :, :] = my_walls / self.engine.MAX_WALLS
        obs[5, :, :] = opp_walls / self.engine.MAX_WALLS
        obs[6, :, :] = 1.0 if curr_player == 0 else -1.0
        
        return obs

    def _get_info(self) -> Dict[str, Any]:
        mask = np.zeros(136, dtype=np.int8)
        state = self.engine.get_state()
        player_idx = state['current_player']
        walls_left = state['p1_walls'] if player_idx == 0 else state['p2_walls']

        valid_moves = self._get_movement_destinations(player_idx)
        for act_idx in valid_moves.keys():
            mask[act_idx] = 1

        if walls_left > 0:
            for w_idx in range(128):
                orientation = 'H' if w_idx < 64 else 'V'
                coord_idx = w_idx % 64
                y, x = coord_idx // 8, coord_idx % 8
                
                if self.engine.is_valid_wall(y, x, orientation):
                    mask[8 + w_idx] = 1

        return {
            "action_mask": mask,
            "current_player": player_idx
        }