import numpy as np
from collections import deque
from typing import Dict

class QuoridorEngine:
    """
    Высокопроизводительный движок игры Quoridor (Wallz.gg).
    Оптимизирован для последующей векторизации в RL-средах.
    """
    
    BOARD_SIZE = 9
    MAX_WALLS = 10

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        """Сброс среды к начальному состоянию."""
        # ФИКС: Убрано ограничение int8. Теперь переполнения памяти не будет.
        self.p1_pos = np.array([0, 4])
        self.p2_pos = np.array([8, 4])
        
        self.p1_walls = self.MAX_WALLS
        self.p2_walls = self.MAX_WALLS
        
        self.walls_h = np.zeros((8, 8), dtype=bool)
        self.walls_v = np.zeros((8, 8), dtype=bool)
        
        self.current_player = 0 
        self.done = False
        self.winner = -1

    def get_state(self) -> Dict:
        """Возвращает текущее состояние в виде словаря."""
        return {
            'p1_pos': self.p1_pos.copy(),
            'p2_pos': self.p2_pos.copy(),
            'p1_walls': self.p1_walls,
            'p2_walls': self.p2_walls,
            'walls_h': self.walls_h.copy(),
            'walls_v': self.walls_v.copy(),
            'current_player': self.current_player
        }

    def _can_move(self, y1: int, x1: int, y2: int, x2: int) -> bool:
        """Проверяет, нет ли стены между двумя соседними клетками."""
        if y1 == y2: # Горизонтальное перемещение
            min_x = min(x1, x2)
            if y1 > 0 and self.walls_v[y1-1, min_x]: return False
            if y1 < 8 and self.walls_v[y1, min_x]: return False
        elif x1 == x2: # Вертикальное перемещение
            min_y = min(y1, y2)
            if x1 > 0 and self.walls_h[min_y, x1-1]: return False
            if x1 < 8 and self.walls_h[min_y, x1]: return False
        return True

    def get_shortest_path_length(self, start_pos: np.ndarray, target_row: int) -> int:
        """
        BFS, возвращающий длину кратчайшего пути.
        Используется для проверки валидности стен и для Reward Shaping.
        Возвращает 999, если путь заблокирован.
        """
        queue = deque([(start_pos[0], start_pos[1], 0)])
        visited = set([(start_pos[0], start_pos[1])])
        
        while queue:
            y, x, dist = queue.popleft()
            
            if y == target_row:
                return dist
                
            moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]
            for dy, dx in moves:
                ny, nx = y + dy, x + dx
                if 0 <= ny < 9 and 0 <= nx < 9 and (ny, nx) not in visited:
                    if self._can_move(y, x, ny, nx):
                        visited.add((ny, nx))
                        queue.append((ny, nx, dist + 1))
                        
        return 999

    def is_valid_wall(self, y: int, x: int, orientation: str) -> bool:
        """
        Проверяет, можно ли поставить стену (границы, пересечения и блокировка пути).
        """
        if y < 0 or y >= 8 or x < 0 or x >= 8:
            return False

        if orientation == 'H':
            if self.walls_h[y, x]: return False
            if x > 0 and self.walls_h[y, x-1]: return False
            if x < 7 and self.walls_h[y, x+1]: return False
            if self.walls_v[y, x]: return False 
            
            self.walls_h[y, x] = True
        else: 
            if self.walls_v[y, x]: return False
            if y > 0 and self.walls_v[y-1, x]: return False
            if y < 7 and self.walls_v[y+1, x]: return False
            if self.walls_h[y, x]: return False 
            
            self.walls_v[y, x] = True

        p1_dist = self.get_shortest_path_length(self.p1_pos, 8)
        p2_dist = self.get_shortest_path_length(self.p2_pos, 0)

        if orientation == 'H':
            self.walls_h[y, x] = False
        else:
            self.walls_v[y, x] = False

        return p1_dist != 999 and p2_dist != 999