import numpy as np
from collections import deque

class QuoridorEngine:
    def __init__(self):
        self.board_size = 9
        self.board = np.zeros((self.board_size, self.board_size), dtype=np.int8)
        self.horizontal_walls = np.zeros((self.board_size - 1, self.board_size - 1), dtype=bool)
        self.vertical_walls = np.zeros((self.board_size - 1, self.board_size - 1), dtype=bool)
        
        self.p1_pos = (4, 8) # Игрок 1 начинает снизу
        self.p2_pos = (4, 0) # Игрок 2 начинает сверху
        self.board[self.p1_pos[1], self.p1_pos[0]] = 1
        self.board[self.p2_pos[1], self.p2_pos[0]] = 2
        
        self.current_player = 1
        self.walls_left = {1: 10, 2: 10}

    def reset(self):
        self.__init__()

    def has_wall_between(self, x1, y1, x2, y2):
        if x1 == x2: # Вертикальное движение (проверка горизонтальных стен)
            y_min = min(y1, y2)
            if x1 < 8 and self.horizontal_walls[y_min, x1]: return True
            if x1 > 0 and self.horizontal_walls[y_min, x1 - 1]: return True
        elif y1 == y2: # Горизонтальное движение (проверка вертикальных стен)
            x_min = min(x1, x2)
            if y1 < 8 and self.vertical_walls[y1, x_min]: return True
            if y1 > 0 and self.vertical_walls[y1 - 1, x_min]: return True
        return False

    def get_valid_moves(self, player_id):
        pos = self.p1_pos if player_id == 1 else self.p2_pos
        x, y = pos
        moves = []
        directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]
        
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.board_size and 0 <= ny < self.board_size:
                if not self.has_wall_between(x, y, nx, ny):
                    # Если клетка пустая (без учета прыжков через игрока для базы)
                    if self.board[ny, nx] == 0:
                        moves.append((nx, ny))
        return moves

    def get_bfs_distance(self, start_pos, target_row):
        queue = deque([(start_pos[0], start_pos[1], 0)])
        visited = set([start_pos])
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        
        while queue:
            x, y, dist = queue.popleft()
            if y == target_row:
                return dist
                
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.board_size and 0 <= ny < self.board_size:
                    if not self.has_wall_between(x, y, nx, ny) and (nx, ny) not in visited:
                        visited.add((nx, ny))
                        queue.append((nx, ny, dist + 1))
        return 999

    def can_place_wall(self, player_id, r, c, orientation):
        if self.walls_left[player_id] <= 0:
            return False
        if r < 0 or r > 7 or c < 0 or c > 7:
            return False
            
        # Проверка наложения стен
        if orientation == 'H':
            if self.horizontal_walls[r, c]: return False
            if c > 0 and self.horizontal_walls[r, c - 1]: return False
            if c < 7 and self.horizontal_walls[r, c + 1]: return False
            if self.vertical_walls[r, c]: return False # Крест-накрест
        else: # 'V'
            if self.vertical_walls[r, c]: return False
            if r > 0 and self.vertical_walls[r - 1, c]: return False
            if r < 7 and self.vertical_walls[r + 1, c]: return False
            if self.horizontal_walls[r, c]: return False # Крест-накрест

        # Временная установка стены для проверки блокировки путей (правило Quoridor)
        if orientation == 'H': self.horizontal_walls[r, c] = True
        else: self.vertical_walls[r, c] = True

        p1_dist = self.get_bfs_distance(self.p1_pos, 0)  # Цель 1-го игрока - строка 0
        p2_dist = self.get_bfs_distance(self.p2_pos, 8)  # Цель 2-го игрока - строка 8

        # Откат временной стены
        if orientation == 'H': self.horizontal_walls[r, c] = False
        else: self.vertical_walls[r, c] = False

        # Если кто-то заблокирован намертво - ход запрещен
        if p1_dist == 999 or p2_dist == 999:
            return False

        return True

    def place_wall(self, player_id, r, c, orientation):
        if orientation == 'H':
            self.horizontal_walls[r, c] = True
        else:
            self.vertical_walls[r, c] = True
        self.walls_left[player_id] -= 1