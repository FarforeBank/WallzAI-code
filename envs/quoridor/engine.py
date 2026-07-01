import numpy as np
from collections import deque


ORTHOGONAL_DIRS = [(0, -1), (0, 1), (-1, 0), (1, 0)]


class QuoridorEngine:
    def __init__(self):
        self.board_size = 9
        self.board = np.zeros((self.board_size, self.board_size), dtype=np.int8)
        self.horizontal_walls = np.zeros((self.board_size - 1, self.board_size - 1), dtype=bool)
        self.vertical_walls = np.zeros((self.board_size - 1, self.board_size - 1), dtype=bool)

        self.p1_pos = (4, 8)  # Player 1 starts at the bottom
        self.p2_pos = (4, 0)  # Player 2 starts at the top
        self.board[self.p1_pos[1], self.p1_pos[0]] = 1
        self.board[self.p2_pos[1], self.p2_pos[0]] = 2

        self.current_player = 1
        self.walls_left = {1: 10, 2: 10}

    def reset(self):
        self.__init__()

    def _in_bounds(self, x, y):
        return 0 <= x < self.board_size and 0 <= y < self.board_size

    def has_wall_between(self, x1, y1, x2, y2):
        if not self._in_bounds(x1, y1) or not self._in_bounds(x2, y2):
            return True
        if abs(x1 - x2) + abs(y1 - y2) != 1:
            return True

        if x1 == x2:  # Vertical movement: check horizontal walls.
            y_min = min(y1, y2)
            if x1 < 8 and self.horizontal_walls[y_min, x1]:
                return True
            if x1 > 0 and self.horizontal_walls[y_min, x1 - 1]:
                return True
        elif y1 == y2:  # Horizontal movement: check vertical walls.
            x_min = min(x1, x2)
            if y1 < 8 and self.vertical_walls[y1, x_min]:
                return True
            if y1 > 0 and self.vertical_walls[y1 - 1, x_min]:
                return True
        return False

    def _pawn_pos(self, player_id):
        return self.p1_pos if player_id == 1 else self.p2_pos

    def _opponent_pos(self, player_id):
        return self.p2_pos if player_id == 1 else self.p1_pos

    def get_valid_moves(self, player_id):
        """Return legal pawn target cells, including jumps and diagonals.

        Quoridor movement rules:
        - Move one orthogonal cell if there is no wall.
        - If the adjacent cell contains the opponent, jump straight over them when possible.
        - If the straight jump is blocked by wall/edge, move diagonally around the opponent.
        """
        x, y = self._pawn_pos(player_id)
        opponent = self._opponent_pos(player_id)
        moves = []

        for dx, dy in ORTHOGONAL_DIRS:
            ax, ay = x + dx, y + dy
            if not self._in_bounds(ax, ay):
                continue
            if self.has_wall_between(x, y, ax, ay):
                continue

            if (ax, ay) != opponent:
                if self.board[ay, ax] == 0:
                    moves.append((ax, ay))
                continue

            # Opponent is adjacent: try straight jump first.
            jx, jy = ax + dx, ay + dy
            can_jump = (
                self._in_bounds(jx, jy)
                and not self.has_wall_between(ax, ay, jx, jy)
                and self.board[jy, jx] == 0
            )
            if can_jump:
                moves.append((jx, jy))
                continue

            # Straight jump is blocked by wall/edge: diagonal around opponent.
            if dx == 0:
                perpendicular = [(-1, 0), (1, 0)]
            else:
                perpendicular = [(0, -1), (0, 1)]

            for pdx, pdy in perpendicular:
                tx, ty = ax + pdx, ay + pdy
                if not self._in_bounds(tx, ty):
                    continue
                if self.has_wall_between(ax, ay, tx, ty):
                    continue
                if self.board[ty, tx] == 0:
                    moves.append((tx, ty))

        # Preserve order but remove duplicates, because diagonal paths can converge.
        seen = set()
        unique_moves = []
        for move in moves:
            if move not in seen:
                unique_moves.append(move)
                seen.add(move)
        return unique_moves

    def get_bfs_distance(self, start_pos, target_row):
        queue = deque([(start_pos[0], start_pos[1], 0)])
        visited = {start_pos}

        while queue:
            x, y, dist = queue.popleft()
            if y == target_row:
                return dist

            for dx, dy in ORTHOGONAL_DIRS:
                nx, ny = x + dx, y + dy
                if self._in_bounds(nx, ny):
                    if not self.has_wall_between(x, y, nx, ny) and (nx, ny) not in visited:
                        visited.add((nx, ny))
                        queue.append((nx, ny, dist + 1))
        return 999

    def can_place_wall(self, player_id, r, c, orientation):
        if player_id not in self.walls_left or self.walls_left[player_id] <= 0:
            return False
        if r < 0 or r > 7 or c < 0 or c > 7:
            return False
        if orientation not in {"H", "V"}:
            return False

        # Overlap/crossing checks.
        if orientation == "H":
            if self.horizontal_walls[r, c]:
                return False
            if c > 0 and self.horizontal_walls[r, c - 1]:
                return False
            if c < 7 and self.horizontal_walls[r, c + 1]:
                return False
            if self.vertical_walls[r, c]:
                return False
        else:
            if self.vertical_walls[r, c]:
                return False
            if r > 0 and self.vertical_walls[r - 1, c]:
                return False
            if r < 7 and self.vertical_walls[r + 1, c]:
                return False
            if self.horizontal_walls[r, c]:
                return False

        # Temporarily place the wall to ensure both players still have a path.
        if orientation == "H":
            self.horizontal_walls[r, c] = True
        else:
            self.vertical_walls[r, c] = True

        p1_dist = self.get_bfs_distance(self.p1_pos, 0)
        p2_dist = self.get_bfs_distance(self.p2_pos, 8)

        if orientation == "H":
            self.horizontal_walls[r, c] = False
        else:
            self.vertical_walls[r, c] = False

        return p1_dist != 999 and p2_dist != 999

    def place_wall(self, player_id, r, c, orientation):
        if not self.can_place_wall(player_id, r, c, orientation):
            return False

        if orientation == "H":
            self.horizontal_walls[r, c] = True
        else:
            self.vertical_walls[r, c] = True
        self.walls_left[player_id] -= 1
        return True
