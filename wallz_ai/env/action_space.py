"""Fixed Wallz action encoding.

0..80     pawn move to target square index row * 9 + col
81..144   horizontal wall at 8x8 coordinate
145..208  vertical wall at 8x8 coordinate
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple

BOARD_SIZE = 9
WALL_GRID_SIZE = BOARD_SIZE - 1
BOARD_CELLS = BOARD_SIZE * BOARD_SIZE
H_WALL_OFFSET = BOARD_CELLS
V_WALL_OFFSET = H_WALL_OFFSET + WALL_GRID_SIZE * WALL_GRID_SIZE
ACTION_SIZE = V_WALL_OFFSET + WALL_GRID_SIZE * WALL_GRID_SIZE

Position = Tuple[int, int]
WallCoord = Tuple[int, int]


class WallOrientation(str, Enum):
    HORIZONTAL = "H"
    VERTICAL = "V"


@dataclass(frozen=True)
class DecodedAction:
    kind: str
    target: Optional[Position] = None
    wall: Optional[WallCoord] = None
    orientation: Optional[WallOrientation] = None


def square_to_action(pos: Position) -> int:
    row, col = pos
    if not (0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE):
        raise ValueError(f"Square out of bounds: {pos}")
    return row * BOARD_SIZE + col


def action_to_square(action: int) -> Position:
    if not (0 <= action < BOARD_CELLS):
        raise ValueError(f"Not a pawn target action: {action}")
    return divmod(action, BOARD_SIZE)


def wall_to_action(row: int, col: int, orientation: WallOrientation | str) -> int:
    orientation = WallOrientation(orientation)
    if not (0 <= row < WALL_GRID_SIZE and 0 <= col < WALL_GRID_SIZE):
        raise ValueError(f"Wall coordinate out of bounds: {(row, col)}")
    idx = row * WALL_GRID_SIZE + col
    if orientation == WallOrientation.HORIZONTAL:
        return H_WALL_OFFSET + idx
    return V_WALL_OFFSET + idx


def action_to_wall(action: int) -> tuple[int, int, WallOrientation]:
    if H_WALL_OFFSET <= action < V_WALL_OFFSET:
        idx = action - H_WALL_OFFSET
        row, col = divmod(idx, WALL_GRID_SIZE)
        return row, col, WallOrientation.HORIZONTAL
    if V_WALL_OFFSET <= action < ACTION_SIZE:
        idx = action - V_WALL_OFFSET
        row, col = divmod(idx, WALL_GRID_SIZE)
        return row, col, WallOrientation.VERTICAL
    raise ValueError(f"Not a wall action: {action}")


def decode_action(action: int) -> DecodedAction:
    if 0 <= action < BOARD_CELLS:
        return DecodedAction(kind="move", target=action_to_square(action))
    row, col, orientation = action_to_wall(action)
    return DecodedAction(kind="wall", wall=(row, col), orientation=orientation)
