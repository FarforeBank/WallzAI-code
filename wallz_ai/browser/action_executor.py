from __future__ import annotations

from wallz_ai.env.action_space import action_to_square, action_to_wall
from wallz_ai.env.rules import WallzState, is_legal_action


class WallzActionExecutor:
    def __init__(self, page, dry_run: bool = True):
        self.page = page
        self.dry_run = dry_run

    async def execute(self, state: WallzState, action: int) -> None:
        if not is_legal_action(state, action):
            raise ValueError(f"Refusing to execute illegal browser action: {action}")
        if self.dry_run:
            print(self.describe_action(action))
            return
        raise NotImplementedError("Live clicking is disabled by default. Add a private-game executor and pass an explicit allow flag.")

    @staticmethod
    def describe_action(action: int) -> str:
        if 0 <= action < 81:
            row, col = action_to_square(action)
            return f"MOVE pawn to row={row}, col={col} (action={action})"
        row, col, orientation = action_to_wall(action)
        return f"PLACE {orientation.value} wall at row={row}, col={col} (action={action})"
