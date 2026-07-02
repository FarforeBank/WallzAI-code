from __future__ import annotations

from typing import Any

from wallz_ai.env.rules import WallzState


class WallzStateReader:
    def __init__(self, page):
        self.page = page

    async def read_state(self) -> WallzState:
        data = await self._read_structured_state()
        if data is None:
            raise RuntimeError("Could not read structured Wallz state. Canvas visual parsing is intentionally not enabled yet.")
        return self._state_from_payload(data)

    async def _read_structured_state(self) -> dict[str, Any] | None:
        return await self.page.evaluate(
            """
            () => {
              const candidates = [window.__WALLZ_STATE__, window.wallzState, window.gameState,
                window.store && window.store.getState && window.store.getState()].filter(Boolean);
              for (const c of candidates) return JSON.parse(JSON.stringify(c));
              for (let i = 0; i < localStorage.length; i++) {
                const key = localStorage.key(i);
                if (!key || !key.toLowerCase().includes('wallz')) continue;
                try { return JSON.parse(localStorage.getItem(key)); } catch (e) {}
              }
              return null;
            }
            """
        )

    def _state_from_payload(self, data: dict[str, Any]) -> WallzState:
        pawns = data.get("pawns") or data.get("players") or data.get("pawnPositions")
        if pawns is None:
            raise ValueError(f"Unknown Wallz payload schema: missing pawns in keys={list(data.keys())}")
        state = WallzState()
        parsed = []
        for p in pawns[:2]:
            if isinstance(p, dict):
                parsed.append((int(p.get("row", p.get("y"))), int(p.get("col", p.get("x")))))
            else:
                parsed.append((int(p[0]), int(p[1])))
        state.pawn_positions = parsed
        state.current_player = int(data.get("currentPlayer", data.get("sideToMove", 0)))
        state.walls_remaining = list(data.get("wallsRemaining", state.walls_remaining))[:2]
        for wall in data.get("walls", []):
            row = int(wall.get("row", wall.get("y")))
            col = int(wall.get("col", wall.get("x")))
            orient = str(wall.get("orientation", wall.get("dir", ""))).upper()[0]
            if orient == "H":
                state.horizontal_walls[row, col] = True
            elif orient == "V":
                state.vertical_walls[row, col] = True
            else:
                raise ValueError(f"Unknown wall orientation in payload: {wall}")
        return state
