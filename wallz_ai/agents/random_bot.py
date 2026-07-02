from __future__ import annotations

import numpy as np

from wallz_ai.env.rules import WallzState, legal_actions


class RandomLegalBot:
    def __init__(self, seed: int | None = None):
        self.rng = np.random.default_rng(seed)

    def act(self, state: WallzState) -> int:
        actions = legal_actions(state)
        if not actions:
            raise RuntimeError("No legal actions available")
        return int(self.rng.choice(actions))
