from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class RolloutBuffer:
    obs: list[np.ndarray] = field(default_factory=list)
    masks: list[np.ndarray] = field(default_factory=list)
    actions: list[int] = field(default_factory=list)
    log_probs: list[float] = field(default_factory=list)
    values: list[float] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)

    def add(self, obs, mask, action: int, log_prob: float, value: float, reward: float) -> None:
        self.obs.append(np.asarray(obs, dtype=np.float32))
        self.masks.append(np.asarray(mask, dtype=bool))
        self.actions.append(int(action))
        self.log_probs.append(float(log_prob))
        self.values.append(float(value))
        self.rewards.append(float(reward))

    def __len__(self) -> int:
        return len(self.actions)

    def compute_returns_advantages(self, gamma: float = 0.997) -> tuple[np.ndarray, np.ndarray]:
        returns = np.zeros(len(self.rewards), dtype=np.float32)
        running = 0.0
        for i in range(len(self.rewards) - 1, -1, -1):
            running = self.rewards[i] - gamma * running
            returns[i] = running
        advantages = returns - np.asarray(self.values, dtype=np.float32)
        return returns, advantages

    def to_arrays(self, gamma: float = 0.997):
        returns, advantages = self.compute_returns_advantages(gamma)
        return {
            "obs": np.stack(self.obs).astype(np.float32),
            "masks": np.stack(self.masks).astype(bool),
            "actions": np.asarray(self.actions, dtype=np.int64),
            "old_log_probs": np.asarray(self.log_probs, dtype=np.float32),
            "returns": returns,
            "advantages": advantages.astype(np.float32),
        }
