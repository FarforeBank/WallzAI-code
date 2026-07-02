from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from wallz_ai.env.action_space import ACTION_SIZE
from wallz_ai.env.wallz_env import OBS_CHANNELS


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(4, channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(4, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = F.relu(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        return F.relu(x + residual)


class WallzPolicyValueNet(nn.Module):
    def __init__(self, channels: int = 64, residual_blocks: int = 3):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(OBS_CHANNELS, channels, 3, padding=1), nn.GroupNorm(4, channels), nn.ReLU())
        self.trunk = nn.Sequential(*[ResidualBlock(channels) for _ in range(residual_blocks)])
        self.policy_head = nn.Sequential(nn.Conv2d(channels, 16, 1), nn.ReLU(), nn.Flatten(), nn.Linear(16 * 9 * 9, ACTION_SIZE))
        self.value_head = nn.Sequential(nn.Conv2d(channels, 8, 1), nn.ReLU(), nn.Flatten(), nn.Linear(8 * 9 * 9, 128), nn.ReLU(), nn.Linear(128, 1), nn.Tanh())

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.trunk(self.stem(obs))
        return self.policy_head(x), self.value_head(x).squeeze(-1)


def masked_logits(logits: torch.Tensor, legal_mask: torch.Tensor) -> torch.Tensor:
    if legal_mask.dtype != torch.bool:
        legal_mask = legal_mask.bool()
    return logits.masked_fill(~legal_mask, -1e9)


def masked_distribution(logits: torch.Tensor, legal_mask: torch.Tensor) -> torch.distributions.Categorical:
    return torch.distributions.Categorical(logits=masked_logits(logits, legal_mask))


def select_device(requested: str = "auto") -> torch.device:
    requested = requested.lower()
    if requested == "auto":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    if requested == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        return torch.device("cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(requested)
