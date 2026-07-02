from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from wallz_ai.agents.model import WallzPolicyValueNet, masked_logits
from wallz_ai.env.rules import WallzState, apply_action, legal_action_mask, legal_actions
from wallz_ai.env.wallz_env import WallzEnv


@dataclass
class MCTSNode:
    state: WallzState
    prior: float = 1.0
    visits: int = 0
    value_sum: float = 0.0
    children: dict[int, "MCTSNode"] = field(default_factory=dict)

    @property
    def value(self) -> float:
        return 0.0 if self.visits == 0 else self.value_sum / self.visits


class MCTS:
    def __init__(self, model: WallzPolicyValueNet, device: torch.device, simulations: int = 64, c_puct: float = 1.5):
        self.model = model
        self.device = device
        self.simulations = int(simulations)
        self.c_puct = float(c_puct)

    def search(self, state: WallzState) -> np.ndarray:
        root = MCTSNode(state.clone())
        self._expand(root)
        for _ in range(self.simulations):
            path = [root]
            node = root
            while node.children and not node.state.terminal:
                _, node = self._select_child(node)
                path.append(node)
            value = self._terminal_value(node.state) if node.state.terminal else self._expand(node)
            for n in reversed(path):
                n.visits += 1
                n.value_sum += value
                value = -value
        policy = np.zeros(209, dtype=np.float32)
        for action, child in root.children.items():
            policy[action] = child.visits
        if policy.sum() > 0:
            policy /= policy.sum()
        return policy

    def _select_child(self, node: MCTSNode) -> tuple[int, MCTSNode]:
        sqrt_parent = np.sqrt(max(1, node.visits))
        best_score = -1e9
        best = None
        for action, child in node.children.items():
            score = -child.value + self.c_puct * child.prior * sqrt_parent / (1 + child.visits)
            if score > best_score:
                best_score = score
                best = (action, child)
        assert best is not None
        return best

    def _expand(self, node: MCTSNode) -> float:
        env = WallzEnv(max_moves=node.state.max_moves)
        env.state = node.state.clone()
        obs = torch.as_tensor(env.observation()[None], dtype=torch.float32, device=self.device)
        mask = torch.as_tensor(legal_action_mask(node.state)[None], dtype=torch.bool, device=self.device)
        with torch.no_grad():
            logits, value = self.model(obs)
            probs = torch.softmax(masked_logits(logits, mask), dim=-1)[0].detach().cpu().numpy()
        for action in legal_actions(node.state):
            node.children[action] = MCTSNode(apply_action(node.state, action), prior=float(probs[action]))
        return float(value.item())

    @staticmethod
    def _terminal_value(state: WallzState) -> float:
        return 0.0 if state.winner is None else 1.0
