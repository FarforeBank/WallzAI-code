from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from .model import WallzPolicyValueNet, masked_distribution


@dataclass
class PPOBatch:
    obs: np.ndarray
    masks: np.ndarray
    actions: np.ndarray
    old_log_probs: np.ndarray
    returns: np.ndarray
    advantages: np.ndarray


def sample_action(model: WallzPolicyValueNet, obs: np.ndarray, mask: np.ndarray, device: torch.device, deterministic: bool = False):
    model.eval()
    obs_t = torch.as_tensor(obs[None], dtype=torch.float32, device=device)
    mask_t = torch.as_tensor(mask[None], dtype=torch.bool, device=device)
    with torch.no_grad():
        logits, value = model(obs_t)
        dist = masked_distribution(logits, mask_t)
        action = torch.argmax(dist.logits, dim=-1) if deterministic else dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
    return int(action.item()), float(log_prob.item()), float(value.item()), float(entropy.item())


def ppo_update(model, optimizer, batch: PPOBatch, device: torch.device, epochs: int = 4, minibatch_size: int = 256, clip_range: float = 0.2, value_coef: float = 0.5, entropy_coef: float = 0.01, max_grad_norm: float = 1.0) -> dict[str, float]:
    model.train()
    n = len(batch.actions)
    idxs = np.arange(n)
    metrics = {"policy_loss": [], "value_loss": [], "entropy": [], "total_loss": []}
    obs = torch.as_tensor(batch.obs, dtype=torch.float32, device=device)
    masks = torch.as_tensor(batch.masks, dtype=torch.bool, device=device)
    actions = torch.as_tensor(batch.actions, dtype=torch.long, device=device)
    old_log_probs = torch.as_tensor(batch.old_log_probs, dtype=torch.float32, device=device)
    returns = torch.as_tensor(batch.returns, dtype=torch.float32, device=device)
    advantages = torch.as_tensor(batch.advantages, dtype=torch.float32, device=device)
    advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
    for _ in range(epochs):
        np.random.shuffle(idxs)
        for start in range(0, n, minibatch_size):
            mb = idxs[start : start + minibatch_size]
            logits, values = model(obs[mb])
            dist = masked_distribution(logits, masks[mb])
            new_log_probs = dist.log_prob(actions[mb])
            ratio = torch.exp(new_log_probs - old_log_probs[mb])
            unclipped = ratio * advantages[mb]
            clipped = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range) * advantages[mb]
            policy_loss = -torch.min(unclipped, clipped).mean()
            value_loss = F.mse_loss(values, returns[mb])
            entropy = dist.entropy().mean()
            loss = policy_loss + value_coef * value_loss - entropy_coef * entropy
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            metrics["policy_loss"].append(float(policy_loss.detach().cpu()))
            metrics["value_loss"].append(float(value_loss.detach().cpu()))
            metrics["entropy"].append(float(entropy.detach().cpu()))
            metrics["total_loss"].append(float(loss.detach().cpu()))
    return {k: float(np.mean(v)) if v else 0.0 for k, v in metrics.items()}
