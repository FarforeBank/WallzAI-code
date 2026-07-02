import numpy as np
import torch

from wallz_ai.agents.model import WallzPolicyValueNet, masked_distribution
from wallz_ai.env.wallz_env import WallzEnv
from wallz_ai.training.self_play import play_self_play_games


def test_model_mask_zeroes_invalid_probabilities():
    env = WallzEnv(max_moves=20)
    obs, _ = env.reset(seed=7)
    mask = env.legal_action_mask()
    model = WallzPolicyValueNet(channels=16, residual_blocks=1)
    with torch.no_grad():
        logits, value = model(torch.as_tensor(obs[None], dtype=torch.float32))
        dist = masked_distribution(logits, torch.as_tensor(mask[None], dtype=torch.bool))
        probs = dist.probs[0].numpy()
    assert np.allclose(probs[~mask], 0.0)
    assert probs[mask].sum() > 0.999
    assert -1.0 <= float(value.item()) <= 1.0


def test_self_play_smoke_collects_rollout_without_invalid_actions():
    device = torch.device("cpu")
    model = WallzPolicyValueNet(channels=16, residual_blocks=1).to(device)
    buffer, stats = play_self_play_games(model, device, games=1, max_moves=4)
    assert len(buffer) > 0
    assert stats.invalid_actions == 0
