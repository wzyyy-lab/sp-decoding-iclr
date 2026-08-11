from __future__ import annotations

import torch
from torch.nn import functional as F

from train_plc_acceptance import (
    gold_competitor_margin_loss,
    survival_continuation_weights,
)


def test_survival_continuation_weights_emphasize_early_positions() -> None:
    probabilities = torch.tensor([[0.8, 0.7, 0.6, 0.5]])
    weights = survival_continuation_weights(
        probabilities,
        torch.tensor([True]),
        survival_floor=0.5,
        unreachable_weight=0.05,
    )
    assert weights.shape == probabilities.shape
    assert torch.all(weights[:, :-1] > weights[:, 1:])
    torch.testing.assert_close(weights.mean(), torch.tensor(1.0))


def test_unreachable_prefix_retains_small_training_signal() -> None:
    probabilities = torch.full((2, 3), 0.5)
    weights = survival_continuation_weights(
        probabilities,
        torch.tensor([True, False]),
        survival_floor=0.5,
        unreachable_weight=0.1,
    )
    ratio = weights[1] / weights[0]
    torch.testing.assert_close(ratio, torch.full_like(ratio, 0.1))


def test_margin_uses_best_non_gold_competitor() -> None:
    logits = torch.tensor([[[3.0, 2.0, 1.0], [3.0, 2.0, 1.0]]])
    gold = torch.tensor([[0, 1]])
    losses = gold_competitor_margin_loss(
        logits, gold, offset=0.0, temperature=1.0
    )
    torch.testing.assert_close(losses[0, 0], F.softplus(torch.tensor(-1.0)))
    torch.testing.assert_close(losses[0, 1], F.softplus(torch.tensor(1.0)))
