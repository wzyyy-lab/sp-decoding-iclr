from __future__ import annotations

from pathlib import Path
import sys

import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from train_domino_cached_head import (  # noqa: E402
    auf_reach_mask,
    best_competitor_margin_loss,
    collate,
    dpace_weights,
    objective_loss,
)


def test_collate_accepts_training_cache_without_domino_rollout_fields() -> None:
    record = {
        "sample_id": "train-0",
        "domain": "code",
        "anchor_token_id": 7,
        "gold_ids": torch.tensor([8, 9, 10]),
        "parallel_hidden": torch.randn(3, 4),
    }
    batch = collate([record])
    assert batch["cached_released_lengths"].tolist() == [-1]
    assert batch["cached_released_ids"].tolist() == [[-1, -1, -1]]
    assert batch["gold"].tolist() == [[8, 9, 10]]


def test_auf_mask_includes_first_mismatch_and_stops_after() -> None:
    predicted = torch.tensor([[1, 2, 9, 4, 5], [9, 2, 3, 4, 5]])
    gold = torch.tensor([[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]])
    assert auf_reach_mask(predicted, gold).tolist() == [
        [1.0, 1.0, 1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0, 0.0],
    ]


def test_dpace_weights_are_positive_and_front_loaded() -> None:
    logits = torch.zeros((2, 4, 5))
    gold = torch.zeros((2, 4), dtype=torch.long)
    weights = dpace_weights(logits, gold, smoothing=0.5)
    assert weights.shape == gold.shape
    assert torch.all(weights > 0)
    assert torch.all(weights[:, :-1] > weights[:, 1:])


def test_dpace_suffix_weights_include_first_draft_candidate_probability() -> None:
    gold = torch.zeros((1, 4), dtype=torch.long)
    high_first = torch.zeros((1, 4, 3))
    low_first = high_first.clone()
    high_first[0, 0, 0] = 4.0
    low_first[0, 0, 1] = 4.0
    high_weights = dpace_weights(high_first, gold, smoothing=0.5)[:, 1:]
    low_weights = dpace_weights(low_first, gold, smoothing=0.5)[:, 1:]
    assert torch.all(high_weights > low_weights)


def test_margin_loss_targets_best_non_gold_logit() -> None:
    logits = torch.tensor([[[5.0, 4.0, 1.0], [1.0, 3.0, 4.0]]])
    gold = torch.tensor([[0, 1]])
    loss = best_competitor_margin_loss(
        logits, gold, temperature=1.0, offset=0.0
    )
    expected = torch.nn.functional.softplus(torch.tensor([-1.0, 1.0]))
    assert torch.allclose(loss[0], expected)


def test_all_objectives_have_finite_gradient() -> None:
    torch.manual_seed(3)
    logits = torch.randn((2, 5, 7), requires_grad=True)
    gold = torch.randint(0, 7, (2, 5))
    for objective in [
        "decay_ce",
        "dpace",
        "dpace_normalized",
        "auf",
        "auf_decay",
        "breaker",
        "breaker_margin",
    ]:
        current = logits.clone().detach().requires_grad_(True)
        loss, diagnostics = objective_loss(
            all_logits=current,
            gold=gold,
            objective=objective,
            gamma=7.0,
            dpace_smoothing=0.5,
        )
        loss.backward()
        assert torch.isfinite(loss)
        assert current.grad is not None and torch.isfinite(current.grad).all()
        assert diagnostics["weight_sum"] >= 0.0


def test_head_objectives_zero_when_frozen_first_token_is_wrong() -> None:
    logits = torch.zeros((2, 4, 5), requires_grad=True)
    gold = torch.tensor([[1, 0, 0, 0], [0, 0, 0, 0]])
    # Argmax is token 0, so the first example is unreachable by the head.
    for objective in ["auf", "auf_decay", "breaker", "breaker_margin"]:
        current = logits.clone().detach().requires_grad_(True)
        loss, _ = objective_loss(
            all_logits=current,
            gold=gold,
            objective=objective,
            gamma=7.0,
            dpace_smoothing=0.5,
        )
        loss.backward()
        assert torch.equal(current.grad[0], torch.zeros_like(current.grad[0]))


def test_standard_decay_and_dpace_keep_suffix_signal_when_first_is_wrong() -> None:
    logits = torch.zeros((1, 4, 5), requires_grad=True)
    gold = torch.tensor([[1, 0, 0, 0]])
    for objective in ["decay_ce", "dpace", "dpace_normalized"]:
        current = logits.clone().detach().requires_grad_(True)
        loss, _ = objective_loss(
            all_logits=current,
            gold=gold,
            objective=objective,
            gamma=7.0,
            dpace_smoothing=0.5,
        )
        loss.backward()
        assert torch.count_nonzero(current.grad[:, 1:]) > 0
