from __future__ import annotations

import itertools

import pytest
import torch

from sph.japd import (
    BLOCK_LENGTH,
    CANDIDATES,
    FULL_PREFIX_NORMALIZER,
    candidate_gold_ranks,
    clean_support,
    fixed_prompt_balanced_batch_loss,
    japd_per_block_loss,
    matched_candidate_dpace_per_block_loss,
    strict_joint_two_frontier_metric,
)


def make_lattice(
    ranks: list[list[int]],
) -> tuple[torch.Tensor, torch.Tensor]:
    batch = len(ranks)
    ids = torch.arange(
        batch * BLOCK_LENGTH * CANDIDATES, dtype=torch.long
    ).reshape(batch, BLOCK_LENGTH, CANDIDATES)
    gold = torch.empty((batch, BLOCK_LENGTH), dtype=torch.long)
    for b, row_ranks in enumerate(ranks):
        assert len(row_ranks) == BLOCK_LENGTH
        for position, rank in enumerate(row_ranks):
            if rank < 0:
                gold[b, position] = 10_000_000 + b * BLOCK_LENGTH + position
            else:
                gold[b, position] = ids[b, position, rank]
    return ids, gold


def test_clean_support_stops_before_first_missing_or_geometry_mismatch() -> None:
    ids, gold = make_lattice([[0] * BLOCK_LENGTH, [0] * BLOCK_LENGTH])
    gold[0, 5] = -1
    ranks = candidate_gold_ranks(ids, gold)
    target_matches = torch.ones((2, BLOCK_LENGTH), dtype=torch.bool)
    target_matches[1, 7] = False
    support, horizons = clean_support(ranks, target_matches)
    assert horizons.tolist() == [5, 7]
    assert support[0, :5].all() and not support[0, 5:].any()
    assert support[1, :7].all() and not support[1, 7:].any()


def test_japd_fixed_normalizer_and_nonvanishing_later_gradient() -> None:
    ranks = [[1, 2] + [0] * (BLOCK_LENGTH - 2)]
    ids, gold = make_lattice(ranks)
    scores = torch.zeros((1, BLOCK_LENGTH, CANDIDATES), requires_grad=True)
    target = torch.zeros_like(scores)
    matches = torch.ones((1, BLOCK_LENGTH), dtype=torch.bool)
    output = japd_per_block_loss(scores, ids, gold, target, matches)
    assert output.horizons.tolist() == [BLOCK_LENGTH]
    assert output.multi_repair_mask.tolist() == [True]
    assert output.second_error_positions.tolist() == [1]
    assert FULL_PREFIX_NORMALIZER == 136
    output.all_prefix_loss.sum().backward()
    # Position 1 receives direct gold gradient even though position 0 has not
    # been repaired; no prefix-probability product can starve it.
    assert scores.grad is not None
    assert abs(float(scores.grad[0, 1, 2])) > 1e-6


def test_joint_certificate_is_conservative_and_not_mean_diluted() -> None:
    ranks = [[1, 1] + [0] * (BLOCK_LENGTH - 2)]
    ids, gold = make_lattice(ranks)
    scores = torch.full((1, BLOCK_LENGTH, CANDIDATES), -5.0)
    scores[..., 0] = 0.0
    # At the two base errors, make one gold margin -1 and the other +5.
    scores[0, 0, 1] = -1.0
    scores[0, 1, 1] = 5.0
    target = torch.zeros_like(scores)
    matches = torch.ones((1, BLOCK_LENGTH), dtype=torch.bool)
    output = japd_per_block_loss(scores, ids, gold, target, matches)
    prefix = output.joint_prefix_mask[0]
    minimum_margin = output.target_margins[0, prefix].min()
    certificate = -torch.logsumexp(
        -output.target_margins[0, prefix], dim=0
    )
    assert certificate <= minimum_margin
    assert float(certificate) < 0.0
    assert float(output.joint_two_frontier_loss[0]) > 1.0


def test_joint_term_is_zero_without_two_base_errors() -> None:
    ids, gold = make_lattice([[1] + [0] * (BLOCK_LENGTH - 1)])
    scores = torch.zeros((1, BLOCK_LENGTH, CANDIDATES))
    output = japd_per_block_loss(
        scores,
        ids,
        gold,
        torch.zeros_like(scores),
        torch.ones((1, BLOCK_LENGTH), dtype=torch.bool),
    )
    assert output.multi_repair_mask.tolist() == [False]
    assert output.joint_two_frontier_loss.tolist() == [0.0]


def test_horizon_zero_has_exact_zero_loss() -> None:
    ids, gold = make_lattice([[-1] + [0] * (BLOCK_LENGTH - 1)])
    scores = torch.randn((1, BLOCK_LENGTH, CANDIDATES))
    output = japd_per_block_loss(
        scores,
        ids,
        gold,
        torch.randn_like(scores),
        torch.ones((1, BLOCK_LENGTH), dtype=torch.bool),
    )
    assert output.horizons.tolist() == [0]
    assert output.per_block_loss.tolist() == [0.0]


def test_strict_j2_includes_the_second_error_position() -> None:
    ranks = torch.tensor([[1, 0, 2] + [0] * (BLOCK_LENGTH - 3)])
    predicted = ranks.clone()
    matches = torch.ones((1, BLOCK_LENGTH), dtype=torch.bool)
    success = strict_joint_two_frontier_metric(predicted, ranks, matches)
    assert success.denominator == 1 and success.numerator == 1
    predicted[0, 2] = 0
    failure = strict_joint_two_frontier_metric(predicted, ranks, matches)
    assert failure.denominator == 1 and failure.numerator == 0


def test_prompt_balanced_fixed_estimator_is_unbiased() -> None:
    # Prompt A has one block with loss 2.  Prompt B has three blocks with
    # losses 1,3,5.  Exact prompt mean is (2 + 3) / 2 = 2.5.
    losses = torch.tensor([2.0, 1.0, 3.0, 5.0])
    counts = torch.tensor([1.0, 3.0, 3.0, 3.0])
    estimates = []
    for indices in itertools.combinations(range(4), 2):
        index = torch.tensor(indices)
        estimates.append(
            fixed_prompt_balanced_batch_loss(
                losses[index],
                counts[index],
                total_effective_blocks=4,
                total_effective_prompts=2,
            )
        )
    assert torch.stack(estimates).mean().item() == pytest.approx(2.5)


def test_frozen_hyperparameters_reject_posthoc_changes() -> None:
    ids, gold = make_lattice([[0] * BLOCK_LENGTH])
    scores = torch.zeros((1, BLOCK_LENGTH, CANDIDATES))
    kwargs = dict(
        scores=scores,
        candidate_ids=ids,
        gold_ids=gold,
        target_candidate_logits=torch.zeros_like(scores),
        target_matches_gold=torch.ones((1, BLOCK_LENGTH), dtype=torch.bool),
    )
    with pytest.raises(ValueError, match="temperature"):
        japd_per_block_loss(**kwargs, temperature=1.0)
    with pytest.raises(ValueError, match="mixture"):
        japd_per_block_loss(**kwargs, hard_target_mixture=0.8)
    with pytest.raises(ValueError, match="normalizer"):
        japd_per_block_loss(**kwargs, full_prefix_normalizer=120)


def test_candidate_dpace_control_uses_identical_clean_support() -> None:
    ranks = torch.tensor([[1, 2, 3] + [0] * (BLOCK_LENGTH - 3)])
    scores = torch.zeros((1, BLOCK_LENGTH, CANDIDATES), requires_grad=True)
    matches = torch.ones((1, BLOCK_LENGTH), dtype=torch.bool)
    matches[0, 2] = False
    loss = matched_candidate_dpace_per_block_loss(
        scores, ranks, matches, alpha=0.5
    )
    loss.sum().backward()
    assert scores.grad is not None
    assert scores.grad[0, :2].abs().sum() > 0
    assert torch.equal(
        scores.grad[0, 2:], torch.zeros_like(scores.grad[0, 2:])
    )


def test_candidate_dpace_control_freezes_alpha() -> None:
    ranks = torch.zeros((1, BLOCK_LENGTH), dtype=torch.long)
    scores = torch.zeros((1, BLOCK_LENGTH, CANDIDATES))
    matches = torch.ones((1, BLOCK_LENGTH), dtype=torch.bool)
    with pytest.raises(ValueError, match="alpha=0.5"):
        matched_candidate_dpace_per_block_loss(
            scores, ranks, matches, alpha=0.6
        )
