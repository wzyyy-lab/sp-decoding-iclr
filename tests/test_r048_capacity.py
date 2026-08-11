from __future__ import annotations

import torch

from sph.r048_capacity import (
    decode_earliest_threshold,
    exact_earliest_one_lengths,
    r048_capacity_loss,
    select_zero_harm_threshold,
)


def test_capacity_loss_masks_suffix_and_has_finite_gradient() -> None:
    torch.manual_seed(2)
    batch, positions, candidates = 2, 4, 3
    candidate_ids = torch.tensor(
        [
            [[1, 9, 8], [2, 9, 8], [3, 9, 8], [4, 9, 8]],
            [[1, 9, 8], [2, 9, 8], [3, 9, 8], [4, 9, 8]],
        ]
    )
    proposal = torch.tensor([[1, 2, 9, 9], [1, 9, 9, 9]])
    gold = torch.tensor([[1, 2, 3, 4], [1, 2, 3, 4]])
    base = torch.randn(batch, positions, candidates)
    # Make each stored proposal the base winner.
    for row in range(batch):
        for position in range(positions):
            index = int(candidate_ids[row, position].eq(proposal[row, position]).nonzero()[0])
            base[row, position, index] = 4.0
    delta = torch.zeros_like(base, requires_grad=True)
    target = torch.randn_like(base)
    valid = torch.ones(batch, positions, dtype=torch.bool)
    output = r048_capacity_loss(
        base_scores=base,
        lens_delta=delta,
        candidate_ids=candidate_ids,
        proposal=proposal,
        target_candidate_logits=target,
        valid_teacher_mask=valid,
        accepted=torch.tensor([2, 1]),
        oracle_accepted=torch.tensor([4, 3]),
    )
    assert output.active_rows.item() == 5
    assert torch.isfinite(output.loss)
    output.loss.backward()
    assert delta.grad is not None
    # Rows after each original frontier are completely unsupervised.
    assert torch.count_nonzero(delta.grad[0, 3]).item() == 0
    assert torch.count_nonzero(delta.grad[1, 2:]).item() == 0


def test_threshold_selection_requires_zero_harm() -> None:
    proposal = torch.tensor([[1, 9, 3], [1, 2, 9]])
    gold = torch.tensor([[1, 2, 3], [1, 2, 3]])
    candidate_ids = torch.tensor(
        [
            [[1, 7], [9, 2], [3, 8]],
            [[1, 7], [2, 8], [9, 3]],
        ]
    )
    scores = torch.tensor(
        [
            [[2.0, 1.0], [1.0, 3.0], [2.0, 1.0]],
            [[2.0, 1.0], [2.0, 1.0], [1.0, 2.5]],
        ]
    )
    report = select_zero_harm_threshold(
        sample_ids=["a", "b"],
        proposal=proposal,
        verifier_top1=gold,
        candidate_ids=candidate_ids,
        adjusted_scores=scores,
        baseline_lengths=torch.tensor([1, 2]),
        oracle_lengths=torch.tensor([3, 3]),
    )
    assert report["harmful_blocks"] == 0
    assert report["eal_prompt_balanced"] == 3.0
    assert report["oracle_gain_recovery"] == 1.0


def test_decode_threshold_changes_only_earliest_position() -> None:
    proposal = torch.tensor([[1, 4, 7]])
    candidates = torch.tensor([[[1, 2], [4, 5], [7, 8]]])
    scores = torch.tensor([[[2.0, 1.0], [1.0, 3.0], [1.0, 4.0]]])
    decoded = decode_earliest_threshold(
        proposal=proposal,
        candidate_ids=candidates,
        adjusted_scores=scores,
        threshold=0.0,
    )
    assert decoded.tolist() == [[1, 5, 7]]


def test_exact_lengths_use_rerun_oracle_only_for_correct_frontier() -> None:
    proposal = torch.tensor(
        [[1, 9, 3, 4], [1, 2, 9, 4], [1, 2, 9, 4], [1, 2, 9, 4]]
    )
    verifier = torch.tensor(
        [[1, 2, 8, 4], [1, 2, 3, 8], [1, 2, 3, 8], [1, 2, 3, 8]]
    )
    decoded = torch.tensor(
        [
            [7, 9, 3, 4],  # false positive before frontier -> immediate harm
            [1, 2, 3, 4],  # correct frontier repair -> rerun oracle
            [1, 2, 7, 4],  # wrong frontier token -> unchanged acceptance
            [1, 2, 9, 8],  # change after frontier -> unchanged acceptance
        ]
    )
    lengths = exact_earliest_one_lengths(
        proposal=proposal,
        decoded=decoded,
        verifier_top1=verifier,
        baseline_lengths=torch.tensor([1, 2, 2, 2]),
        oracle_lengths=torch.tensor([3, 4, 4, 4]),
    )
    assert lengths.tolist() == [0, 4, 2, 2]
