from __future__ import annotations

import torch

from evaluate_r050_target_seeded import (
    paired_summary,
    split_unsplit_numerical_parity,
    split_verifier_logits,
)


def test_split_logits_align_anchor_then_all_suffix_rows_including_bonus() -> None:
    anchor = torch.tensor([[10.0, 1.0, 0.0]])
    suffix = torch.arange(2 * 16 * 3, dtype=torch.float32).view(2, 16, 3)
    anchor = anchor.expand(2, -1)
    aligned = split_verifier_logits(anchor, suffix)
    assert aligned.shape == (2, 17, 3)
    assert torch.equal(aligned[:, 0], anchor)
    assert torch.equal(aligned[:, 1:16], suffix[:, :15])
    # Parity row 16 is the emitted bonus decision from suffix row 15.
    assert torch.equal(aligned[:, 16], suffix[:, 15])


def test_parity_ignores_row_constants_and_marks_real_mismatch_stable() -> None:
    unsplit = torch.tensor([[[5.0, 1.0, 0.0], [4.0, 3.0, 0.0]]])
    split = unsplit + 7.0
    parity = split_unsplit_numerical_parity(split, unsplit)
    assert parity["stable"].all()
    assert parity["matches"].all()
    assert torch.count_nonzero(parity["epsilon"]).item() == 0

    changed = split.clone()
    changed[0, 1] = torch.tensor([7.0, 12.0, 7.0])
    parity = split_unsplit_numerical_parity(changed, unsplit)
    assert parity["stable"][0, 1]
    assert not parity["matches"][0, 1]


def test_paired_summary_counts_gain_and_loss() -> None:
    assert paired_summary([1, 4, 3], [3, 2, 3]) == {
        "gained_blocks": 1,
        "lost_blocks": 1,
        "unchanged_blocks": 1,
        "gained_tokens": 2,
        "lost_tokens": 2,
        "net_tokens": 0,
    }
