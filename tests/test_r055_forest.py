from __future__ import annotations

import torch

from sph.r055_forest import (
    pack_padded_forest,
    structural_forest_acceptance,
    traverse_padded_forest,
)


def _logits(posterior: list[int], vocab: int = 32) -> torch.Tensor:
    values = torch.full((1, len(posterior), vocab), -10.0)
    for row, token in enumerate(posterior):
        values[0, row, token] = 10.0
    return values


def test_pack_padded_forest_has_block_diagonal_visibility() -> None:
    paths = torch.tensor([[2, 3, 4], [2, 8, 9]], dtype=torch.long)
    inputs, positions, mask = pack_padded_forest(
        paths,
        anchor_token_id=1,
        prefix_length=2,
        mask_dtype=torch.float32,
    )
    assert inputs.tolist() == [[1, 2, 3, 4, 2, 8, 9]]
    assert positions.tolist() == [[2, 3, 4, 5, 3, 4, 5]]
    visible = mask[0, 0].eq(0)
    # Prefix keys are always visible.
    assert bool(visible[:, :2].all())
    # Anchor sees itself but no forest tokens.
    assert visible[0, 2:].tolist() == [True, False, False, False, False, False, False]
    # Last row of path zero sees anchor and all path-zero rows only.
    assert visible[3, 2:].tolist() == [True, True, True, True, False, False, False]
    # Middle row of path one cannot see either path-zero or its own future row.
    assert visible[5, 2:].tolist() == [True, False, False, False, True, True, False]


def test_traversal_checks_duplicate_first_token_paths_independently() -> None:
    paths = torch.tensor([[2, 3, 4], [2, 8, 9]], dtype=torch.long)
    # rows: anchor, path0 tokens, path1 tokens.  Both accept token 2, but only
    # path1 continues with 8,9; choosing the first duplicate child would fail.
    result = traverse_padded_forest(paths, _logits([2, 8, 0, 0, 8, 9, 7]))
    assert result.per_path_accepted.tolist() == [1, 3]
    assert result.accepted.tolist() == [3]
    assert result.selected_path.tolist() == [1]
    assert result.next_token.tolist() == [7]


def test_traversal_emits_anchor_posterior_when_no_path_starts() -> None:
    paths = torch.tensor([[2, 3], [4, 5]], dtype=torch.long)
    result = traverse_padded_forest(paths, _logits([7, 0, 0, 0, 0]))
    assert result.per_path_accepted.tolist() == [0, 0]
    assert result.accepted.tolist() == [0]
    assert result.selected_path.tolist() == [0]
    assert result.next_token.tolist() == [7]


def test_traversal_uses_rejecting_row_as_next_token() -> None:
    paths = torch.tensor([[2, 3, 4], [6, 7, 8]], dtype=torch.long)
    # Path zero accepts 2,3 then row for token3 predicts 11 instead of token4.
    result = traverse_padded_forest(paths, _logits([2, 3, 11, 0, 0, 0, 0]))
    assert result.accepted.tolist() == [2]
    assert result.next_token.tolist() == [11]


def test_structural_acceptance_selects_longest_clean_prefix() -> None:
    paths = torch.tensor([[2, 3, 0], [2, 8, 9], [5, 8, 9]])
    target = torch.tensor([2, 8, 7])
    assert int(structural_forest_acceptance(paths, target)) == 2
