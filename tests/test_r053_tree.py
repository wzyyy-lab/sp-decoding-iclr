from __future__ import annotations

import torch
from torch.nn import functional as F

from sph.fast_r048 import fast_candidate_domino_decode_from_base
from sph.r053_tree import (
    build_budgeted_trie,
    fast_candidate_domino_beam_from_base,
    full_pool_oracle_acceptance,
    hindsight_budget_acceptance,
    pack_tree_tensors,
    simulated_tree_acceptance,
    traverse_tree_logits,
    traverse_tree_logits_path,
    traverse_tree_logits_tensor,
)


class _ToyDomino(torch.nn.Module):
    def __init__(self, hidden_width: int, state_width: int, code_width: int, vocab: int):
        super().__init__()
        self.use_bias_norm = False
        self.prefix_gru = torch.nn.GRU(
            hidden_width, state_width, batch_first=True
        )
        self.embed_proj = torch.nn.Sequential(
            torch.nn.Linear(hidden_width + state_width, code_width, bias=False),
            torch.nn.SiLU(),
            torch.nn.Linear(code_width, vocab, bias=False),
        )


def _fixture():
    paths = torch.tensor(
        [
            [1, 2, 3, 4],
            [1, 2, 8, 9],
            [1, 7, 8, 9],
            [6, 7, 8, 9],
        ]
    )
    scores = torch.tensor([-1.0, -2.0, -3.0, -4.0])
    trunk = paths[0]
    return paths, scores, trunk


def test_budget_retains_complete_trunk_and_is_prefix_closed():
    paths, scores, trunk = _fixture()
    tree = build_budgeted_trie(paths, scores, trunk, budget=7)
    assert tree.used_nodes_including_anchor == 7
    assert tree.full_nodes_including_anchor == 14
    for depth in range(1, 5):
        assert tuple(trunk[:depth].tolist()) in tree.selected_prefixes
    for prefix in tree.selected_prefixes:
        if prefix:
            assert prefix[:-1] in tree.selected_prefixes


def test_budgeted_tree_acceptance_uses_only_selected_target_prefix():
    paths, scores, trunk = _fixture()
    tree = build_budgeted_trie(paths, scores, trunk, budget=7)
    assert simulated_tree_acceptance(tree, trunk) == 4
    alternative = torch.tensor([1, 2, 8, 9])
    accepted = simulated_tree_acceptance(tree, alternative)
    assert 2 <= accepted <= 4
    assert simulated_tree_acceptance(tree, torch.tensor([6, 0, 0, 0])) == 0


def test_packed_mask_contains_only_cache_and_ancestors():
    paths, scores, trunk = _fixture()
    tree = build_budgeted_trie(paths, scores, trunk, budget=9)
    inputs, positions, mask = pack_tree_tensors(
        tree,
        anchor_token_id=42,
        prefix_length=5,
        device=torch.device("cpu"),
        mask_dtype=torch.bfloat16,
    )
    assert inputs.shape == (1, 9)
    assert positions.shape == (1, 9)
    assert mask.shape == (1, 1, 9, 14)
    assert mask.dtype == torch.bfloat16
    assert torch.isfinite(mask[0, 0, :, :5]).all()
    for row in range(9):
        visible = set(
            (torch.isfinite(mask[0, 0, row, 5:])).nonzero()[:, 0].tolist()
        )
        expected = set()
        cursor = row
        while cursor >= 0:
            expected.add(cursor)
            cursor = tree.packed_parent_rows[cursor]
        assert visible == expected
        assert int(positions[0, row]) == 5 + tree.packed_depths[row]


def test_logit_traversal_matches_simulated_prefix_and_bonus():
    paths, scores, trunk = _fixture()
    tree = build_budgeted_trie(paths, scores, trunk, budget=7)
    vocab = 16
    logits = torch.full((1, tree.used_nodes_including_anchor, vocab), -10.0)
    target = trunk.tolist()

    tokens_by_row = [None, *tree.packed_token_ids]
    children = [dict() for _ in tokens_by_row]
    for row in range(1, len(tokens_by_row)):
        children[tree.packed_parent_rows[row]][tokens_by_row[row]] = row
    current = 0
    for token in target:
        logits[0, current, token] = 10.0
        current = children[current][token]
    logits[0, current, 11] = 10.0
    accepted, bonus = traverse_tree_logits(tree, logits)
    assert accepted == 4
    assert bonus == 11
    tensor_accepted, tensor_bonus = traverse_tree_logits_tensor(tree, logits)
    assert int(tensor_accepted[0]) == accepted
    assert int(tensor_bonus[0]) == bonus
    path, path_bonus, rows = traverse_tree_logits_path(tree, logits)
    assert path == tuple(target)
    assert path_bonus == bonus
    assert len(rows) == accepted


def test_invalid_budget_cannot_drop_trunk():
    paths, scores, trunk = _fixture()
    try:
        build_budgeted_trie(paths, scores, trunk, budget=4)
    except ValueError as error:
        assert "full trunk" in str(error)
    else:
        raise AssertionError("budget smaller than anchor+trunk must fail")


def test_full_pool_and_hindsight_budget_are_separate_upper_bounds():
    paths, scores, trunk = _fixture()
    target = torch.tensor([1, 2, 8, 9])
    assert full_pool_oracle_acceptance(paths, target) == 4
    # Root+trunk costs five rows.  Sharing the first two target prefixes with
    # trunk means one extra node reaches depth three and two reach depth four.
    assert hindsight_budget_acceptance(paths, trunk, target, budget=5) == 2
    assert hindsight_budget_acceptance(paths, trunk, target, budget=6) == 3
    assert hindsight_budget_acceptance(paths, trunk, target, budget=7) == 4


def test_fast_beam_protects_exact_k64_trunk_with_fixed_k16_support():
    torch.manual_seed(7)
    vocab, width, state_width, code_width, horizon = 80, 8, 6, 5, 4
    target_weight = torch.randn(vocab, width)
    hidden = torch.randn(1, horizon, width)
    anchors = torch.tensor([3])
    domino = _ToyDomino(width, state_width, code_width, vocab).eval()
    base_logits = F.linear(hidden, target_weight)
    beam = fast_candidate_domino_beam_from_base(
        domino=domino,
        target_weight=target_weight,
        anchors=anchors,
        hidden=hidden,
        base_logits=base_logits,
        candidate_pool_topk=64,
        tree_support_size=16,
        beam_width=16,
    )
    control = fast_candidate_domino_decode_from_base(
        domino=domino,
        target_weight=target_weight,
        anchors=anchors,
        hidden=hidden,
        base_logits=base_logits,
        candidate_topk=64,
    )
    assert beam.token_ids.shape == (1, 16, horizon)
    assert beam.candidate_ids.shape == (1, horizon, 16)
    assert torch.equal(beam.trunk_token_ids, control.token_ids)
    assert (beam.token_ids[0] == beam.trunk_token_ids[0]).all(dim=1).any()
    canonical_top15 = base_logits.float().topk(16, dim=-1).indices[..., :15]
    assert torch.equal(beam.candidate_ids[..., :15], canonical_top15)
    assert beam.candidate_ids.eq(
        beam.trunk_token_ids.unsqueeze(-1)
    ).any(dim=-1).all()
