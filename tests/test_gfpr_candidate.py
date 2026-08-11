from __future__ import annotations

import torch
from torch import nn

from sph.gfpr import all_position_onpolicy_decode, all_position_teacher_logits
from sph.gfpr_candidate import (
    GFPRCandidateHead,
    candidate_dense_dpace_loss,
    candidate_dense_margin_loss,
    candidate_frontier_margin_loss,
    candidate_target_distillation_loss,
    select_anchor_early_exit_feature,
)


class TinyDomino(nn.Module):
    def __init__(self, width: int, state: int, code: int, vocab: int) -> None:
        super().__init__()
        self.prefix_gru = nn.GRU(width, state, batch_first=True, bias=False)
        self.embed_proj = nn.Sequential(
            nn.Linear(width + state, code, bias=False),
            nn.SiLU(),
            nn.Linear(code, vocab, bias=False),
        )
        self.use_bias_norm = False
        self.use_bias_gate = False


def test_anchor_early_exit_feature_uses_requested_layer_and_current_anchor() -> None:
    hidden_states = tuple(
        torch.full((1, 7, 5), float(layer))
        + torch.arange(7).view(1, 7, 1)
        for layer in range(6)
    )
    selected = select_anchor_early_exit_feature(
        hidden_states,
        context_length=3,
        early_layers=4,
    )
    assert torch.equal(selected, torch.full((5,), 7.0))
    assert not torch.equal(selected, hidden_states[4][0, 2])


def test_candidate_head_matches_domino_scores_inside_lattice() -> None:
    torch.manual_seed(12)
    width, state, code, vocab, positions, candidates = 7, 5, 3, 19, 4, 6
    domino = TinyDomino(width, state, code, vocab)
    target_weight = torch.randn(vocab, width)
    hidden = torch.randn(2, positions, width)
    anchors = torch.tensor([2, 4])
    gold = torch.tensor([[1, 3, 5, 7], [2, 6, 8, 9]])
    base = torch.nn.functional.linear(hidden, target_weight)
    candidate_logits, candidate_ids = base.topk(candidates, dim=-1)
    head = GFPRCandidateHead.from_domino(
        domino,
        target_weight,
        positions=positions,
        candidates=candidates,
    )
    scores = head.teacher_scores(
        anchors=anchors,
        gold=gold,
        hidden=hidden,
        candidate_ids=candidate_ids,
        candidate_logits=candidate_logits,
    )
    full = all_position_teacher_logits(
        domino=domino,
        target_weight=target_weight,
        anchors=anchors,
        gold=gold,
        hidden=hidden,
        position_zero_scale=0.0,
    )
    assert torch.allclose(scores, full.gather(-1, candidate_ids), atol=1e-5)


def test_candidate_teacher_and_selected_token_rollout_agree() -> None:
    torch.manual_seed(21)
    width, state, code, vocab, positions, candidates = 6, 4, 3, 17, 5, 5
    domino = TinyDomino(width, state, code, vocab)
    target_weight = torch.randn(vocab, width)
    hidden = torch.randn(3, positions, width)
    anchors = torch.tensor([1, 2, 3])
    base = torch.nn.functional.linear(hidden, target_weight)
    candidate_logits, candidate_ids = base.topk(candidates, dim=-1)
    head = GFPRCandidateHead.from_domino(
        domino,
        target_weight,
        positions=positions,
        candidates=candidates,
    )
    decoded = head.decode(
        anchors=anchors,
        hidden=hidden,
        candidate_ids=candidate_ids,
        candidate_logits=candidate_logits,
    )
    assert torch.equal(decoded.token_ids[:, 0], candidate_ids[:, 0, 0])
    teacher = head.teacher_scores(
        anchors=anchors,
        gold=decoded.token_ids,
        hidden=hidden,
        candidate_ids=candidate_ids,
        candidate_logits=candidate_logits,
    )
    teacher_tokens = candidate_ids.gather(
        -1, teacher.argmax(dim=-1, keepdim=True)
    ).squeeze(-1)
    assert torch.equal(teacher_tokens, decoded.token_ids)


def test_candidate_frontier_skips_unavailable_gold() -> None:
    candidate_ids = torch.tensor([[[0, 1, 2], [3, 4, 5]]])
    scores = torch.tensor([[[3.0, 2.0, 1.0], [3.0, 2.0, 1.0]]])
    gold = torch.tensor([[9, 3]])
    result = candidate_frontier_margin_loss(scores, candidate_ids, gold)
    assert result.frontier.tolist() == [0]
    assert result.repairable_blocks.tolist() == [False]
    assert result.gold_available_at_frontier.tolist() == [False]
    assert float(result.loss.detach()) == 0.0


def test_dense_candidate_loss_censors_first_missing_gold_and_suffix() -> None:
    candidate_ids = torch.tensor(
        [[[0, 1, 2], [3, 4, 5], [6, 7, 8]]]
    )
    scores = torch.zeros(1, 3, 3, requires_grad=True)
    gold = torch.tensor([[1, 9, 7]])
    result = candidate_dense_dpace_loss(scores, candidate_ids, gold, alpha=0.5)
    assert result.active_positions.tolist() == [[True, False, False]]
    result.loss.backward()
    assert bool(scores.grad[0, 0].abs().sum() > 0)
    assert float(scores.grad[0, 1:].abs().sum()) == 0.0


def test_dense_candidate_loss_front_loads_reachable_positions() -> None:
    candidate_ids = torch.tensor(
        [[[0, 1], [2, 3], [4, 5]]]
    )
    scores = torch.zeros(1, 3, 2)
    gold = torch.tensor([[0, 2, 4]])
    result = candidate_dense_dpace_loss(scores, candidate_ids, gold, alpha=0.5)
    weights = result.position_weights[0]
    assert bool(weights[0] > weights[1] > weights[2] > 0)


def test_dense_margin_is_zero_for_safe_gold_and_censors_suffix() -> None:
    candidate_ids = torch.tensor(
        [[[0, 1, 2], [3, 4, 5], [6, 7, 8]]]
    )
    scores = torch.tensor(
        [[[3.0, 1.0, 0.0], [2.0, 1.0, 0.0], [2.0, 1.0, 0.0]]],
        requires_grad=True,
    )
    gold = torch.tensor([[0, 9, 7]])
    result = candidate_dense_margin_loss(
        scores, candidate_ids, gold, margin=0.05
    )
    assert result.active_positions.tolist() == [[True, False, False]]
    assert float(result.loss.detach()) == 0.0


def test_zero_initialized_adapter_preserves_domino_candidate_scores() -> None:
    torch.manual_seed(31)
    domino = TinyDomino(6, 4, 3, 17)
    target_weight = torch.randn(17, 6)
    head = GFPRCandidateHead.from_domino(
        domino, target_weight, positions=4, candidates=5
    )
    assert int(torch.count_nonzero(head.residual_up.weight)) == 0
    assert head.adapter_rank == 3


def test_released_union_is_exact_domino_identity_at_zero_residual() -> None:
    torch.manual_seed(44)
    width, state, code, vocab, positions, candidates = 7, 5, 4, 31, 5, 6
    domino = TinyDomino(width, state, code, vocab)
    target_weight = torch.randn(vocab, width)
    hidden = torch.randn(3, positions, width)
    anchors = torch.tensor([1, 2, 3])
    base_ids = torch.nn.functional.linear(hidden, target_weight).topk(
        candidates, dim=-1
    ).indices
    head = GFPRCandidateHead.from_domino(
        domino,
        target_weight,
        positions=positions,
        candidates=candidates,
    )
    reference = all_position_onpolicy_decode(
        domino=domino,
        target_weight=target_weight,
        anchors=anchors,
        hidden=hidden,
        position_zero_scale=0.0,
        topk=candidates,
    )
    decoded = head.decode_with_released_union(
        anchors=anchors,
        hidden=hidden,
        base_candidate_ids=base_ids,
    )
    assert torch.equal(decoded.released_token_ids, reference.token_ids)
    assert torch.equal(decoded.token_ids, reference.token_ids)
    assert bool(
        decoded.candidate_ids.eq(reference.token_ids.unsqueeze(-1)).any(dim=-1).all()
    )
    teacher = head.teacher_stored_union_scores(
        anchors=anchors,
        gold=reference.token_ids,
        hidden=hidden,
        base_candidate_ids=base_ids,
        released_token_ids=reference.token_ids,
    )
    selected = teacher.candidate_ids.gather(
        -1, teacher.scores.argmax(dim=-1, keepdim=True)
    ).squeeze(-1)
    assert torch.equal(selected, reference.token_ids)


def test_candidate_basis_is_frozen_and_not_counted() -> None:
    domino = TinyDomino(6, 4, 3, 101)
    head = GFPRCandidateHead.from_domino(
        domino, torch.randn(101, 6), positions=4, candidates=5
    )
    names = {name for name, _ in head.named_parameters()}
    assert "candidate_basis" not in names
    assert "token_embeddings" not in names
    assert head.trainable_parameter_count < sum(
        parameter.numel() for parameter in domino.parameters()
    )


def test_target_distillation_uses_only_released_reachable_frontier() -> None:
    union_ids = torch.tensor(
        [[[0, 1, 2], [3, 4, 8], [6, 7, 5]]]
    )
    base_ids = torch.tensor(
        [[[0, 1, 2], [3, 4, 5], [6, 7, 8]]]
    )
    released = torch.tensor([[0, 8, 5]])
    gold = torch.tensor([[0, 4, 7]])
    target_base = torch.tensor(
        [[[4.0, 1.0, 0.0], [0.0, 3.0, 1.0], [0.0, 2.0, 1.0]]]
    )
    target_released = torch.tensor([[4.0, 0.5, 0.5]])
    student = torch.zeros(1, 3, 3, requires_grad=True)
    result = candidate_target_distillation_loss(
        student,
        union_ids,
        base_ids,
        target_base,
        released,
        target_released,
        gold,
        torch.tensor([1]),
    )
    assert result.active_positions.tolist() == [[True, True, False]]
    assert result.raw_teacher_top1_matches_gold.tolist() == [
        [True, True, True]
    ]
    (result.kl_loss + result.advantage_loss).backward()
    assert bool(student.grad[0, :2].abs().sum() > 0)
    assert float(student.grad[0, 2].abs().sum()) == 0.0


def test_target_boundary_adapter_is_zero_init_domino_identity() -> None:
    torch.manual_seed(73)
    width, state, code, vocab, positions, candidates = 7, 5, 4, 31, 5, 6
    domino = TinyDomino(width, state, code, vocab)
    target_weight = torch.randn(vocab, width)
    hidden = torch.randn(2, positions, width)
    anchors = torch.tensor([1, 2])
    base_ids = torch.nn.functional.linear(hidden, target_weight).topk(
        candidates, dim=-1
    ).indices
    boundary = torch.randn(2, 13)
    head = GFPRCandidateHead.from_domino(
        domino,
        target_weight,
        positions=positions,
        candidates=candidates,
        boundary_width=13,
    )
    reference = all_position_onpolicy_decode(
        domino=domino,
        target_weight=target_weight,
        anchors=anchors,
        hidden=hidden,
        position_zero_scale=0.0,
        topk=candidates,
    )
    decoded = head.decode_with_released_union(
        anchors=anchors,
        hidden=hidden,
        base_candidate_ids=base_ids,
        target_boundary=boundary,
    )
    assert torch.equal(decoded.token_ids, reference.token_ids)
    assert head.boundary_down is not None
    assert head.boundary_width == 13


def test_anchor_context_adapter_reaches_all_new_projections_after_zero_step() -> None:
    torch.manual_seed(79)
    width, state, code, vocab, positions, candidates = 7, 5, 4, 31, 5, 6
    domino = TinyDomino(width, state, code, vocab)
    target_weight = torch.randn(vocab, width)
    hidden = torch.randn(2, positions, width)
    anchors = torch.tensor([1, 2])
    gold = torch.randint(0, vocab, (2, positions))
    base_ids = torch.nn.functional.linear(hidden, target_weight).topk(
        candidates, dim=-1
    ).indices
    context = torch.randn(2, 13)
    head = GFPRCandidateHead.from_domino(
        domino,
        target_weight,
        positions=positions,
        candidates=candidates,
        adapter_rank=3,
        boundary_width=13,
    )
    head.requires_grad_(False)
    for module in (head.residual_down, head.boundary_down, head.residual_up):
        assert module is not None
        module.requires_grad_(True)

    first = head.teacher_union_scores(
        anchors=anchors,
        gold=gold,
        hidden=hidden,
        base_candidate_ids=base_ids,
        target_boundary=context,
    ).scores
    first.float().square().mean().backward()
    assert head.residual_up.weight.grad is not None
    assert bool(head.residual_up.weight.grad.abs().sum() > 0)
    with torch.no_grad():
        head.residual_up.weight.add_(-0.01 * head.residual_up.weight.grad)
    head.zero_grad(set_to_none=True)

    second = head.teacher_union_scores(
        anchors=anchors,
        gold=gold,
        hidden=hidden,
        base_candidate_ids=base_ids,
        target_boundary=context,
    ).scores
    second.float().square().mean().backward()
    for module in (head.residual_down, head.boundary_down, head.residual_up):
        assert module is not None
        assert module.weight.grad is not None
        assert bool(torch.isfinite(module.weight.grad).all())
        assert bool(module.weight.grad.abs().sum() > 0)
