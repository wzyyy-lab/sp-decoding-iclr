from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from sph.fast_r048 import (
    R048TunedLens,
    candidate_union_with_proposal,
    earliest_one_decision,
    fast_candidate_domino_decode,
    fast_candidate_domino_decode_from_base,
    repair_earliest_frontier,
    sequential_perfect_frontier_repairs,
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


def test_candidate_decode_matches_full_vocab_when_k_is_vocab() -> None:
    torch.manual_seed(12)
    batch, positions, width, state, code, vocab = 2, 5, 7, 4, 3, 13
    domino = TinyDomino(width, state, code, vocab)
    target_weight = torch.randn(vocab, width)
    hidden = torch.randn(batch, positions, width)
    anchors = torch.tensor([1, 3])
    output = fast_candidate_domino_decode(
        domino=domino,
        target_weight=target_weight,
        anchors=anchors,
        hidden=hidden,
        candidate_topk=vocab,
    )

    base = F.linear(hidden, target_weight)
    expected = [base[:, 0].argmax(dim=-1)]
    prefix = torch.stack([anchors, expected[0]], dim=-1)
    _, recurrent = domino.prefix_gru(F.embedding(prefix, target_weight))
    for position in range(1, positions):
        code_i = domino.embed_proj[1](
            domino.embed_proj[0](
                torch.cat(
                    [hidden[:, position : position + 1], recurrent.transpose(0, 1)],
                    dim=-1,
                )
            )
        )
        token = (
            base[:, position]
            + F.linear(code_i[:, 0], domino.embed_proj[2].weight)
        ).argmax(dim=-1)
        expected.append(token)
        if position + 1 < positions:
            _, recurrent = domino.prefix_gru(
                F.embedding(token[:, None], target_weight), recurrent
            )
    assert torch.equal(output.token_ids, torch.stack(expected, dim=-1))
    from_base = fast_candidate_domino_decode_from_base(
        domino=domino,
        target_weight=target_weight,
        anchors=anchors,
        hidden=hidden,
        base_logits=base,
        candidate_topk=vocab,
    )
    assert torch.equal(from_base.token_ids, output.token_ids)


def test_candidate_ids_rank_promoted_logits_but_scores_keep_storage_dtype() -> None:
    # The implementation contract is important on BF16 near-ties: IDs follow
    # FP32 ranking of the materialized base logits, while the score add remains
    # in the checkpoint dtype.
    torch.manual_seed(5)
    domino = TinyDomino(width=4, state=3, code=2, vocab=9).to(torch.bfloat16)
    target_weight = torch.randn(9, 4, dtype=torch.bfloat16)
    hidden = torch.randn(1, 3, 4, dtype=torch.bfloat16)
    output = fast_candidate_domino_decode(
        domino=domino,
        target_weight=target_weight,
        anchors=torch.tensor([2]),
        hidden=hidden,
        candidate_topk=4,
    )
    base = F.linear(hidden, target_weight)
    expected_ids = base.float().topk(4, dim=-1).indices
    assert torch.equal(output.candidate_ids, expected_ids)
    assert output.candidate_base_logits.dtype == torch.bfloat16
    assert torch.equal(
        output.candidate_base_logits, base.gather(-1, expected_ids)
    )


def test_position_zero_uses_vocab_argmax_not_topk_tie_order() -> None:
    domino = TinyDomino(width=2, state=2, code=2, vocab=4)
    target_weight = torch.zeros(4, 2)
    hidden = torch.zeros(1, 2, 2)
    output = fast_candidate_domino_decode(
        domino=domino,
        target_weight=target_weight,
        anchors=torch.tensor([1]),
        hidden=hidden,
        candidate_topk=2,
    )
    # torch.argmax's first-index tie contract is the released position-zero
    # policy even if Top-K returns another valid ordering of equal entries.
    assert output.token_ids[0, 0].item() == 0


def test_forced_position_zero_seeds_the_frozen_gru_suffix() -> None:
    torch.manual_seed(21)
    domino = TinyDomino(width=5, state=4, code=3, vocab=11)
    target_weight = torch.randn(11, 5)
    hidden = torch.randn(1, 4, 5)
    anchors = torch.tensor([2])
    forced = torch.tensor([7])
    output = fast_candidate_domino_decode(
        domino=domino,
        target_weight=target_weight,
        anchors=anchors,
        hidden=hidden,
        candidate_topk=6,
        forced_first=forced,
    )
    assert output.token_ids[0, 0].item() == 7
    assert output.candidate_ids[0, 0].eq(7).any()

    base = F.linear(hidden, target_weight)
    candidate_ids = base.float().topk(6, dim=-1).indices
    candidate_base = base.gather(-1, candidate_ids)
    present = candidate_ids[:, 0].eq(forced[:, None]).any(dim=-1)
    candidate_ids[:, 0, -1] = torch.where(present, candidate_ids[:, 0, -1], forced)
    forced_score = base[:, 0].gather(1, forced[:, None])[:, 0]
    candidate_base[:, 0, -1] = torch.where(
        present, candidate_base[:, 0, -1], forced_score
    )
    expected = [forced]
    _, recurrent = domino.prefix_gru(
        F.embedding(torch.stack([anchors, forced], dim=-1), target_weight)
    )
    for position in range(1, hidden.shape[1]):
        code = domino.embed_proj[1](
            domino.embed_proj[0](
                torch.cat(
                    [hidden[:, position : position + 1], recurrent.transpose(0, 1)],
                    dim=-1,
                )
            )
        )[:, 0]
        basis = F.embedding(candidate_ids[:, position], domino.embed_proj[2].weight)
        scores = candidate_base[:, position] + torch.einsum("bd,bkd->bk", code, basis)
        best = scores.float().argmax(dim=-1)
        token = candidate_ids[:, position].gather(1, best[:, None])[:, 0]
        expected.append(token)
        if position + 1 < hidden.shape[1]:
            _, recurrent = domino.prefix_gru(
                F.embedding(token[:, None], target_weight), recurrent
            )
    assert torch.equal(output.token_ids, torch.stack(expected, dim=-1))


def test_forced_prefix_seeds_the_frozen_gru_suffix_and_retains_support() -> None:
    torch.manual_seed(37)
    batch, positions, width, state, code, vocab = 2, 6, 5, 4, 3, 17
    domino = TinyDomino(width, state, code, vocab)
    target_weight = torch.randn(vocab, width)
    hidden = torch.randn(batch, positions, width)
    anchors = torch.tensor([2, 4])
    forced = torch.tensor([[16, 15, 14], [13, 12, 11]])
    base = F.linear(hidden, target_weight)
    output = fast_candidate_domino_decode_from_base(
        domino=domino,
        target_weight=target_weight,
        anchors=anchors,
        hidden=hidden,
        base_logits=base,
        candidate_topk=4,
        forced_prefix=forced,
    )

    assert torch.equal(output.token_ids[:, :3], forced)
    assert output.candidate_ids[:, :3].eq(forced[:, :, None]).any(dim=-1).all()

    candidate_ids = base.float().topk(4, dim=-1).indices
    candidate_base = base.gather(-1, candidate_ids)
    for position in range(forced.shape[1]):
        token = forced[:, position]
        present = candidate_ids[:, position].eq(token[:, None]).any(dim=-1)
        forced_score = base[:, position].gather(1, token[:, None])[:, 0]
        candidate_ids[:, position, -1] = torch.where(
            present, candidate_ids[:, position, -1], token
        )
        candidate_base[:, position, -1] = torch.where(
            present, candidate_base[:, position, -1], forced_score
        )

    expected = list(forced.unbind(dim=1))
    _, recurrent = domino.prefix_gru(
        F.embedding(torch.cat([anchors[:, None], forced], dim=1), target_weight)
    )
    for position in range(forced.shape[1], positions):
        joined = torch.cat(
            [hidden[:, position : position + 1], recurrent.transpose(0, 1)],
            dim=-1,
        )
        correction_code = domino.embed_proj[1](domino.embed_proj[0](joined))[:, 0]
        basis = F.embedding(candidate_ids[:, position], domino.embed_proj[2].weight)
        scores = candidate_base[:, position] + torch.einsum(
            "bd,bkd->bk", correction_code, basis
        )
        best = scores.float().argmax(dim=-1)
        token = candidate_ids[:, position].gather(1, best[:, None])[:, 0]
        expected.append(token)
        if position + 1 < positions:
            _, recurrent = domino.prefix_gru(
                F.embedding(token[:, None], target_weight), recurrent
            )
    assert torch.equal(output.token_ids, torch.stack(expected, dim=-1))


def test_forced_prefix_contract_rejects_ambiguous_or_invalid_inputs() -> None:
    domino = TinyDomino(width=3, state=2, code=2, vocab=7)
    target_weight = torch.randn(7, 3)
    hidden = torch.randn(1, 4, 3)
    common = dict(
        domino=domino,
        target_weight=target_weight,
        anchors=torch.tensor([1]),
        hidden=hidden,
        candidate_topk=3,
    )
    with torch.no_grad():
        try:
            fast_candidate_domino_decode(
                **common,
                forced_first=torch.tensor([2]),
                forced_prefix=torch.tensor([[2, 3]]),
            )
        except ValueError as error:
            assert "mutually exclusive" in str(error)
        else:
            raise AssertionError("ambiguous forced seed API should fail")
        try:
            fast_candidate_domino_decode(
                **common,
                forced_prefix=torch.tensor([[2, 3, 4, 5, 6]]),
            )
        except ValueError as error:
            assert "1..positions" in str(error)
        else:
            raise AssertionError("overlong forced prefix should fail")


def test_candidate_union_retains_an_outside_proposal() -> None:
    candidates = torch.tensor([[[1, 2, 3, 4], [5, 6, 7, 8]]])
    proposal = torch.tensor([[9, 6]])
    support = candidate_union_with_proposal(candidates, proposal, support_size=3)
    assert support.tolist() == [[[1, 2, 9], [5, 6, 7]]]


def test_one_repair_changes_only_frontier_and_counts_existing_suffix() -> None:
    proposal = torch.tensor(
        [
            [1, 9, 3, 4],
            [1, 2, 9, 4],
            [1, 2, 3, 4],
        ]
    )
    gold = torch.tensor([[1, 2, 3, 4]]).expand_as(proposal)
    candidates = torch.tensor(
        [
            [[1, 7], [2, 8], [3, 8], [4, 8]],
            [[1, 7], [2, 8], [5, 8], [4, 8]],
            [[1, 7], [2, 8], [3, 8], [4, 8]],
        ]
    )
    result = repair_earliest_frontier(
        proposal, gold, candidate_ids=candidates
    )
    assert result.accepted_before.tolist() == [1, 2, 4]
    assert result.accepted_after.tolist() == [4, 2, 4]
    assert result.repair_available.tolist() == [True, False, False]
    assert result.token_ids[0].tolist() == [1, 2, 3, 4]
    assert result.token_ids[1].tolist() == proposal[1].tolist()


def test_two_repairs_advance_two_distinct_frontiers() -> None:
    proposal = torch.tensor([[1, 9, 3, 8, 5]])
    gold = torch.tensor([[1, 2, 3, 4, 5]])
    candidates = torch.stack([gold, torch.zeros_like(gold)], dim=-1)
    outputs = sequential_perfect_frontier_repairs(
        proposal, gold, candidate_ids=candidates, repairs=2
    )
    assert outputs[0].accepted_after.tolist() == [3]
    assert outputs[1].accepted_after.tolist() == [5]


def test_tuned_lens_is_zero_fallback_and_exact_parameter_count() -> None:
    torch.manual_seed(3)
    basis = torch.randn(11, 5)
    lens = R048TunedLens(hidden_width=7, rank=4, candidate_basis=basis)
    assert lens.trainable_parameter_count == 7 * 4 + 4 * 5
    states = torch.randn(2, 3, 7)
    candidates = torch.randint(0, 11, (2, 3, 6))
    scores = lens(states, candidates)
    assert scores.shape == (2, 3, 6)
    assert torch.count_nonzero(scores).item() == 0


def test_earliest_one_decision_keeps_identity_then_changes_one_token() -> None:
    candidate_ids = torch.tensor([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]])
    base_scores = torch.tensor([[[3.0, 2.0, 1.0], [3.0, 2.0, 1.0], [3.0, 2.0, 1.0]]])
    proposal = torch.tensor([[1, 4, 7]])
    keep = earliest_one_decision(
        candidate_ids=candidate_ids,
        candidate_scores=base_scores,
        lens_delta=torch.zeros_like(base_scores),
        proposal=proposal,
        threshold=0.0,
    )
    assert torch.equal(keep.token_ids, proposal)
    assert keep.selected_position.tolist() == [3]

    delta = torch.zeros_like(base_scores)
    delta[0, 1, 1] = 1.5
    delta[0, 2, 1] = 2.0
    changed = earliest_one_decision(
        candidate_ids=candidate_ids,
        candidate_scores=base_scores,
        lens_delta=delta,
        proposal=proposal,
        threshold=0.25,
    )
    assert changed.selected_position.tolist() == [1]
    assert changed.token_ids.tolist() == [[1, 5, 7]]

    tied = earliest_one_decision(
        candidate_ids=candidate_ids,
        candidate_scores=base_scores,
        lens_delta=torch.zeros_like(base_scores),
        proposal=proposal,
        threshold=-1.0,
    )
    assert torch.equal(tied.token_ids, proposal)


def test_position_zero_argmax_is_retained_in_candidate_support_on_tie() -> None:
    domino = TinyDomino(width=2, state=2, code=2, vocab=4)
    target_weight = torch.zeros(4, 2)
    output = fast_candidate_domino_decode(
        domino=domino,
        target_weight=target_weight,
        anchors=torch.tensor([1]),
        hidden=torch.zeros(1, 2, 2),
        candidate_topk=2,
    )
    assert output.candidate_ids[0, 0].eq(output.token_ids[0, 0]).any()
