from __future__ import annotations

import torch
from torch.nn import functional as F

from sph.global_lookahead_causal_selector import (
    GlobalLookaheadCausalSelector,
)


def make_selector(*, positions: int = 4) -> GlobalLookaheadCausalSelector:
    torch.manual_seed(41)
    vocabulary, hidden, state, code = 29, 8, 5, 4
    return GlobalLookaheadCausalSelector(
        token_embeddings=torch.randn(vocabulary, hidden),
        candidate_basis=torch.randn(vocabulary, code),
        gru_weight_ih=torch.randn(3 * state, hidden) * 0.1,
        gru_weight_hh=torch.randn(3 * state, state) * 0.1,
        hidden_projection=torch.randn(code, hidden) * 0.1,
        state_projection=torch.randn(code, state) * 0.1,
        max_positions=positions,
        candidates=3,
        global_width=8,
        global_heads=2,
        global_layers=1,
        global_modes=2,
        feed_forward_width=16,
    )


def make_inputs(*, positions: int = 4) -> dict[str, torch.Tensor]:
    torch.manual_seed(53)
    batch, candidates = 2, 3
    ids = torch.randint(0, 29, (batch, positions, candidates))
    logits = torch.randn(batch, positions, candidates).sort(
        dim=-1, descending=True
    ).values
    fixed = torch.tensor([3, 4])
    preceding = torch.randint(0, 29, (batch, positions - 1))
    return {
        "parallel_hiddens": torch.randn(batch, positions, 8),
        "candidate_ids": ids,
        "candidate_logits": logits,
        "anchor_ids": torch.tensor([1, 2]),
        "fixed_prefix_ids": fixed,
        "previous_ids": torch.cat([fixed[:, None], preceding], dim=-1),
    }


def test_zero_lookahead_exactly_matches_restricted_domino_scores() -> None:
    selector = make_selector()
    inputs = make_inputs()
    states = selector.prefix_states(
        anchor_ids=inputs["anchor_ids"],
        previous_ids=inputs["previous_ids"],
    )
    output = selector.teacher_forward(
        parallel_hiddens=inputs["parallel_hiddens"],
        candidate_ids=inputs["candidate_ids"],
        candidate_logits=inputs["candidate_logits"],
        anchor_ids=inputs["anchor_ids"],
        previous_ids=inputs["previous_ids"],
    )
    correction = F.silu(
        F.linear(
            inputs["parallel_hiddens"], selector.hidden_projection.weight
        )
        + F.linear(states, selector.state_projection.weight)
    )
    basis = F.embedding(inputs["candidate_ids"], selector.candidate_basis)
    expected = inputs["candidate_logits"].float() + torch.einsum(
        "blc,blkc->blk", correction, basis
    ).float()
    assert torch.equal(output.candidate_scores, expected)


def test_teacher_forward_matches_its_own_onpolicy_rollout() -> None:
    selector = make_selector().eval()
    inputs = make_inputs()
    decoded = selector.decode(
        parallel_hiddens=inputs["parallel_hiddens"],
        candidate_ids=inputs["candidate_ids"],
        candidate_logits=inputs["candidate_logits"],
        anchor_ids=inputs["anchor_ids"],
        fixed_prefix_ids=inputs["fixed_prefix_ids"],
    )
    previous = torch.cat(
        [inputs["fixed_prefix_ids"][:, None], decoded.token_ids[:, :-1]], dim=-1
    )
    teacher = selector.teacher_forward(
        parallel_hiddens=inputs["parallel_hiddens"],
        candidate_ids=inputs["candidate_ids"],
        candidate_logits=inputs["candidate_logits"],
        anchor_ids=inputs["anchor_ids"],
        previous_ids=previous,
    )
    assert torch.equal(decoded.token_ids, teacher.token_ids)
    assert torch.allclose(
        decoded.candidate_scores, teacher.candidate_scores, atol=1e-6, rtol=1e-6
    )


def test_selected_token_changes_later_causal_scores() -> None:
    selector = make_selector().eval()
    inputs = make_inputs()
    changed = inputs["previous_ids"].clone()
    changed[:, 1] = (changed[:, 1] + 7) % selector.vocabulary
    original = selector.teacher_forward(
        parallel_hiddens=inputs["parallel_hiddens"],
        candidate_ids=inputs["candidate_ids"],
        candidate_logits=inputs["candidate_logits"],
        anchor_ids=inputs["anchor_ids"],
        previous_ids=inputs["previous_ids"],
    )
    modified = selector.teacher_forward(
        parallel_hiddens=inputs["parallel_hiddens"],
        candidate_ids=inputs["candidate_ids"],
        candidate_logits=inputs["candidate_logits"],
        anchor_ids=inputs["anchor_ids"],
        previous_ids=changed,
    )
    # Position zero has the same visible prefix.  The changed selected token is
    # first visible to position one.
    assert torch.equal(
        original.candidate_scores[:, :1], modified.candidate_scores[:, :1]
    )
    assert not torch.allclose(
        original.candidate_scores[:, 1:], modified.candidate_scores[:, 1:]
    )


def test_future_lattice_can_change_an_earlier_decision_score() -> None:
    selector = make_selector().eval()
    inputs = make_inputs()
    with torch.no_grad():
        selector.global_to_code.weight.normal_(std=0.2)
    original = selector.teacher_forward(
        parallel_hiddens=inputs["parallel_hiddens"],
        candidate_ids=inputs["candidate_ids"],
        candidate_logits=inputs["candidate_logits"],
        anchor_ids=inputs["anchor_ids"],
        previous_ids=inputs["previous_ids"],
    )
    future_ids = inputs["candidate_ids"].clone()
    future_ids[:, -1] = (future_ids[:, -1] + 11) % selector.vocabulary
    modified = selector.teacher_forward(
        parallel_hiddens=inputs["parallel_hiddens"],
        candidate_ids=future_ids,
        candidate_logits=inputs["candidate_logits"],
        anchor_ids=inputs["anchor_ids"],
        previous_ids=inputs["previous_ids"],
    )
    assert not torch.allclose(
        original.candidate_scores[:, 0], modified.candidate_scores[:, 0]
    )


def test_preprojected_inference_table_matches_training_lexical_path() -> None:
    selector = make_selector().eval()
    ids = make_inputs()["candidate_ids"]
    expected = selector._lexical_nodes(ids)
    selector.prepare_inference()
    actual = selector._lexical_nodes(ids)
    assert torch.allclose(expected, actual, atol=1e-6, rtol=1e-6)


def test_preprojected_gru_rollout_matches_module_gru_rollout() -> None:
    selector = make_selector().eval()
    inputs = make_inputs()
    reference = selector.decode(
        parallel_hiddens=inputs["parallel_hiddens"],
        candidate_ids=inputs["candidate_ids"],
        candidate_logits=inputs["candidate_logits"],
        anchor_ids=inputs["anchor_ids"],
        fixed_prefix_ids=inputs["fixed_prefix_ids"],
    )
    selector.prepare_inference()
    projected = selector.decode(
        parallel_hiddens=inputs["parallel_hiddens"],
        candidate_ids=inputs["candidate_ids"],
        candidate_logits=inputs["candidate_logits"],
        anchor_ids=inputs["anchor_ids"],
        fixed_prefix_ids=inputs["fixed_prefix_ids"],
    )
    assert torch.equal(reference.token_ids, projected.token_ids)
    assert torch.allclose(
        reference.candidate_scores,
        projected.candidate_scores,
        atol=1e-5,
        rtol=1e-5,
    )


def test_decode_uses_position_specific_rank_calibration() -> None:
    selector = make_selector(positions=2).eval()
    inputs = make_inputs(positions=2)
    with torch.no_grad():
        selector.candidate_basis.zero_()
        selector.rank_bias.zero_()
        selector.rank_bias[0, 1] = 3.0
        selector.rank_bias[1, 2] = 4.0
    inputs["candidate_logits"].zero_()
    decoded = selector.decode(
        parallel_hiddens=inputs["parallel_hiddens"],
        candidate_ids=inputs["candidate_ids"],
        candidate_logits=inputs["candidate_logits"],
        anchor_ids=inputs["anchor_ids"],
        fixed_prefix_ids=inputs["fixed_prefix_ids"],
    )
    expected = torch.stack(
        [inputs["candidate_ids"][:, 0, 1], inputs["candidate_ids"][:, 1, 2]],
        dim=-1,
    )
    assert torch.equal(decoded.token_ids, expected)
