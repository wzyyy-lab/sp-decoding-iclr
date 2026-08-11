from __future__ import annotations

from pathlib import Path
import sys

import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from train_domino_causal_lattice_decoder import (  # noqa: E402
    CausalLatticeDecoder,
    candidate_action_features,
    candidate_set,
    decode_ids,
    teacher_forward,
    training_loss,
)


def test_candidate_set_keeps_base_order_and_released_action() -> None:
    base = torch.tensor([[[9.0, 8.0, 7.0, 6.0, 5.0]]])
    fixed = torch.tensor([[[1.0, 2.0, 3.0, 10.0, 0.0]]])
    ids = candidate_set(base_logits=base, fixed_logits=fixed, topk=3)
    assert ids.tolist() == [[[0, 1, 3]]]

    features = candidate_action_features(
        base_logits=base, fixed_logits=fixed, ids=ids
    )
    assert features.tolist() == [[[[0.0, 0.0], [0.0, 0.0], [1.0, 1.0]]]]


def test_zero_causal_lattice_residual_preserves_fixed_candidate_scores() -> None:
    torch.manual_seed(2)
    batch, length, candidates, hidden_size = 2, 4, 3, 8
    model = CausalLatticeDecoder(
        hidden_size=hidden_size,
        positions=length,
        candidates=candidates,
        model_dim=8,
        num_heads=2,
        lattice_layers=1,
        decoder_layers=1,
    )
    hidden = torch.randn(batch, length, hidden_size)
    lattice_embeddings = torch.randn(
        batch, length, candidates, hidden_size
    )
    lattice_logits = torch.randn(batch, length, candidates)
    lattice_lse = torch.logsumexp(torch.randn(batch, length, 9), dim=-1)
    memory, local = model.encode_lattice(
        hidden=hidden,
        candidate_embeddings=lattice_embeddings,
        candidate_logits=lattice_logits,
        full_logsumexp=lattice_lse,
    )
    prefix = model.encode_prefix(
        prefix_embeddings=torch.randn(batch, length, hidden_size),
        memory=memory,
    )
    fixed = torch.randn(batch, length, candidates)
    fixed_lse = torch.logsumexp(torch.randn(batch, length, 9), dim=-1)
    scores, residual = model.score_candidates(
        prefix_states=prefix,
        local_hidden=local,
        candidate_embeddings=torch.randn(
            batch, length, candidates, hidden_size
        ),
        fixed_candidate_logits=fixed,
        fixed_logsumexp=fixed_lse,
    )
    expected = fixed.float() - fixed_lse.float()[..., None]
    assert torch.equal(residual, torch.zeros_like(residual))
    assert torch.equal(scores, expected)

    with torch.no_grad():
        model.residual_projection.weight.normal_(std=0.2)
        model.residual_scale.zero_()
    gated_scores, gated_residual = model.score_candidates(
        prefix_states=prefix,
        local_hidden=local,
        candidate_embeddings=torch.randn(
            batch, length, candidates, hidden_size
        ),
        fixed_candidate_logits=fixed,
        fixed_logsumexp=fixed_lse,
    )
    assert torch.equal(gated_residual, torch.zeros_like(gated_residual))
    assert torch.equal(gated_scores, expected)


def test_causal_lattice_losses_have_finite_gradient() -> None:
    candidate_ids = torch.tensor(
        [[[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]]
    )
    gold = torch.tensor([[1, 5, 9, 10]])
    for objective in ["decay_ce", "breaker_margin"]:
        scores = torch.randn(1, 4, 3, requires_grad=True)
        loss, diagnostics = training_loss(
            scores=scores,
            candidate_ids=candidate_ids,
            gold=gold,
            objective=objective,
            gamma=7.0,
            prefix_weight=0.5,
            margin_temperature=1.0,
            margin_offset=0.0,
        )
        loss.backward()
        assert torch.isfinite(loss)
        assert scores.grad is not None and torch.isfinite(scores.grad).all()
        assert diagnostics["weight_sum"] > 0


def test_causal_lattice_loss_censors_suffix_after_missing_gold() -> None:
    candidate_ids = torch.tensor(
        [[[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]]
    )
    gold = torch.tensor([[1, 99, 8, 10]])
    scores = torch.randn(1, 4, 3, requires_grad=True)
    loss, _ = training_loss(
        scores=scores,
        candidate_ids=candidate_ids,
        gold=gold,
        objective="decay_ce",
        gamma=7.0,
        prefix_weight=0.5,
        margin_temperature=1.0,
        margin_offset=0.0,
    )
    loss.backward()
    assert torch.count_nonzero(scores.grad[:, 0]) > 0
    assert torch.count_nonzero(scores.grad[:, 1:]) == 0


def test_zero_decoder_matches_sequential_domino_rollout() -> None:
    torch.manual_seed(5)
    hidden_size, state_size, vocabulary, positions = 8, 5, 17, 4

    class FakeDomino(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.prefix_gru = torch.nn.GRU(
                hidden_size, state_size, batch_first=True
            )
            self.embed_proj = torch.nn.Linear(
                hidden_size + state_size, vocabulary, bias=False
            )

    domino = FakeDomino()
    target_weight = torch.randn(vocabulary, hidden_size)
    hidden = torch.randn(2, positions, hidden_size)
    anchors = torch.tensor([1, 2])
    decoder = CausalLatticeDecoder(
        hidden_size=hidden_size,
        positions=positions,
        candidates=3,
        model_dim=8,
        num_heads=2,
        lattice_layers=1,
        decoder_layers=1,
    )
    actual = decode_ids(
        domino=domino,
        decoder=decoder,
        target_weight=target_weight,
        anchors=anchors,
        hidden=hidden,
        topk=3,
    )

    base = torch.nn.functional.linear(hidden, target_weight)
    expected = torch.empty_like(actual)
    first = base[:, :1].argmax(dim=-1)
    expected[:, 0] = first[:, 0]
    _, state = domino.prefix_gru(
        torch.nn.functional.embedding(
            torch.cat([anchors[:, None], first], dim=-1), target_weight
        )
    )
    for position in range(1, positions):
        correction = domino.embed_proj(
            torch.cat(
                [hidden[:, position : position + 1], state.transpose(0, 1)],
                dim=-1,
            )
        )
        token = (base[:, position : position + 1] + correction).argmax(dim=-1)
        expected[:, position] = token[:, 0]
        if position + 1 < positions:
            _, state = domino.prefix_gru(
                torch.nn.functional.embedding(token, target_weight), state
            )
    assert torch.equal(actual, expected)


def test_teacher_scores_match_forced_gold_sequential_scores() -> None:
    torch.manual_seed(7)
    hidden_size, state_size, vocabulary, positions, topk = 8, 5, 19, 4, 3

    class FakeDomino(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.prefix_gru = torch.nn.GRU(
                hidden_size, state_size, batch_first=True
            )
            self.embed_proj = torch.nn.Linear(
                hidden_size + state_size, vocabulary, bias=False
            )

    domino = FakeDomino()
    target_weight = torch.randn(vocabulary, hidden_size)
    hidden = torch.randn(2, positions, hidden_size)
    anchors = torch.tensor([1, 2])
    gold = torch.tensor([[3, 4, 5, 6], [7, 8, 9, 10]])
    decoder = CausalLatticeDecoder(
        hidden_size=hidden_size,
        positions=positions,
        candidates=topk,
        model_dim=8,
        num_heads=2,
        lattice_layers=1,
        decoder_layers=1,
    )
    with torch.no_grad():
        decoder.residual_projection.weight.normal_(std=0.1)

    teacher_scores, _, teacher_ids = teacher_forward(
        domino=domino,
        decoder=decoder,
        target_weight=target_weight,
        anchors=anchors,
        gold=gold,
        hidden=hidden,
        topk=topk,
    )

    base = torch.nn.functional.linear(hidden, target_weight)
    base_ids = base.topk(topk, dim=-1).indices
    memory, local_hidden = decoder.encode_lattice(
        hidden=hidden,
        candidate_embeddings=torch.nn.functional.embedding(
            base_ids, target_weight
        ),
        candidate_logits=base.gather(-1, base_ids),
        full_logsumexp=torch.logsumexp(base.float(), dim=-1),
    )
    sequential_scores = []
    sequential_ids = []
    for position in range(positions):
        prefix_ids = torch.cat(
            [anchors[:, None], gold[:, :position]], dim=-1
        )
        if position == 0:
            fixed = base[:, :1]
        else:
            _, state = domino.prefix_gru(
                torch.nn.functional.embedding(prefix_ids, target_weight)
            )
            fixed = base[:, position : position + 1] + domino.embed_proj(
                torch.cat(
                    [
                        hidden[:, position : position + 1],
                        state.transpose(0, 1),
                    ],
                    dim=-1,
                )
            )
        ids = candidate_set(
            base_logits=base[:, position : position + 1],
            fixed_logits=fixed,
            topk=topk,
        )
        prefix_states = decoder.encode_prefix(
            prefix_embeddings=torch.nn.functional.embedding(
                prefix_ids, target_weight
            ),
            memory=memory,
        )
        scores, _ = decoder.score_candidates(
            prefix_states=prefix_states[:, -1:],
            local_hidden=local_hidden[:, position : position + 1],
            candidate_embeddings=torch.nn.functional.embedding(
                ids, target_weight
            ),
            fixed_candidate_logits=fixed.gather(-1, ids),
            fixed_logsumexp=torch.logsumexp(fixed.float(), dim=-1),
            action_features=candidate_action_features(
                base_logits=base[:, position : position + 1],
                fixed_logits=fixed,
                ids=ids,
            ),
        )
        sequential_ids.append(ids)
        sequential_scores.append(scores)

    assert torch.equal(teacher_ids, torch.cat(sequential_ids, dim=1))
    assert torch.allclose(
        teacher_scores, torch.cat(sequential_scores, dim=1), atol=1e-5, rtol=1e-5
    )
