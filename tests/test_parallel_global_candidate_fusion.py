from __future__ import annotations

import inspect

import pytest
import torch

from sph.parallel_global_candidate_fusion import (
    BLOCK_LENGTH,
    CANDIDATES,
    DEFAULT_PARAMETER_COUNT,
    MatchedLocalCandidateFusionHead,
    ParallelGlobalCandidateFusionHead,
    PGCFOutput,
    parallel_prefix_utility_loss,
    pgcf_loss_weights,
    pgcf_training_loss,
    rms_normalize,
    supported_candidate_cross_entropy,
    target_candidate_kl_loss,
)


def make_inputs(hidden_size: int = 32) -> dict[str, torch.Tensor]:
    torch.manual_seed(17)
    batch = 2
    logits = torch.randn(batch, BLOCK_LENGTH, CANDIDATES).sort(
        dim=-1, descending=True
    ).values
    return {
        "hidden": torch.randn(batch, BLOCK_LENGTH, hidden_size),
        "candidate_logits": logits,
        "anchor_embeddings": torch.randn(batch, hidden_size),
        "candidate_embeddings": torch.randn(
            batch, BLOCK_LENGTH, CANDIDATES, hidden_size
        ),
    }


def make_small_head(*, local: bool = False):
    head_type = (
        MatchedLocalCandidateFusionHead
        if local
        else ParallelGlobalCandidateFusionHead
    )
    return head_type(
        hidden_size=32,
        model_dim=32,
        num_heads=4,
        num_layers=2,
        ff_multiplier=2,
    )


def test_default_parameter_count_is_exact() -> None:
    model = ParallelGlobalCandidateFusionHead()
    assert sum(parameter.numel() for parameter in model.parameters()) == (
        DEFAULT_PARAMETER_COUNT
    )
    assert model.residual_projection.bias is None
    assert model.scalar_projection.bias is not None
    for block in model.blocks:
        assert block.qkv.bias is None
        assert block.attention_out.bias is None
        assert block.feed_forward_up.bias is None
        assert block.feed_forward_down.bias is None


def test_one_parallel_forward_has_identity_scores_and_one_chain() -> None:
    model = make_small_head()
    inputs = make_inputs()
    output = model(**inputs)
    assert output.scores.shape == (2, BLOCK_LENGTH, CANDIDATES)
    assert output.candidate_states.shape == (2, BLOCK_LENGTH, CANDIDATES, 32)
    assert torch.equal(output.scores, inputs["candidate_logits"].float())
    selected = model.proposal_candidate_indices(output)
    assert selected.shape == (2, BLOCK_LENGTH)
    assert torch.equal(selected, inputs["candidate_logits"].argmax(dim=-1))


def test_preprojected_candidate_path_matches_training_path() -> None:
    model = make_small_head().eval()
    inputs = make_inputs()
    with torch.no_grad():
        projected = model.token_projection(
            rms_normalize(inputs["candidate_embeddings"])
        )
        raw = model(**inputs)
        preprojected = model(
            inputs["hidden"],
            inputs["candidate_logits"],
            inputs["anchor_embeddings"],
            projected_candidate_embeddings=projected,
        )
    assert torch.equal(raw.scores, preprojected.scores)
    assert torch.equal(raw.candidate_states, preprojected.candidate_states)


def test_chunked_vocabulary_projection_is_exact_and_frozen() -> None:
    model = make_small_head().eval()
    embedding = torch.randn(33, 32)
    projected = model.project_vocabulary(embedding, chunk_size=7)
    expected = model.token_projection(rms_normalize(embedding))
    assert torch.equal(projected, expected)
    assert not projected.requires_grad


def test_global_head_uses_remote_position_and_local_control_does_not() -> None:
    global_head = make_small_head().eval()
    local_head = make_small_head(local=True).eval()
    local_head.load_state_dict(global_head.state_dict())
    with torch.no_grad():
        probe = torch.linspace(-1.0, 1.0, global_head.model_dim)[None]
        global_head.residual_projection.weight.copy_(probe)
        local_head.residual_projection.weight.copy_(probe)

    inputs = make_inputs()
    perturbed = {key: value.clone() for key, value in inputs.items()}
    perturbed["hidden"][:, 15] += 7.0
    perturbed["candidate_embeddings"][:, 15] *= -3.0
    perturbed["candidate_logits"][:, 15] += torch.linspace(
        -2.0, 2.0, CANDIDATES
    )

    with torch.no_grad():
        global_before = global_head(**inputs).scores[:, 0]
        global_after = global_head(**perturbed).scores[:, 0]
        local_before = local_head(**inputs).scores[:, 0]
        local_after = local_head(**perturbed).scores[:, 0]
    assert not torch.allclose(global_before, global_after, atol=1e-7, rtol=0)
    assert torch.equal(local_before, local_after)

    reverse = {key: value.clone() for key, value in inputs.items()}
    reverse["hidden"][:, 0] -= 5.0
    with torch.no_grad():
        global_last_before = global_head(**inputs).scores[:, 15]
        global_last_after = global_head(**reverse).scores[:, 15]
    assert not torch.allclose(
        global_last_before, global_last_after, atol=1e-7, rtol=0
    )


def test_global_and_local_have_identical_seeded_trainable_initialization() -> None:
    torch.manual_seed(20260810)
    global_head = ParallelGlobalCandidateFusionHead()
    torch.manual_seed(20260810)
    local_head = MatchedLocalCandidateFusionHead()
    assert global_head.state_dict().keys() == local_head.state_dict().keys()
    for key, global_value in global_head.state_dict().items():
        assert torch.equal(global_value, local_head.state_dict()[key]), key


def test_forward_signature_has_online_features_only() -> None:
    parameters = set(
        inspect.signature(ParallelGlobalCandidateFusionHead.forward).parameters
    )
    assert parameters == {
        "self",
        "hidden",
        "candidate_logits",
        "anchor_embeddings",
        "candidate_embeddings",
        "projected_candidate_embeddings",
    }
    assert not parameters & {
        "gold_ids",
        "gold_candidate_ranks",
        "policy_ids",
        "target_logits",
        "selected_tokens",
        "previous_token",
    }


def test_safe_prefix_loss_handles_out_of_k_and_censors_suffix() -> None:
    scores = torch.zeros(2, BLOCK_LENGTH, CANDIDATES, requires_grad=True)
    ranks = torch.full((2, BLOCK_LENGTH), -1, dtype=torch.long)
    ranks[0, :2] = torch.tensor([0, 1])
    loss, support = parallel_prefix_utility_loss(scores, ranks)
    assert torch.isfinite(loss)
    assert support[0, :2].all()
    assert not support[0, 2:].any()
    assert not support[1].any()
    loss.backward()
    assert torch.isfinite(scores.grad).all()
    assert torch.equal(scores.grad[0, 2:], torch.zeros_like(scores.grad[0, 2:]))
    assert torch.equal(scores.grad[1], torch.zeros_like(scores.grad[1]))


def test_supported_dense_ce_masks_unsupported_rows_before_gather() -> None:
    scores = torch.zeros(1, BLOCK_LENGTH, CANDIDATES, requires_grad=True)
    ranks = torch.full((1, BLOCK_LENGTH), -1, dtype=torch.long)
    ranks[0, 0] = 3
    ranks[0, 7] = 5
    loss, support = supported_candidate_cross_entropy(scores, ranks)
    assert support.sum().item() == 2
    assert float(loss.detach()) == pytest.approx(torch.log(torch.tensor(16.0)).item())
    loss.backward()
    assert scores.grad[0, 0].abs().sum() > 0
    assert scores.grad[0, 7].abs().sum() > 0
    assert torch.equal(scores.grad[0, 1:7], torch.zeros_like(scores.grad[0, 1:7]))
    assert torch.equal(scores.grad[0, 8:], torch.zeros_like(scores.grad[0, 8:]))


def test_empty_target_kl_is_zero_without_branching() -> None:
    scores = torch.randn(2, BLOCK_LENGTH, CANDIDATES, requires_grad=True)
    target = torch.randn_like(scores)
    support = torch.zeros(2, BLOCK_LENGTH, dtype=torch.bool)
    matches = torch.zeros_like(support)
    loss, valid = target_candidate_kl_loss(scores, target, support, matches)
    assert float(loss.detach()) == pytest.approx(0.0, abs=0.0)
    assert not valid.any()
    loss.backward()
    assert torch.equal(scores.grad, torch.zeros_like(scores.grad))


def test_target_kl_masks_entire_suffix_after_first_replay_mismatch() -> None:
    scores = torch.randn(1, BLOCK_LENGTH, CANDIDATES)
    target = torch.randn_like(scores)
    support = torch.ones(1, BLOCK_LENGTH, dtype=torch.bool)
    matches = torch.ones_like(support)
    matches[:, 3] = False
    _, valid = target_candidate_kl_loss(scores, target, support, matches)
    assert valid[0, :3].all()
    assert not valid[0, 3:].any()


@pytest.mark.parametrize(
    ("progress", "expected"),
    [
        (0.0, (0.0, 0.0, 1.0)),
        (0.099, (0.0, 0.0, 1.0)),
        (0.10, (0.0, 0.0, 1.0)),
        (0.20, (0.5, 0.025, 0.5)),
        (0.30, (1.0, 0.05, 0.0)),
        (1.0, (1.0, 0.05, 0.0)),
    ],
)
def test_curriculum_weights(progress: float, expected: tuple[float, ...]) -> None:
    assert pgcf_loss_weights(progress) == pytest.approx(expected)


def test_training_loss_keeps_labels_outside_forward_and_is_finite() -> None:
    scores = torch.randn(2, BLOCK_LENGTH, CANDIDATES, requires_grad=True)
    output = PGCFOutput(
        scores=scores,
        residual_scores=torch.zeros_like(scores),
        candidate_states=torch.empty(2, BLOCK_LENGTH, CANDIDATES, 1),
    )
    gold_ranks = torch.randint(0, CANDIDATES, (2, BLOCK_LENGTH))
    gold_ranks[:, 7] = -1
    target = torch.randn_like(scores)
    target_matches = torch.ones(2, BLOCK_LENGTH, dtype=torch.bool)
    teacher_ranks = torch.randint(0, CANDIDATES, (2, BLOCK_LENGTH))
    teacher_ranks[:, 5] = -1
    result = pgcf_training_loss(
        output,
        gold_ranks,
        progress=0.20,
        target_candidate_logits=target,
        target_matches_gold=target_matches,
        teacher_candidate_ranks=teacher_ranks,
    )
    assert torch.isfinite(result.loss)
    assert result.lambda_prefix == pytest.approx(0.5)
    assert result.lambda_target_kl == pytest.approx(0.025)
    assert result.lambda_teacher == pytest.approx(0.5)
    result.loss.backward()
    assert torch.isfinite(scores.grad).all()


def test_input_shapes_and_mutually_exclusive_embedding_paths_fail_closed() -> None:
    model = make_small_head()
    inputs = make_inputs()
    with pytest.raises(ValueError, match="exactly one"):
        model(
            inputs["hidden"],
            inputs["candidate_logits"],
            inputs["anchor_embeddings"],
        )
    with pytest.raises(ValueError, match="exactly one"):
        model(
            **inputs,
            projected_candidate_embeddings=torch.randn(
                2, BLOCK_LENGTH, CANDIDATES, 32
            ),
        )
