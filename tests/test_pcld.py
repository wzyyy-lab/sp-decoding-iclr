from __future__ import annotations

import inspect

import pytest
import torch

from sph.pcld import (
    BLOCK_LENGTH,
    CANDIDATES,
    EXPECTED_PARAMETER_COUNT,
    HIDDEN_SIZE,
    PCLD16Head,
    PCLDOutput,
    assert_frozen_architecture,
    calibrate_numeric_epsilon,
    continuous_clean_support,
    count_trainable_parameters,
    latent_alpha,
    parameter_free_rms_norm,
    pcld_per_block_loss,
    stable_teacher_rows,
)


def test_frozen_architecture_one_chain_and_exact_identity() -> None:
    torch.manual_seed(4)
    model = PCLD16Head(scope="global").eval()
    assert_frozen_architecture(model)
    assert count_trainable_parameters(model) == EXPECTED_PARAMETER_COUNT
    assert model._encoder_mask(torch.device("cpu")) is None
    assert model._query_mask(torch.device("cpu")) is None
    assert set(inspect.signature(model.forward).parameters) == {
        "hidden",
        "candidate_lm_rows",
        "candidate_logits",
        "base_logsumexp",
    }

    hidden = torch.randn(1, BLOCK_LENGTH, HIDDEN_SIZE, dtype=torch.bfloat16)
    rows = torch.randn(
        1, BLOCK_LENGTH, CANDIDATES, HIDDEN_SIZE, dtype=torch.bfloat16
    )
    logits = torch.randn(1, BLOCK_LENGTH, CANDIDATES)
    lse = torch.logsumexp(torch.randn(1, BLOCK_LENGTH, 64), dim=-1)
    output = model(hidden, rows, logits, lse)
    assert output.scores.shape == (1, 16, 16)
    assert output.predicted_residual.shape == (1, 16, 2560)
    assert torch.equal(output.scores, logits.float())
    candidate_ids = torch.arange(256).view(1, 16, 16)
    assert model.proposal_ids(candidate_ids, output).shape == (1, 16)


def test_local_control_masks_both_node_and_query_paths() -> None:
    model = PCLD16Head(scope="local")
    encoder = model._encoder_mask(torch.device("cpu"))
    query = model._query_mask(torch.device("cpu"))
    assert encoder is not None and encoder.shape == (256, 256)
    assert query is not None and query.shape == (16, 256)
    assert not bool(encoder[0, 15])
    assert bool(encoder[0, 16])
    assert not bool(query[0, 15])
    assert bool(query[0, 16])
    with pytest.raises(ValueError):
        PCLD16Head(scope="causal")


def test_parameter_free_rms_norm() -> None:
    value = torch.tensor([[3.0, 4.0]])
    normalized = parameter_free_rms_norm(value)
    assert normalized.shape == value.shape
    assert torch.allclose(normalized.square().mean(dim=-1), torch.ones(1), atol=1e-5)


def test_numeric_epsilon_excludes_disagreeing_rows_and_freezes_stability() -> None:
    authoritative = torch.tensor([[1, 2, 3, 4]])
    replay = torch.tensor([[1, 9, 3, 4]])
    errors = torch.tensor([[0.01, 99.0, 0.03, 0.02]])
    epsilon = calibrate_numeric_epsilon(authoritative, replay, errors)
    assert epsilon.item() == pytest.approx(0.03)
    margins = torch.tensor([[0.061, 100.0, 0.059, 0.2]])
    stable = stable_teacher_rows(authoritative, replay, margins, epsilon)
    assert stable.tolist() == [[True, False, False, True]]


def test_continuous_support_masks_first_failure_and_suffix() -> None:
    ranks = torch.tensor([[0, 2, 1, -1, 0, 0]])
    ranks = torch.cat([ranks, torch.zeros(1, 10, dtype=torch.long)], dim=1)
    gold = torch.arange(16).view(1, 16)
    top1 = gold.clone()
    stable = torch.ones_like(gold, dtype=torch.bool)
    support, horizon = continuous_clean_support(ranks, top1, gold, stable)
    assert horizon.tolist() == [3]
    assert support[0, :3].all()
    assert not support[0, 3:].any()


def test_pcld_loss_uses_one_shared_prefix_mask() -> None:
    batch = 1
    candidate_ids = torch.arange(16).view(1, 1, 16).expand(batch, 16, 16).clone()
    gold = torch.zeros(batch, 16, dtype=torch.long)
    # First three rows are valid; position 3 is outside K and censors suffix.
    gold[0, 3] = 99
    base = torch.zeros(batch, 16, 16)
    scores = base.clone()
    scores[..., 0] = 0.5
    output = PCLDOutput(
        scores=scores,
        corrections=scores - base,
        predicted_residual=torch.zeros(batch, 16, 2560),
        global_states=torch.zeros(batch, 16, 256),
        base_scores=base,
    )
    residual = torch.randn(batch, 16, 2560)
    teacher = torch.randn(batch, 16, 16)
    target_top1 = gold.clone()
    stable = torch.ones(batch, 16, dtype=torch.bool)
    scale = torch.ones(2560)
    first = pcld_per_block_loss(
        output,
        candidate_ids,
        gold,
        residual,
        teacher,
        target_top1,
        stable,
        scale,
        alpha=0.7,
    )
    assert first.horizons.tolist() == [3]
    modified_residual = residual.clone()
    modified_teacher = teacher.clone()
    modified_residual[:, 3:] += 10_000
    modified_teacher[:, 3:] += torch.randn_like(modified_teacher[:, 3:]) * 10_000
    second = pcld_per_block_loss(
        output,
        candidate_ids,
        gold,
        modified_residual,
        modified_teacher,
        target_top1,
        stable,
        scale,
        alpha=0.7,
    )
    assert torch.allclose(first.per_block_loss, second.per_block_loss)
    assert torch.isfinite(first.per_block_loss).all()


def test_latent_alpha_contract() -> None:
    assert latent_alpha(0, 100) == pytest.approx(1.0)
    assert latent_alpha(30, 100) == pytest.approx(0.1)
    assert latent_alpha(100, 100) == pytest.approx(0.1)

