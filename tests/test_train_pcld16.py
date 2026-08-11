from __future__ import annotations

import argparse
import copy
from pathlib import Path

import pytest
import torch

from sph.pcld import EXPECTED_PARAMETER_COUNT, PCLDOutput
from train_pcld16 import (
    gate_checks,
    pcld_block_diagnostics,
    prompt_mean,
    selection_evaluation_step,
    serialized_config,
    validate_pcld_checkpoint,
)


def test_capacity_gate_requires_all_four_science_metrics() -> None:
    passing = {
        "candidate_accuracy": 0.99,
        "oracle_gap_recovered": 0.95,
        "harmed_fraction": 0.01,
        "legacy_j2_prompt_balanced": 0.99,
        "stable_j2_prompt_balanced": 1.0,
        "stable_j2_denominator": 314,
    }
    passing["legacy_j2_denominator"] = 411
    assert all(
        gate_checks(
            passing, "capacity", expected_j2_denominator=411
        ).values()
    )
    for key in passing:
        failing = dict(passing)
        if key in {"stable_j2_prompt_balanced", "stable_j2_denominator"}:
            # Stable J2 is diagnostic and must not alter the binding gate.
            continue
        if key == "legacy_j2_denominator":
            failing[key] -= 1
        elif key == "harmed_fraction":
            failing[key] += 1e-4
        else:
            failing[key] -= 1e-4
        assert not all(
            gate_checks(
                failing, "capacity", expected_j2_denominator=411
            ).values()
        )

    substituted = dict(passing)
    substituted["legacy_j2_denominator"] = 314
    substituted["stable_j2_denominator"] = 411
    assert not all(
        gate_checks(
            substituted, "capacity", expected_j2_denominator=411
        ).values()
    )


def test_selection_cadence_excludes_step_one_and_includes_final() -> None:
    assert not selection_evaluation_step(1, total_steps=8000, eval_every_steps=250)
    assert selection_evaluation_step(250, total_steps=8000, eval_every_steps=250)
    assert selection_evaluation_step(8000, total_steps=8000, eval_every_steps=250)


def test_prompt_mean_is_block_then_prompt_not_position_micro() -> None:
    assert prompt_mean([1.0, 1.0, 0.0], ["a", "a", "b"]) == 0.5


def test_block_diagnostics_use_only_the_shared_support_prefix() -> None:
    scores = torch.zeros(1, 16, 16)
    output = PCLDOutput(
        scores=scores,
        corrections=torch.zeros_like(scores),
        predicted_residual=torch.zeros(1, 16, 2560),
        global_states=torch.zeros(1, 16, 256),
        base_scores=scores,
    )
    batch = {
        "target_residual": torch.ones(1, 16, 2560),
        "target_candidate_logits": torch.ones(1, 16, 16),
        "gold_candidate_ranks": torch.zeros(1, 16, dtype=torch.long),
    }
    support = torch.zeros(1, 16, dtype=torch.bool)
    support[:, :2] = True
    first = pcld_block_diagnostics(output, batch, support, torch.full((2560,), 2.0))
    assert first["raw_hidden_rmse"].item() == 1.0
    assert first["whitened_hidden_rmse"].item() == 0.5
    assert first["candidate_correction_rmse"].item() == 1.0
    assert first["teacher_margin_sign_agreement"].item() == 1.0
    assert first["teacher_candidate_agreement"].item() == 1.0

    changed = dict(batch)
    changed_residual = batch["target_residual"].clone()
    changed_logits = batch["target_candidate_logits"].clone()
    changed_residual[:, 2:] = 10000
    changed_logits[:, 2:] = 10000
    changed["target_residual"] = changed_residual
    changed["target_candidate_logits"] = changed_logits
    second = pcld_block_diagnostics(
        output, changed, support, torch.full((2560,), 2.0)
    )
    for name in first:
        assert torch.equal(first[name], second[name])


def test_best_checkpoint_reload_is_fail_closed_field_by_field(tmp_path: Path) -> None:
    args = argparse.Namespace(
        scope="global",
        target=tmp_path / "target",
        train_rollout=tmp_path / "rollout",
        seed=0,
    )
    metrics = {"model_eal": 7.5, "j2_denominator": 411}
    scale = torch.ones(2560)
    checkpoint = {
        "format": "pcld16_checkpoint_v1",
        "model": {"weight": torch.ones(1)},
        "step": 250,
        "metrics": metrics,
        "epsilon_num": 0.25,
        "latent_scale": scale.clone(),
        "parameter_count": EXPECTED_PARAMETER_COUNT,
        "config": serialized_config(args),
    }
    state = validate_pcld_checkpoint(
        checkpoint,
        args=args,
        expected_step=250,
        expected_metrics=metrics,
        epsilon_num=0.25,
        latent_scale=scale,
    )
    assert "weight" in state

    corruptions = {
        "format": "wrong",
        "parameter_count": 1,
        "step": 251,
        "metrics": {"model_eal": 0.0},
        "epsilon_num": 0.5,
        "latent_scale": torch.zeros(2560),
        "config": {**serialized_config(args), "scope": "local"},
        "model": {},
    }
    for field, value in corruptions.items():
        broken = copy.deepcopy(checkpoint)
        broken[field] = value
        with pytest.raises(RuntimeError):
            validate_pcld_checkpoint(
                broken,
                args=args,
                expected_step=250,
                expected_metrics=metrics,
                epsilon_num=0.25,
                latent_scale=scale,
            )
