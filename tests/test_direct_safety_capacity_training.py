from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from scripts.train_direct_safety_capacity import (
    BATCH_SIZE,
    PEAK_LEARNING_RATE,
    TOTAL_UPDATES,
    WARMUP_UPDATES,
    _training_tensors,
    _write_json,
    build_capacity_order_manifest,
    capacity_learning_rate,
    capacity_minibatch_objective,
    capacity_training_step,
    evaluate_capacity,
    load_capacity_order_manifest,
)
from sph.direct_safety_artifacts import sha256_file
from sph.direct_safety_gate import DirectSafetySidecar
from sph.direct_safety_protocol import BlockKey


@pytest.fixture(scope="module", autouse=True)
def _single_threaded_small_tensor_checks():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def _capacity_records() -> list[dict[str, object]]:
    features = torch.zeros(15, 200, dtype=torch.float32)
    features[:, 196] = torch.arange(15, dtype=torch.float32) / 14.0
    features[0, 195] = 1 / 15
    features[0, 197] = 1.0
    path = torch.zeros(15, dtype=torch.int64)
    path[0] = 1
    changed = path.ne(0)
    records: list[dict[str, object]] = []
    for outcome, count, lengths in (
        ("harmful", 128, (1, 0)),
        ("changed-neutral", 128, (0, 0)),
        ("beneficial", 256, (0, 1)),
    ):
        for index in range(count):
            records.append(
                {
                    "sample_id": f"{outcome}:{index:04d}",
                    "anchor_offset": index,
                    "context_length": 1000 + index,
                    "position_features": features,
                    "direct_path": path,
                    "change_mask": changed,
                    "normalized_gain": (lengths[1] - lengths[0]) / 15.0,
                    "base_length": lengths[0],
                    "direct_length": lengths[1],
                    "base_first_token_correct": lengths[0] > 0,
                    "direct_first_token_correct": lengths[1] > 0,
                }
            )
    return records


def test_binding_capacity_learning_rate_warmup_and_cosine_endpoints() -> None:
    assert capacity_learning_rate(0) == pytest.approx(
        PEAK_LEARNING_RATE / WARMUP_UPDATES
    )
    assert capacity_learning_rate(WARMUP_UPDATES - 1) == PEAK_LEARNING_RATE
    assert capacity_learning_rate(TOTAL_UPDATES - 1) == pytest.approx(0.0)
    assert capacity_learning_rate(WARMUP_UPDATES) < PEAK_LEARNING_RATE
    with pytest.raises(ValueError):
        capacity_learning_rate(TOTAL_UPDATES)
    with pytest.raises(ValueError):
        capacity_learning_rate(True)  # type: ignore[arg-type]


def test_exact_record_mass_minibatch_objective() -> None:
    scores = torch.tensor([0.0, 0.0], requires_grad=True)
    gains = torch.tensor([1.0, -1.0])
    optimizer_loss, contribution = capacity_minibatch_objective(scores, gains)
    assert contribution.item() == pytest.approx(2 / 512)
    assert optimizer_loss.item() == pytest.approx(1.0)
    optimizer_loss.backward()
    torch.testing.assert_close(scores.grad, torch.tensor([-0.5, 0.5]))


def test_epoch_zero_metrics_and_one_training_step_are_finite() -> None:
    records = _capacity_records()
    tensors = _training_tensors(records, torch.device("cpu"))
    model = DirectSafetySidecar(initialization_seed=0)
    metrics, saved = evaluate_capacity(
        model, tensors, records, gradients_finite=True
    )
    assert len(saved) == 512
    assert metrics["prompt_weighted_loss"] == pytest.approx(0.05)
    assert metrics["beneficial_apply_count"] == 0
    assert metrics["harmful_apply_count"] == 0
    assert metrics["oracle_recovery"] == 0.0
    assert metrics["utility_optimal_count"] == 256
    assert metrics["values_finite"] is True

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=PEAK_LEARNING_RATE,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
    )
    diagnostics = capacity_training_step(
        model,
        optimizer,
        tensors,
        list(range(BATCH_SIZE)),
        update_index=0,
    )
    assert diagnostics["gradients_finite"] is True
    assert math.isfinite(float(diagnostics["gradient_norm_before_clip"]))
    assert diagnostics["learning_rate"] == capacity_learning_rate(0)


def test_capacity_cli_has_no_noncapacity_data_surface() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "scripts/train_direct_safety_capacity.py"
    ).read_text(encoding="utf-8")
    assert '"--capacity-bundle"' in source
    for forbidden in (
        '"--data"',
        '"--fit-bundle"',
        '"--checkpoint-bundle"',
        '"--falsifier',
        '"--target"',
        '"--direct-run"',
    ):
        assert forbidden not in source


def test_all_320_pass_orders_are_frozen_and_reloaded_before_training(
    tmp_path,
) -> None:
    keys = [BlockKey(f"prompt:{index:04d}", index, 1000 + index) for index in range(512)]
    training_hash = "a" * 64
    manifest = build_capacity_order_manifest(keys, training_hash)
    assert len(manifest["orders"]) == 320
    assert manifest["orders"][0]["ordered_block_keys_sha256"] != (
        manifest["orders"][1]["ordered_block_keys_sha256"]
    )
    path = tmp_path / "orders.json"
    _write_json(path, manifest)
    digest = sha256_file(path)
    loaded = load_capacity_order_manifest(
        path,
        expected_sha256=digest,
        expected_training_manifest_sha256=training_hash,
        expected_keys=keys,
    )
    assert len(loaded) == 320
    assert set(loaded[0]) == set(keys)

    manifest["orders"][0]["block_keys"][0], manifest["orders"][0]["block_keys"][1] = (
        manifest["orders"][0]["block_keys"][1],
        manifest["orders"][0]["block_keys"][0],
    )
    _write_json(path, manifest)
    with pytest.raises(RuntimeError, match="semantic hash"):
        load_capacity_order_manifest(
            path,
            expected_sha256=sha256_file(path),
            expected_training_manifest_sha256=training_hash,
            expected_keys=keys,
        )
