from __future__ import annotations

import copy
from pathlib import Path
import unittest

import torch

from scripts.evaluate_direct_one_edit import (
    EXPECTED_DIRECT_CONFIG,
    EXPECTED_DIRECT_HEAD_SHA256,
    EXPECTED_DIRECT_METRICS,
    EXPECTED_DIRECT_TRAINER_SHA256,
    EXPECTED_SOURCE_DATA_SHA256,
    EXPECTED_TRAIN_DATA,
    EXPECTED_TRAIN_METADATA_SHA256,
    build_direct_model,
    validate_direct_checkpoint_contract,
    validate_isolated_data_contract,
)


def _small_config() -> dict[str, object]:
    return {
        "scope": "global",
        "mixer": "axial",
        "node_encoder": "additive",
        "candidate_k": 4,
        "model_dim": 8,
        "num_heads": 2,
        "num_layers": 1,
        "dropout": 0.0,
        "seed": 3,
        "validation_split": "validation_select",
        "skip_gate": True,
    }


def _valid_run_contract() -> tuple[dict[str, object], dict[str, object]]:
    direct_run = "/tmp/frozen-direct-run"
    config = {**copy.deepcopy(EXPECTED_DIRECT_CONFIG), "output": direct_run}
    provenance = {
        "data_metadata_sha256": EXPECTED_SOURCE_DATA_SHA256,
        "trainer_sha256": EXPECTED_DIRECT_TRAINER_SHA256,
        "trainer_sha256_at_end": EXPECTED_DIRECT_TRAINER_SHA256,
        "head_source_sha256": EXPECTED_DIRECT_HEAD_SHA256,
        "head_source_sha256_at_end": EXPECTED_DIRECT_HEAD_SHA256,
        "external_train_data": [
            {"path": path, "metadata_sha256": digest}
            for path, digest in zip(
                EXPECTED_TRAIN_DATA,
                EXPECTED_TRAIN_METADATA_SHA256,
                strict=True,
            )
        ],
    }
    metrics = {
        "config": config,
        "selected_epoch": 2,
        **copy.deepcopy(EXPECTED_DIRECT_METRICS),
        "provenance": provenance,
    }
    checkpoint = {
        "args": config,
        "epoch": 2,
        "parameter_count": EXPECTED_DIRECT_METRICS["parameter_count"],
    }
    return metrics, checkpoint


class DirectOneEditEvaluationTest(unittest.TestCase):
    def test_checkpoint_contract_accepts_only_frozen_gate2_run(self) -> None:
        metrics, checkpoint = _valid_run_contract()
        self.assertIs(
            validate_direct_checkpoint_contract(
                metrics,
                checkpoint,
                direct_run=Path("/tmp/frozen-direct-run"),
            ),
            metrics["config"],
        )

    def test_model_reconstruction(self) -> None:
        config = _small_config()
        model = build_direct_model(config, hidden_size=8, block_length=3)
        rebuilt = build_direct_model(config, hidden_size=8, block_length=3)
        rebuilt.load_state_dict(model.state_dict(), strict=True)
        for left, right in zip(model.parameters(), rebuilt.parameters(), strict=True):
            self.assertTrue(torch.equal(left, right))

    def test_checkpoint_contract_fails_on_config_mismatch(self) -> None:
        metrics, checkpoint = _valid_run_contract()
        metrics = copy.deepcopy(metrics)
        metrics["config"]["candidate_k"] = 8
        with self.assertRaisesRegex(RuntimeError, "args differ"):
            validate_direct_checkpoint_contract(
                metrics,
                checkpoint,
                direct_run=Path("/tmp/frozen-direct-run"),
            )

    def test_self_consistent_wrong_architecture_is_rejected(self) -> None:
        metrics, checkpoint = _valid_run_contract()
        wrong = copy.deepcopy(metrics["config"])
        wrong.update(
            {
                "scope": "local",
                "mixer": "flat",
                "model_dim": 640,
                "num_heads": 10,
                "num_layers": 4,
                "candidate_k": 3,
                "dropout": 0.5,
                "seed": 9,
            }
        )
        metrics["config"] = wrong
        checkpoint["args"] = copy.deepcopy(wrong)
        with self.assertRaisesRegex(RuntimeError, "frozen Gate-2 scope"):
            validate_direct_checkpoint_contract(
                metrics,
                checkpoint,
                direct_run=Path("/tmp/frozen-direct-run"),
            )

    def test_wrong_budget_or_provenance_is_rejected(self) -> None:
        metrics, checkpoint = _valid_run_contract()
        metrics["total_steps"] = 37220
        with self.assertRaisesRegex(RuntimeError, "total_steps"):
            validate_direct_checkpoint_contract(
                metrics,
                checkpoint,
                direct_run=Path("/tmp/frozen-direct-run"),
            )
        metrics, checkpoint = _valid_run_contract()
        metrics["provenance"]["trainer_sha256_at_end"] = "wrong"
        with self.assertRaisesRegex(RuntimeError, "trainer_sha256_at_end"):
            validate_direct_checkpoint_contract(
                metrics,
                checkpoint,
                direct_run=Path("/tmp/frozen-direct-run"),
            )

    def test_isolated_data_must_match_split_and_source_identity(self) -> None:
        metadata = {
            "provenance": {
                "split_materialization": {
                    "source_collection": {"metadata_sha256": "source-hash"}
                }
            }
        }
        records = [
            {"split": "validation_select"},
            {"split": "validation_select"},
        ]
        metrics = {"provenance": {"data_metadata_sha256": "source-hash"}}
        validate_isolated_data_contract(
            metadata,
            records,
            metrics,
            validation_split="validation_select",
            metadata_sha256="isolated-hash",
        )
        records.append({"split": "validation_gate"})
        with self.assertRaisesRegex(RuntimeError, "not physically isolated"):
            validate_isolated_data_contract(
                metadata,
                records,
                metrics,
                validation_split="validation_select",
                metadata_sha256="isolated-hash",
            )


if __name__ == "__main__":
    unittest.main()
