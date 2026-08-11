from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from scripts.summarize_gcls_v3_reach_capacity import (
    EXPECTED_THRESHOLDS,
    EXPECTED_WEIGHTS,
    summarize_capacity,
)


class ReachCapacitySummaryTest(unittest.TestCase):
    @staticmethod
    def write_cell(
        root: Path,
        label: str,
        *,
        passed: object,
        weight: float | None = None,
        config_overrides: dict[str, object] | None = None,
    ) -> None:
        output = root / f"{label}_seed0"
        output.mkdir(parents=True)
        actual_weight = EXPECTED_WEIGHTS[label] if weight is None else weight
        provenance = {
            "trainer_sha256": "a" * 64,
            "trainer_sha256_at_end": "a" * 64,
            "head_source_sha256": "b" * 64,
            "head_source_sha256_at_end": "b" * 64,
            "data_metadata_sha256": "c" * 64,
            "verified_target_embedding_files": [
                {
                    "path": "model.safetensors",
                    "bytes": 123,
                    "sha256": "e" * 64,
                }
            ],
        }
        config = {
                "loss_weighting": "reachable_dpace",
                "post_break_weight": actual_weight,
                "dpace_alpha": 0.5,
                "base_safety_weight": 0.0,
                "base_safety_margin": 0.1,
                "exponential_gamma": 7.0,
                "scope": "global",
                "mixer": "axial",
                "node_encoder": "additive",
                "candidate_k": 16,
                "model_dim": 64,
                "num_heads": 4,
                "num_layers": 1,
                "dropout": 0.0,
                "batch_size": 32,
                "epochs": 320,
                "learning_rate": 0.0006,
                "weight_decay": 0.0,
                "warmup_ratio": 0.04,
                "gradient_clip": 1.0,
                "seed": 0,
                "max_train_prompts": 0,
                "train_subset_seed": 20260730,
                "train_split": "train",
                "skip_gate": True,
                "memorization_blocks": 128,
                "memorization_opportunity_fraction": 0.5,
                "require_capacity_gate": False,
                "min_candidate_accuracy": 0.99,
                "min_hard_candidate_accuracy": 0.97,
                "min_first_miss_repair_rate": 0.95,
                "min_oracle_gap_recovered": 0.95,
                "max_harmed_fraction": 0.01,
                "evidence_tier": "capacity_probe",
                "calibrate_margin": False,
                "data": "/data",
                "target": "/target",
                "output": str(output),
            }
        if config_overrides:
            config.update(config_overrides)
        metrics_pass = passed is True
        values = {
            "candidate_accuracy": 1.0 if metrics_pass else 0.98,
            "hard_candidate_accuracy": 1.0,
            "first_miss_repair_rate": 1.0,
            "oracle_gap_recovered": 1.0,
            "harmed_fraction": 0.0,
        }
        checks = {
            "candidate_accuracy": values["candidate_accuracy"] >= 0.99,
            "hard_candidate_accuracy": True,
            "first_miss_repair_rate": True,
            "oracle_gap_recovered": True,
            "harmed_fraction": True,
        }
        report = {
            "config": config,
            "selected_epoch": 10,
            "parameter_count": 433772,
            "total_steps": 1280,
            "train_blocks": 128,
            "train_prompt_set_sha256": "d" * 64,
            "seconds": 1.0,
            "capacity_gate": {
                "passed": passed,
                "values": values,
                "thresholds": EXPECTED_THRESHOLDS,
                "checks": checks,
            },
            "provenance": provenance,
        }
        (output / "metrics.json").write_text(
            json.dumps(report), encoding="utf-8"
        )

    def test_requires_every_cell_to_pass(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for label in EXPECTED_WEIGHTS:
                self.write_cell(root, label, passed=label != "post0_hard")
            summary = summarize_capacity(root)
            self.assertFalse(summary["passed"])
            self.assertEqual(summary["status"], "scientific_negative")

    def test_all_cells_open_gate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for label in EXPECTED_WEIGHTS:
                self.write_cell(root, label, passed=True)
            self.assertTrue(summarize_capacity(root)["passed"])

    def test_rejects_post_break_weight_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for label in EXPECTED_WEIGHTS:
                self.write_cell(
                    root,
                    label,
                    passed=True,
                    weight=0.2 if label == "post0p1_soft" else None,
                )
            with self.assertRaisesRegex(RuntimeError, "config mismatch"):
                summarize_capacity(root)

    def test_rejects_string_false_verdict(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for label in EXPECTED_WEIGHTS:
                self.write_cell(
                    root,
                    label,
                    passed="false" if label == "post0_hard" else True,
                )
            with self.assertRaisesRegex(RuntimeError, "must be boolean"):
                summarize_capacity(root)

    def test_rejects_relaxed_gate_threshold(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for label in EXPECTED_WEIGHTS:
                self.write_cell(root, label, passed=True)
            path = root / "post0_hard_seed0" / "metrics.json"
            report = json.loads(path.read_text(encoding="utf-8"))
            report["capacity_gate"]["thresholds"]["candidate_accuracy"] = 0.9
            path.write_text(json.dumps(report), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "threshold mismatch"):
                summarize_capacity(root)

    def test_rejects_capacity_config_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for label in EXPECTED_WEIGHTS:
                self.write_cell(
                    root,
                    label,
                    passed=True,
                    config_overrides=(
                        {"learning_rate": 0.001}
                        if label == "post0p1_soft"
                        else None
                    ),
                )
            with self.assertRaisesRegex(RuntimeError, "config mismatch"):
                summarize_capacity(root)

    def test_rejects_missing_provenance_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for label in EXPECTED_WEIGHTS:
                self.write_cell(root, label, passed=True)
            path = root / "post0_hard_seed0" / "metrics.json"
            report = json.loads(path.read_text(encoding="utf-8"))
            report["provenance"].pop("trainer_sha256")
            report["provenance"].pop("trainer_sha256_at_end")
            path.write_text(json.dumps(report), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "SHA-256"):
                summarize_capacity(root)

    def test_rejects_missing_capacity_prompt_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for label in EXPECTED_WEIGHTS:
                self.write_cell(root, label, passed=True)
            path = root / "post0p1_soft_seed0" / "metrics.json"
            report = json.loads(path.read_text(encoding="utf-8"))
            report.pop("train_prompt_set_sha256")
            path.write_text(json.dumps(report), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "prompt set SHA-256"):
                summarize_capacity(root)


if __name__ == "__main__":
    unittest.main()
