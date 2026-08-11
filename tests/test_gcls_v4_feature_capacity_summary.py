from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from scripts.summarize_gcls_v3_reach_capacity import EXPECTED_THRESHOLDS
from scripts.summarize_gcls_v4_feature_capacity import (
    LABEL,
    summarize_feature_capacity,
)


class FeatureCapacitySummaryTest(unittest.TestCase):
    @staticmethod
    def write_report(
        root: Path,
        *,
        passed: bool,
        config_overrides: dict[str, object] | None = None,
    ) -> Path:
        output = root / LABEL
        output.mkdir(parents=True)
        config = {
            "loss_weighting": "candidate_dpace",
            "post_break_weight": 1.0,
            "dpace_alpha": 0.5,
            "base_safety_weight": 0.0,
            "base_safety_margin": 0.1,
            "exponential_gamma": 7.0,
            "scope": "global",
            "mixer": "flat",
            "node_encoder": "compatibility",
            "candidate_k": 16,
            "model_dim": 640,
            "num_heads": 10,
            "num_layers": 4,
            "dropout": 0.0,
            "batch_size": 32,
            "epochs": 120,
            "learning_rate": 0.0003,
            "weight_decay": 0.0,
            "warmup_ratio": 0.04,
            "gradient_clip": 1.0,
            "seed": 0,
            "max_train_prompts": 0,
            "train_subset_seed": 20260730,
            "train_split": "train",
            "skip_gate": True,
            "memorization_blocks": 512,
            "memorization_opportunity_fraction": 0.5,
            "require_capacity_gate": False,
            "min_candidate_accuracy": 0.99,
            "min_hard_candidate_accuracy": 0.97,
            "min_first_miss_repair_rate": 0.95,
            "min_oracle_gap_recovered": 0.95,
            "max_harmed_fraction": 0.01,
            "evidence_tier": "capacity_probe",
            "calibrate_margin": False,
        }
        if config_overrides:
            config.update(config_overrides)
        values = {
            "candidate_accuracy": 1.0 if passed else 0.98,
            "hard_candidate_accuracy": 1.0,
            "first_miss_repair_rate": 1.0,
            "oracle_gap_recovered": 1.0,
            "harmed_fraction": 0.0,
        }
        checks = {
            "candidate_accuracy": passed,
            "hard_candidate_accuracy": True,
            "first_miss_repair_rate": True,
            "oracle_gap_recovered": True,
            "harmed_fraction": True,
        }
        report = {
            "config": config,
            "parameter_count": 27_482_160,
            "total_steps": 1_920,
            "train_blocks": 512,
            "train_prompt_set_sha256": "d" * 64,
            "selected_epoch": 100,
            "capacity_gate": {
                "passed": passed,
                "values": values,
                "thresholds": EXPECTED_THRESHOLDS,
                "checks": checks,
            },
            "provenance": {
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
            },
        }
        path = output / "metrics.json"
        path.write_text(json.dumps(report), encoding="utf-8")
        return path

    def test_pass_is_positive_only_capacity_witness(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.write_report(root, passed=True)
            summary = summarize_feature_capacity(root)
            self.assertTrue(summary["passed"])
            self.assertEqual(
                summary["interpretation"],
                "tested_high_capacity_function_class_can_fit_same_subset",
            )

    def test_scientific_negative_is_not_information_ceiling(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.write_report(root, passed=False)
            summary = summarize_feature_capacity(root)
            self.assertFalse(summary["passed"])
            self.assertEqual(
                summary["interpretation"],
                "engineering_stop_only_not_an_information_ceiling",
            )

    def test_rejects_architecture_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.write_report(
                root,
                passed=True,
                config_overrides={"model_dim": 512},
            )
            with self.assertRaisesRegex(RuntimeError, "config mismatch"):
                summarize_feature_capacity(root)

    def test_rejects_missing_source_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = self.write_report(root, passed=True)
            report = json.loads(path.read_text(encoding="utf-8"))
            report["provenance"].pop("head_source_sha256")
            path.write_text(json.dumps(report), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "SHA-256"):
                summarize_feature_capacity(root)


if __name__ == "__main__":
    unittest.main()
