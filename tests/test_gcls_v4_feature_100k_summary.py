from __future__ import annotations

import json
from pathlib import Path
import statistics
import tempfile
import unittest

from scripts.summarize_gcls_v4_feature_100k import (
    CELLS,
    EXPECTED_EXTERNAL_METADATA,
    EXPECTED_HEAD_SHA256,
    EXPECTED_PROMPT_HASH,
    EXPECTED_TARGET_FILES,
    EXPECTED_TRAINER_SHA256,
    EXPECTED_VALIDATION_METADATA_SHA256,
    summarize_feature_100k,
)


class Feature100KSummaryTest(unittest.TestCase):
    @staticmethod
    def write_cell(
        root: Path,
        label: str,
        *,
        desired_eal: float,
        config_overrides: dict[str, object] | None = None,
        drop_part: str | None = None,
    ) -> Path:
        output = root / label
        output.mkdir(parents=True)
        sample_ids = [
            sample_id
            for prompt in range(146)
            for sample_id in [f"p{prompt}"] * 8
        ] + ["p146"] * 7
        direct_values = [5] * len(sample_ids)
        net_prompt_units = round((desired_eal - 5.0) * 147 * 8)
        for index in range(net_prompt_units):
            direct_values[index] = 6
        examples = [
            {
                "sample_id": sample_id,
                "domain": "chat",
                "accepted_draft_tokens": {
                    "base": 5,
                    "direct": direct_value,
                },
                "first_token_correct": {"base": True, "direct": True},
                "oracle_accepted_draft_tokens": 15,
                "candidate_path_indices": {
                    "base": [0] * 15,
                    "direct": [0] * 15,
                },
            }
            for sample_id, direct_value in zip(
                sample_ids, direct_values, strict=True
            )
        ]
        grouped: dict[str, list[int]] = {}
        for sample_id, direct_value in zip(
            sample_ids, direct_values, strict=True
        ):
            grouped.setdefault(sample_id, []).append(direct_value)
        direct_eal = statistics.fmean(
            statistics.fmean(values) for values in grouped.values()
        )
        validation = {
            "base": {
                "mean_accepted_draft_tokens_prompt_balanced": 5.0,
                "first_token_accuracy": 1.0,
            },
            "direct": {
                "mean_accepted_draft_tokens_prompt_balanced": direct_eal,
                "first_token_accuracy": 1.0,
            },
            "oracle": {
                "mean_accepted_draft_tokens_prompt_balanced": 15.0,
                "first_token_accuracy": 1.0,
            },
            "direct_diagnostics": {
                "harmed_fraction": 0.0,
                "first_miss_repair_rate_given_k": 0.2,
                "oracle_gap_recovered": (direct_eal - 5.0) / 10.0,
            },
            "examples": examples,
        }
        cell = CELLS[label]
        config = {
            "loss_weighting": "candidate_dpace",
            "post_break_weight": 1.0,
            "dpace_alpha": 0.5,
            "base_safety_weight": 0.0,
            "base_safety_margin": 0.1,
            "exponential_gamma": 7.0,
            "scope": "global",
            "candidate_k": 16,
            "dropout": 0.0,
            "batch_size": 64,
            "epochs": 3,
            "weight_decay": 0.0,
            "warmup_ratio": 0.04,
            "gradient_clip": 1.0,
            "seed": 0,
            "max_train_prompts": 0,
            "train_subset_seed": 20260730,
            "train_split": "train",
            "validation_split": "validation_select",
            "skip_gate": True,
            "memorization_blocks": 0,
            "evidence_tier": "development",
            "calibrate_margin": True,
            "max_calibration_first_token_drop": 0.001,
            "max_calibration_domain_drop": 0.0,
            "mixer": cell["mixer"],
            "node_encoder": cell["node_encoder"],
            "model_dim": cell["model_dim"],
            "num_heads": cell["num_heads"],
            "num_layers": cell["num_layers"],
            "learning_rate": cell["learning_rate"],
            "data": "/validation",
            "train_data": [
                f"/data/{part}" for part in EXPECTED_EXTERNAL_METADATA
            ],
            "target": "/target",
            "output": str(output),
        }
        if config_overrides:
            config.update(config_overrides)
        parts = [
            part for part in EXPECTED_EXTERNAL_METADATA if part != drop_part
        ]
        provenance = {
            "trainer_sha256": EXPECTED_TRAINER_SHA256,
            "trainer_sha256_at_end": EXPECTED_TRAINER_SHA256,
            "head_source_sha256": EXPECTED_HEAD_SHA256,
            "head_source_sha256_at_end": EXPECTED_HEAD_SHA256,
            "data_metadata_sha256": EXPECTED_VALIDATION_METADATA_SHA256,
            "external_train_data": [
                {
                    "path": f"/data/{part}",
                    "metadata_sha256": EXPECTED_EXTERNAL_METADATA[part],
                }
                for part in parts
            ],
            "verified_target_embedding_files": EXPECTED_TARGET_FILES,
            "verified_external_target_embedding_files": [
                {
                    "data": f"/data/{part}",
                    "target_fingerprint_matches_base_collection": True,
                    "draft_fingerprint_matches_base_collection": True,
                }
                for part in parts
            ],
        }
        report = {
            "config": config,
            "train_prompts": 99_356,
            "train_blocks": 793_989,
            "train_prompt_set_sha256": EXPECTED_PROMPT_HASH,
            "total_steps": 37_221,
            "validation_prompts": 147,
            "validation_blocks": 1_175,
            "gate_blocks": 0,
            "parameter_count": cell["parameter_count"],
            "selected_epoch": 3,
            "seconds": 1.0,
            "peak_cuda_memory_gib": 2.0,
            "final_validation": validation,
            "provenance": provenance,
        }
        path = output / "metrics.json"
        path.write_text(json.dumps(report), encoding="utf-8")
        return path

    def write_pair(self, root: Path, *, probe_eal: float) -> None:
        self.write_cell(
            root,
            "compact_axial_additive_d64_full_seed0",
            desired_eal=5.2,
        )
        self.write_cell(
            root,
            "probe_flat_compat_d640_full_seed0",
            desired_eal=probe_eal,
        )

    def test_material_probe_is_positive_witness(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.write_pair(root, probe_eal=5.7)
            summary = summarize_feature_100k(
                root, bootstrap_repetitions=20, bootstrap_seed=1
            )
            self.assertTrue(summary["passed"])
            self.assertTrue(summary["ten_k_is_not_a_prerequisite"])
            self.assertEqual(
                summary["next_stage"],
                "start_separately_preregistered_distillation_project",
            )

    def test_subthreshold_probe_is_bounded_engineering_stop(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.write_pair(root, probe_eal=5.5)
            summary = summarize_feature_100k(
                root, bootstrap_repetitions=20, bootstrap_seed=1
            )
            self.assertFalse(summary["passed"])
            self.assertEqual(summary["status"], "engineering_stop")
            self.assertEqual(
                summary["positive_only_interpretation"],
                "engineering_stop_only_not_an_information_ceiling",
            )

    def test_rejects_missing_external_part(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.write_cell(
                root,
                "compact_axial_additive_d64_full_seed0",
                desired_eal=5.2,
            )
            self.write_cell(
                root,
                "probe_flat_compat_d640_full_seed0",
                desired_eal=5.7,
                drop_part="part-007",
            )
            with self.assertRaisesRegex(RuntimeError, "external provenance"):
                summarize_feature_100k(root, bootstrap_repetitions=10)

    def test_rejects_reported_metric_disagreement(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.write_cell(
                root,
                "compact_axial_additive_d64_full_seed0",
                desired_eal=5.2,
            )
            path = self.write_cell(
                root,
                "probe_flat_compat_d640_full_seed0",
                desired_eal=5.7,
            )
            report = json.loads(path.read_text(encoding="utf-8"))
            report["final_validation"]["direct"][
                "mean_accepted_draft_tokens_prompt_balanced"
            ] += 1.0
            path.write_text(json.dumps(report), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "prompt examples"):
                summarize_feature_100k(root, bootstrap_repetitions=10)

    def test_rejects_cross_cell_common_config_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.write_cell(
                root,
                "compact_axial_additive_d64_full_seed0",
                desired_eal=5.2,
            )
            self.write_cell(
                root,
                "probe_flat_compat_d640_full_seed0",
                desired_eal=5.7,
                config_overrides={"future_toggle": True},
            )
            with self.assertRaisesRegex(RuntimeError, "common config"):
                summarize_feature_100k(root, bootstrap_repetitions=10)

    def test_rejects_identically_substituted_reviewed_source(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.write_pair(root, probe_eal=5.7)
            for label in CELLS:
                path = root / label / "metrics.json"
                report = json.loads(path.read_text(encoding="utf-8"))
                report["provenance"]["trainer_sha256"] = "f" * 64
                report["provenance"]["trainer_sha256_at_end"] = "f" * 64
                path.write_text(json.dumps(report), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "reviewed source"):
                summarize_feature_100k(root, bootstrap_repetitions=10)

    def test_rejects_invalid_selected_epoch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.write_pair(root, probe_eal=5.7)
            path = root / "probe_flat_compat_d640_full_seed0" / "metrics.json"
            report = json.loads(path.read_text(encoding="utf-8"))
            report["selected_epoch"] = 4
            path.write_text(json.dumps(report), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "selected epoch"):
                summarize_feature_100k(root, bootstrap_repetitions=10)

    def test_rejects_prompt_domain_inconsistency(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.write_pair(root, probe_eal=5.7)
            path = root / "probe_flat_compat_d640_full_seed0" / "metrics.json"
            report = json.loads(path.read_text(encoding="utf-8"))
            report["final_validation"]["examples"][0]["domain"] = "math"
            path.write_text(json.dumps(report), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "multiple domains"):
                summarize_feature_100k(root, bootstrap_repetitions=10)

    def test_rejects_target_signature_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.write_pair(root, probe_eal=5.7)
            for label in CELLS:
                path = root / label / "metrics.json"
                report = json.loads(path.read_text(encoding="utf-8"))
                report["provenance"]["verified_target_embedding_files"][0][
                    "sha256"
                ] = "f" * 64
                path.write_text(json.dumps(report), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "reviewed source"):
                summarize_feature_100k(root, bootstrap_repetitions=10)

    def test_calibration_cannot_pass_the_raw_gate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.write_pair(root, probe_eal=5.5)
            path = root / "probe_flat_compat_d640_full_seed0" / "metrics.json"
            report = json.loads(path.read_text(encoding="utf-8"))
            report["final_validation"]["calibrated"] = {
                "mean_accepted_draft_tokens_prompt_balanced": 15.0
            }
            path.write_text(json.dumps(report), encoding="utf-8")
            summary = summarize_feature_100k(
                root, bootstrap_repetitions=10, bootstrap_seed=1
            )
            self.assertFalse(summary["passed"])


if __name__ == "__main__":
    unittest.main()
