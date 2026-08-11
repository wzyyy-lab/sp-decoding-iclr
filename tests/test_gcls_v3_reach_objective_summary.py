from __future__ import annotations

import json
from pathlib import Path
import statistics
import tempfile
import unittest

from scripts.summarize_gcls_v3_reach_objective import (
    EXPECTED_PROMPT_HASH,
    EXPECTED_WEIGHTS,
    summarize_objective,
)


class ReachObjectiveSummaryTest(unittest.TestCase):
    @staticmethod
    def write_cell(
        root: Path,
        label: str,
        *,
        direct_eal: float,
        harm: float,
        first_token: float = 1.0,
        weighted_suffix_loss: float | None = None,
        config_overrides: dict[str, object] | None = None,
    ) -> None:
        output = root / f"{label}_seed0"
        output.mkdir(parents=True)
        weight = EXPECTED_WEIGHTS[label]
        if weighted_suffix_loss is None:
            weighted_suffix_loss = 0.0 if weight == 0.0 else 0.1
        if first_token != 1.0:
            raise ValueError("synthetic fixture freezes first-token accuracy")
        sample_ids = [
            sample_id
            for prompt in range(146)
            for sample_id in [f"p{prompt}"] * 8
        ] + ["p146"] * 7
        direct_values = [5] * len(sample_ids)
        harmed_blocks = round(harm * len(sample_ids))
        net_prompt_units = round((direct_eal - 5.0) * 147 * 8)
        improved_blocks = harmed_blocks + net_prompt_units
        for index in range(harmed_blocks):
            direct_values[index] = 4
        for index in range(harmed_blocks, harmed_blocks + improved_blocks):
            direct_values[index] = 6
        examples = [
            {
                "sample_id": sample_id,
                "domain": "chat",
                "accepted_draft_tokens": {
                    "base": 5,
                    "direct": direct_value,
                },
                "first_token_correct": {
                    "base": True,
                    "direct": True,
                },
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
        prompt_values: dict[str, list[int]] = {}
        for sample_id, direct_value in zip(
            sample_ids, direct_values, strict=True
        ):
            prompt_values.setdefault(sample_id, []).append(direct_value)
        actual_direct_eal = statistics.fmean(
            statistics.fmean(values) for values in prompt_values.values()
        )
        actual_harm = harmed_blocks / len(sample_ids)
        actual_domain_eal = statistics.fmean(direct_values)
        validation = {
            "base": {
                "mean_accepted_draft_tokens_prompt_balanced": 5.0,
                "first_token_accuracy": 1.0,
            },
            "direct": {
                "mean_accepted_draft_tokens_prompt_balanced": actual_direct_eal,
                "first_token_accuracy": first_token,
            },
            "candidate_classification": {
                "accuracy": 0.7,
                "non_top1_accuracy": 0.1,
            },
            "direct_diagnostics": {
                "harmed_fraction": actual_harm,
                "first_miss_repair_rate_given_k": 0.2,
                "oracle_gap_recovered": 0.1,
            },
            "loss": {
                "components": {
                    "reachable_fraction_of_coverage": 0.6,
                    "post_break_positions_per_block": 3.0,
                    "post_break_suffix_loss": 0.2,
                    "weighted_post_break_suffix_loss": weighted_suffix_loss,
                }
            },
            "by_domain": {
                "chat": {
                    "base": {"mean_accepted_draft_tokens": 5.0},
                    "direct": {
                        "mean_accepted_draft_tokens": actual_domain_eal
                    },
                }
            },
            "examples": examples,
        }
        provenance = {
            "trainer_sha256": "a" * 64,
            "trainer_sha256_at_end": "a" * 64,
            "head_source_sha256": "b" * 64,
            "head_source_sha256_at_end": "b" * 64,
            "data_metadata_sha256": "c" * 64,
            "external_train_data": [
                {"path": "train", "metadata_sha256": "d" * 64}
            ],
            "verified_target_embedding_files": [
                {
                    "path": "model.safetensors",
                    "bytes": 123,
                    "sha256": "e" * 64,
                }
            ],
            "verified_external_target_embedding_files": [
                {
                    "data": "train",
                    "target_fingerprint_matches_base_collection": True,
                    "draft_fingerprint_matches_base_collection": True,
                }
            ],
        }
        config = {
                "loss_weighting": "reachable_dpace",
                "post_break_weight": weight,
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
                "batch_size": 64,
                "epochs": 12,
                "learning_rate": 0.0006,
                "weight_decay": 0.0,
                "warmup_ratio": 0.04,
                "gradient_clip": 1.0,
                "seed": 0,
                "max_train_prompts": 25000,
                "train_subset_seed": 20260730,
                "train_split": "train",
                "validation_split": "validation_select",
                "skip_gate": True,
                "evidence_tier": "development",
                "calibrate_margin": True,
                "max_calibration_first_token_drop": 0.001,
                "max_calibration_domain_drop": 0.0,
                "data": "/validation",
                "train_data": ["/train"],
                "target": "/target",
                "output": str(output),
            }
        if config_overrides:
            config.update(config_overrides)
        report = {
            "config": config,
            "train_prompts": 25000,
            "train_blocks": 199818,
            "train_prompt_set_sha256": EXPECTED_PROMPT_HASH,
            "total_steps": 37476,
            "validation_prompts": 147,
            "validation_blocks": 1175,
            "gate_blocks": 0,
            "selected_epoch": 4,
            "parameter_count": 433772,
            "seconds": 1.0,
            "final_validation": validation,
            "provenance": provenance,
        }
        (output / "metrics.json").write_text(
            json.dumps(report), encoding="utf-8"
        )

    def test_passes_only_material_safe_effect(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.write_cell(root, "post1_control", direct_eal=5.10, harm=0.05)
            self.write_cell(root, "post0_hard", direct_eal=5.16, harm=0.06)
            self.write_cell(root, "post0p1_soft", direct_eal=5.14, harm=0.05)
            summary = summarize_objective(
                root, bootstrap_repetitions=100, bootstrap_seed=1
            )
            self.assertTrue(summary["passed"])
            self.assertEqual(summary["selected_label"], "post0_hard")

    def test_subthreshold_gain_is_scientific_negative(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.write_cell(root, "post1_control", direct_eal=5.10, harm=0.05)
            self.write_cell(root, "post0_hard", direct_eal=5.14, harm=0.05)
            self.write_cell(root, "post0p1_soft", direct_eal=5.13, harm=0.05)
            summary = summarize_objective(
                root, bootstrap_repetitions=100, bootstrap_seed=1
            )
            self.assertFalse(summary["passed"])
            self.assertEqual(
                summary["architecture_decision"],
                "close_reachable_support_route",
            )

    def test_hard_cell_must_have_zero_weighted_suffix_loss(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.write_cell(root, "post1_control", direct_eal=5.10, harm=0.05)
            self.write_cell(
                root,
                "post0_hard",
                direct_eal=5.20,
                harm=0.05,
                weighted_suffix_loss=1e-8,
            )
            self.write_cell(root, "post0p1_soft", direct_eal=5.15, harm=0.05)
            summary = summarize_objective(
                root, bootstrap_repetitions=100, bootstrap_seed=1
            )
            self.assertFalse(
                summary["checks"]["hard_cell_weighted_suffix_loss_zero"]
            )

    def test_rejects_frozen_optimizer_config_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for label in EXPECTED_WEIGHTS:
                self.write_cell(
                    root,
                    label,
                    direct_eal=5.2,
                    harm=0.05,
                    config_overrides=(
                        {"warmup_ratio": 0.1}
                        if label == "post0p1_soft"
                        else None
                    ),
                )
            with self.assertRaisesRegex(RuntimeError, "config mismatch"):
                summarize_objective(root, bootstrap_repetitions=10)

    def test_rejects_unanticipated_cross_cell_config_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for label in EXPECTED_WEIGHTS:
                self.write_cell(
                    root,
                    label,
                    direct_eal=5.2,
                    harm=0.05,
                    config_overrides=(
                        {"future_protocol_toggle": True}
                        if label == "post0_hard"
                        else None
                    ),
                )
            with self.assertRaisesRegex(RuntimeError, "signature"):
                summarize_objective(root, bootstrap_repetitions=10)

    def test_rejects_missing_provenance_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for label in EXPECTED_WEIGHTS:
                self.write_cell(root, label, direct_eal=5.2, harm=0.05)
            path = root / "post0_hard_seed0" / "metrics.json"
            report = json.loads(path.read_text(encoding="utf-8"))
            report["provenance"].pop("data_metadata_sha256")
            path.write_text(json.dumps(report), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "SHA-256"):
                summarize_objective(root, bootstrap_repetitions=10)

    def test_rejects_nonfinite_gate_metric(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for label in EXPECTED_WEIGHTS:
                self.write_cell(root, label, direct_eal=5.2, harm=0.05)
            path = root / "post0_hard_seed0" / "metrics.json"
            report = json.loads(path.read_text(encoding="utf-8"))
            report["final_validation"]["direct"][
                "mean_accepted_draft_tokens_prompt_balanced"
            ] = float("inf")
            path.write_text(json.dumps(report), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "finite"):
                summarize_objective(root, bootstrap_repetitions=10)

    def test_rejects_scalar_example_disagreement(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for label in EXPECTED_WEIGHTS:
                self.write_cell(root, label, direct_eal=5.2, harm=0.05)
            path = root / "post0p1_soft_seed0" / "metrics.json"
            report = json.loads(path.read_text(encoding="utf-8"))
            report["final_validation"]["direct"][
                "mean_accepted_draft_tokens_prompt_balanced"
            ] += 0.5
            path.write_text(json.dumps(report), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "prompt examples"):
                summarize_objective(root, bootstrap_repetitions=10)

    def test_rejects_out_of_range_rate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for label in EXPECTED_WEIGHTS:
                self.write_cell(root, label, direct_eal=5.2, harm=0.05)
            path = root / "post1_control_seed0" / "metrics.json"
            report = json.loads(path.read_text(encoding="utf-8"))
            report["final_validation"]["direct_diagnostics"][
                "harmed_fraction"
            ] = 1.1
            path.write_text(json.dumps(report), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "above 1.0"):
                summarize_objective(root, bootstrap_repetitions=10)


if __name__ == "__main__":
    unittest.main()
