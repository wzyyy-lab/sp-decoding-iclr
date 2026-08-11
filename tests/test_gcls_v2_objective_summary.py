from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from scripts.summarize_gcls_v2_objective import (
    EXPECTED_LABELS,
    summarize_objective,
)


class ObjectiveSummaryTest(unittest.TestCase):
    @staticmethod
    def write_condition(
        root: Path,
        label: str,
        *,
        direct_eal: float,
        harm: float,
        first_token: float = 0.9,
    ) -> None:
        output = root / f"{label}_seed0"
        output.mkdir(parents=True)
        validation = {
            "base": {
                "mean_accepted_draft_tokens_prompt_balanced": 5.0,
                "first_token_accuracy": 0.9,
            },
            "direct": {
                "mean_accepted_draft_tokens_prompt_balanced": direct_eal,
                "first_token_accuracy": first_token,
            },
            "direct_diagnostics": {
                "harmed_fraction": harm,
                "first_miss_repair_rate_given_k": 0.2,
                "oracle_gap_recovered": 0.1,
            },
            "by_domain": {
                "chat": {
                    "base": {"mean_accepted_draft_tokens": 5.0},
                    "direct": {"mean_accepted_draft_tokens": direct_eal},
                }
            },
        }
        (output / "metrics.json").write_text(
            json.dumps(
                {
                    "selected_epoch": 3,
                    "seconds": 1.0,
                    "final_validation": validation,
                }
            ),
            encoding="utf-8",
        )

    def test_selects_reach_by_declared_lexicographic_rule(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            values = {
                "historical_dpace_a0p5": (5.1, 0.08),
                "reach_lam0": (5.2, 0.07),
                "reach_lam0p1": (5.3, 0.06),
                "reach_lam0p25": (5.25, 0.01),
            }
            for label in EXPECTED_LABELS:
                direct_eal, harm = values[label]
                self.write_condition(
                    root, label, direct_eal=direct_eal, harm=harm
                )
            summary = summarize_objective(root)
            self.assertTrue(summary["passed"])
            self.assertEqual(
                summary["selected_reach"]["label"], "reach_lam0p1"
            )
            self.assertEqual(summary["selected_safety_weight"], 0.1)

    def test_reports_scientific_negative_when_gain_costs_harm(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for label in EXPECTED_LABELS:
                self.write_condition(
                    root,
                    label,
                    direct_eal=5.1 if label.startswith("historical") else 5.2,
                    harm=0.01 if label.startswith("historical") else 0.02,
                )
            summary = summarize_objective(root)
            self.assertFalse(summary["passed"])
            self.assertFalse(summary["checks"]["harm_not_above_historical"])

    def test_missing_condition_is_artifact_error(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for label in EXPECTED_LABELS[:-1]:
                self.write_condition(root, label, direct_eal=5.2, harm=0.01)
            with self.assertRaises(FileNotFoundError):
                summarize_objective(root)


if __name__ == "__main__":
    unittest.main()
