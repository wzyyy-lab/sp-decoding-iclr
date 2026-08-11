from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from scripts.summarize_gcls_v2_capacity import (
    EXPECTED_LABELS,
    aggregate_capacity,
)


class CapacitySummaryTest(unittest.TestCase):
    @staticmethod
    def write_condition(root: Path, label: str, passed: bool) -> None:
        output = root / f"{label}_seed0"
        output.mkdir(parents=True)
        (output / "metrics.json").write_text(
            json.dumps(
                {
                    "selected_epoch": 7,
                    "parameter_count": 123,
                    "seconds": 1.0,
                    "capacity_gate": {
                        "passed": passed,
                        "values": {},
                        "thresholds": {},
                        "checks": {},
                    },
                }
            ),
            encoding="utf-8",
        )

    def test_any_passing_lambda_opens_stage_gate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for index, label in enumerate(EXPECTED_LABELS):
                self.write_condition(root, label, passed=index == 1)
            summary = aggregate_capacity(root)
            self.assertTrue(summary["passed"])
            self.assertEqual(summary["passing_conditions"], ["reach_lam0p1"])

    def test_complete_negative_is_not_an_artifact_error(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for label in EXPECTED_LABELS:
                self.write_condition(root, label, passed=False)
            summary = aggregate_capacity(root)
            self.assertFalse(summary["passed"])
            self.assertEqual(summary["status"], "scientific_negative")

    def test_missing_condition_is_an_artifact_error(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for label in EXPECTED_LABELS[:-1]:
                self.write_condition(root, label, passed=True)
            with self.assertRaises(FileNotFoundError):
                aggregate_capacity(root)


if __name__ == "__main__":
    unittest.main()
