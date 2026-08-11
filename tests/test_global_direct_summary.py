from __future__ import annotations

import unittest

from scripts.summarize_global_direct_v1 import (
    accepted_prefix,
    calibrated_prompt_eal,
    prompt_cluster_bootstrap,
)


class GlobalDirectSummaryTest(unittest.TestCase):
    def test_accepted_prefix_stops_at_first_error(self) -> None:
        self.assertEqual(accepted_prefix([True, True, False, True]), 2)
        self.assertEqual(accepted_prefix([True, True]), 2)
        self.assertEqual(accepted_prefix([False, True]), 0)

    def test_calibrated_prompt_eal_uses_margin_and_prompt_balance(self) -> None:
        report = {
            "final_validation": {
                "calibrated": {"threshold": 0.5},
                "examples": [
                    {
                        "sample_id": "a",
                        "candidate_path_indices": {
                            "direct": [1, 0, 0],
                        },
                        "direct_margin_over_base": [0.8, 0.0, 0.0],
                        "base_position_correct": [False, False, False],
                        "direct_position_correct": [True, False, False],
                    },
                    {
                        "sample_id": "a",
                        "candidate_path_indices": {
                            "direct": [0, 1, 0],
                        },
                        "direct_margin_over_base": [0.0, 0.2, 0.0],
                        "base_position_correct": [True, True, False],
                        "direct_position_correct": [True, False, False],
                    },
                    {
                        "sample_id": "b",
                        "candidate_path_indices": {
                            "direct": [0, 0, 0],
                        },
                        "direct_margin_over_base": [0.0, 0.0, 0.0],
                        "base_position_correct": [True, True, True],
                        "direct_position_correct": [True, True, True],
                    },
                ],
            }
        }
        self.assertEqual(
            calibrated_prompt_eal(report),
            {"a": 1.5, "b": 3.0},
        )

    def test_prompt_bootstrap_is_reproducible(self) -> None:
        deltas = {"a": -1.0, "b": 1.0, "c": 3.0}
        first = prompt_cluster_bootstrap(
            deltas, repetitions=200, seed=7
        )
        second = prompt_cluster_bootstrap(
            deltas, repetitions=200, seed=7
        )
        self.assertEqual(first, second)
        self.assertAlmostEqual(first["estimate"], 1.0)
        self.assertEqual(first["unit"], "prompt")


if __name__ == "__main__":
    unittest.main()
