from __future__ import annotations

import unittest

from scripts.summarize_gcls_v2_capacity_rescue import rescue_decision


class CapacityRescueDecisionTest(unittest.TestCase):
    def test_arr_pass_resumes_with_longer_budget(self) -> None:
        result = rescue_decision(
            {
                "compat_arr_budget": True,
                "compat_cdpace05": True,
                "additive_cdpace05": True,
            }
        )
        self.assertEqual(
            result["decision"], "resume_arr_with_1280_step_capacity_budget"
        )

    def test_cdpace_pass_deletes_arr_claim(self) -> None:
        result = rescue_decision(
            {
                "compat_arr_budget": False,
                "compat_cdpace05": True,
                "additive_cdpace05": True,
            }
        )
        self.assertEqual(
            result["diagnosis"], "unsmoothed_arr_gradient_starvation"
        )
        self.assertTrue(result["route_continues"])

    def test_additive_only_pass_stops_compatibility_thesis(self) -> None:
        result = rescue_decision(
            {
                "compat_arr_budget": False,
                "compat_cdpace05": False,
                "additive_cdpace05": True,
            }
        )
        self.assertEqual(
            result["diagnosis"], "compatibility_encoder_bottleneck"
        )
        self.assertFalse(result["route_continues"])

    def test_all_fail_stops_without_fourth_rescue(self) -> None:
        result = rescue_decision(
            {
                "compat_arr_budget": False,
                "compat_cdpace05": False,
                "additive_cdpace05": False,
            }
        )
        self.assertEqual(result["decision"], "stop_route_no_fourth_rescue")
        self.assertFalse(result["route_continues"])


if __name__ == "__main__":
    unittest.main()
