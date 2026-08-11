from __future__ import annotations

import unittest
from contextlib import redirect_stdout
import io
from pathlib import Path
import tempfile
from unittest.mock import patch

from scripts.summarize_gcls_v2_representation import (
    main,
    representation_decision,
)


def row(
    direct_eal: float,
    *,
    base_eal: float = 5.0,
    harm: float = 0.04,
    first_token: float = 0.95,
) -> dict[str, float]:
    return {
        "base_eal": base_eal,
        "direct_eal": direct_eal,
        "raw_eal_delta": direct_eal - base_eal,
        "harmed_fraction": harm,
        "first_token_accuracy": first_token,
    }


class RepresentationDecisionTest(unittest.TestCase):
    def test_selects_compatibility_only_after_both_ordered_comparisons(self) -> None:
        result = representation_decision(
            {
                "axial_additive_cdpace05": row(5.20),
                "flat_additive_cdpace05": row(5.31),
                "flat_compat_cdpace05": row(5.40),
            }
        )
        self.assertEqual(
            result["architecture_decision"], "select_flat_compatibility"
        )
        self.assertTrue(result["enter_scope_confirmation"])

    def test_compatibility_failure_simplifies_to_flat_additive(self) -> None:
        result = representation_decision(
            {
                "axial_additive_cdpace05": row(5.20),
                "flat_additive_cdpace05": row(5.40),
                "flat_compat_cdpace05": row(5.35),
            }
        )
        self.assertEqual(
            result["architecture_decision"],
            "select_flat_additive_delete_compatibility",
        )
        self.assertFalse(result["compatibility_supported"])
        self.assertTrue(result["enter_scope_confirmation"])

    def test_flat_mixer_failure_stops_even_if_compatibility_is_high(self) -> None:
        result = representation_decision(
            {
                "axial_additive_cdpace05": row(5.35),
                "flat_additive_cdpace05": row(5.30),
                "flat_compat_cdpace05": row(5.50),
            }
        )
        self.assertEqual(
            result["architecture_decision"], "stop_full_lattice_route"
        )
        self.assertIsNone(result["selected_label"])
        self.assertFalse(result["enter_scope_confirmation"])

    def test_development_gate_is_strict_and_preserves_harm(self) -> None:
        result = representation_decision(
            {
                "axial_additive_cdpace05": row(5.20, harm=0.03),
                "flat_additive_cdpace05": row(5.30, harm=0.04),
                "flat_compat_cdpace05": row(5.40, harm=0.04),
            }
        )
        self.assertFalse(
            result["confirmation_checks"]["harm_not_above_axial"]
        )
        self.assertFalse(result["enter_scope_confirmation"])

    def test_first_token_tolerance_is_relative_to_axial(self) -> None:
        result = representation_decision(
            {
                "axial_additive_cdpace05": row(5.20, first_token=0.95),
                "flat_additive_cdpace05": row(5.40, first_token=0.948),
                "flat_compat_cdpace05": row(5.35, first_token=0.95),
            }
        )
        self.assertFalse(
            result["confirmation_checks"][
                "first_token_within_axial_tolerance"
            ]
        )


class RepresentationArtifactExitTest(unittest.TestCase):
    def run_main(self, root: Path) -> tuple[int, str]:
        stream = io.StringIO()
        with patch(
            "sys.argv",
            ["summarize_gcls_v2_representation.py", "--run-root", str(root)],
        ), redirect_stdout(stream):
            with self.assertRaises(SystemExit) as raised:
                main()
        return int(raised.exception.code), stream.getvalue()

    def test_missing_artifact_exits_two(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            code, output = self.run_main(Path(temporary))
        self.assertEqual(code, 2)
        self.assertIn('"status": "artifact_error"', output)

    def test_malformed_schema_exits_two(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "axial_additive_cdpace05_seed0" / "metrics.json"
            path.parent.mkdir(parents=True)
            path.write_text("{}\n", encoding="utf-8")
            code, output = self.run_main(root)
        self.assertEqual(code, 2)
        self.assertIn('"status": "artifact_error"', output)


if __name__ == "__main__":
    unittest.main()
