from __future__ import annotations

from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from compare_same_anchor_domino import paired_summary  # noqa: E402


def test_paired_summary_matches_prompt_cluster_estimand() -> None:
    records = [
        {"sample_id": "many", "left": 2, "right": 0},
        {"sample_id": "many", "left": 2, "right": 0},
        {"sample_id": "one", "left": 0, "right": 1},
    ]
    summary = paired_summary(
        records, "left", "right", draws=100, seed=7
    )
    assert summary["mean_difference"] == 0.5
    assert summary["mean_difference_prompt_balanced"] == 0.5
    assert summary["mean_difference_round_weighted"] == 1.0
