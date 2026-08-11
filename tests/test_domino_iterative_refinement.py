from __future__ import annotations

from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from evaluate_domino_iterative_refinement import (  # noqa: E402
    paired_prompt_bootstrap,
    summarize,
)


def test_iterative_summary_is_prompt_balanced() -> None:
    records = [
        {"sample_id": "a", "domain": "chat", "pass_1": 1, "pass_2": 4},
        {"sample_id": "a", "domain": "chat", "pass_1": 3, "pass_2": 4},
        {"sample_id": "b", "domain": "code", "pass_1": 10, "pass_2": 12},
    ]
    result = summarize(records, ["pass_1", "pass_2"], horizon=15)
    assert result["pass_1"]["mean_accepted_draft_tokens_round_weighted"] == 14 / 3
    assert result["pass_1"]["mean_accepted_draft_tokens_prompt_balanced"] == 6
    assert result["pass_2"]["mean_accepted_draft_tokens_prompt_balanced"] == 8


def test_iterative_paired_point_uses_prompt_clusters() -> None:
    records = [
        {"sample_id": "a", "pass_1": 1, "pass_2": 3},
        {"sample_id": "a", "pass_1": 2, "pass_2": 4},
        {"sample_id": "b", "pass_1": 5, "pass_2": 6},
    ]
    paired = paired_prompt_bootstrap(
        records, "pass_2", "pass_1", draws=100, seed=7
    )
    assert paired["point"] == 1.5
    assert paired["ci95_low"] <= paired["point"] <= paired["ci95_high"]
