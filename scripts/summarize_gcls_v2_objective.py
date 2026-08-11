#!/usr/bin/env python3
"""Select the GCLS-v2 development objective under the frozen rule."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


EXPECTED_LABELS = (
    "historical_dpace_a0p5",
    "reach_lam0",
    "reach_lam0p1",
    "reach_lam0p25",
)
REACH_LABELS = EXPECTED_LABELS[1:]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-first-token-drop", type=float, default=0.001)
    return parser.parse_args()


def load_row(run_root: Path, label: str) -> dict[str, Any]:
    path = run_root / f"{label}_seed0" / "metrics.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing objective artifact: {path}")
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"corrupt objective artifact: {path}: {error}") from error
    validation = report.get("final_validation")
    if not isinstance(validation, dict):
        raise RuntimeError(f"objective artifact has no final_validation: {path}")
    try:
        base = validation["base"]
        direct = validation["direct"]
        diagnostics = validation["direct_diagnostics"]
        minimum_domain_delta = min(
            float(domain_metrics["direct"]["mean_accepted_draft_tokens"])
            - float(domain_metrics["base"]["mean_accepted_draft_tokens"])
            for domain_metrics in validation["by_domain"].values()
        )
        base_eal = float(base["mean_accepted_draft_tokens_prompt_balanced"])
        direct_eal = float(
            direct["mean_accepted_draft_tokens_prompt_balanced"]
        )
        base_first = float(base["first_token_accuracy"])
        direct_first = float(direct["first_token_accuracy"])
        harmed_fraction = float(diagnostics["harmed_fraction"])
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError(f"malformed objective metrics: {path}: {error}") from error
    return {
        "label": label,
        "metrics_path": str(path.resolve()),
        "selected_epoch": report.get("selected_epoch"),
        "seconds": report.get("seconds"),
        "base_eal": base_eal,
        "direct_eal": direct_eal,
        "raw_eal_delta": direct_eal - base_eal,
        "harmed_fraction": harmed_fraction,
        "first_token_accuracy": direct_first,
        "first_token_delta": direct_first - base_first,
        "minimum_domain_delta": minimum_domain_delta,
        "first_miss_repair_rate": diagnostics.get(
            "first_miss_repair_rate_given_k"
        ),
        "oracle_gap_recovered": diagnostics.get("oracle_gap_recovered"),
    }


def selection_key(row: dict[str, Any]) -> tuple[float, ...]:
    """Frozen development ordering: raw EAL, harm, domain, first token."""

    return (
        float(row["direct_eal"]),
        -float(row["harmed_fraction"]),
        float(row["minimum_domain_delta"]),
        float(row["first_token_accuracy"]),
    )


def summarize_objective(
    run_root: Path, *, max_first_token_drop: float = 0.001
) -> dict[str, Any]:
    if max_first_token_drop < 0:
        raise ValueError("max_first_token_drop cannot be negative")
    rows = {label: load_row(run_root, label) for label in EXPECTED_LABELS}
    historical = rows[EXPECTED_LABELS[0]]
    selected = max((rows[label] for label in REACH_LABELS), key=selection_key)
    checks = {
        "raw_eal_exceeds_historical": (
            selected["direct_eal"] > historical["direct_eal"]
        ),
        "harm_not_above_historical": (
            selected["harmed_fraction"]
            <= historical["harmed_fraction"]
        ),
        "first_token_within_tolerance": (
            selected["first_token_accuracy"]
            >= historical["first_token_accuracy"] - max_first_token_drop
        ),
    }
    passed = all(checks.values())
    return {
        "status": "passed" if passed else "scientific_negative",
        "selection_rule": (
            "lexicographic_direct_eal_then_negative_harm_then_"
            "minimum_domain_delta_then_first_token_accuracy"
        ),
        "passed": passed,
        "checks": checks,
        "historical": historical,
        "selected_reach": selected,
        "selected_safety_weight": {
            "reach_lam0": 0.0,
            "reach_lam0p1": 0.1,
            "reach_lam0p25": 0.25,
        }[selected["label"]],
        "rows": [rows[label] for label in EXPECTED_LABELS],
    }


def main() -> None:
    args = parse_args()
    try:
        summary = summarize_objective(
            args.run_root,
            max_first_token_drop=args.max_first_token_drop,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        print(json.dumps({"status": "artifact_error", "error": str(error)}))
        raise SystemExit(2) from error
    output = args.output or args.run_root / "objective_selection.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    raise SystemExit(0 if summary["passed"] else 1)


if __name__ == "__main__":
    main()
