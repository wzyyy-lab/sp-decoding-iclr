#!/usr/bin/env python3
"""Aggregate the three GCLS-v2 capacity conditions into one stage gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


EXPECTED_LABELS = ("reach_lam0", "reach_lam0p1", "reach_lam0p25")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def load_condition(run_root: Path, label: str) -> dict[str, Any]:
    path = run_root / f"{label}_seed0" / "metrics.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing capacity artifact: {path}")
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"corrupt capacity artifact: {path}: {error}") from error
    gate = report.get("capacity_gate")
    if not isinstance(gate, dict) or "passed" not in gate:
        raise RuntimeError(f"capacity artifact has no gate report: {path}")
    return {
        "label": label,
        "metrics_path": str(path.resolve()),
        "selected_epoch": report.get("selected_epoch"),
        "parameter_count": report.get("parameter_count"),
        "seconds": report.get("seconds"),
        "gate": gate,
    }


def aggregate_capacity(run_root: Path) -> dict[str, Any]:
    """Require a complete array and apply the declared any-lambda rule."""

    conditions = [
        load_condition(run_root, label) for label in EXPECTED_LABELS
    ]
    passing = [item["label"] for item in conditions if item["gate"]["passed"]]
    return {
        "status": "passed" if passing else "scientific_negative",
        "aggregate_rule": "at_least_one_lambda_passes_complete_capacity_gate",
        "passed": bool(passing),
        "passing_conditions": passing,
        "conditions": conditions,
    }


def main() -> None:
    args = parse_args()
    try:
        summary = aggregate_capacity(args.run_root)
    except (FileNotFoundError, RuntimeError) as error:
        print(json.dumps({"status": "artifact_error", "error": str(error)}))
        raise SystemExit(2) from error

    output = args.output or args.run_root / "capacity_gate_summary.json"
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
