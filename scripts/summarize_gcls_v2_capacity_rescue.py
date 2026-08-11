#!/usr/bin/env python3
"""Apply the predeclared one-shot capacity-rescue decision table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from scripts.summarize_gcls_v2_capacity import load_condition


EXPECTED_LABELS = (
    "compat_arr_budget",
    "compat_cdpace05",
    "additive_cdpace05",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def rescue_decision(passed: dict[str, bool]) -> dict[str, Any]:
    arr = passed["compat_arr_budget"]
    compatibility_cdpace = passed["compat_cdpace05"]
    additive_cdpace = passed["additive_cdpace05"]
    if arr:
        return {
            "decision": "resume_arr_with_1280_step_capacity_budget",
            "route_continues": True,
            "diagnosis": "original_arr_horizon_was_insufficient",
        }
    if compatibility_cdpace:
        return {
            "decision": "delete_arr_claim_and_refreeze_smoothed_cdpace",
            "route_continues": True,
            "diagnosis": "unsmoothed_arr_gradient_starvation",
        }
    if additive_cdpace:
        return {
            "decision": "stop_compatibility_full_lattice_thesis",
            "route_continues": False,
            "diagnosis": "compatibility_encoder_bottleneck",
        }
    return {
        "decision": "stop_route_no_fourth_rescue",
        "route_continues": False,
        "diagnosis": "tested_route_failed_capacity",
    }


def summarize_rescue(run_root: Path) -> dict[str, Any]:
    conditions = [load_condition(run_root, label) for label in EXPECTED_LABELS]
    passed = {
        item["label"]: bool(item["gate"]["passed"])
        for item in conditions
    }
    decision = rescue_decision(passed)
    return {
        "status": (
            "route_continues"
            if decision["route_continues"]
            else "scientific_negative"
        ),
        "predeclared_order": list(EXPECTED_LABELS),
        "gate_counts": {
            "active_correct_minimum": "1297/1310",
            "hard_correct_minimum": "213/219",
            "first_miss_repairs_minimum": "61/64",
            "harmed_blocks_maximum": "1/128",
        },
        "passed_conditions": passed,
        **decision,
        "conditions": conditions,
    }


def main() -> None:
    args = parse_args()
    try:
        summary = summarize_rescue(args.run_root)
    except (FileNotFoundError, RuntimeError) as error:
        print(json.dumps({"status": "artifact_error", "error": str(error)}))
        raise SystemExit(2) from error
    output = args.output or args.run_root / "capacity_rescue_summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    raise SystemExit(0 if summary["route_continues"] else 1)


if __name__ == "__main__":
    main()
