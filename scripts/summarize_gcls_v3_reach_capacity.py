#!/usr/bin/env python3
"""Aggregate the frozen reachable-support capacity matrix."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


EXPECTED_WEIGHTS = {
    "post1_control": 1.0,
    "post0_hard": 0.0,
    "post0p1_soft": 0.1,
}
EXPECTED_THRESHOLDS = {
    "candidate_accuracy": 0.99,
    "hard_candidate_accuracy": 0.97,
    "first_miss_repair_rate": 0.95,
    "oracle_gap_recovered": 0.95,
    "harmed_fraction": 0.01,
}
EXPECTED_REPORT_FIELDS = {
    "parameter_count": 433_772,
    "total_steps": 1_280,
    "train_blocks": 128,
}
HEX_DIGITS = frozenset("0123456789abcdef")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _require_sha256(value: Any, *, field: str, path: Path) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in HEX_DIGITS for character in value)
    ):
        raise RuntimeError(f"missing/invalid {field} SHA-256: {path}")
    return value


def _target_file_signature(provenance: dict[str, Any], *, path: Path) -> list[dict[str, Any]]:
    files = provenance.get("verified_target_embedding_files")
    if not isinstance(files, list) or not files:
        raise RuntimeError(f"missing verified target embedding files: {path}")
    normalized = []
    for index, entry in enumerate(files):
        if not isinstance(entry, dict):
            raise RuntimeError(f"malformed target file provenance {index}: {path}")
        relative_path = entry.get("path")
        byte_count = entry.get("bytes")
        if not isinstance(relative_path, str) or not relative_path:
            raise RuntimeError(f"missing target file path {index}: {path}")
        if type(byte_count) is not int or byte_count < 1:
            raise RuntimeError(f"invalid target file size {index}: {path}")
        normalized.append(
            {
                "path": relative_path,
                "bytes": byte_count,
                "sha256": _require_sha256(
                    entry.get("sha256"),
                    field=f"target embedding file {index}",
                    path=path,
                ),
            }
        )
    return normalized


def _validate_gate(gate: dict[str, Any], *, path: Path) -> None:
    expected_keys = set(EXPECTED_THRESHOLDS)
    values = gate.get("values")
    thresholds = gate.get("thresholds")
    checks = gate.get("checks")
    if not all(isinstance(item, dict) for item in (values, thresholds, checks)):
        raise RuntimeError(f"malformed capacity gate dictionaries: {path}")
    if any(set(item) != expected_keys for item in (values, thresholds, checks)):
        raise RuntimeError(f"capacity gate fields do not match protocol: {path}")
    threshold_mismatches = {
        key: {
            "expected": expected,
            "actual": thresholds.get(key),
        }
        for key, expected in EXPECTED_THRESHOLDS.items()
        if (
            isinstance(thresholds.get(key), bool)
            or not isinstance(thresholds.get(key), (int, float))
            or not math.isfinite(float(thresholds[key]))
            or float(thresholds[key]) != expected
        )
    }
    if threshold_mismatches:
        raise RuntimeError(
            f"capacity gate threshold mismatch: {threshold_mismatches}"
        )
    for key, value in values.items():
        if value is not None and (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            raise RuntimeError(f"invalid capacity gate value {key}: {value!r}")

    def at_least(key: str) -> bool:
        value = values[key]
        return value is not None and float(value) >= EXPECTED_THRESHOLDS[key]

    expected_checks = {
        "candidate_accuracy": at_least("candidate_accuracy"),
        "hard_candidate_accuracy": at_least("hard_candidate_accuracy"),
        "first_miss_repair_rate": at_least("first_miss_repair_rate"),
        "oracle_gap_recovered": at_least("oracle_gap_recovered"),
        "harmed_fraction": (
            values["harmed_fraction"] is not None
            and float(values["harmed_fraction"])
            <= EXPECTED_THRESHOLDS["harmed_fraction"]
        ),
    }
    if any(type(value) is not bool for value in checks.values()):
        raise RuntimeError(f"capacity gate checks must be booleans: {path}")
    if checks != expected_checks:
        raise RuntimeError(
            f"capacity gate checks disagree with metrics: {path}"
        )
    if type(gate.get("passed")) is not bool:
        raise RuntimeError(f"capacity gate verdict must be boolean: {path}")
    if gate["passed"] != all(expected_checks.values()):
        raise RuntimeError(
            f"capacity gate verdict disagrees with fixed checks: {path}"
        )


def _load(run_root: Path, label: str) -> tuple[dict[str, Any], dict[str, Any]]:
    path = run_root / f"{label}_seed0" / "metrics.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing reach-capacity artifact: {path}")
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"corrupt reach-capacity artifact: {path}") from error
    config = report.get("config")
    gate = report.get("capacity_gate")
    provenance = report.get("provenance")
    if not isinstance(config, dict) or not isinstance(gate, dict):
        raise RuntimeError(f"missing config/capacity gate: {path}")
    if not isinstance(provenance, dict):
        raise RuntimeError(f"missing provenance: {path}")
    expected_weight = EXPECTED_WEIGHTS[label]
    frozen = {
        "loss_weighting": "reachable_dpace",
        "post_break_weight": expected_weight,
        "dpace_alpha": 0.5,
        "base_safety_weight": 0.0,
        "scope": "global",
        "mixer": "axial",
        "node_encoder": "additive",
        "candidate_k": 16,
        "model_dim": 64,
        "num_heads": 4,
        "num_layers": 1,
        "dropout": 0.0,
        "batch_size": 32,
        "epochs": 320,
        "learning_rate": 0.0006,
        "weight_decay": 0.0,
        "warmup_ratio": 0.04,
        "gradient_clip": 1.0,
        "exponential_gamma": 7.0,
        "base_safety_margin": 0.1,
        "seed": 0,
        "max_train_prompts": 0,
        "train_subset_seed": 20260730,
        "train_split": "train",
        "skip_gate": True,
        "memorization_blocks": 128,
        "memorization_opportunity_fraction": 0.5,
        "require_capacity_gate": False,
        "min_candidate_accuracy": 0.99,
        "min_hard_candidate_accuracy": 0.97,
        "min_first_miss_repair_rate": 0.95,
        "min_oracle_gap_recovered": 0.95,
        "max_harmed_fraction": 0.01,
        "evidence_tier": "capacity_probe",
        "calibrate_margin": False,
    }
    mismatches = {
        key: {"expected": value, "actual": config.get(key)}
        for key, value in frozen.items()
        if config.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"frozen capacity config mismatch {label}: {mismatches}")
    report_mismatches = {
        key: {"expected": expected, "actual": report.get(key)}
        for key, expected in EXPECTED_REPORT_FIELDS.items()
        if report.get(key) != expected
    }
    if report_mismatches:
        raise RuntimeError(
            f"capacity budget/model mismatch {label}: {report_mismatches}"
        )
    _validate_gate(gate, path=path)
    trainer_hash = _require_sha256(
        provenance.get("trainer_sha256"),
        field="trainer source",
        path=path,
    )
    trainer_hash_at_end = _require_sha256(
        provenance.get("trainer_sha256_at_end"),
        field="end-of-run trainer source",
        path=path,
    )
    head_hash = _require_sha256(
        provenance.get("head_source_sha256"),
        field="head source",
        path=path,
    )
    head_hash_at_end = _require_sha256(
        provenance.get("head_source_sha256_at_end"),
        field="end-of-run head source",
        path=path,
    )
    data_hash = _require_sha256(
        provenance.get("data_metadata_sha256"),
        field="data metadata",
        path=path,
    )
    prompt_hash = _require_sha256(
        report.get("train_prompt_set_sha256"),
        field="capacity train prompt set",
        path=path,
    )
    target_files = _target_file_signature(provenance, path=path)
    source_stable = (
        trainer_hash == trainer_hash_at_end
        and head_hash == head_hash_at_end
    )
    if not source_stable:
        raise RuntimeError(f"source changed during capacity run: {path}")
    row = {
        "label": label,
        "post_break_weight": expected_weight,
        "metrics_path": str(path.resolve()),
        "selected_epoch": report.get("selected_epoch"),
        "parameter_count": report.get("parameter_count"),
        "total_steps": report.get("total_steps"),
        "train_blocks": report.get("train_blocks"),
        "train_prompt_set_sha256": report.get("train_prompt_set_sha256"),
        "seconds": report.get("seconds"),
        "gate": gate,
    }
    signature = {
        "trainer_sha256": trainer_hash,
        "head_source_sha256": head_hash,
        "data_metadata_sha256": data_hash,
        "target_embedding_files": target_files,
        "parameter_count": report.get("parameter_count"),
        "total_steps": report.get("total_steps"),
        "train_blocks": report.get("train_blocks"),
        "train_prompt_set_sha256": prompt_hash,
        "matched_config": {
            key: value
            for key, value in config.items()
            if key not in {"output", "post_break_weight"}
        },
    }
    return row, signature


def summarize_capacity(run_root: Path) -> dict[str, Any]:
    rows = []
    signatures = []
    for label in EXPECTED_WEIGHTS:
        row, signature = _load(run_root, label)
        rows.append(row)
        signatures.append(signature)
    if any(signature != signatures[0] for signature in signatures[1:]):
        raise RuntimeError("capacity cells do not share source/data/budget signature")
    all_passed = all(row["gate"]["passed"] is True for row in rows)
    return {
        "status": "passed" if all_passed else "scientific_negative",
        "passed": all_passed,
        "aggregate_rule": "all_three_support_coefficients_pass_fixed_coverage_gate",
        "classification_denominator": "fixed_gold_in_k_coverage_prefix",
        "expected_post_break_weights": EXPECTED_WEIGHTS,
        "rows": rows,
    }


def main() -> None:
    args = parse_args()
    try:
        summary = summarize_capacity(args.run_root)
    except (FileNotFoundError, RuntimeError) as error:
        print(json.dumps({"status": "artifact_error", "error": str(error)}))
        raise SystemExit(2) from error
    output = args.output or args.run_root / "reach_capacity_summary.json"
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
