#!/usr/bin/env python3
"""Validate the frozen D640 positive-only feature-probe capacity stage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    from scripts.summarize_gcls_v3_reach_capacity import (
        _require_sha256,
        _target_file_signature,
        _validate_gate,
    )
except ModuleNotFoundError:
    from summarize_gcls_v3_reach_capacity import (  # type: ignore[no-redef]
        _require_sha256,
        _target_file_signature,
        _validate_gate,
    )


LABEL = "d640_flat_compat_cdpace_seed0"
EXPECTED_REPORT_FIELDS = {
    "parameter_count": 27_482_160,
    "total_steps": 1_920,
    "train_blocks": 512,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def summarize_feature_capacity(run_root: Path) -> dict[str, Any]:
    path = run_root / LABEL / "metrics.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing feature-capacity artifact: {path}")
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"corrupt feature-capacity artifact: {path}") from error
    config = report.get("config")
    gate = report.get("capacity_gate")
    provenance = report.get("provenance")
    if not all(isinstance(item, dict) for item in (config, gate, provenance)):
        raise RuntimeError(f"missing config/gate/provenance: {path}")
    frozen = {
        "loss_weighting": "candidate_dpace",
        "post_break_weight": 1.0,
        "dpace_alpha": 0.5,
        "base_safety_weight": 0.0,
        "base_safety_margin": 0.1,
        "exponential_gamma": 7.0,
        "scope": "global",
        "mixer": "flat",
        "node_encoder": "compatibility",
        "candidate_k": 16,
        "model_dim": 640,
        "num_heads": 10,
        "num_layers": 4,
        "dropout": 0.0,
        "batch_size": 32,
        "epochs": 120,
        "learning_rate": 0.0003,
        "weight_decay": 0.0,
        "warmup_ratio": 0.04,
        "gradient_clip": 1.0,
        "seed": 0,
        "max_train_prompts": 0,
        "train_subset_seed": 20260730,
        "train_split": "train",
        "skip_gate": True,
        "memorization_blocks": 512,
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
        key: {"expected": expected, "actual": config.get(key)}
        for key, expected in frozen.items()
        if config.get(key) != expected
    }
    if mismatches:
        raise RuntimeError(f"frozen feature config mismatch: {mismatches}")
    report_mismatches = {
        key: {"expected": expected, "actual": report.get(key)}
        for key, expected in EXPECTED_REPORT_FIELDS.items()
        if report.get(key) != expected
    }
    if report_mismatches:
        raise RuntimeError(
            f"feature capacity budget/model mismatch: {report_mismatches}"
        )
    _validate_gate(gate, path=path)
    trainer_hash = _require_sha256(
        provenance.get("trainer_sha256"), field="trainer source", path=path
    )
    trainer_hash_at_end = _require_sha256(
        provenance.get("trainer_sha256_at_end"),
        field="end-of-run trainer source",
        path=path,
    )
    head_hash = _require_sha256(
        provenance.get("head_source_sha256"), field="head source", path=path
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
        field="feature-capacity train prompt set",
        path=path,
    )
    target_files = _target_file_signature(provenance, path=path)
    if trainer_hash != trainer_hash_at_end or head_hash != head_hash_at_end:
        raise RuntimeError(f"source changed during feature-capacity run: {path}")
    passed = gate["passed"] is True
    return {
        "status": "passed" if passed else "scientific_negative",
        "passed": passed,
        "evidence_tier": "capacity_probe",
        "interpretation": (
            "tested_high_capacity_function_class_can_fit_same_subset"
            if passed
            else "engineering_stop_only_not_an_information_ceiling"
        ),
        "label": LABEL,
        "metrics_path": str(path.resolve()),
        "selected_epoch": report.get("selected_epoch"),
        **EXPECTED_REPORT_FIELDS,
        "train_prompt_set_sha256": prompt_hash,
        "source_signature": {
            "trainer_sha256": trainer_hash,
            "head_source_sha256": head_hash,
            "data_metadata_sha256": data_hash,
            "target_embedding_files": target_files,
        },
        "gate": gate,
    }


def main() -> None:
    args = parse_args()
    try:
        summary = summarize_feature_capacity(args.run_root)
    except (FileNotFoundError, RuntimeError) as error:
        print(json.dumps({"status": "artifact_error", "error": str(error)}))
        raise SystemExit(2) from error
    output = args.output or args.run_root / "feature_capacity_summary.json"
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
