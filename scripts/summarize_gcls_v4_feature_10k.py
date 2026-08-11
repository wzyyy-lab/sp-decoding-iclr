#!/usr/bin/env python3
"""Apply the positive-only held-out gate to the frozen 10K feature probe."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import statistics
from typing import Any

try:
    from scripts.summarize_gcls_v3_reach_objective import (
        _example_metrics,
        _finite_float,
        _paired_bootstrap,
        _require_sha256,
        _target_file_signature,
    )
except ModuleNotFoundError:
    from summarize_gcls_v3_reach_objective import (  # type: ignore[no-redef]
        _example_metrics,
        _finite_float,
        _paired_bootstrap,
        _require_sha256,
        _target_file_signature,
    )


CELLS = {
    "compact_axial_additive_d64_seed0": {
        "mixer": "axial",
        "node_encoder": "additive",
        "model_dim": 64,
        "num_heads": 4,
        "num_layers": 1,
        "learning_rate": 0.0006,
        "parameter_count": 433_772,
    },
    "probe_flat_compat_d640_seed0": {
        "mixer": "flat",
        "node_encoder": "compatibility",
        "model_dim": 640,
        "num_heads": 10,
        "num_layers": 4,
        "learning_rate": 0.0003,
        "parameter_count": 27_482_160,
    },
}
EXPECTED_PROMPT_HASH = (
    "7cecb2289e172df0642056c3d5cc78f99f10093ec08ba79f7df30a57d89047e9"
)
EXPECTED_TRAIN_BLOCKS = 79_931
EXPECTED_TOTAL_STEPS = 37_470
MINIMUM_RAW_DELTA = 0.6
MINIMUM_ORACLE_GAP_RECOVERED = 0.15


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--bootstrap-repetitions", type=int, default=20_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260804)
    return parser.parse_args()


def _oracle_prompt_eal(report: dict[str, Any]) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for record in report["final_validation"]["examples"]:
        grouped[str(record["sample_id"])].append(
            float(record["oracle_accepted_draft_tokens"])
        )
    return {
        sample_id: statistics.fmean(values)
        for sample_id, values in grouped.items()
    }


def _load(
    run_root: Path, label: str
) -> tuple[dict[str, Any], dict[str, Any], dict[str, float]]:
    path = run_root / label / "metrics.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing feature-probe artifact: {path}")
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"corrupt feature-probe artifact: {path}") from error
    config = report.get("config")
    provenance = report.get("provenance")
    validation = report.get("final_validation")
    if not all(isinstance(item, dict) for item in (config, provenance, validation)):
        raise RuntimeError(f"missing config/provenance/validation: {path}")
    cell = CELLS[label]
    frozen = {
        "loss_weighting": "candidate_dpace",
        "post_break_weight": 1.0,
        "dpace_alpha": 0.5,
        "base_safety_weight": 0.0,
        "base_safety_margin": 0.1,
        "exponential_gamma": 7.0,
        "scope": "global",
        "candidate_k": 16,
        "dropout": 0.0,
        "batch_size": 64,
        "epochs": 30,
        "weight_decay": 0.0,
        "warmup_ratio": 0.04,
        "gradient_clip": 1.0,
        "seed": 0,
        "max_train_prompts": 10_000,
        "train_subset_seed": 20260730,
        "train_split": "train",
        "validation_split": "validation_select",
        "skip_gate": True,
        "memorization_blocks": 0,
        "evidence_tier": "development",
        "calibrate_margin": True,
        "max_calibration_first_token_drop": 0.001,
        "max_calibration_domain_drop": 0.0,
        **{
            key: cell[key]
            for key in (
                "mixer",
                "node_encoder",
                "model_dim",
                "num_heads",
                "num_layers",
                "learning_rate",
            )
        },
    }
    mismatches = {
        key: {"expected": expected, "actual": config.get(key)}
        for key, expected in frozen.items()
        if config.get(key) != expected
    }
    if mismatches:
        raise RuntimeError(f"frozen 10K config mismatch {label}: {mismatches}")
    expected_report = {
        "train_prompts": 10_000,
        "train_blocks": EXPECTED_TRAIN_BLOCKS,
        "train_prompt_set_sha256": EXPECTED_PROMPT_HASH,
        "total_steps": EXPECTED_TOTAL_STEPS,
        "validation_prompts": 147,
        "validation_blocks": 1_175,
        "gate_blocks": 0,
        "parameter_count": cell["parameter_count"],
    }
    report_mismatches = {
        key: {"expected": expected, "actual": report.get(key)}
        for key, expected in expected_report.items()
        if report.get(key) != expected
    }
    if report_mismatches:
        raise RuntimeError(
            f"10K budget/data/model mismatch {label}: {report_mismatches}"
        )
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
        field="validation data metadata",
        path=path,
    )
    target_files = _target_file_signature(provenance, path=path)
    external_train_data = provenance.get("external_train_data")
    external_identity = provenance.get("verified_external_target_embedding_files")
    if (
        not isinstance(external_train_data, list)
        or len(external_train_data) != 1
        or not isinstance(external_train_data[0], dict)
        or not isinstance(external_identity, list)
        or len(external_identity) != 1
        or not isinstance(external_identity[0], dict)
    ):
        raise RuntimeError(f"malformed external training provenance: {path}")
    external_path = external_train_data[0].get("path")
    if not isinstance(external_path, str) or not external_path:
        raise RuntimeError(f"missing external training path: {path}")
    external_hash = _require_sha256(
        external_train_data[0].get("metadata_sha256"),
        field="external training metadata",
        path=path,
    )
    if (
        external_identity[0].get("data") != external_path
        or external_identity[0].get(
            "target_fingerprint_matches_base_collection"
        )
        is not True
        or external_identity[0].get("draft_fingerprint_matches_base_collection")
        is not True
    ):
        raise RuntimeError(f"external target/draft identity check failed: {path}")
    if trainer_hash != trainer_hash_at_end or head_hash != head_hash_at_end:
        raise RuntimeError(f"source changed during 10K feature probe: {path}")

    examples = _example_metrics(report)
    if examples["examples"] != 1_175 or examples["prompts"] != 147:
        raise RuntimeError(f"validation example counts disagree: {path}")
    try:
        base = validation["base"]
        direct = validation["direct"]
        oracle = validation["oracle"]
        diagnostics = validation["direct_diagnostics"]
        base_eal = _finite_float(
            base["mean_accepted_draft_tokens_prompt_balanced"],
            field="base prompt-balanced EAL",
            minimum=0.0,
        )
        direct_eal = _finite_float(
            direct["mean_accepted_draft_tokens_prompt_balanced"],
            field="direct prompt-balanced EAL",
            minimum=0.0,
        )
        oracle_eal = _finite_float(
            oracle["mean_accepted_draft_tokens_prompt_balanced"],
            field="oracle prompt-balanced EAL",
            minimum=0.0,
        )
        first_token_accuracy = _finite_float(
            direct["first_token_accuracy"],
            field="direct first-token accuracy",
            minimum=0.0,
            maximum=1.0,
        )
        harmed_fraction = _finite_float(
            diagnostics["harmed_fraction"],
            field="harmed fraction",
            minimum=0.0,
            maximum=1.0,
        )
        oracle_gap_recovered = _finite_float(
            diagnostics["oracle_gap_recovered"],
            field="oracle gap recovered",
            maximum=1.0,
        )
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError(f"malformed 10K feature metrics: {path}") from error
    assert base_eal is not None and direct_eal is not None and oracle_eal is not None
    assert first_token_accuracy is not None and harmed_fraction is not None
    assert oracle_gap_recovered is not None
    prompt_eal = examples["prompt_eal"]
    oracle_prompt_eal = _oracle_prompt_eal(report)
    recomputed = {
        "base": statistics.fmean(prompt_eal["base"].values()),
        "direct": statistics.fmean(prompt_eal["direct"].values()),
        "oracle": statistics.fmean(oracle_prompt_eal.values()),
        "first_token": examples["direct_first_token_accuracy"],
        "harm": examples["harmed_fraction"],
    }
    reported = {
        "base": base_eal,
        "direct": direct_eal,
        "oracle": oracle_eal,
        "first_token": first_token_accuracy,
        "harm": harmed_fraction,
    }
    inconsistent = {
        key: {"reported": reported[key], "recomputed": value}
        for key, value in recomputed.items()
        if not math.isclose(
            reported[key], value, rel_tol=0.0, abs_tol=1e-12
        )
    }
    if inconsistent:
        raise RuntimeError(
            f"10K metrics disagree with prompt examples: {inconsistent}"
        )
    denominator = oracle_eal - base_eal
    expected_gap = (
        (direct_eal - base_eal) / denominator if denominator > 0 else None
    )
    if expected_gap is None or not math.isclose(
        oracle_gap_recovered,
        expected_gap,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise RuntimeError(f"oracle-gap metric disagrees with EAL: {path}")
    row = {
        "label": label,
        "metrics_path": str(path.resolve()),
        "selected_epoch": report.get("selected_epoch"),
        "parameter_count": report["parameter_count"],
        "seconds": report.get("seconds"),
        "peak_cuda_memory_gib": report.get("peak_cuda_memory_gib"),
        "base_eal": base_eal,
        "direct_eal": direct_eal,
        "oracle_eal": oracle_eal,
        "raw_eal_delta": direct_eal - base_eal,
        "oracle_gap_recovered": oracle_gap_recovered,
        "harmed_fraction": harmed_fraction,
        "first_token_accuracy": first_token_accuracy,
        "first_miss_repair_rate": diagnostics.get(
            "first_miss_repair_rate_given_k"
        ),
    }
    excluded = {
        "output",
        "mixer",
        "node_encoder",
        "model_dim",
        "num_heads",
        "num_layers",
        "learning_rate",
    }
    signature = {
        "trainer_sha256": trainer_hash,
        "head_source_sha256": head_hash,
        "data_metadata_sha256": data_hash,
        "external_path": external_path,
        "external_metadata_sha256": external_hash,
        "target_embedding_files": target_files,
        "external_target_identity": external_identity,
        "matched_config": {
            key: value for key, value in config.items() if key not in excluded
        },
    }
    return row, signature, prompt_eal["direct"]


def summarize_feature_10k(
    run_root: Path,
    *,
    bootstrap_repetitions: int = 20_000,
    bootstrap_seed: int = 20260804,
) -> dict[str, Any]:
    rows = {}
    signatures = []
    predictions = {}
    for label in CELLS:
        row, signature, prompt_eal = _load(run_root, label)
        rows[label] = row
        signatures.append(signature)
        predictions[label] = prompt_eal
    if signatures[1] != signatures[0]:
        raise RuntimeError("10K cells do not share source/data/common config")
    compact = rows["compact_axial_additive_d64_seed0"]
    probe = rows["probe_flat_compat_d640_seed0"]
    checks = {
        "raw_delta_at_least_0p6": (
            float(probe["raw_eal_delta"]) >= MINIMUM_RAW_DELTA
        ),
        "oracle_gap_recovered_at_least_0p15": (
            float(probe["oracle_gap_recovered"])
            >= MINIMUM_ORACLE_GAP_RECOVERED
        ),
    }
    passed = any(checks.values())
    comparison = _paired_bootstrap(
        predictions["probe_flat_compat_d640_seed0"],
        predictions["compact_axial_additive_d64_seed0"],
        repetitions=bootstrap_repetitions,
        seed=bootstrap_seed,
    )
    return {
        "status": "positive_witness" if passed else "scientific_negative",
        "passed": passed,
        "evidence_tier": "development_diagnostic",
        "positive_only_interpretation": (
            "tested_frozen_inputs_and_high_capacity_function_class_are_sufficient_for_material_heldout_gain"
            if passed
            else "engineering_stop_only_not_an_information_ceiling"
        ),
        "next_stage": "run_100k_probe" if passed else "stop_frozen_probe_route",
        "minimum_raw_delta": MINIMUM_RAW_DELTA,
        "minimum_oracle_gap_recovered": MINIMUM_ORACLE_GAP_RECOVERED,
        "checks": checks,
        "probe_minus_compact_raw_eal": (
            float(probe["direct_eal"]) - float(compact["direct_eal"])
        ),
        "probe_minus_compact_prompt_bootstrap": comparison,
        "compact": compact,
        "probe": probe,
    }


def main() -> None:
    args = parse_args()
    try:
        summary = summarize_feature_10k(
            args.run_root,
            bootstrap_repetitions=args.bootstrap_repetitions,
            bootstrap_seed=args.bootstrap_seed,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        print(json.dumps({"status": "artifact_error", "error": str(error)}))
        raise SystemExit(2) from error
    output = args.output or args.run_root / "feature_10k_summary.json"
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
