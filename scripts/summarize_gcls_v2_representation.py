#!/usr/bin/env python3
"""Aggregate the frozen three-cell GCLS-v2 representation screen."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


EXPECTED_CELLS = {
    "axial_additive_cdpace05": {
        "mixer": "axial",
        "node_encoder": "additive",
        "parameter_count": 1_467_440,
    },
    "flat_additive_cdpace05": {
        "mixer": "flat",
        "node_encoder": "additive",
        "parameter_count": 1_071_968,
    },
    "flat_compat_cdpace05": {
        "mixer": "flat",
        "node_encoder": "compatibility",
        "parameter_count": 1_235_808,
    },
}

FROZEN_CONFIG = {
    "scope": "global",
    "candidate_k": 16,
    "model_dim": 128,
    "num_heads": 8,
    "num_layers": 2,
    "dropout": 0.0,
    "batch_size": 64,
    "epochs": 9,
    "learning_rate": 0.0006,
    "weight_decay": 0.0,
    "warmup_ratio": 0.04,
    "gradient_clip": 1.0,
    "loss_weighting": "candidate_dpace",
    "dpace_alpha": 0.5,
    "base_safety_weight": 0.0,
    "base_safety_margin": 0.1,
    "seed": 0,
    "max_train_prompts": 25_000,
    "train_subset_seed": 20_260_730,
    "train_split": "train",
    "validation_split": "validation_select",
    "skip_gate": True,
    "memorization_blocks": 0,
    "evidence_tier": "development",
    "calibrate_margin": True,
    "max_calibration_first_token_drop": 0.001,
    "max_calibration_domain_drop": 0.0,
}

RAW_EAL_FIELD = "mean_accepted_draft_tokens_prompt_balanced"
MINIMUM_RAW_DELTA = 0.285
MAXIMUM_FIRST_TOKEN_DROP = 0.001
EXPECTED_TRAIN_PROMPT_SET_SHA256 = (
    "a3d25eba926ea8dc474d59b8a4bf3eabef6953d198bf2630d525344c3236fa73"
)
EXPECTED_TRAIN_PROMPTS = 25_000
EXPECTED_TRAIN_BLOCKS = 199_818
EXPECTED_TOTAL_STEPS = 28_107


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _require_number(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(f"{field} is not numeric")
    result = float(value)
    if not math.isfinite(result):
        raise RuntimeError(f"{field} is not finite")
    return result


def _minimum_domain_delta(validation: dict[str, Any]) -> float:
    domains = validation.get("by_domain")
    if not isinstance(domains, dict) or not domains:
        raise RuntimeError("final_validation.by_domain is missing or empty")
    deltas = []
    for domain, metrics in domains.items():
        try:
            direct = metrics["direct"]
            base = metrics["base"]
            direct_value = direct.get(
                RAW_EAL_FIELD, direct["mean_accepted_draft_tokens"]
            )
            base_value = base.get(
                RAW_EAL_FIELD, base["mean_accepted_draft_tokens"]
            )
        except (KeyError, TypeError) as error:
            raise RuntimeError(
                f"malformed domain metrics for {domain}"
            ) from error
        deltas.append(
            _require_number(direct_value, field=f"{domain}.direct_eal")
            - _require_number(base_value, field=f"{domain}.base_eal")
        )
    return min(deltas)


def _validate_config(
    label: str, report: dict[str, Any], path: Path
) -> None:
    config = report.get("config")
    if not isinstance(config, dict):
        raise RuntimeError(f"missing config: {path}")
    expected_cell = EXPECTED_CELLS[label]
    for key in ("mixer", "node_encoder"):
        if config.get(key) != expected_cell[key]:
            raise RuntimeError(
                f"{label} has {key}={config.get(key)!r}, expected "
                f"{expected_cell[key]!r}: {path}"
            )
    for key, expected in FROZEN_CONFIG.items():
        if config.get(key) != expected:
            raise RuntimeError(
                f"{label} violates frozen {key}: found "
                f"{config.get(key)!r}, expected {expected!r}: {path}"
            )
    if report.get("parameter_count") != expected_cell["parameter_count"]:
        raise RuntimeError(
            f"{label} parameter count changed: found "
            f"{report.get('parameter_count')!r}, expected "
            f"{expected_cell['parameter_count']}: {path}"
        )
    if report.get("scope") != "global":
        raise RuntimeError(f"{label} report scope is not global: {path}")
    if report.get("evidence_tier") != "development":
        raise RuntimeError(
            f"{label} is not labeled development evidence: {path}"
        )
    if report.get("split_protocol") != (
        "prompt_disjoint_external_train_development"
    ):
        raise RuntimeError(
            f"{label} has unexpected split protocol: {path}"
        )
    if report.get("final_gate") is not None:
        raise RuntimeError(f"{label} unexpectedly opened the gate split: {path}")


def load_row(run_root: Path, label: str) -> tuple[dict[str, Any], dict[str, Any]]:
    path = run_root / f"{label}_seed0" / "metrics.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing representation artifact: {path}")
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(
            f"corrupt representation artifact: {path}: {error}"
        ) from error
    if not isinstance(report, dict):
        raise RuntimeError(f"representation artifact is not an object: {path}")
    _validate_config(label, report, path)
    validation = report.get("final_validation")
    if not isinstance(validation, dict):
        raise RuntimeError(f"missing final_validation: {path}")
    try:
        base = validation["base"]
        direct = validation["direct"]
        diagnostics = validation["direct_diagnostics"]
        classification = validation["candidate_classification"]
        base_eal = _require_number(base[RAW_EAL_FIELD], field="base_eal")
        direct_eal = _require_number(
            direct[RAW_EAL_FIELD], field="direct_eal"
        )
        base_first = _require_number(
            base["first_token_accuracy"], field="base_first_token_accuracy"
        )
        direct_first = _require_number(
            direct["first_token_accuracy"],
            field="direct_first_token_accuracy",
        )
        harm = _require_number(
            diagnostics["harmed_fraction"], field="harmed_fraction"
        )
    except (KeyError, TypeError) as error:
        raise RuntimeError(f"malformed representation metrics: {path}") from error
    calibrated = validation.get("calibrated")
    calibrated_delta = None
    calibration_threshold = None
    if calibrated is not None:
        if not isinstance(calibrated, dict):
            raise RuntimeError(f"malformed calibrated metrics: {path}")
        calibrated_delta = _require_number(
            calibrated[RAW_EAL_FIELD], field="calibrated_eal"
        ) - base_eal
        calibration_threshold = calibrated.get("threshold")
    row = {
        "label": label,
        "metrics_path": str(path.resolve()),
        "mixer": report["config"]["mixer"],
        "node_encoder": report["config"]["node_encoder"],
        "parameter_count": int(report["parameter_count"]),
        "train_prompts": int(report["train_prompts"]),
        "train_blocks": int(report["train_blocks"]),
        "validation_prompts": int(report["validation_prompts"]),
        "validation_blocks": int(report["validation_blocks"]),
        "total_steps": int(report["total_steps"]),
        "selected_epoch": int(report["selected_epoch"]),
        "seconds": _require_number(report["seconds"], field="seconds"),
        "peak_cuda_memory_gib": _require_number(
            report["peak_cuda_memory_gib"], field="peak_cuda_memory_gib"
        ),
        "peak_cuda_reserved_gib": _require_number(
            report["peak_cuda_reserved_gib"],
            field="peak_cuda_reserved_gib",
        ),
        "base_eal": base_eal,
        "direct_eal": direct_eal,
        "raw_eal_delta": direct_eal - base_eal,
        "calibrated_eal_delta": calibrated_delta,
        "calibration_threshold": calibration_threshold,
        "harmed_fraction": harm,
        "first_token_accuracy": direct_first,
        "first_token_delta_vs_base": direct_first - base_first,
        "minimum_domain_delta": _minimum_domain_delta(validation),
        "first_miss_repair_rate": diagnostics.get(
            "first_miss_repair_rate_given_k"
        ),
        "oracle_gap_recovered": diagnostics.get("oracle_gap_recovered"),
        "candidate_accuracy": classification.get("accuracy"),
        "hard_candidate_accuracy": classification.get("non_top1_accuracy"),
    }
    return row, report


def _assert_matched_reports(reports: dict[str, dict[str, Any]]) -> None:
    match_fields = (
        "train_prompts",
        "train_blocks",
        "train_prompt_set_sha256",
        "validation_prompts",
        "validation_blocks",
        "total_steps",
        "warmup_steps",
    )
    for field in match_fields:
        values = {report.get(field) for report in reports.values()}
        if len(values) != 1:
            raise RuntimeError(
                f"representation reports are unmatched on {field}: {values}"
            )
    exact_values = {
        "train_prompts": EXPECTED_TRAIN_PROMPTS,
        "train_blocks": EXPECTED_TRAIN_BLOCKS,
        "train_prompt_set_sha256": EXPECTED_TRAIN_PROMPT_SET_SHA256,
        "total_steps": EXPECTED_TOTAL_STEPS,
    }
    for field, expected in exact_values.items():
        value = next(iter(reports.values())).get(field)
        if value != expected:
            raise RuntimeError(
                f"representation {field} differs from the frozen subset: "
                f"found {value!r}, expected {expected!r}"
            )
    normalized_configs = set()
    for report in reports.values():
        config = dict(report["config"])
        for treatment_field in ("output", "mixer", "node_encoder"):
            config.pop(treatment_field, None)
        normalized_configs.add(json.dumps(config, sort_keys=True))
    if len(normalized_configs) != 1:
        raise RuntimeError(
            "representation reports differ outside mixer/node encoder/output"
        )
    signatures = set()
    base_eals = set()
    for label, report in reports.items():
        provenance = report.get("provenance")
        if not isinstance(provenance, dict):
            raise RuntimeError(f"{label} has no provenance")
        if provenance.get("trainer_sha256") != provenance.get(
            "trainer_sha256_at_end"
        ) or provenance.get("head_source_sha256") != provenance.get(
            "head_source_sha256_at_end"
        ):
            raise RuntimeError(f"source changed during {label}")
        signatures.add(
            json.dumps(
                {
                    "project_commit": provenance.get("project_commit"),
                    "data_metadata_sha256": provenance.get(
                        "data_metadata_sha256"
                    ),
                    "external_train_data": provenance.get(
                        "external_train_data"
                    ),
                    "trainer_sha256": provenance.get("trainer_sha256"),
                    "head_source_sha256": provenance.get(
                        "head_source_sha256"
                    ),
                    "target_files": provenance.get(
                        "verified_target_embedding_files"
                    ),
                },
                sort_keys=True,
            )
        )
        base_eals.add(
            report["final_validation"]["base"][RAW_EAL_FIELD]
        )
    if len(signatures) != 1:
        raise RuntimeError("source, target, or data provenance differs across cells")
    if len(base_eals) != 1:
        raise RuntimeError("DFlash base EAL differs across matched cells")


def representation_decision(
    rows: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    axial = rows["axial_additive_cdpace05"]
    additive = rows["flat_additive_cdpace05"]
    compatibility = rows["flat_compat_cdpace05"]
    flat_mixer_supported = additive["direct_eal"] > axial["direct_eal"]
    compatibility_supported = (
        flat_mixer_supported
        and compatibility["direct_eal"] > additive["direct_eal"]
    )
    if not flat_mixer_supported:
        selected_label = None
        architecture_decision = "stop_full_lattice_route"
    elif compatibility_supported:
        selected_label = "flat_compat_cdpace05"
        architecture_decision = "select_flat_compatibility"
    else:
        selected_label = "flat_additive_cdpace05"
        architecture_decision = "select_flat_additive_delete_compatibility"

    selected = None if selected_label is None else rows[selected_label]
    checks = {
        "flat_mixer_supported": flat_mixer_supported,
        "raw_delta_exceeds_existing_best": (
            selected is not None
            and selected["raw_eal_delta"] > MINIMUM_RAW_DELTA
        ),
        "harm_not_above_axial": (
            selected is not None
            and selected["harmed_fraction"] <= axial["harmed_fraction"]
        ),
        "first_token_within_axial_tolerance": (
            selected is not None
            and selected["first_token_accuracy"]
            >= axial["first_token_accuracy"] - MAXIMUM_FIRST_TOKEN_DROP
        ),
    }
    enter_scope_confirmation = all(checks.values())
    return {
        "architecture_decision": architecture_decision,
        "selected_label": selected_label,
        "flat_mixer_supported": flat_mixer_supported,
        "flat_additive_minus_axial_raw_eal": (
            additive["direct_eal"] - axial["direct_eal"]
        ),
        "compatibility_supported": compatibility_supported,
        "compatibility_raw_eal_difference": (
            compatibility["direct_eal"] - additive["direct_eal"]
        ),
        "confirmation_checks": checks,
        "enter_scope_confirmation": enter_scope_confirmation,
    }


def summarize_representation(
    run_root: Path,
) -> dict[str, Any]:
    loaded = {
        label: load_row(run_root, label) for label in EXPECTED_CELLS
    }
    rows = {label: item[0] for label, item in loaded.items()}
    reports = {label: item[1] for label, item in loaded.items()}
    _assert_matched_reports(reports)
    decision = representation_decision(rows)
    passed = bool(decision["enter_scope_confirmation"])
    return {
        "status": "passed" if passed else "scientific_negative",
        "passed": passed,
        "evidence_tier": "development",
        "primary_metric": "raw_prompt_balanced_eal",
        "thresholds": {
            "minimum_raw_delta_strictly_greater_than": MINIMUM_RAW_DELTA,
            "maximum_first_token_drop": MAXIMUM_FIRST_TOKEN_DROP,
            "maximum_harm": "matched_axial_harmed_fraction",
        },
        "integrity": {
            "matched_prompt_hash_counts_steps": True,
            "matched_source_target_data_provenance": True,
            "source_stable_during_runs": True,
            "gate_split_opened": False,
        },
        **decision,
        "rows": [rows[label] for label in EXPECTED_CELLS],
    }


def main() -> None:
    args = parse_args()
    try:
        summary = summarize_representation(args.run_root)
    except (
        FileNotFoundError,
        RuntimeError,
        ValueError,
        KeyError,
        TypeError,
        OverflowError,
        AttributeError,
        IndexError,
    ) as error:
        print(json.dumps({"status": "artifact_error", "error": str(error)}))
        raise SystemExit(2) from error
    output = args.output or args.run_root / "representation_summary.json"
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
