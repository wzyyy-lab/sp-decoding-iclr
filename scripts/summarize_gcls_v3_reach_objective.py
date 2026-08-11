#!/usr/bin/env python3
"""Apply the frozen OPB-25K reachable-support development gate."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import random
import statistics
from typing import Any


EXPECTED_WEIGHTS = {
    "post1_control": 1.0,
    "post0_hard": 0.0,
    "post0p1_soft": 0.1,
}
EXPECTED_PROMPT_HASH = (
    "a3d25eba926ea8dc474d59b8a4bf3eabef6953d198bf2630d525344c3236fa73"
)
EXPECTED_TRAIN_PROMPTS = 25_000
EXPECTED_TRAIN_BLOCKS = 199_818
EXPECTED_TOTAL_STEPS = 37_476
MINIMUM_RAW_EFFECT = 0.05
MAXIMUM_HARM_INCREASE = 0.01
MAXIMUM_FIRST_TOKEN_DROP = 0.001
HEX_DIGITS = frozenset("0123456789abcdef")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--bootstrap-repetitions", type=int, default=20_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260804)
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


def _finite_float(
    value: Any,
    *,
    field: str,
    minimum: float | None = None,
    maximum: float | None = None,
    allow_none: bool = False,
) -> float | None:
    if value is None and allow_none:
        return None
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise RuntimeError(f"{field} must be finite numeric data")
    result = float(value)
    if minimum is not None and result < minimum:
        raise RuntimeError(f"{field} is below {minimum}")
    if maximum is not None and result > maximum:
        raise RuntimeError(f"{field} is above {maximum}")
    return result


def _example_metrics(report: dict[str, Any]) -> dict[str, Any]:
    grouped: dict[str, dict[str, list[float]]] = {
        "base": defaultdict(list),
        "direct": defaultdict(list),
    }
    domain_values: dict[str, dict[str, list[float]]] = {
        "base": defaultdict(list),
        "direct": defaultdict(list),
    }
    examples = report["final_validation"].get("examples")
    if not isinstance(examples, list) or not examples:
        raise RuntimeError("final validation lacks prompt-level examples")
    direct_first_correct = 0
    harmed = 0
    for index, record in enumerate(examples):
        try:
            sample_id = record["sample_id"]
            domain = record["domain"]
            accepted = record["accepted_draft_tokens"]
            first_correct = record["first_token_correct"]
            base_value = accepted["base"]
            direct_value = accepted["direct"]
            oracle_value = record["oracle_accepted_draft_tokens"]
            paths = record["candidate_path_indices"]
            base_path = paths["base"]
            direct_path = paths["direct"]
        except (KeyError, TypeError) as error:
            raise RuntimeError(
                f"malformed validation example at index {index}"
            ) from error
        if not isinstance(sample_id, str) or not sample_id:
            raise RuntimeError(f"invalid sample_id at example {index}")
        if not isinstance(domain, str) or not domain:
            raise RuntimeError(f"invalid domain at example {index}")
        if (
            not isinstance(base_path, list)
            or not isinstance(direct_path, list)
            or not base_path
            or len(base_path) != len(direct_path)
            or any(type(candidate) is not int for candidate in base_path)
            or any(type(candidate) is not int for candidate in direct_path)
        ):
            raise RuntimeError(f"invalid candidate paths at example {index}")
        max_accepted = len(direct_path)
        for method, value in (("base", base_value), ("direct", direct_value)):
            if type(value) is not int or not 0 <= value <= max_accepted:
                raise RuntimeError(
                    f"invalid {method} accepted length at example {index}"
                )
            grouped[method][sample_id].append(float(value))
            domain_values[method][domain].append(float(value))
        if (
            type(oracle_value) is not int
            or not 0 <= oracle_value <= max_accepted
            or base_value > oracle_value
            or direct_value > oracle_value
        ):
            raise RuntimeError(f"invalid oracle bound at example {index}")
        if (
            not isinstance(first_correct, dict)
            or type(first_correct.get("base")) is not bool
            or type(first_correct.get("direct")) is not bool
            or first_correct["base"] != (base_value > 0)
            or first_correct["direct"] != (direct_value > 0)
        ):
            raise RuntimeError(
                f"invalid first-token flags at example {index}"
            )
        direct_first_correct += int(first_correct["direct"])
        harmed += int(direct_value < base_value)
    prompt_eal = {
        method: {
            sample_id: statistics.fmean(values)
            for sample_id, values in by_prompt.items()
        }
        for method, by_prompt in grouped.items()
    }
    return {
        "examples": len(examples),
        "prompts": len(prompt_eal["direct"]),
        "prompt_eal": prompt_eal,
        "domain_eal": {
            method: {
                domain: statistics.fmean(values)
                for domain, values in by_domain.items()
            }
            for method, by_domain in domain_values.items()
        },
        "direct_first_token_accuracy": direct_first_correct / len(examples),
        "harmed_fraction": harmed / len(examples),
    }


def _paired_bootstrap(
    left: dict[str, float],
    right: dict[str, float],
    *,
    repetitions: int,
    seed: int,
) -> dict[str, Any]:
    if repetitions < 1:
        raise ValueError("bootstrap repetitions must be positive")
    if left.keys() != right.keys():
        raise RuntimeError("objective cells have different validation prompts")
    deltas = [left[key] - right[key] for key in left]
    generator = random.Random(seed)
    draws = [
        statistics.fmean(
            deltas[generator.randrange(len(deltas))] for _ in deltas
        )
        for _ in range(repetitions)
    ]
    draws.sort()
    return {
        "estimate": statistics.fmean(deltas),
        "ci95": [
            draws[int(0.025 * (repetitions - 1))],
            draws[int(0.975 * (repetitions - 1))],
        ],
        "prompts": len(deltas),
        "repetitions": repetitions,
        "unit": "prompt",
    }


def _load(run_root: Path, label: str) -> tuple[dict[str, Any], dict[str, Any], dict[str, float]]:
    path = run_root / f"{label}_seed0" / "metrics.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing reach-objective artifact: {path}")
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"corrupt reach-objective artifact: {path}") from error
    config = report.get("config")
    provenance = report.get("provenance")
    validation = report.get("final_validation")
    if not all(isinstance(item, dict) for item in (config, provenance, validation)):
        raise RuntimeError(f"missing config/provenance/validation: {path}")
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
        "batch_size": 64,
        "epochs": 12,
        "learning_rate": 0.0006,
        "weight_decay": 0.0,
        "warmup_ratio": 0.04,
        "gradient_clip": 1.0,
        "exponential_gamma": 7.0,
        "base_safety_margin": 0.1,
        "seed": 0,
        "max_train_prompts": 25_000,
        "train_subset_seed": 20260730,
        "train_split": "train",
        "validation_split": "validation_select",
        "skip_gate": True,
        "evidence_tier": "development",
        "calibrate_margin": True,
        "max_calibration_first_token_drop": 0.001,
        "max_calibration_domain_drop": 0.0,
    }
    mismatches = {
        key: {"expected": value, "actual": config.get(key)}
        for key, value in frozen.items()
        if config.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"frozen objective config mismatch {label}: {mismatches}")
    expected_report = {
        "train_prompts": EXPECTED_TRAIN_PROMPTS,
        "train_blocks": EXPECTED_TRAIN_BLOCKS,
        "train_prompt_set_sha256": EXPECTED_PROMPT_HASH,
        "total_steps": EXPECTED_TOTAL_STEPS,
        "validation_prompts": 147,
        "validation_blocks": 1175,
        "gate_blocks": 0,
        "parameter_count": 433_772,
    }
    report_mismatches = {
        key: {"expected": value, "actual": report.get(key)}
        for key, value in expected_report.items()
        if report.get(key) != value
    }
    if report_mismatches:
        raise RuntimeError(f"objective budget/provenance mismatch {label}: {report_mismatches}")
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
        field="validation data metadata",
        path=path,
    )
    external_train_data = provenance.get("external_train_data")
    if not isinstance(external_train_data, list) or not external_train_data:
        raise RuntimeError(f"missing external training provenance: {path}")
    for index, entry in enumerate(external_train_data):
        if not isinstance(entry, dict):
            raise RuntimeError(
                f"malformed external training provenance {index}: {path}"
            )
        external_path = entry.get("path")
        if not isinstance(external_path, str) or not external_path:
            raise RuntimeError(
                f"missing external training path {index}: {path}"
            )
        _require_sha256(
            entry.get("metadata_sha256"),
            field=f"external training metadata {index}",
            path=path,
        )
    target_files = _target_file_signature(provenance, path=path)
    external_target_checks = provenance.get(
        "verified_external_target_embedding_files"
    )
    if (
        not isinstance(external_target_checks, list)
        or len(external_target_checks) != len(external_train_data)
    ):
        raise RuntimeError(
            f"missing external target/draft identity checks: {path}"
        )
    for index, (source, identity) in enumerate(
        zip(external_train_data, external_target_checks, strict=True)
    ):
        if (
            not isinstance(identity, dict)
            or identity.get("data") != source["path"]
            or identity.get("target_fingerprint_matches_base_collection")
            is not True
            or identity.get("draft_fingerprint_matches_base_collection")
            is not True
        ):
            raise RuntimeError(
                f"external target/draft identity check failed {index}: {path}"
            )
    if trainer_hash != trainer_hash_at_end or head_hash != head_hash_at_end:
        raise RuntimeError(f"source changed during objective run: {path}")
    example_metrics = _example_metrics(report)
    if (
        example_metrics["examples"] != report["validation_blocks"]
        or example_metrics["prompts"] != report["validation_prompts"]
    ):
        raise RuntimeError(
            f"validation examples disagree with report counts: {path}"
        )
    try:
        base = validation["base"]
        direct = validation["direct"]
        diagnostics = validation["direct_diagnostics"]
        components = validation["loss"]["components"]
        classification = validation["candidate_classification"]
        by_domain = validation["by_domain"]
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
        first_miss_repair_rate = _finite_float(
            diagnostics.get("first_miss_repair_rate_given_k"),
            field="first-miss repair rate",
            minimum=0.0,
            maximum=1.0,
            allow_none=True,
        )
        oracle_gap_recovered = _finite_float(
            diagnostics.get("oracle_gap_recovered"),
            field="oracle gap recovered",
            maximum=1.0,
            allow_none=True,
        )
        candidate_accuracy = _finite_float(
            classification["accuracy"],
            field="candidate accuracy",
            minimum=0.0,
            maximum=1.0,
            allow_none=True,
        )
        hard_candidate_accuracy = _finite_float(
            classification["non_top1_accuracy"],
            field="hard candidate accuracy",
            minimum=0.0,
            maximum=1.0,
            allow_none=True,
        )
        reachable_fraction = _finite_float(
            components["reachable_fraction_of_coverage"],
            field="reachable fraction of coverage",
            minimum=0.0,
            maximum=1.0,
        )
        post_break_positions = _finite_float(
            components["post_break_positions_per_block"],
            field="post-break positions per block",
            minimum=0.0,
        )
        post_break_suffix_loss = _finite_float(
            components["post_break_suffix_loss"],
            field="post-break suffix loss",
            minimum=0.0,
        )
        weighted_post_break_suffix_loss = _finite_float(
            components["weighted_post_break_suffix_loss"],
            field="weighted post-break suffix loss",
            minimum=0.0,
        )
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError(f"malformed objective metrics: {path}") from error
    assert base_eal is not None and direct_eal is not None
    assert first_token_accuracy is not None and harmed_fraction is not None
    prompt_eal = example_metrics["prompt_eal"]
    recomputed_base_eal = statistics.fmean(prompt_eal["base"].values())
    recomputed_direct_eal = statistics.fmean(prompt_eal["direct"].values())
    scalar_checks = {
        "base prompt-balanced EAL": (base_eal, recomputed_base_eal),
        "direct prompt-balanced EAL": (direct_eal, recomputed_direct_eal),
        "direct first-token accuracy": (
            first_token_accuracy,
            example_metrics["direct_first_token_accuracy"],
        ),
        "harmed fraction": (
            harmed_fraction,
            example_metrics["harmed_fraction"],
        ),
    }
    inconsistent = {
        name: {"reported": reported, "recomputed": recomputed}
        for name, (reported, recomputed) in scalar_checks.items()
        if not math.isclose(
            reported,
            recomputed,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    }
    if inconsistent:
        raise RuntimeError(
            f"objective metrics disagree with prompt examples: {inconsistent}"
        )
    expected_domains = set(example_metrics["domain_eal"]["direct"])
    if not isinstance(by_domain, dict) or set(by_domain) != expected_domains:
        raise RuntimeError(f"domain metrics disagree with examples: {path}")
    domain_deltas = []
    for domain in sorted(expected_domains):
        try:
            reported_base = _finite_float(
                by_domain[domain]["base"]["mean_accepted_draft_tokens"],
                field=f"{domain} base EAL",
                minimum=0.0,
            )
            reported_direct = _finite_float(
                by_domain[domain]["direct"]["mean_accepted_draft_tokens"],
                field=f"{domain} direct EAL",
                minimum=0.0,
            )
        except (KeyError, TypeError, ValueError) as error:
            raise RuntimeError(f"malformed domain metrics: {path}") from error
        assert reported_base is not None and reported_direct is not None
        expected_base = example_metrics["domain_eal"]["base"][domain]
        expected_direct = example_metrics["domain_eal"]["direct"][domain]
        if not (
            math.isclose(
                reported_base, expected_base, rel_tol=0.0, abs_tol=1e-12
            )
            and math.isclose(
                reported_direct,
                expected_direct,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ):
            raise RuntimeError(
                f"domain metrics disagree with validation examples: {path}"
            )
        domain_deltas.append(reported_direct - reported_base)
    minimum_domain_delta = min(domain_deltas)
    row = {
        "label": label,
        "post_break_weight": expected_weight,
        "metrics_path": str(path.resolve()),
        "selected_epoch": report.get("selected_epoch"),
        "parameter_count": report.get("parameter_count"),
        "seconds": report.get("seconds"),
        "base_eal": base_eal,
        "direct_eal": direct_eal,
        "raw_eal_delta": direct_eal - base_eal,
        "harmed_fraction": harmed_fraction,
        "first_token_accuracy": first_token_accuracy,
        "minimum_domain_delta": minimum_domain_delta,
        "first_miss_repair_rate": first_miss_repair_rate,
        "oracle_gap_recovered": oracle_gap_recovered,
        "candidate_accuracy": candidate_accuracy,
        "hard_candidate_accuracy": hard_candidate_accuracy,
        "reachable_fraction_of_coverage": reachable_fraction,
        "post_break_positions_per_block": post_break_positions,
        "post_break_suffix_loss": post_break_suffix_loss,
        "weighted_post_break_suffix_loss": weighted_post_break_suffix_loss,
    }
    signature = {
        "trainer_sha256": trainer_hash,
        "head_source_sha256": head_hash,
        "data_metadata_sha256": data_hash,
        "external_train_data": external_train_data,
        "target_embedding_files": target_files,
        "external_target_identity": external_target_checks,
        "parameter_count": report.get("parameter_count"),
        "matched_config": {
            key: value
            for key, value in config.items()
            if key not in {"output", "post_break_weight"}
        },
    }
    return row, signature, prompt_eal["direct"]


def summarize_objective(
    run_root: Path,
    *,
    bootstrap_repetitions: int = 20_000,
    bootstrap_seed: int = 20260804,
) -> dict[str, Any]:
    rows: dict[str, dict[str, Any]] = {}
    signatures = []
    predictions: dict[str, dict[str, float]] = {}
    for label in EXPECTED_WEIGHTS:
        row, signature, prompt_predictions = _load(run_root, label)
        rows[label] = row
        signatures.append(signature)
        predictions[label] = prompt_predictions
    if any(signature != signatures[0] for signature in signatures[1:]):
        raise RuntimeError("objective cells do not share source/data signature")
    control = rows["post1_control"]
    selected = max(
        (rows["post0_hard"], rows["post0p1_soft"]),
        key=lambda item: (
            float(item["direct_eal"]),
            -float(item["harmed_fraction"]),
            float(item["minimum_domain_delta"]),
            float(item["first_token_accuracy"]),
        ),
    )
    effect = float(selected["direct_eal"]) - float(control["direct_eal"])
    hard_zero = rows["post0_hard"]["weighted_post_break_suffix_loss"]
    checks = {
        "raw_effect_at_least_0p05": effect >= MINIMUM_RAW_EFFECT,
        "harm_within_control_plus_0p01": (
            float(selected["harmed_fraction"])
            <= float(control["harmed_fraction"]) + MAXIMUM_HARM_INCREASE
        ),
        "first_token_within_control_tolerance": (
            float(selected["first_token_accuracy"])
            >= float(control["first_token_accuracy"]) - MAXIMUM_FIRST_TOKEN_DROP
        ),
        "hard_cell_weighted_suffix_loss_zero": (
            hard_zero is not None and float(hard_zero) == 0.0
        ),
    }
    passed = all(checks.values())
    comparisons = {
        label: _paired_bootstrap(
            predictions[label],
            predictions["post1_control"],
            repetitions=bootstrap_repetitions,
            seed=bootstrap_seed + index,
        )
        for index, label in enumerate(("post0_hard", "post0p1_soft"))
    }
    return {
        "status": "passed" if passed else "scientific_negative",
        "passed": passed,
        "evidence_tier": "development",
        "selected_label": selected["label"] if passed else None,
        "architecture_decision": (
            "enter_full_data_confirmation"
            if passed
            else "close_reachable_support_route"
        ),
        "minimum_raw_effect": MINIMUM_RAW_EFFECT,
        "selected_minus_control_raw_eal": effect,
        "checks": checks,
        "control": control,
        "selected_treatment": selected,
        "paired_prompt_bootstrap": comparisons,
        "rows": [rows[label] for label in EXPECTED_WEIGHTS],
    }


def main() -> None:
    args = parse_args()
    try:
        summary = summarize_objective(
            args.run_root,
            bootstrap_repetitions=args.bootstrap_repetitions,
            bootstrap_seed=args.bootstrap_seed,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        print(json.dumps({"status": "artifact_error", "error": str(error)}))
        raise SystemExit(2) from error
    output = args.output or args.run_root / "reach_objective_summary.json"
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
