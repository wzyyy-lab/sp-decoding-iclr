#!/usr/bin/env python3
"""Apply the adaptive positive-only gate to the fixed-step 100K probe."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics
from typing import Any

try:
    from scripts.summarize_gcls_v4_feature_10k import (
        _example_metrics,
        _finite_float,
        _oracle_prompt_eal,
        _paired_bootstrap,
        _require_sha256,
        _target_file_signature,
    )
except ModuleNotFoundError:
    from summarize_gcls_v4_feature_10k import (  # type: ignore[no-redef]
        _example_metrics,
        _finite_float,
        _oracle_prompt_eal,
        _paired_bootstrap,
        _require_sha256,
        _target_file_signature,
    )


CELLS = {
    "compact_axial_additive_d64_full_seed0": {
        "mixer": "axial",
        "node_encoder": "additive",
        "model_dim": 64,
        "num_heads": 4,
        "num_layers": 1,
        "learning_rate": 0.0006,
        "parameter_count": 433_772,
    },
    "probe_flat_compat_d640_full_seed0": {
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
    "45471a62f93a488f3f7653c096bebcddb0ddae3773f6c99744bd070e348a9405"
)
EXPECTED_TRAIN_PROMPTS = 99_356
EXPECTED_TRAIN_BLOCKS = 793_989
EXPECTED_TOTAL_STEPS = 37_221
EXPECTED_TRAINER_SHA256 = (
    "e104ba65faa2ab94fdf210a1ed8c313d482443bc6a0f89c2136e736dea073110"
)
EXPECTED_HEAD_SHA256 = (
    "f57eb27ce6486d9d5f7edb71639dfa116cdd6591cd413d97066754568ebd1b06"
)
EXPECTED_VALIDATION_METADATA_SHA256 = (
    "0dbca3e9667f6578b02559e2934ee79a677e45d058c68271c94db6e8f4338320"
)
EXPECTED_TARGET_FILES = [
    {
        "path": "config.json",
        "bytes": 726,
        "sha256": "8ba006f74fecfaaeb392872a60f4a480e7ec9860153d2e1b769ec81f9a147f8a",
    },
    {
        "path": "model.safetensors.index.json",
        "bytes": 32_819,
        "sha256": "6dc0981b8829fead746441f68f38f24c5ca4a3a66351f652c26c6df0efc43ab2",
    },
    {
        "path": "model-00001-of-00003.safetensors",
        "bytes": 3_957_900_840,
        "sha256": "328a91d3122359d5547f9d79521205bc0a46e1f79a792dfe650e99fc2d651223",
    },
]
EXPECTED_EXTERNAL_METADATA = {
    "part-000": "d64492233e6112daeee5f54c88cca16dffb3a2b4f98a54ad6c5f11b877935856",
    "part-001": "a55331a31c9dc6efa4a376896ec5d6f7de828f104e3ffcb10da68546425240de",
    "part-002": "4d78caf63e0a70c012d7382e20abacd8abd2029cc27de9885de88043628fd7e5",
    "part-003": "0e447e1a3a40635525128ea36569b3e4a7424d1056aa683a81f0943b32f74d50",
    "part-004": "d7597e41b7b16f533522f0f9f0d741f8de9ad1e3c8bcc3bf8557a24d8d31b42c",
    "part-005": "03099d833d1ca9eaa32dcc1f468038ab0d58d8952c4220bcd0ed5e897ba5ec7a",
    "part-006": "b9dcf53d64b38b1658831e09130cfb9a1a265685ede5a45e562b42e1be016c02",
    "part-007": "48f015efdb40bc8ed32a41af0e9682d2090dd3dd9bb90e6e0906dfe8c6803585",
}
MINIMUM_RAW_DELTA = 0.6
MINIMUM_ORACLE_GAP_RECOVERED = 0.15


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--bootstrap-repetitions", type=int, default=20_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260804)
    return parser.parse_args()


def _external_signature(
    provenance: dict[str, Any], *, path: Path
) -> list[dict[str, str]]:
    external = provenance.get("external_train_data")
    identities = provenance.get("verified_external_target_embedding_files")
    if (
        not isinstance(external, list)
        or len(external) != len(EXPECTED_EXTERNAL_METADATA)
        or not isinstance(identities, list)
        or len(identities) != len(external)
    ):
        raise RuntimeError(f"malformed full-data external provenance: {path}")
    identity_by_path: dict[str, dict[str, Any]] = {}
    for identity in identities:
        if not isinstance(identity, dict):
            raise RuntimeError(f"malformed external identity: {path}")
        data = identity.get("data")
        if not isinstance(data, str) or not data or data in identity_by_path:
            raise RuntimeError(f"invalid/duplicate external identity path: {path}")
        identity_by_path[data] = identity
    result = []
    seen_parts: set[str] = set()
    for entry in external:
        if not isinstance(entry, dict):
            raise RuntimeError(f"malformed external metadata entry: {path}")
        external_path = entry.get("path")
        if not isinstance(external_path, str) or not external_path:
            raise RuntimeError(f"missing external training path: {path}")
        part = Path(external_path).name
        if part not in EXPECTED_EXTERNAL_METADATA or part in seen_parts:
            raise RuntimeError(f"unexpected/duplicate full-data part {part}: {path}")
        seen_parts.add(part)
        metadata_hash = _require_sha256(
            entry.get("metadata_sha256"),
            field=f"external training metadata {part}",
            path=path,
        )
        if metadata_hash != EXPECTED_EXTERNAL_METADATA[part]:
            raise RuntimeError(f"external metadata hash mismatch {part}: {path}")
        identity = identity_by_path.get(external_path)
        if (
            identity is None
            or identity.get("target_fingerprint_matches_base_collection")
            is not True
            or identity.get("draft_fingerprint_matches_base_collection")
            is not True
        ):
            raise RuntimeError(f"external target/draft identity failed {part}: {path}")
        result.append({"part": part, "metadata_sha256": metadata_hash})
    if seen_parts != set(EXPECTED_EXTERNAL_METADATA):
        raise RuntimeError(f"full-data part set incomplete: {path}")
    return sorted(result, key=lambda item: item["part"])


def _prompt_domains(report: dict[str, Any]) -> dict[str, str]:
    result: dict[str, str] = {}
    for index, record in enumerate(report["final_validation"]["examples"]):
        sample_id = record.get("sample_id")
        domain = record.get("domain")
        if not isinstance(sample_id, str) or not isinstance(domain, str):
            raise RuntimeError(f"invalid prompt/domain at example {index}")
        previous = result.setdefault(sample_id, domain)
        if previous != domain:
            raise RuntimeError(f"prompt spans multiple domains: {sample_id}")
    return result


def _load(
    run_root: Path, label: str
) -> tuple[
    dict[str, Any], dict[str, Any], dict[str, float], dict[str, str]
]:
    path = run_root / label / "metrics.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing full feature-probe artifact: {path}")
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"corrupt full feature-probe artifact: {path}") from error
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
        "epochs": 3,
        "weight_decay": 0.0,
        "warmup_ratio": 0.04,
        "gradient_clip": 1.0,
        "seed": 0,
        "max_train_prompts": 0,
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
        raise RuntimeError(f"frozen 100K config mismatch {label}: {mismatches}")
    expected_report = {
        "train_prompts": EXPECTED_TRAIN_PROMPTS,
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
            f"100K budget/data/model mismatch {label}: {report_mismatches}"
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
    external = _external_signature(provenance, path=path)
    if trainer_hash != trainer_hash_at_end or head_hash != head_hash_at_end:
        raise RuntimeError(f"source changed during 100K feature probe: {path}")
    pinned_mismatches = {}
    if trainer_hash != EXPECTED_TRAINER_SHA256:
        pinned_mismatches["trainer_sha256"] = trainer_hash
    if head_hash != EXPECTED_HEAD_SHA256:
        pinned_mismatches["head_source_sha256"] = head_hash
    if data_hash != EXPECTED_VALIDATION_METADATA_SHA256:
        pinned_mismatches["data_metadata_sha256"] = data_hash
    if target_files != EXPECTED_TARGET_FILES:
        pinned_mismatches["target_embedding_files"] = target_files
    if pinned_mismatches:
        raise RuntimeError(
            f"100K run differs from reviewed source/data/target: "
            f"{pinned_mismatches}"
        )

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
        raise RuntimeError(f"malformed 100K feature metrics: {path}") from error
    assert base_eal is not None and direct_eal is not None and oracle_eal is not None
    assert first_token_accuracy is not None and harmed_fraction is not None
    assert oracle_gap_recovered is not None
    prompt_eal = examples["prompt_eal"]
    prompt_domains = _prompt_domains(report)
    if prompt_domains.keys() != prompt_eal["direct"].keys():
        raise RuntimeError(f"prompt/domain support mismatch: {path}")
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
        if not math.isclose(reported[key], value, rel_tol=0.0, abs_tol=1e-12)
    }
    if inconsistent:
        raise RuntimeError(
            f"100K metrics disagree with prompt examples: {inconsistent}"
        )
    denominator = oracle_eal - base_eal
    expected_gap = (
        (direct_eal - base_eal) / denominator if denominator > 0 else None
    )
    if expected_gap is None or not math.isclose(
        oracle_gap_recovered, expected_gap, rel_tol=0.0, abs_tol=1e-12
    ):
        raise RuntimeError(f"oracle-gap metric disagrees with EAL: {path}")
    domain_eal_block_balanced = examples["domain_eal"]
    if (
        domain_eal_block_balanced["base"].keys()
        != domain_eal_block_balanced["direct"].keys()
    ):
        raise RuntimeError(f"base/direct domain support mismatch: {path}")
    domains = sorted(set(prompt_domains.values()))
    domain_eal_prompt_balanced = {
        method: {
            domain: statistics.fmean(
                value
                for sample_id, value in prompt_eal[method].items()
                if prompt_domains[sample_id] == domain
            )
            for domain in domains
        }
        for method in ("base", "direct")
    }
    domain_raw_eal_delta = {
        domain: domain_eal_prompt_balanced["direct"][domain] - base_value
        for domain, base_value in domain_eal_prompt_balanced["base"].items()
    }
    selected_epoch = report.get("selected_epoch")
    if type(selected_epoch) is not int or not 0 <= selected_epoch <= 3:
        raise RuntimeError(f"invalid selected epoch: {path}")
    row = {
        "label": label,
        "metrics_path": str(path.resolve()),
        "selected_epoch": selected_epoch,
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
        "domain_eal_prompt_balanced": domain_eal_prompt_balanced,
        "domain_raw_eal_delta_prompt_balanced": domain_raw_eal_delta,
        "domain_eal_block_balanced_diagnostic": domain_eal_block_balanced,
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
        "external_metadata": external,
        "target_embedding_files": target_files,
        "matched_config": {
            key: value for key, value in config.items() if key not in excluded
        },
    }
    return row, signature, prompt_eal["direct"], prompt_domains


def summarize_feature_100k(
    run_root: Path,
    *,
    bootstrap_repetitions: int = 20_000,
    bootstrap_seed: int = 20260804,
) -> dict[str, Any]:
    rows = {}
    signatures = []
    predictions = {}
    prompt_domains = []
    for label in CELLS:
        row, signature, prompt_eal, domains = _load(run_root, label)
        rows[label] = row
        signatures.append(signature)
        predictions[label] = prompt_eal
        prompt_domains.append(domains)
    if signatures[1] != signatures[0]:
        raise RuntimeError("100K cells do not share source/data/common config")
    if prompt_domains[1] != prompt_domains[0]:
        raise RuntimeError("100K cells do not share prompt/domain assignments")
    compact_label = "compact_axial_additive_d64_full_seed0"
    probe_label = "probe_flat_compat_d640_full_seed0"
    compact = rows[compact_label]
    probe = rows[probe_label]
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
        predictions[probe_label],
        predictions[compact_label],
        repetitions=bootstrap_repetitions,
        seed=bootstrap_seed,
    )
    comparison_by_domain = {}
    for domain in sorted(set(prompt_domains[0].values())):
        keys = sorted(
            key for key, value in prompt_domains[0].items() if value == domain
        )
        comparison_by_domain[domain] = _paired_bootstrap(
            {key: predictions[probe_label][key] for key in keys},
            {key: predictions[compact_label][key] for key in keys},
            repetitions=bootstrap_repetitions,
            seed=bootstrap_seed,
        )
    return {
        "status": "positive_witness" if passed else "engineering_stop",
        "passed": passed,
        "evidence_tier": "adaptive_development_diagnostic",
        "protocol_amendment": (
            "refine-logs/feature-probe/"
            "FIXED_STEP_PROMPT_DIVERSITY_AMENDMENT.md"
        ),
        "ten_k_is_not_a_prerequisite": True,
        "positive_only_interpretation": (
            "tested_frozen_inputs_and_high_capacity_function_class_are_sufficient_for_material_prompt_diverse_heldout_gain"
            if passed
            else "engineering_stop_only_not_an_information_ceiling"
        ),
        "next_stage": (
            "start_separately_preregistered_distillation_project"
            if passed
            else "stop_frozen_selector_family"
        ),
        "minimum_raw_delta": MINIMUM_RAW_DELTA,
        "minimum_oracle_gap_recovered": MINIMUM_ORACLE_GAP_RECOVERED,
        "checks": checks,
        "probe_minus_compact_raw_eal": (
            float(probe["direct_eal"]) - float(compact["direct_eal"])
        ),
        "probe_minus_compact_prompt_bootstrap": comparison,
        "probe_minus_compact_prompt_bootstrap_by_domain": comparison_by_domain,
        "compact": compact,
        "probe": probe,
    }


def main() -> None:
    args = parse_args()
    try:
        summary = summarize_feature_100k(
            args.run_root,
            bootstrap_repetitions=args.bootstrap_repetitions,
            bootstrap_seed=args.bootstrap_seed,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        print(json.dumps({"status": "artifact_error", "error": str(error)}))
        raise SystemExit(2) from error
    output = args.output or args.run_root / "feature_100k_summary.json"
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
