#!/usr/bin/env python3
"""Evaluate a frozen Direct checkpoint with the preregistered one-edit decoder."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import platform
import time
from typing import Any

import torch

from sph.data import CanonicalBlockDataset
from sph.first_miss_action_selector import FirstMissActionSelector
from sph.global_direct_selector import GlobalDirectCandidateSelector

try:
    import train_first_miss_action_selector as fmas
    import train_global_direct_selector as direct
except ModuleNotFoundError:  # Imported as ``scripts.*`` in CPU tests.
    from scripts import train_first_miss_action_selector as fmas
    from scripts import train_global_direct_selector as direct


PROJECT = Path(__file__).resolve().parents[1]
EXPECTED_SOURCE_DATA = PROJECT / "artifacts/canonical/qwen3_4b_phase3_tier1_10035436"
EXPECTED_TRAIN_ROOT = (
    PROJECT / "artifacts/canonical/qwen3_4b_open_perfectblend_100k_10099770"
)
EXPECTED_TRAIN_DATA = [
    str((EXPECTED_TRAIN_ROOT / f"part-{index:03d}").resolve())
    for index in range(8)
]
EXPECTED_TRAIN_METADATA_SHA256 = [
    "d64492233e6112daeee5f54c88cca16dffb3a2b4f98a54ad6c5f11b877935856",
    "a55331a31c9dc6efa4a376896ec5d6f7de828f104e3ffcb10da68546425240de",
    "4d78caf63e0a70c012d7382e20abacd8abd2029cc27de9885de88043628fd7e5",
    "0e447e1a3a40635525128ea36569b3e4a7424d1056aa683a81f0943b32f74d50",
    "d7597e41b7b16f533522f0f9f0d741f8de9ad1e3c8bcc3bf8557a24d8d31b42c",
    "03099d833d1ca9eaa32dcc1f468038ab0d58d8952c4220bcd0ed5e897ba5ec7a",
    "b9dcf53d64b38b1658831e09130cfb9a1a265685ede5a45e562b42e1be016c02",
    "48f015efdb40bc8ed32a41af0e9682d2090dd3dd9bb90e6e0906dfe8c6803585",
]
EXPECTED_DIRECT_CONFIG: dict[str, Any] = {
    "data": str(EXPECTED_SOURCE_DATA.resolve()),
    "train_data": EXPECTED_TRAIN_DATA,
    "target": "/hpc2hdd/home/zwang668/sp decoding/hf_assets/models/Qwen3-4B",
    "scope": "global",
    "mixer": "axial",
    "node_encoder": "additive",
    "candidate_k": 16,
    "model_dim": 64,
    "num_heads": 4,
    "num_layers": 1,
    "dropout": 0.0,
    "batch_size": 64,
    "epochs": 3,
    "learning_rate": 0.0006,
    "weight_decay": 0.0,
    "warmup_ratio": 0.04,
    "gradient_clip": 1.0,
    "loss_weighting": "candidate_dpace",
    "dpace_alpha": 0.5,
    "post_break_weight": 1.0,
    "exponential_gamma": 7.0,
    "base_safety_weight": 0.0,
    "base_safety_margin": 0.1,
    "seed": 0,
    "max_train_prompts": 0,
    "train_subset_seed": 20260730,
    "train_split": "train",
    "validation_split": "validation_select",
    "gate_split": "validation_gate",
    "skip_gate": True,
    "memorization_blocks": 0,
    "memorization_opportunity_fraction": 0.5,
    "require_capacity_gate": False,
    "min_candidate_accuracy": 0.99,
    "min_hard_candidate_accuracy": 0.97,
    "min_first_miss_repair_rate": 0.95,
    "min_oracle_gap_recovered": 0.95,
    "max_harmed_fraction": 0.01,
    "evidence_tier": "development",
    "calibrate_margin": True,
    "max_calibration_first_token_drop": 0.001,
    "max_calibration_domain_drop": 0.0,
}
EXPECTED_DIRECT_METRICS: dict[str, Any] = {
    "split_protocol": "prompt_disjoint_external_train_development",
    "train_blocks": 793989,
    "train_prompts": 99356,
    "train_prompt_set_sha256": (
        "45471a62f93a488f3f7653c096bebcddb0ddae3773f6c99744bd070e348a9405"
    ),
    "validation_blocks": 1175,
    "validation_prompts": 147,
    "total_steps": 37221,
    "parameter_count": 433772,
}
EXPECTED_SOURCE_DATA_SHA256 = (
    "0dbca3e9667f6578b02559e2934ee79a677e45d058c68271c94db6e8f4338320"
)
EXPECTED_DIRECT_TRAINER_SHA256 = (
    "e104ba65faa2ab94fdf210a1ed8c313d482443bc6a0f89c2136e736dea073110"
)
EXPECTED_DIRECT_HEAD_SHA256 = (
    "f57eb27ce6486d9d5f7edb71639dfa116cdd6591cd413d97066754568ebd1b06"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--direct-run", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validation-split", default="validation_select")
    parser.add_argument("--batch-size", type=int, default=64)
    return parser.parse_args()


def validate_direct_checkpoint_contract(
    metrics: dict[str, Any],
    checkpoint: dict[str, Any],
    *,
    direct_run: Path,
) -> dict[str, Any]:
    """Fail closed unless metrics and checkpoint identify the same run."""

    config = metrics.get("config")
    if not isinstance(config, dict):
        raise RuntimeError("direct metrics are missing the frozen config")
    if checkpoint.get("args") != config:
        raise RuntimeError("direct checkpoint args differ from metrics config")
    if int(checkpoint.get("epoch", -1)) != int(metrics.get("selected_epoch", -2)):
        raise RuntimeError("direct checkpoint epoch differs from metrics")
    if int(checkpoint.get("parameter_count", -1)) != int(
        metrics.get("parameter_count", -2)
    ):
        raise RuntimeError("direct checkpoint parameter count differs from metrics")
    for key, expected in EXPECTED_DIRECT_CONFIG.items():
        if config.get(key) != expected:
            raise RuntimeError(
                f"direct config differs from frozen Gate-2 {key}: "
                f"{config.get(key)!r} != {expected!r}"
            )
    if Path(str(config.get("output", ""))).resolve() != direct_run.resolve():
        raise RuntimeError("direct config output differs from --direct-run")
    for key, expected in EXPECTED_DIRECT_METRICS.items():
        if metrics.get(key) != expected:
            raise RuntimeError(
                f"direct metrics differ from frozen Gate-2 {key}: "
                f"{metrics.get(key)!r} != {expected!r}"
            )
    provenance = metrics.get("provenance", {})
    expected_source_hashes = {
        "data_metadata_sha256": EXPECTED_SOURCE_DATA_SHA256,
        "trainer_sha256": EXPECTED_DIRECT_TRAINER_SHA256,
        "trainer_sha256_at_end": EXPECTED_DIRECT_TRAINER_SHA256,
        "head_source_sha256": EXPECTED_DIRECT_HEAD_SHA256,
        "head_source_sha256_at_end": EXPECTED_DIRECT_HEAD_SHA256,
    }
    for key, expected in expected_source_hashes.items():
        if provenance.get(key) != expected:
            raise RuntimeError(
                f"direct provenance differs from frozen Gate-2 {key}"
            )
    external = provenance.get("external_train_data")
    if not isinstance(external, list) or len(external) != 8:
        raise RuntimeError("direct provenance lacks eight frozen train parts")
    observed_external = [
        (str(item.get("path", "")), str(item.get("metadata_sha256", "")))
        for item in external
    ]
    expected_external = list(
        zip(EXPECTED_TRAIN_DATA, EXPECTED_TRAIN_METADATA_SHA256, strict=True)
    )
    if observed_external != expected_external:
        raise RuntimeError("direct external-train provenance differs from Gate 2")
    return config


def build_direct_model(
    config: dict[str, Any], *, hidden_size: int, block_length: int
) -> GlobalDirectCandidateSelector:
    return GlobalDirectCandidateSelector(
        hidden_size=hidden_size,
        max_positions=block_length,
        max_candidates=int(config["candidate_k"]),
        model_dim=int(config["model_dim"]),
        num_heads=int(config["num_heads"]),
        num_layers=int(config["num_layers"]),
        scope=str(config["scope"]),
        mixer=str(config["mixer"]),
        node_encoder=str(config.get("node_encoder", "additive")),
        dropout=float(config["dropout"]),
        initialization_seed=int(config["seed"]),
    )


def validate_isolated_data_contract(
    metadata: dict[str, Any],
    records: list[dict[str, Any]],
    metrics: dict[str, Any],
    *,
    validation_split: str,
    metadata_sha256: str,
) -> None:
    observed_splits = {str(record["split"]) for record in records}
    if observed_splits != {validation_split}:
        raise RuntimeError(
            "one-edit evaluation data are not physically isolated: "
            f"{sorted(observed_splits)}"
        )
    direct_data_sha256 = str(
        metrics.get("provenance", {}).get("data_metadata_sha256", "")
    )
    source_data_sha256 = str(
        metadata.get("provenance", {})
        .get("split_materialization", {})
        .get("source_collection", {})
        .get("metadata_sha256", "")
    )
    if direct_data_sha256 not in {metadata_sha256, source_data_sha256}:
        raise RuntimeError(
            "isolated evaluation data do not descend from the direct run data"
        )


def _assert_method_summary_matches(
    observed: dict[str, Any], expected: dict[str, Any], *, label: str
) -> None:
    for field in (
        "mean_accepted_draft_tokens",
        "mean_verification_advance",
        "mean_accepted_draft_tokens_prompt_balanced",
        "mean_verification_advance_prompt_balanced",
        "first_token_accuracy",
    ):
        if field not in expected:
            raise RuntimeError(f"direct metrics lack {label}.{field}")
        if not math.isclose(
            float(observed[field]),
            float(expected[field]),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise RuntimeError(
                f"recomputed {label}.{field} differs from direct metrics: "
                f"{observed[field]} != {expected[field]}"
            )


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Direct one-edit evaluation requires CUDA")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")

    metrics_path = args.direct_run / "metrics.json"
    checkpoint_path = args.direct_run / "best.pt"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = validate_direct_checkpoint_contract(
        metrics, checkpoint, direct_run=args.direct_run
    )
    if str(config["validation_split"]) != args.validation_split:
        raise RuntimeError("requested validation split differs from direct config")

    source_paths = {
        "evaluator": Path(__file__).resolve(),
        "fmas_evaluator": PROJECT / "scripts/train_first_miss_action_selector.py",
        "direct_trainer": PROJECT / "scripts/train_global_direct_selector.py",
        "fmas_head": PROJECT / "src/sph/first_miss_action_selector.py",
        "direct_head": PROJECT / "src/sph/global_direct_selector.py",
        "canonical_data": PROJECT / "src/sph/data.py",
    }
    source_hashes_start = {
        f"{name}_sha256": direct.sha256_file(path)
        for name, path in source_paths.items()
    }
    start = time.perf_counter()
    collection = CanonicalBlockDataset(args.data)
    metadata_path = args.data / "metadata.json"
    metadata_sha256 = direct.sha256_file(metadata_path)
    validate_isolated_data_contract(
        collection.metadata,
        collection.records,
        metrics,
        validation_split=args.validation_split,
        metadata_sha256=metadata_sha256,
    )
    expected_blocks = int(metrics.get("validation_blocks", -1))
    expected_prompts = int(metrics.get("validation_prompts", -1))
    observed_prompts = len(
        {str(record["sample_id"]) for record in collection.records}
    )
    if len(collection) != expected_blocks or observed_prompts != expected_prompts:
        raise RuntimeError(
            "isolated validation cardinality differs from the direct run: "
            f"{len(collection)}/{observed_prompts} != "
            f"{expected_blocks}/{expected_prompts}"
        )
    direct.validate_target_embedding_identity(collection.metadata, args.target)

    device = torch.device("cuda:0")
    target_embedding = (
        direct.load_target_embedding(args.target)
        .to(device=device, dtype=torch.bfloat16)
        .detach()
    )
    target_embedding.requires_grad_(False)
    block_length = int(collection.records[0]["gold_ids"].numel())
    backbone = build_direct_model(
        config,
        hidden_size=int(target_embedding.shape[1]),
        block_length=block_length,
    )
    backbone.load_state_dict(checkpoint["model"], strict=True)
    backbone = backbone.to(device)
    parameter_count = sum(parameter.numel() for parameter in backbone.parameters())
    if parameter_count != int(checkpoint["parameter_count"]):
        raise RuntimeError("reconstructed direct model parameter count differs")
    wrapper = FirstMissActionSelector(backbone)
    loader = direct.make_loader(
        collection,
        candidate_k=int(config["candidate_k"]),
        batch_size=args.batch_size,
        shuffle=False,
    )
    evaluation = fmas.evaluate(
        wrapper,
        loader,
        target_embedding,
        device,
        candidate_k=int(config["candidate_k"]),
        include_examples=True,
    )
    final_direct = metrics.get("final_validation", {})
    _assert_method_summary_matches(
        evaluation["base"], final_direct.get("base", {}), label="base"
    )
    _assert_method_summary_matches(
        evaluation["direct_native"],
        final_direct.get("direct", {}),
        label="direct_native",
    )

    source_hashes_end = {
        f"{name}_sha256_at_end": direct.sha256_file(path)
        for name, path in source_paths.items()
    }
    for name in source_paths:
        if source_hashes_start[f"{name}_sha256"] != source_hashes_end[
            f"{name}_sha256_at_end"
        ]:
            raise RuntimeError(f"source changed during evaluation: {name}")
    report = {
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "hostname": platform.node(),
        "device": torch.cuda.get_device_name(device),
        "evidence_tier": "development_control",
        "decoder": "keep_or_global_max_margin_single_edit",
        "direct_run": str(args.direct_run.resolve()),
        "direct_metrics_sha256": direct.sha256_file(metrics_path),
        "direct_checkpoint_sha256": direct.sha256_file(checkpoint_path),
        "direct_selected_epoch": int(checkpoint["epoch"]),
        "data": str(args.data.resolve()),
        "data_metadata_sha256": metadata_sha256,
        "validation_split": args.validation_split,
        "validation_blocks": len(collection),
        "validation_prompts": observed_prompts,
        "seconds": time.perf_counter() - start,
        "direct_native_identity_check": "exact_method_summary_match",
        "base": evaluation["base"],
        "direct_native": evaluation["direct_native"],
        "direct_one_edit": evaluation["fmas"],
        "single_edit_oracle": evaluation["single_edit_oracle"],
        "one_edit_diagnostics": evaluation["fmas_diagnostics"],
        "action_diagnostics": evaluation["action_classification"],
        "by_domain": {
            domain: {
                "base": values["base"],
                "direct_native": values["direct_native"],
                "direct_one_edit": values["fmas"],
                "single_edit_oracle": values["single_edit_oracle"],
            }
            for domain, values in evaluation["by_domain"].items()
        },
        "examples": evaluation["examples"],
        "provenance": {
            **source_hashes_start,
            **source_hashes_end,
            "target_identity_verified": True,
            "isolated_data_identity_verified": True,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "direct_native": report["direct_native"],
                "direct_one_edit": report["direct_one_edit"],
                "one_edit_diagnostics": report["one_edit_diagnostics"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
