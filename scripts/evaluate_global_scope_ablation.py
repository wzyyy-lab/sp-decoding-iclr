#!/usr/bin/env python3
"""Evaluate one trained global selector with cross-position paths masked."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import torch

from sph.data import CanonicalBlockDataset
from sph.global_direct_selector import GlobalDirectCandidateSelector
from scripts.train_global_direct_selector import (
    RecordDataset,
    evaluate,
    load_target_embedding,
    make_loader,
    summarize_margin_calibration,
    validate_target_embedding_identity,
)


EAL_FIELD = "mean_accepted_draft_tokens_prompt_balanced"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compact_evaluation(
    evaluation: dict[str, Any], calibrated: dict[str, Any]
) -> dict[str, Any]:
    base = evaluation["base"]
    direct = evaluation["direct"]
    return {
        "base_eal": base[EAL_FIELD],
        "raw_eal": direct[EAL_FIELD],
        "raw_delta_vs_base": direct[EAL_FIELD] - base[EAL_FIELD],
        "calibrated_eal": calibrated[EAL_FIELD],
        "calibrated_delta_vs_base": calibrated[EAL_FIELD] - base[EAL_FIELD],
        "raw_first_token_delta": (
            direct["first_token_accuracy"] - base["first_token_accuracy"]
        ),
        "calibrated_first_token_delta": (
            calibrated["first_token_accuracy"]
            - base["first_token_accuracy"]
        ),
        "raw_direct_diagnostics": evaluation["direct_diagnostics"],
        "calibrated_diagnostics": calibrated["diagnostics"],
        "candidate_classification": evaluation["candidate_classification"],
        "calibrated_by_domain": calibrated["by_domain"],
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("scope ablation requires CUDA")
    report = json.loads(args.metrics.read_text(encoding="utf-8"))
    checkpoint = torch.load(
        args.checkpoint, map_location="cpu", weights_only=False
    )
    config = checkpoint["args"]
    if config["scope"] != "global" or config["mixer"] != "axial":
        raise ValueError("ablation requires an axial-global checkpoint")
    if report["scope"] != "global" or report["final_gate"] is not None:
        raise ValueError("metrics are not an unopened-gate global run")

    collection = CanonicalBlockDataset(args.data)
    metadata = collection.metadata
    validate_target_embedding_identity(metadata, args.target)
    validation_records = [
        record
        for record in collection.records
        if str(record["split"]) == str(config["validation_split"])
    ]
    validation = RecordDataset(validation_records, metadata)
    loader = make_loader(
        validation,
        candidate_k=int(config["candidate_k"]),
        batch_size=args.batch_size,
        shuffle=False,
    )

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    target_embedding = (
        load_target_embedding(args.target)
        .to(device=device, dtype=torch.bfloat16)
        .detach()
    )
    target_embedding.requires_grad_(False)
    block_length = int(validation.records[0]["gold_ids"].numel())
    model = GlobalDirectCandidateSelector(
        hidden_size=int(target_embedding.shape[1]),
        max_positions=block_length,
        max_candidates=int(config["candidate_k"]),
        model_dim=int(config["model_dim"]),
        num_heads=int(config["num_heads"]),
        num_layers=int(config["num_layers"]),
        scope="global",
        mixer="axial",
        dropout=float(config["dropout"]),
    ).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)

    threshold = report["calibration_threshold"]
    if threshold is None:
        raise ValueError("checkpoint has no frozen calibration threshold")
    results = {}
    for scope in ("global", "causal", "local"):
        model.scope = scope
        evaluation = evaluate(
            model,
            loader,
            target_embedding,
            device,
            candidate_k=int(config["candidate_k"]),
            loss_weighting=str(config["loss_weighting"]),
            dpace_alpha=float(config["dpace_alpha"]),
            exponential_gamma=float(config["exponential_gamma"]),
            include_examples=True,
        )
        calibrated = summarize_margin_calibration(
            evaluation, threshold=float(threshold)
        )
        results[scope] = compact_evaluation(evaluation, calibrated)

    stored_raw = report["final_validation"]["direct"][EAL_FIELD]
    reproduced_raw = results["global"]["raw_eal"]
    output = {
        "schema_version": 1,
        "evidence_tier": "development_context_ablation",
        "ablation": (
            "At inference only, replace the trained global attention mask "
            "with causal or same-position-local masks; weights and frozen "
            "margin threshold remain unchanged."
        ),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "metrics": str(args.metrics.resolve()),
        "metrics_sha256": sha256_file(args.metrics),
        "device": torch.cuda.get_device_name(0),
        "validation_prompts": report["validation_prompts"],
        "validation_blocks": report["validation_blocks"],
        "calibration_threshold": threshold,
        "stored_global_raw_eal": stored_raw,
        "reproduced_global_raw_eal": reproduced_raw,
        "absolute_reproduction_difference": abs(
            float(stored_raw) - float(reproduced_raw)
        ),
        "results": results,
        "sealed_gate_evaluated": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, args.output)
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
