#!/usr/bin/env python3
"""Audit selected development checkpoints on train/validation/test splits.

This script is intentionally diagnostic.  It does not select a checkpoint or
authorize a paper claim; it characterizes whether a failed development probe
underfit or memorized its small training collection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import time
from pathlib import Path
from typing import Any

import torch

from sph.data import CanonicalBlockDataset
from sph.survival_path_head import (
    BidirectionalSurvivalPathHead,
    SurvivalPathHead,
)
from train_survival_head import evaluate, load_target_embedding, make_loader


PROJECT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--runs-root",
        type=Path,
        action="append",
        required=True,
        help="Root containing one or more recursively discoverable best.pt files.",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_revision(path: Path) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def git_is_dirty(path: Path) -> bool | None:
    result = subprocess.run(
        ["git", "-C", str(path), "status", "--porcelain"],
        check=False,
        capture_output=True,
        text=True,
    )
    return bool(result.stdout.strip()) if result.returncode == 0 else None


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing audit: {path}")
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def build_head(
    config: dict[str, Any], hidden_size: int, block_length: int
) -> torch.nn.Module:
    if config["head_type"] == "no_mixer":
        return SurvivalPathHead(hidden_size, rank=int(config["rank"]))
    return BidirectionalSurvivalPathHead(
        hidden_size,
        rank=int(config["rank"]),
        model_dim=int(config["model_dim"]),
        num_heads=int(config["num_heads"]),
        max_positions=block_length,
    )


def compact_metrics(report: dict[str, Any]) -> dict[str, Any]:
    methods = [
        "base",
        "local",
        "local_map",
        "local_survival",
        "global_map",
        "global_survival",
    ]
    return {
        "nll": report["nll"],
        "local_nll": report["local_nll"],
        "absorbing_crf_nll": report["absorbing_crf_nll"],
        "accepted_draft_tokens": {
            name: report[name]["mean_accepted_draft_tokens"] for name in methods
        },
        "first_token_accuracy": {
            name: report[name]["first_token_accuracy"] for name in methods
        },
        "decoder_disagreement": report["decoder_disagreement"],
        "score_diagnostics": report["score_diagnostics"],
        "by_domain": report["by_domain"],
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("checkpoint audit requires CUDA")
    checkpoints = sorted(
        checkpoint.resolve()
        for root in args.runs_root
        for checkpoint in root.glob("**/best.pt")
    )
    if not checkpoints:
        raise ValueError("no best.pt checkpoints found")

    provenance = {
        "project_commit": git_revision(PROJECT),
        "project_dirty_at_start": git_is_dirty(PROJECT),
        "audit_script_sha256": sha256_file(Path(__file__)),
        "trainer_sha256": sha256_file(PROJECT / "scripts" / "train_survival_head.py"),
        "head_source_sha256": sha256_file(
            PROJECT / "src" / "sph" / "survival_path_head.py"
        ),
    }
    device = torch.device("cuda:0")
    embedding_cache: dict[Path, torch.Tensor] = {}
    loader_cache: dict[tuple[Path, str, int, int], Any] = {}
    dataset_cache: dict[tuple[Path, str], CanonicalBlockDataset] = {}
    records = []
    start = time.perf_counter()

    for checkpoint_path in checkpoints:
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
        config = checkpoint["args"]
        data_path = Path(config["data"]).resolve()
        target_path = Path(config["target"]).resolve()
        candidate_k = int(config["candidate_k"])
        batch_size = int(config["batch_size"])
        if target_path not in embedding_cache:
            embedding_cache[target_path] = load_target_embedding(target_path).to(
                device=device, dtype=torch.bfloat16
            )
        embedding = embedding_cache[target_path]

        split_names = {
            "train": config["train_split"],
            "validation": config["validation_split"],
            "test": config["test_split"],
        }
        for split_name in split_names.values():
            dataset_key = (data_path, split_name)
            if dataset_key not in dataset_cache:
                dataset_cache[dataset_key] = CanonicalBlockDataset(
                    data_path, split=split_name
                )
            loader_key = (data_path, split_name, candidate_k, batch_size)
            if loader_key not in loader_cache:
                loader_cache[loader_key] = make_loader(
                    dataset_cache[dataset_key],
                    candidate_k=candidate_k,
                    batch_size=batch_size,
                    shuffle=False,
                )

        train_dataset = dataset_cache[(data_path, split_names["train"])]
        block_length = int(train_dataset.records[0]["gold_ids"].numel())
        head = build_head(config, int(embedding.shape[1]), block_length).to(device)
        head.load_state_dict(checkpoint["model"])
        reports = {
            label: compact_metrics(
                evaluate(
                    head,
                    loader_cache[(data_path, split, candidate_k, batch_size)],
                    embedding,
                    device,
                    config["normalization"],
                )
            )
            for label, split in split_names.items()
        }
        source_metrics = checkpoint_path.with_name("metrics.json")
        source_report = (
            json.loads(source_metrics.read_text()) if source_metrics.exists() else None
        )
        records.append(
            {
                "checkpoint": str(checkpoint_path),
                "checkpoint_sha256": sha256_file(checkpoint_path),
                "source_metrics": (
                    str(source_metrics) if source_metrics.exists() else None
                ),
                "source_metrics_sha256": (
                    sha256_file(source_metrics) if source_metrics.exists() else None
                ),
                "source_job_id": source_report.get("job_id") if source_report else None,
                "source_project_commit": (
                    source_report.get("provenance", {}).get("project_commit")
                    if source_report
                    else None
                ),
                "head_type": config["head_type"],
                "normalization": config["normalization"],
                "survival_loss_weight": float(config["survival_loss_weight"]),
                "seed": int(config["seed"]),
                "selected_epoch": int(checkpoint["epoch"]),
                "data": str(data_path),
                "data_metadata_sha256": sha256_file(data_path / "metadata.json"),
                "target": str(target_path),
                "split_blocks": {
                    label: len(dataset_cache[(data_path, split)])
                    for label, split in split_names.items()
                },
                "reports": reports,
            }
        )
        del head

    payload = {
        "evidence_tier": "development_diagnostic_only",
        "formal_claim_allowed": False,
        "selection_allowed": False,
        "reason": (
            "The 12-prompt test split was already observed by the failed probes. "
            "This audit only distinguishes underfitting from memorization and must "
            "not be used for further hyperparameter selection."
        ),
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "hostname": platform.node(),
        "device": torch.cuda.get_device_name(device),
        "seconds": time.perf_counter() - start,
        "provenance": provenance,
        "records": records,
    }
    atomic_write_json(args.output, payload)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "records": len(records),
                "seconds": payload["seconds"],
                "provenance": provenance,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
