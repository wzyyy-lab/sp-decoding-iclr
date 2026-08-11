#!/usr/bin/env python3
"""Train the direct local/causal/global DFlash candidate selector."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
import os
import platform
from pathlib import Path
import random
import shutil
import subprocess
import time
from typing import Any, Iterable

from safetensors import safe_open
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from sph.candidate_ceiling import accepted_draft_prefix_lengths
from sph.data import CanonicalBlockDataset, collate_canonical_blocks
from sph.global_direct_selector import (
    GlobalDirectCandidateSelector,
    global_direct_candidate_loss,
    prefix_candidate_mask,
)


PROJECT = Path(__file__).resolve().parents[1]
PRIMARY_METHOD = "direct"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument(
        "--train-data",
        type=Path,
        action="append",
        default=[],
        help=(
            "Optional train-only canonical collection; repeat for sharded "
            "large-data collection. Validation still comes from --data."
        ),
    )
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--scope",
        choices=["local", "causal", "global"],
        default="global",
    )
    parser.add_argument(
        "--mixer",
        choices=["flat", "axial"],
        default="flat",
    )
    parser.add_argument(
        "--node-encoder",
        choices=["additive", "compatibility"],
        default="additive",
    )
    parser.add_argument("--candidate-k", type=int, default=16)
    parser.add_argument("--model-dim", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=6e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.04)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument(
        "--loss-weighting",
        choices=[
            "uniform",
            "exponential",
            "dpace",
            "candidate_dpace",
            "reachable_dpace",
            "accepted_reach",
        ],
        default="dpace",
    )
    parser.add_argument("--dpace-alpha", type=float, default=0.5)
    parser.add_argument(
        "--post-break-weight",
        type=float,
        default=1.0,
        help=(
            "For reachable_dpace only, retain this fraction of the matched "
            "Candidate-D-PACE loss after the current greedy breaker."
        ),
    )
    parser.add_argument("--exponential-gamma", type=float, default=7.0)
    parser.add_argument("--base-safety-weight", type=float, default=0.0)
    parser.add_argument("--base-safety-margin", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--max-train-prompts",
        type=int,
        default=0,
        help="Use a deterministic nested prompt subset; zero uses all prompts.",
    )
    parser.add_argument("--train-subset-seed", type=int, default=20260730)
    parser.add_argument("--train-split", default="train")
    parser.add_argument(
        "--validation-split", default="validation_select"
    )
    parser.add_argument("--gate-split", default="validation_gate")
    parser.add_argument(
        "--skip-gate",
        action="store_true",
        help="Do not evaluate the sealed prompt-disjoint development gate.",
    )
    parser.add_argument(
        "--memorization-blocks",
        type=int,
        default=0,
        help="Use one deterministic subset for both train and validation.",
    )
    parser.add_argument(
        "--memorization-opportunity-fraction",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--require-capacity-gate", action="store_true"
    )
    parser.add_argument("--min-candidate-accuracy", type=float, default=0.99)
    parser.add_argument(
        "--min-hard-candidate-accuracy", type=float, default=0.97
    )
    parser.add_argument(
        "--min-first-miss-repair-rate", type=float, default=0.95
    )
    parser.add_argument(
        "--min-oracle-gap-recovered", type=float, default=0.95
    )
    parser.add_argument(
        "--max-harmed-fraction", type=float, default=0.01
    )
    parser.add_argument(
        "--evidence-tier",
        choices=["smoke", "capacity_probe", "development"],
        default="development",
    )
    parser.add_argument("--calibrate-margin", action="store_true")
    parser.add_argument(
        "--max-calibration-first-token-drop",
        type=float,
        default=0.001,
    )
    parser.add_argument(
        "--max-calibration-domain-drop",
        type=float,
        default=0.05,
    )
    return parser.parse_args()


class RecordDataset(Dataset[dict[str, Any]]):
    """A cheap view over already integrity-checked canonical records."""

    def __init__(
        self,
        records: Iterable[dict[str, Any]],
        metadata: dict[str, Any],
    ) -> None:
        self.records = list(records)
        self.metadata = metadata
        if not self.records:
            raise ValueError("record dataset cannot be empty")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.records[index]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def serializable_config(args: argparse.Namespace) -> dict[str, Any]:
    """Return a JSON-safe, immutable snapshot of the CLI configuration."""

    def convert(value: Any) -> Any:
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, list):
            return [convert(item) for item in value]
        return value

    return {
        key: convert(value) for key, value in vars(args).items()
    }


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


def load_target_embedding(target: Path) -> Tensor:
    index_path = target / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    key = "model.embed_tokens.weight"
    shard_name = index["weight_map"][key]
    with safe_open(
        target / shard_name, framework="pt", device="cpu"
    ) as handle:
        return handle.get_tensor(key)


def validate_target_embedding_identity(
    data_metadata: dict[str, Any],
    target: Path,
) -> list[dict[str, Any]]:
    """Bind frozen token semantics to the collection target checkpoint."""

    if int(data_metadata.get("format_version", 1)) < 2:
        return []
    expected_records = data_metadata.get("provenance", {}).get(
        "target_files"
    )
    if not isinstance(expected_records, list):
        raise RuntimeError("protocol-v2 data lacks target fingerprints")
    expected = {
        str(record["path"]): record for record in expected_records
    }
    index_path = target / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    embedding_shard = str(
        index["weight_map"]["model.embed_tokens.weight"]
    )
    verified = []
    for name in ["config.json", index_path.name, embedding_shard]:
        if name not in expected:
            raise RuntimeError(
                f"collection fingerprint is missing target file {name}"
            )
        path = target / name
        actual = {
            "path": name,
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        reference = expected[name]
        if actual["bytes"] != int(reference["bytes"]):
            raise RuntimeError(
                f"training target size differs from collection: {name}"
            )
        if actual["sha256"] != str(reference["sha256"]):
            raise RuntimeError(
                f"training target hash differs from collection: {name}"
            )
        verified.append(actual)
    return verified


def _fingerprint_map(records: Any, label: str) -> dict[str, tuple[int, str]]:
    if not isinstance(records, list):
        raise RuntimeError(f"protocol-v2 data lacks {label} fingerprints")
    return {
        str(record["path"]): (
            int(record["bytes"]),
            str(record["sha256"]),
        )
        for record in records
    }


def assert_canonical_collection_compatible(
    reference: dict[str, Any],
    candidate: dict[str, Any],
    *,
    path: Path,
) -> None:
    """Reject external train shards collected with different model semantics."""

    for key in (
        "format_version",
        "block_size",
        "draft_positions",
        "attention_implementation",
        "dtype",
        "target_layer_ids",
    ):
        if candidate.get(key) != reference.get(key):
            raise RuntimeError(
                f"external canonical metadata differs for {key}: {path}"
            )
    for label in ("target_files", "draft_files"):
        reference_files = _fingerprint_map(
            reference.get("provenance", {}).get(label), label
        )
        candidate_files = _fingerprint_map(
            candidate.get("provenance", {}).get(label), label
        )
        if candidate_files != reference_files:
            raise RuntimeError(
                f"external canonical {label} differ: {path}"
            )


def assert_prompt_disjoint_splits(
    named_datasets: dict[str, RecordDataset | None],
) -> None:
    prompt_sets = {
        name: {
            str(record["sample_id"]) for record in dataset.records
        }
        for name, dataset in named_datasets.items()
        if dataset is not None
    }
    names = list(prompt_sets)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            overlap = prompt_sets[left] & prompt_sets[right]
            if overlap:
                raise RuntimeError(
                    f"prompt leakage between {left} and {right}: "
                    f"{sorted(overlap)[:3]}"
                )


def make_loader(
    dataset: RecordDataset,
    *,
    candidate_k: int,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=True,
        collate_fn=lambda records: collate_canonical_blocks(
            records, candidate_k
        ),
    )


def to_device(
    batch: dict[str, Any], device: torch.device
) -> dict[str, Any]:
    return {
        key: (
            value.to(device, non_blocking=True)
            if isinstance(value, Tensor)
            else value
        )
        for key, value in batch.items()
    }


def deterministic_capacity_subset(
    records: list[dict[str, Any]],
    *,
    count: int,
    seed: int,
    opportunity_fraction: float,
    candidate_k: int,
) -> list[dict[str, Any]]:
    """Repeatable mixture of natural and reachable-repair blocks."""

    if not 0.0 <= opportunity_fraction <= 1.0:
        raise ValueError("opportunity fraction must be in [0, 1]")
    if count > len(records):
        raise ValueError(
            f"requested {count} blocks from only {len(records)}"
        )
    generator = random.Random(seed)
    opportunities = []
    ordinary = []
    for record in records:
        topk = record["base_topk_ids"][:, :candidate_k].long()
        gold = record["gold_ids"].long()
        top1_correct = topk[:, 0] == gold
        if bool((~top1_correct).any()):
            first_miss = int(
                (~top1_correct).to(torch.int64).argmax().item()
            )
            repairable = bool(
                (topk[first_miss] == gold[first_miss]).any()
            )
        else:
            repairable = False
        (opportunities if repairable else ordinary).append(record)
    generator.shuffle(opportunities)
    generator.shuffle(ordinary)
    desired_opportunities = min(
        len(opportunities), round(count * opportunity_fraction)
    )
    desired_ordinary = min(
        len(ordinary), count - desired_opportunities
    )
    selected = (
        opportunities[:desired_opportunities]
        + ordinary[:desired_ordinary]
    )
    if len(selected) < count:
        selected_ids = {id(record) for record in selected}
        remainder = [
            record
            for record in records
            if id(record) not in selected_ids
        ]
        generator.shuffle(remainder)
        selected.extend(remainder[: count - len(selected)])
    generator.shuffle(selected)
    return selected


def deterministic_prompt_subset(
    records: list[dict[str, Any]],
    *,
    max_prompts: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Select a hash-ranked, nested prompt subset in original record order."""

    if max_prompts < 1:
        raise ValueError("max_prompts must be positive")
    prompt_domains: dict[str, str] = {}
    for record in records:
        sample_id = str(record["sample_id"])
        domain = str(record["domain"])
        previous = prompt_domains.setdefault(sample_id, domain)
        if previous != domain:
            raise RuntimeError(
                f"prompt {sample_id!r} occurs in multiple domains"
            )
    if max_prompts > len(prompt_domains):
        raise ValueError(
            f"requested {max_prompts} train prompts from only "
            f"{len(prompt_domains)}"
        )

    def rank(sample_id: str) -> bytes:
        return hashlib.sha256(
            f"{seed}\0{sample_id}".encode("utf-8")
        ).digest()

    selected_ids = set(
        sorted(prompt_domains, key=rank)[:max_prompts]
    )
    selected = [
        record
        for record in records
        if str(record["sample_id"]) in selected_ids
    ]
    if len({str(record["sample_id"]) for record in selected}) != max_prompts:
        raise AssertionError("prompt subset cardinality invariant failed")
    return selected


def realized_prefix(
    path: Tensor,
    candidate_ids: Tensor,
    gold_ids: Tensor,
) -> Tensor:
    selected_ids = candidate_ids.gather(
        -1, path.unsqueeze(-1)
    ).squeeze(-1)
    return accepted_draft_prefix_lengths(selected_ids == gold_ids)


def _prompt_balanced_mean(
    records: list[dict[str, Any]], key: str
) -> float:
    grouped: dict[str, list[float]] = defaultdict(list)
    for record in records:
        grouped[str(record["sample_id"])].append(float(record[key]))
    return sum(
        sum(values) / len(values) for values in grouped.values()
    ) / len(grouped)


def _method_summary(
    records: list[dict[str, Any]], method: str
) -> dict[str, float]:
    accepted = [
        float(record["accepted_draft_tokens"][method])
        for record in records
    ]
    first = [
        float(record["first_token_correct"][method])
        for record in records
    ]
    prompt_records = [
        {
            "sample_id": record["sample_id"],
            "value": record["accepted_draft_tokens"][method],
        }
        for record in records
    ]
    prompt_mean = _prompt_balanced_mean(prompt_records, "value")
    return {
        "mean_accepted_draft_tokens": sum(accepted) / len(accepted),
        "mean_verification_advance": sum(accepted) / len(accepted)
        + 1.0,
        "mean_accepted_draft_tokens_prompt_balanced": prompt_mean,
        "mean_verification_advance_prompt_balanced": prompt_mean + 1.0,
        "first_token_accuracy": sum(first) / len(first),
    }


def _rank_bucket(rank: int, candidate_k: int) -> str:
    if rank == 1:
        return "rank1"
    if rank == 2:
        return "rank2"
    if rank <= 4:
        return "rank3_4"
    if rank <= 8:
        return "rank5_8"
    if rank <= candidate_k:
        return f"rank9_{candidate_k}"
    return "outside"


def _position_metrics(
    correct: Tensor,
    active: Tensor,
    gold_indices: Tensor,
) -> list[dict[str, Any]]:
    metrics = []
    for position in range(correct.shape[1]):
        position_active = active[:, position]
        position_hard = position_active & (
            gold_indices[:, position] > 0
        )
        total = int(position_active.sum())
        hard_total = int(position_hard.sum())
        metrics.append(
            {
                "position": position,
                "accuracy": (
                    float(
                        (
                            correct[:, position] & position_active
                        ).sum()
                    )
                    / total
                    if total
                    else None
                ),
                "positions": total,
                "non_top1_accuracy": (
                    float(
                        (
                            correct[:, position] & position_hard
                        ).sum()
                    )
                    / hard_total
                    if hard_total
                    else None
                ),
                "non_top1_positions": hard_total,
            }
        )
    return metrics


@torch.inference_mode()
def evaluate(
    model: GlobalDirectCandidateSelector,
    loader: DataLoader,
    target_embedding: Tensor,
    device: torch.device,
    *,
    candidate_k: int,
    loss_weighting: str,
    dpace_alpha: float,
    exponential_gamma: float,
    post_break_weight: float = 1.0,
    base_safety_weight: float = 0.0,
    base_safety_margin: float = 0.1,
    include_examples: bool = False,
    require_base_identity: bool = False,
) -> dict[str, Any]:
    model.eval()
    example_records: list[dict[str, Any]] = []
    loss_sum = 0.0
    nll_numerator = 0.0
    active_correct = 0
    active_total = 0
    hard_correct = 0
    hard_total = 0
    residual_abs_sum = 0.0
    residual_count = 0
    residual_abs_max = 0.0
    chosen_rank_counts: dict[str, int] = defaultdict(int)
    position_correct_parts: list[Tensor] = []
    position_active_parts: list[Tensor] = []
    position_gold_parts: list[Tensor] = []
    component_sums: dict[str, float] = defaultdict(float)

    for cpu_batch in loader:
        batch = to_device(cpu_batch, device)
        candidate_embeddings = target_embedding[
            batch["candidate_ids"]
        ]
        anchor_embeddings = target_embedding[batch["anchor_ids"]]
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            output = model(
                batch["hidden"],
                candidate_embeddings,
                batch["candidate_logits"],
                batch["base_logsumexp"],
                anchor_embeddings,
            )
            loss_output = global_direct_candidate_loss(
                output,
                batch["gold_candidate_indices"],
                batch["gold_in_lattice"],
                weighting=loss_weighting,
                dpace_alpha=dpace_alpha,
                exponential_gamma=exponential_gamma,
                post_break_weight=post_break_weight,
                base_safety_weight=base_safety_weight,
                base_safety_margin=base_safety_margin,
            )

        if require_base_identity:
            expected_base_scores = (
                batch["candidate_logits"].float()
                - batch["base_logsumexp"].float().unsqueeze(-1)
            )
            if not torch.equal(output.scores, expected_base_scores):
                max_error = float(
                    (output.scores - expected_base_scores).abs().max()
                )
                raise RuntimeError(
                    "epoch-zero selector scores do not exactly reproduce "
                    f"the canonical DFlash lattice (max error {max_error})"
                )
            if not torch.equal(
                output.scores.argmax(dim=-1),
                torch.zeros_like(output.scores[..., 0], dtype=torch.long),
            ):
                raise RuntimeError(
                    "epoch-zero selector path does not exactly reproduce "
                    "the canonical DFlash rank-zero path"
                )

        batch_size, length, _ = batch["candidate_ids"].shape
        active = loss_output.active_positions
        direct_path = output.scores.argmax(dim=-1)
        base_path = torch.zeros_like(direct_path)
        candidate_correct = (
            direct_path == batch["gold_candidate_indices"]
        )
        hard = active & (batch["gold_candidate_indices"] > 0)
        current_active = int(active.sum())

        loss_sum += float(loss_output.loss) * batch_size
        for name, value in loss_output.components.items():
            component_sums[name] += float(value) * batch_size
        nll_numerator += (
            float(loss_output.unweighted_nll) * current_active
        )
        active_correct += int((candidate_correct & active).sum())
        active_total += current_active
        hard_correct += int((candidate_correct & hard).sum())
        hard_total += int(hard.sum())
        position_correct_parts.append(candidate_correct.cpu())
        position_active_parts.append(active.cpu())
        position_gold_parts.append(
            batch["gold_candidate_indices"].cpu()
        )

        residual_abs = output.residual_scores.abs()
        residual_abs_sum += float(residual_abs.sum())
        residual_count += residual_abs.numel()
        residual_abs_max = max(
            residual_abs_max, float(residual_abs.max())
        )
        for selected_rank in direct_path.reshape(-1).tolist():
            chosen_rank_counts[str(int(selected_rank) + 1)] += 1

        paths = {"base": base_path, "direct": direct_path}
        realized = {
            name: realized_prefix(
                path, batch["candidate_ids"], batch["gold_ids"]
            )
            for name, path in paths.items()
        }
        oracle = accepted_draft_prefix_lengths(
            batch["gold_in_lattice"]
        )
        first_correct = {
            name: (
                batch["candidate_ids"][:, 0]
                .gather(-1, path[:, :1])
                .squeeze(1)
                == batch["gold_ids"][:, 0]
            )
            for name, path in paths.items()
        }

        for item, (sample_id, domain) in enumerate(
            zip(
                batch["sample_ids"],
                batch["domains"],
                strict=True,
            )
        ):
            base_prefix = int(realized["base"][item])
            has_failure = base_prefix < length
            first_miss_rank = (
                int(
                    batch["gold_candidate_indices"][
                        item, base_prefix
                    ]
                )
                + 1
                if has_failure
                and bool(
                    batch["gold_in_lattice"][item, base_prefix]
                )
                else candidate_k + 1
            )
            record: dict[str, Any] = {
                "sample_id": sample_id,
                "domain": domain,
                "accepted_draft_tokens": {
                    name: int(values[item])
                    for name, values in realized.items()
                },
                "first_token_correct": {
                    name: bool(values[item])
                    for name, values in first_correct.items()
                },
                "oracle_accepted_draft_tokens": int(oracle[item]),
                "base_first_miss_position": (
                    base_prefix if has_failure else None
                ),
                "base_first_miss_gold_rank": first_miss_rank,
            }
            if include_examples:
                base_position_correct = (
                    batch["candidate_ids"][item, :, 0]
                    == batch["gold_ids"][item]
                )
                direct_position_correct = (
                    batch["candidate_ids"][item]
                    .gather(
                        -1,
                        direct_path[item].unsqueeze(-1),
                    )
                    .squeeze(-1)
                    == batch["gold_ids"][item]
                )
                direct_margin_over_base = (
                    output.scores[item]
                    .gather(
                        -1,
                        direct_path[item].unsqueeze(-1),
                    )
                    .squeeze(-1)
                    - output.scores[item, :, 0]
                )
                record["candidate_path_indices"] = {
                    name: path[item].detach().cpu().tolist()
                    for name, path in paths.items()
                }
                record["direct_residual_scores"] = (
                    output.residual_scores[item]
                    .detach()
                    .cpu()
                    .tolist()
                )
                record["base_position_correct"] = (
                    base_position_correct.detach().cpu().tolist()
                )
                record["direct_position_correct"] = (
                    direct_position_correct.detach().cpu().tolist()
                )
                record["direct_margin_over_base"] = (
                    direct_margin_over_base.detach().cpu().tolist()
                )
            example_records.append(record)

    candidate_correct_all = torch.cat(position_correct_parts, dim=0)
    active_all = torch.cat(position_active_parts, dim=0)
    gold_all = torch.cat(position_gold_parts, dim=0)
    report: dict[str, Any] = {
        "loss": {
            "objective": loss_sum / len(example_records),
            "unweighted_nll": nll_numerator / max(1, active_total),
            "components": {
                name: value / len(example_records)
                for name, value in sorted(component_sums.items())
            },
        },
        "candidate_classification": {
            "accuracy": (
                active_correct / active_total if active_total else None
            ),
            "correct": active_correct,
            "positions": active_total,
            "non_top1_accuracy": (
                hard_correct / hard_total if hard_total else None
            ),
            "non_top1_correct": hard_correct,
            "non_top1_positions": hard_total,
            "by_position": _position_metrics(
                candidate_correct_all, active_all, gold_all
            ),
        },
        "residual": {
            "mean_absolute": residual_abs_sum / residual_count,
            "max_absolute": residual_abs_max,
        },
        "chosen_candidate_rank_counts": dict(
            sorted(
                chosen_rank_counts.items(),
                key=lambda item: int(item[0]),
            )
        ),
        "blocks": len(example_records),
        "prompts": len(
            {
                str(record["sample_id"])
                for record in example_records
            }
        ),
    }
    for method in ("base", "direct"):
        report[method] = _method_summary(example_records, method)
    oracle_records = [
        {
            "sample_id": record["sample_id"],
            "accepted_draft_tokens": {
                "oracle": record["oracle_accepted_draft_tokens"]
            },
            "first_token_correct": {
                "oracle": record["oracle_accepted_draft_tokens"] > 0
            },
        }
        for record in example_records
    ]
    report["oracle"] = _method_summary(oracle_records, "oracle")

    report["by_domain"] = {}
    for domain in sorted(
        {str(record["domain"]) for record in example_records}
    ):
        subset = [
            record
            for record in example_records
            if record["domain"] == domain
        ]
        report["by_domain"][domain] = {
            method: _method_summary(subset, method)
            for method in ("base", "direct")
        }

    changed = 0
    reachable_changes = 0
    suffix_only_changes = 0
    improved = 0
    harmed = 0
    repair_opportunities = 0
    repairs = 0
    repair_ranks: dict[str, int] = defaultdict(int)
    for record in example_records:
        base_prefix = record["accepted_draft_tokens"]["base"]
        direct_prefix = record["accepted_draft_tokens"]["direct"]
        rank = record["base_first_miss_gold_rank"]
        if rank <= candidate_k:
            repair_opportunities += 1
            if direct_prefix > base_prefix:
                repairs += 1
                repair_ranks[_rank_bucket(rank, candidate_k)] += 1
        if direct_prefix > base_prefix:
            improved += 1
        elif direct_prefix < base_prefix:
            harmed += 1
        path = record.get("candidate_path_indices", {}).get("direct")
        if path is not None:
            different = [
                index
                for index, candidate in enumerate(path)
                if candidate
            ]
            if different:
                changed += 1
                if different[0] <= base_prefix:
                    reachable_changes += 1
                else:
                    suffix_only_changes += 1
    base_eal = report["base"][
        "mean_accepted_draft_tokens_prompt_balanced"
    ]
    direct_eal = report["direct"][
        "mean_accepted_draft_tokens_prompt_balanced"
    ]
    oracle_eal = report["oracle"][
        "mean_accepted_draft_tokens_prompt_balanced"
    ]
    denominator = oracle_eal - base_eal
    report["direct_diagnostics"] = {
        "path_changed_blocks": (
            changed if include_examples else None
        ),
        "reachable_changed_blocks": (
            reachable_changes if include_examples else None
        ),
        "suffix_only_changed_blocks": (
            suffix_only_changes if include_examples else None
        ),
        "improved_blocks": improved,
        "harmed_blocks": harmed,
        "harmed_fraction": harmed / len(example_records),
        "first_miss_repair_opportunities": repair_opportunities,
        "first_miss_repairs": repairs,
        "first_miss_repair_rate_given_k": (
            repairs / repair_opportunities
            if repair_opportunities
            else None
        ),
        "successful_repair_gold_rank": dict(repair_ranks),
        "oracle_gap_recovered": (
            (direct_eal - base_eal) / denominator
            if denominator > 0
            else None
        ),
    }
    if include_examples:
        report["examples"] = example_records
    return report


def summarize_margin_calibration(
    evaluation: dict[str, Any],
    *,
    threshold: float,
) -> dict[str, Any]:
    """Apply a validation-selected per-position alternative margin."""

    examples = evaluation.get("examples")
    if not examples:
        raise ValueError("margin calibration requires evaluation examples")
    paths = torch.tensor(
        [
            record["candidate_path_indices"]["direct"]
            for record in examples
        ],
        dtype=torch.long,
    )
    margins = torch.tensor(
        [record["direct_margin_over_base"] for record in examples],
        dtype=torch.float32,
    )
    base_correct = torch.tensor(
        [record["base_position_correct"] for record in examples],
        dtype=torch.bool,
    )
    direct_correct = torch.tensor(
        [record["direct_position_correct"] for record in examples],
        dtype=torch.bool,
    )
    use_alternative = paths.ne(0) & margins.ge(threshold)
    calibrated_correct = torch.where(
        use_alternative, direct_correct, base_correct
    )
    accepted = accepted_draft_prefix_lengths(calibrated_correct)
    calibrated_records = []
    for index, record in enumerate(examples):
        calibrated_records.append(
            {
                "sample_id": record["sample_id"],
                "domain": record["domain"],
                "accepted_draft_tokens": {
                    "calibrated": int(accepted[index])
                },
                "first_token_correct": {
                    "calibrated": bool(
                        calibrated_correct[index, 0]
                    )
                },
            }
        )
    summary: dict[str, Any] = _method_summary(
        calibrated_records, "calibrated"
    )
    summary["by_domain"] = {}
    for domain in sorted(
        {str(record["domain"]) for record in calibrated_records}
    ):
        subset = [
            record
            for record in calibrated_records
            if str(record["domain"]) == domain
        ]
        summary["by_domain"][domain] = _method_summary(
            subset, "calibrated"
        )
    base_accepted = torch.tensor(
        [
            record["accepted_draft_tokens"]["base"]
            for record in examples
        ]
    )
    summary["diagnostics"] = {
        "improved_blocks": int((accepted > base_accepted).sum()),
        "harmed_blocks": int((accepted < base_accepted).sum()),
        "harmed_fraction": float(
            (accepted < base_accepted).float().mean()
        ),
        "alternative_positions_used": int(use_alternative.sum()),
        "alternative_blocks_used": int(
            use_alternative.any(dim=-1).sum()
        ),
    }
    summary["threshold"] = (
        threshold if math.isfinite(threshold) else "base_only"
    )
    return summary


def tune_margin_threshold(
    evaluation: dict[str, Any],
    *,
    max_first_token_drop: float,
    max_domain_drop: float,
) -> tuple[float, dict[str, Any]]:
    """Select a conservative threshold using validation data only."""

    if max_first_token_drop < 0 or max_domain_drop < 0:
        raise ValueError("calibration safety tolerances cannot be negative")
    examples = evaluation.get("examples")
    if not examples:
        raise ValueError("margin calibration requires evaluation examples")
    paths = torch.tensor(
        [
            record["candidate_path_indices"]["direct"]
            for record in examples
        ],
        dtype=torch.long,
    )
    margins = torch.tensor(
        [record["direct_margin_over_base"] for record in examples],
        dtype=torch.float32,
    )
    base_correct = torch.tensor(
        [record["base_position_correct"] for record in examples],
        dtype=torch.bool,
    )
    direct_correct = torch.tensor(
        [record["direct_position_correct"] for record in examples],
        dtype=torch.bool,
    )
    sample_ids = [str(record["sample_id"]) for record in examples]
    prompt_counts: dict[str, int] = defaultdict(int)
    for sample_id in sample_ids:
        prompt_counts[sample_id] += 1
    prompt_weight = torch.tensor(
        [
            1.0 / (len(prompt_counts) * prompt_counts[sample_id])
            for sample_id in sample_ids
        ],
        dtype=torch.float64,
    )
    domains = [str(record["domain"]) for record in examples]
    domain_masks = {
        domain: torch.tensor(
            [value == domain for value in domains],
            dtype=torch.bool,
        )
        for domain in sorted(set(domains))
    }
    base_first = evaluation["base"]["first_token_accuracy"]
    base_domain_eal = {
        domain: metrics["base"]["mean_accepted_draft_tokens"]
        for domain, metrics in evaluation["by_domain"].items()
    }

    changed_margins = margins[paths.ne(0)]
    regimes = [0.0]
    unique_margins = torch.unique(changed_margins).sort().values
    next_margins = torch.nextafter(
        unique_margins,
        torch.full_like(unique_margins, math.inf),
    )
    regimes.extend(float(value) for value in next_margins)
    regimes.append(math.inf)
    feasible: list[tuple[tuple[float, ...], float]] = []
    for threshold in regimes:
        use_alternative = paths.ne(0) & margins.ge(threshold)
        correct = torch.where(
            use_alternative, direct_correct, base_correct
        )
        accepted = accepted_draft_prefix_lengths(correct).double()
        prompt_eal = float((accepted * prompt_weight).sum())
        first_accuracy = float(correct[:, 0].float().mean())
        domain_deltas = [
            float(accepted[mask].mean())
            - base_domain_eal[domain]
            for domain, mask in domain_masks.items()
        ]
        if (
            first_accuracy
            >= base_first - max_first_token_drop
            and min(domain_deltas) >= -max_domain_drop
        ):
            key = (
                prompt_eal,
                min(domain_deltas),
                first_accuracy,
                threshold,
            )
            feasible.append((key, threshold))
    if not feasible:
        raise RuntimeError("no feasible margin-calibration threshold")
    _, threshold = max(feasible, key=lambda item: item[0])
    return threshold, summarize_margin_calibration(
        evaluation, threshold=threshold
    )


def selection_key(evaluation: dict[str, Any]) -> tuple[float, ...]:
    """Validation-only checkpoint ordering with accepted length first."""

    minimum_domain_delta = min(
        metrics["direct"]["mean_accepted_draft_tokens"]
        - metrics["base"]["mean_accepted_draft_tokens"]
        for metrics in evaluation["by_domain"].values()
    )
    candidate_accuracy = evaluation["candidate_classification"]["accuracy"]
    return (
        evaluation["direct"][
            "mean_accepted_draft_tokens_prompt_balanced"
        ],
        evaluation["direct"]["first_token_accuracy"],
        minimum_domain_delta,
        (
            float(candidate_accuracy)
            if candidate_accuracy is not None
            else -math.inf
        ),
    )


def capacity_gate_report(
    evaluation: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    classification = evaluation["candidate_classification"]
    diagnostics = evaluation["direct_diagnostics"]
    values = {
        "candidate_accuracy": classification["accuracy"],
        "hard_candidate_accuracy": classification[
            "non_top1_accuracy"
        ],
        "first_miss_repair_rate": diagnostics[
            "first_miss_repair_rate_given_k"
        ],
        "oracle_gap_recovered": diagnostics[
            "oracle_gap_recovered"
        ],
        "harmed_fraction": diagnostics["harmed_fraction"],
    }
    thresholds = {
        "candidate_accuracy": args.min_candidate_accuracy,
        "hard_candidate_accuracy": args.min_hard_candidate_accuracy,
        "first_miss_repair_rate": args.min_first_miss_repair_rate,
        "oracle_gap_recovered": args.min_oracle_gap_recovered,
        "harmed_fraction": args.max_harmed_fraction,
    }
    checks = {
        "candidate_accuracy": (
            values["candidate_accuracy"] is not None
            and values["candidate_accuracy"]
            >= thresholds["candidate_accuracy"]
        ),
        "hard_candidate_accuracy": (
            values["hard_candidate_accuracy"] is not None
            and values["hard_candidate_accuracy"]
            >= thresholds["hard_candidate_accuracy"]
        ),
        "first_miss_repair_rate": (
            values["first_miss_repair_rate"] is not None
            and values["first_miss_repair_rate"]
            >= thresholds["first_miss_repair_rate"]
        ),
        "oracle_gap_recovered": (
            values["oracle_gap_recovered"] is not None
            and values["oracle_gap_recovered"]
            >= thresholds["oracle_gap_recovered"]
        ),
        "harmed_fraction": (
            values["harmed_fraction"]
            <= thresholds["harmed_fraction"]
        ),
    }
    return {
        "passed": all(checks.values()),
        "values": values,
        "thresholds": thresholds,
        "checks": checks,
    }


def checkpoint_selection_key(
    evaluation: dict[str, Any], args: argparse.Namespace
) -> tuple[float, ...]:
    """Retain a capacity-passing epoch whenever one has been observed.

    Development runs keep the normal EAL-first ordering.  Same-subset probes
    prepend their complete declared gate so a later higher-EAL but nonpassing
    epoch cannot overwrite an earlier witness that satisfied every capacity
    criterion.
    """

    base_key = selection_key(evaluation)
    if not args.memorization_blocks:
        return base_key
    passed = capacity_gate_report(evaluation, args)["passed"]
    return (float(passed), *base_key)


def cosine_warmup_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    total_steps: int,
    warmup_ratio: float,
) -> tuple[torch.optim.lr_scheduler.LambdaLR, int]:
    if total_steps < 1:
        raise ValueError("total_steps must be positive")
    if not 0.0 <= warmup_ratio < 1.0:
        raise ValueError("warmup_ratio must be in [0, 1)")
    warmup_steps = round(total_steps * warmup_ratio)

    def multiplier(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        decay_steps = max(1, total_steps - warmup_steps)
        progress = min(
            1.0,
            max(0.0, (step - warmup_steps) / decay_steps),
        )
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return (
        torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=multiplier
        ),
        warmup_steps,
    )


def compact_epoch_metrics(
    evaluation: dict[str, Any],
) -> dict[str, Any]:
    return {
        "candidate_accuracy": evaluation[
            "candidate_classification"
        ]["accuracy"],
        "hard_candidate_accuracy": evaluation[
            "candidate_classification"
        ]["non_top1_accuracy"],
        "base_eal": evaluation["base"][
            "mean_accepted_draft_tokens_prompt_balanced"
        ],
        "direct_eal": evaluation["direct"][
            "mean_accepted_draft_tokens_prompt_balanced"
        ],
        "oracle_eal": evaluation["oracle"][
            "mean_accepted_draft_tokens_prompt_balanced"
        ],
        "first_token_accuracy": evaluation["direct"][
            "first_token_accuracy"
        ],
        "repair_rate": evaluation["direct_diagnostics"][
            "first_miss_repair_rate_given_k"
        ],
        "oracle_gap_recovered": evaluation[
            "direct_diagnostics"
        ]["oracle_gap_recovered"],
        "harmed_blocks": evaluation["direct_diagnostics"][
            "harmed_blocks"
        ],
        "mean_absolute_residual": evaluation["residual"][
            "mean_absolute"
        ],
        "soft_expected_accepted_tokens": evaluation["loss"].get(
            "components", {}
        ).get("soft_expected_accepted_tokens"),
        "reachable_fraction_of_coverage": evaluation["loss"].get(
            "components", {}
        ).get("reachable_fraction_of_coverage"),
        "post_break_positions_per_block": evaluation["loss"].get(
            "components", {}
        ).get("post_break_positions_per_block"),
        "post_break_suffix_loss": evaluation["loss"].get(
            "components", {}
        ).get("post_break_suffix_loss"),
    }


def main() -> None:
    args = parse_args()
    if args.candidate_k < 2:
        raise ValueError("--candidate-k must be at least 2")
    if args.model_dim < 1 or args.num_heads < 1 or args.num_layers < 1:
        raise ValueError("model dimensions and layer count must be positive")
    if args.model_dim % args.num_heads:
        raise ValueError("--model-dim must be divisible by --num-heads")
    if args.batch_size < 1 or args.epochs < 1:
        raise ValueError("--batch-size and --epochs must be positive")
    if args.learning_rate <= 0:
        raise ValueError("--learning-rate must be positive")
    if args.weight_decay < 0:
        raise ValueError("--weight-decay cannot be negative")
    if args.gradient_clip <= 0:
        raise ValueError("--gradient-clip must be positive")
    if args.base_safety_weight < 0:
        raise ValueError("--base-safety-weight cannot be negative")
    if args.base_safety_margin < 0:
        raise ValueError("--base-safety-margin cannot be negative")
    if args.max_train_prompts < 0:
        raise ValueError("--max-train-prompts cannot be negative")
    if args.max_train_prompts and args.memorization_blocks:
        raise ValueError(
            "--max-train-prompts cannot be combined with "
            "--memorization-blocks"
        )
    if args.train_data and args.memorization_blocks:
        raise ValueError(
            "--train-data cannot be combined with --memorization-blocks"
        )
    if (
        args.max_calibration_first_token_drop < 0
        or args.max_calibration_domain_drop < 0
    ):
        raise ValueError("calibration safety tolerances cannot be negative")
    if args.require_capacity_gate and not args.memorization_blocks:
        raise ValueError("capacity gate requires --memorization-blocks")
    if (
        args.memorization_blocks
        and args.evidence_tier not in {"smoke", "capacity_probe"}
    ):
        raise ValueError(
            "same-subset results must be labeled smoke/capacity_probe"
        )
    if not 0.0 <= args.dpace_alpha <= 1.0:
        raise ValueError("--dpace-alpha must be in [0, 1]")
    if not 0.0 <= args.post_break_weight <= 1.0:
        raise ValueError("--post-break-weight must be in [0, 1]")
    if (
        args.loss_weighting != "reachable_dpace"
        and args.post_break_weight != 1.0
    ):
        raise ValueError(
            "--post-break-weight differs from 1 only for reachable_dpace"
        )
    if args.loss_weighting == "reachable_dpace":
        if args.dpace_alpha != 0.5:
            raise ValueError("reachable_dpace freezes --dpace-alpha=0.5")
        if args.base_safety_weight != 0.0:
            raise ValueError(
                "reachable_dpace freezes --base-safety-weight=0"
            )
    if not torch.cuda.is_available():
        raise RuntimeError("direct selector training requires CUDA")

    seed_everything(args.seed)
    device = torch.device("cuda:0")
    args.output.mkdir(parents=True, exist_ok=True)
    config_snapshot = serializable_config(args)
    head_source_path = (
        PROJECT / "src" / "sph" / "global_direct_selector.py"
    )
    trainer_source_path = Path(__file__).resolve()
    source_hashes_at_start = {
        "trainer_sha256": sha256_file(trainer_source_path),
        "head_source_sha256": sha256_file(head_source_path),
    }
    source_snapshot_dir = args.output / "source_snapshot"
    source_snapshot_dir.mkdir(exist_ok=True)
    shutil.copy2(
        trainer_source_path,
        source_snapshot_dir / trainer_source_path.name,
    )
    shutil.copy2(
        head_source_path,
        source_snapshot_dir / head_source_path.name,
    )
    start = time.perf_counter()

    metadata_path = args.data / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    if args.candidate_k > int(metadata.get("top_k", 0)):
        raise ValueError(
            f"--candidate-k={args.candidate_k} exceeds stored top-K="
            f"{metadata.get('top_k')}"
        )
    verified_target_files = validate_target_embedding_identity(
        metadata, args.target
    )
    collection = CanonicalBlockDataset(args.data)
    by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in collection.records:
        by_split[str(record["split"])].append(record)
    if args.train_split not in by_split:
        raise ValueError(f"missing train split {args.train_split!r}")

    if args.memorization_blocks:
        capacity_records = deterministic_capacity_subset(
            by_split[args.train_split],
            count=args.memorization_blocks,
            seed=args.seed,
            opportunity_fraction=args.memorization_opportunity_fraction,
            candidate_k=args.candidate_k,
        )
        train_dataset = RecordDataset(capacity_records, metadata)
        validation_dataset = train_dataset
        gate_dataset = None
        split_protocol = "same_subset_capacity_probe"
    else:
        if args.validation_split not in by_split:
            raise ValueError(
                f"missing validation split {args.validation_split!r}"
            )
        external_train_collections = [
            CanonicalBlockDataset(path) for path in args.train_data
        ]
        for path, dataset in zip(
            args.train_data,
            external_train_collections,
            strict=True,
        ):
            assert_canonical_collection_compatible(
                metadata, dataset.metadata, path=path
            )
        verified_external_target_files = [
            {
                "data": str(path.resolve()),
                "target_fingerprint_matches_base_collection": True,
                "draft_fingerprint_matches_base_collection": True,
            }
            for path in args.train_data
        ]
        seen_external_prompts: set[str] = set()
        for path, dataset in zip(
            args.train_data,
            external_train_collections,
            strict=True,
        ):
            external_top_k = int(dataset.metadata.get("top_k", 0))
            if args.candidate_k > external_top_k:
                raise ValueError(
                    f"--candidate-k={args.candidate_k} exceeds stored "
                    f"top-K={external_top_k} in {path}"
                )
            current_prompts = {
                str(record["sample_id"])
                for record in dataset.records
                if str(record["split"]) == args.train_split
            }
            overlap = seen_external_prompts & current_prompts
            if overlap:
                raise RuntimeError(
                    f"prompt overlap between external train collections at "
                    f"{path}: {sorted(overlap)[:3]}"
                )
            seen_external_prompts.update(current_prompts)
        train_records = (
            [
                record
                for dataset in external_train_collections
                for record in dataset.records
                if str(record["split"]) == args.train_split
            ]
            if external_train_collections
            else by_split[args.train_split]
        )
        if external_train_collections and not train_records:
            raise ValueError(
                f"external collections contain no {args.train_split!r} records"
            )
        if args.max_train_prompts:
            train_records = deterministic_prompt_subset(
                train_records,
                max_prompts=args.max_train_prompts,
                seed=args.train_subset_seed,
            )
        train_dataset = RecordDataset(train_records, metadata)
        validation_dataset = RecordDataset(
            by_split[args.validation_split], metadata
        )
        if not args.skip_gate and args.gate_split not in by_split:
            raise ValueError(f"missing gate split {args.gate_split!r}")
        gate_dataset = (
            None
            if args.skip_gate
            else RecordDataset(by_split[args.gate_split], metadata)
        )
        assert_prompt_disjoint_splits(
            {
                "train": train_dataset,
                "validation": validation_dataset,
                "gate": gate_dataset,
            }
        )
        split_protocol = (
            "prompt_disjoint_external_train_development"
            if external_train_collections
            else "prompt_disjoint_development_train_subset"
            if args.max_train_prompts
            else "prompt_disjoint_development"
        )
    if args.memorization_blocks:
        external_train_collections = []
        verified_external_target_files = []

    train_loader = make_loader(
        train_dataset,
        candidate_k=args.candidate_k,
        batch_size=args.batch_size,
        shuffle=True,
    )
    validation_loader = make_loader(
        validation_dataset,
        candidate_k=args.candidate_k,
        batch_size=args.batch_size,
        shuffle=False,
    )
    gate_loader = (
        None
        if gate_dataset is None
        else make_loader(
            gate_dataset,
            candidate_k=args.candidate_k,
            batch_size=args.batch_size,
            shuffle=False,
        )
    )

    target_embedding = (
        load_target_embedding(args.target)
        .to(device=device, dtype=torch.bfloat16)
        .detach()
    )
    target_embedding.requires_grad_(False)
    hidden_size = int(target_embedding.shape[1])
    block_length = int(
        train_dataset.records[0]["gold_ids"].numel()
    )
    expected_block_length = int(
        metadata.get("draft_positions", block_length)
    )
    if block_length != expected_block_length:
        raise RuntimeError(
            f"data block length {block_length} differs from metadata "
            f"{expected_block_length}"
        )
    if any(
        int(record["gold_ids"].numel()) != block_length
        for record in (
            train_dataset.records
            + (
                []
                if validation_dataset is train_dataset
                else validation_dataset.records
            )
            + (
                []
                if gate_dataset is None
                else gate_dataset.records
            )
        )
    ):
        raise RuntimeError("records contain inconsistent draft block lengths")
    model = GlobalDirectCandidateSelector(
        hidden_size=hidden_size,
        max_positions=block_length,
        max_candidates=args.candidate_k,
        model_dim=args.model_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        scope=args.scope,
        mixer=args.mixer,
        node_encoder=args.node_encoder,
        dropout=args.dropout,
        initialization_seed=args.seed,
    ).to(device)
    parameter_count = sum(
        parameter.numel() for parameter in model.parameters()
    )
    named_parameters = list(model.named_parameters())
    decay_parameters = [
        parameter
        for _, parameter in named_parameters
        if parameter.ndim >= 2
    ]
    no_decay_parameters = [
        parameter
        for _, parameter in named_parameters
        if parameter.ndim < 2
    ]
    optimizer = torch.optim.AdamW(
        [
            {
                "params": decay_parameters,
                "weight_decay": args.weight_decay,
            },
            {
                "params": no_decay_parameters,
                "weight_decay": 0.0,
            },
        ],
        lr=args.learning_rate,
    )
    total_steps = args.epochs * len(train_loader)
    scheduler, warmup_steps = cosine_warmup_scheduler(
        optimizer,
        total_steps=total_steps,
        warmup_ratio=args.warmup_ratio,
    )

    evaluation_kwargs = {
        "candidate_k": args.candidate_k,
        "loss_weighting": args.loss_weighting,
        "dpace_alpha": args.dpace_alpha,
        "exponential_gamma": args.exponential_gamma,
        "post_break_weight": args.post_break_weight,
        "base_safety_weight": args.base_safety_weight,
        "base_safety_margin": args.base_safety_margin,
    }
    initial_validation = evaluate(
        model,
        validation_loader,
        target_embedding,
        device,
        include_examples=False,
        require_base_identity=True,
        **evaluation_kwargs,
    )
    history: list[dict[str, Any]] = [
        {
            "epoch": 0,
            "train": None,
            "validation": compact_epoch_metrics(
                initial_validation
            ),
        }
    ]
    best_key = checkpoint_selection_key(initial_validation, args)
    best_path = args.output / "best.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "args": config_snapshot,
            "epoch": 0,
            "selection_key": best_key,
            "parameter_count": parameter_count,
        },
        best_path,
    )
    print(json.dumps(history[0], ensure_ascii=False), flush=True)

    global_step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        objective_sum = 0.0
        nll_numerator = 0.0
        active_total = 0
        weight_sum = 0.0
        grad_norm_sum = 0.0
        component_sums: dict[str, float] = defaultdict(float)
        examples_seen = 0
        for cpu_batch in train_loader:
            batch = to_device(cpu_batch, device)
            batch_size = int(batch["hidden"].shape[0])
            optimizer.zero_grad(set_to_none=True)
            candidate_embeddings = target_embedding[
                batch["candidate_ids"]
            ]
            anchor_embeddings = target_embedding[
                batch["anchor_ids"]
            ]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output = model(
                    batch["hidden"],
                    candidate_embeddings,
                    batch["candidate_logits"],
                    batch["base_logsumexp"],
                    anchor_embeddings,
                )
                losses = global_direct_candidate_loss(
                    output,
                    batch["gold_candidate_indices"],
                    batch["gold_in_lattice"],
                    weighting=args.loss_weighting,
                    dpace_alpha=args.dpace_alpha,
                    exponential_gamma=args.exponential_gamma,
                    post_break_weight=args.post_break_weight,
                    base_safety_weight=args.base_safety_weight,
                    base_safety_margin=args.base_safety_margin,
                )
            losses.loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), args.gradient_clip
            )
            optimizer.step()
            scheduler.step()
            global_step += 1

            current_active = int(losses.active_positions.sum())
            examples_seen += batch_size
            objective_sum += float(losses.loss.detach()) * batch_size
            for name, value in losses.components.items():
                component_sums[name] += float(value.detach()) * batch_size
            nll_numerator += (
                float(losses.unweighted_nll.detach())
                * current_active
            )
            active_total += current_active
            weight_sum += float(
                (
                    losses.position_weights
                    * losses.active_positions.float()
                ).sum()
            )
            grad_norm_sum += float(grad_norm)

        validation = evaluate(
            model,
            validation_loader,
            target_embedding,
            device,
            include_examples=False,
            **evaluation_kwargs,
        )
        train_metrics = {
            "objective": objective_sum / examples_seen,
            "unweighted_nll": nll_numerator / max(1, active_total),
            "mean_position_weight": weight_sum
            / max(1, active_total),
            "mean_preclip_grad_norm": grad_norm_sum
            / len(train_loader),
            "learning_rate": optimizer.param_groups[0]["lr"],
            "steps": global_step,
            "components": {
                name: value / examples_seen
                for name, value in sorted(component_sums.items())
            },
        }
        epoch_record = {
            "epoch": epoch,
            "train": train_metrics,
            "validation": compact_epoch_metrics(validation),
        }
        history.append(epoch_record)
        print(
            json.dumps(epoch_record, ensure_ascii=False), flush=True
        )

        current_key = checkpoint_selection_key(validation, args)
        if current_key > best_key:
            best_key = current_key
            torch.save(
                {
                    "model": model.state_dict(),
                    "args": config_snapshot,
                    "epoch": epoch,
                    "selection_key": best_key,
                    "parameter_count": parameter_count,
                },
                best_path,
            )

    checkpoint = torch.load(
        best_path, map_location=device, weights_only=False
    )
    model.load_state_dict(checkpoint["model"])
    final_validation = evaluate(
        model,
        validation_loader,
        target_embedding,
        device,
        include_examples=True,
        **evaluation_kwargs,
    )
    calibration_threshold: float | None = None
    if args.calibrate_margin:
        (
            calibration_threshold,
            final_validation["calibrated"],
        ) = tune_margin_threshold(
            final_validation,
            max_first_token_drop=(
                args.max_calibration_first_token_drop
            ),
            max_domain_drop=args.max_calibration_domain_drop,
        )
    final_gate = (
        None
        if gate_loader is None
        else evaluate(
            model,
            gate_loader,
            target_embedding,
            device,
            include_examples=True,
            **evaluation_kwargs,
        )
    )
    if (
        final_gate is not None
        and calibration_threshold is not None
    ):
        final_gate["calibrated"] = summarize_margin_calibration(
            final_gate, threshold=calibration_threshold
        )
        final_gate["calibrated"][
            "threshold_selected_on_validation"
        ] = final_validation["calibrated"]["threshold"]
    final_train = (
        final_validation
        if args.memorization_blocks
        else evaluate(
            model,
            make_loader(
                train_dataset,
                candidate_k=args.candidate_k,
                batch_size=args.batch_size,
                shuffle=False,
            ),
            target_embedding,
            device,
            include_examples=False,
            **evaluation_kwargs,
        )
    )
    capacity_gate = (
        capacity_gate_report(final_validation, args)
        if args.memorization_blocks
        else None
    )

    provenance = {
        "project_commit": git_revision(PROJECT),
        "project_dirty": git_is_dirty(PROJECT),
        "data_metadata_sha256": sha256_file(metadata_path),
        "external_train_data": [
            {
                "path": str(path.resolve()),
                "metadata_sha256": sha256_file(path / "metadata.json"),
                "base_greedy_witness_status": (
                    dataset.base_greedy_witness_status
                ),
            }
            for path, dataset in zip(
                args.train_data,
                external_train_collections,
                strict=True,
            )
        ],
        **source_hashes_at_start,
        "trainer_sha256_at_end": sha256_file(trainer_source_path),
        "head_source_sha256_at_end": sha256_file(head_source_path),
        "verified_target_embedding_files": verified_target_files,
        "verified_external_target_embedding_files": (
            verified_external_target_files
        ),
        "base_greedy_witness_status": (
            collection.base_greedy_witness_status
        ),
        "dflash_commit": git_revision(
            PROJECT / "third_party" / "dflash"
        ),
        "domino_commit": git_revision(
            PROJECT / "third_party" / "Domino"
        ),
        "dpace_commit": git_revision(
            PROJECT / "third_party" / "D-PACE"
        ),
        "deepspec_commit": git_revision(
            PROJECT / "third_party" / "DeepSpec"
        ),
        "dels_commit": git_revision(
            PROJECT / "third_party" / "DeLS-Spec"
        ),
    }
    report = {
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
        "array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "hostname": platform.node(),
        "device": torch.cuda.get_device_name(device),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "config": config_snapshot,
        "scope": args.scope,
        "primary_method": PRIMARY_METHOD,
        "evidence_tier": args.evidence_tier,
        "split_protocol": split_protocol,
        "train_blocks": len(train_dataset),
        "train_prompts": len(
            {
                str(record["sample_id"])
                for record in train_dataset.records
            }
        ),
        "train_prompt_set_sha256": hashlib.sha256(
            "\n".join(
                sorted(
                    {
                        str(record["sample_id"])
                        for record in train_dataset.records
                    }
                )
            ).encode("utf-8")
        ).hexdigest(),
        "validation_blocks": len(validation_dataset),
        "validation_prompts": len(
            {
                str(record["sample_id"])
                for record in validation_dataset.records
            }
        ),
        "gate_blocks": (
            len(gate_dataset) if gate_dataset is not None else 0
        ),
        "gate_prompts": (
            len(
                {
                    str(record["sample_id"])
                    for record in gate_dataset.records
                }
            )
            if gate_dataset is not None
            else 0
        ),
        "parameter_count": parameter_count,
        "total_steps": total_steps,
        "warmup_steps": warmup_steps,
        "selected_epoch": int(checkpoint["epoch"]),
        "selection_key": checkpoint["selection_key"],
        "calibration_threshold": (
            calibration_threshold
            if calibration_threshold is None
            or math.isfinite(calibration_threshold)
            else "base_only"
        ),
        "seconds": time.perf_counter() - start,
        "peak_cuda_memory_gib": (
            torch.cuda.max_memory_allocated() / 2**30
        ),
        "peak_cuda_reserved_gib": (
            torch.cuda.max_memory_reserved() / 2**30
        ),
        "initial_validation": initial_validation,
        "final_train_diagnostic": final_train,
        "final_validation": final_validation,
        "final_gate": final_gate,
        "capacity_gate": capacity_gate,
        "history": history,
        "provenance": provenance,
    }
    metrics_path = args.output / "metrics.json"
    temporary_path = metrics_path.with_suffix(".json.tmp")
    temporary_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(metrics_path)
    print(
        json.dumps(
            {
                "metrics": str(metrics_path),
                "scope": args.scope,
                "selected_epoch": int(checkpoint["epoch"]),
                "parameter_count": parameter_count,
                "validation": compact_epoch_metrics(
                    final_validation
                ),
                "gate": (
                    compact_epoch_metrics(final_gate)
                    if final_gate is not None
                    else None
                ),
                "capacity_gate": capacity_gate,
                "calibrated_validation": (
                    final_validation.get("calibrated")
                ),
                "calibrated_gate": (
                    None
                    if final_gate is None
                    else final_gate.get("calibrated")
                ),
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    if (
        args.require_capacity_gate
        and capacity_gate is not None
        and not capacity_gate["passed"]
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
