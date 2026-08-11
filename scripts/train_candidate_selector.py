#!/usr/bin/env python3
"""Train a parameter-matched local/global DFlash candidate-lattice selector."""

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
import subprocess
import time
from typing import Any, Iterable

from safetensors import safe_open
import torch
from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from sph.candidate_ceiling import accepted_draft_prefix_lengths
from sph.candidate_lattice_selector import (
    CandidateLatticeSelector,
    candidate_selector_loss,
    first_divergence_margin,
    prefix_candidate_mask,
    teacher_forced_logits,
    viterbi_decode,
)
from sph.data import CanonicalBlockDataset, collate_canonical_blocks
from sph.survival_path_head import survival_decode


PROJECT = Path(__file__).resolve().parents[1]
PRIMARY_DECODER = "survival"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--scope",
        choices=["local", "causal", "global"],
        default="global",
        help="Candidate-node attention scope; all choices share parameters.",
    )
    parser.add_argument("--candidate-k", type=int, default=16)
    parser.add_argument("--model-dim", type=int, default=128)
    parser.add_argument("--token-dim", type=int, default=64)
    parser.add_argument("--transition-dim", type=int, default=32)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument(
        "--loss-weighting",
        choices=["uniform", "dpace"],
        default="dpace",
    )
    parser.add_argument(
        "--rank-weight-power",
        type=float,
        default=0.5,
        help="Multiply candidate CE by one-indexed gold rank to this power.",
    )
    parser.add_argument("--in-lattice-loss-weight", type=float, default=0.1)
    parser.add_argument("--base-correct-loss-weight", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--validation-split", default="validation_select")
    parser.add_argument("--gate-split", default="validation_gate")
    parser.add_argument(
        "--skip-gate",
        action="store_true",
        help="Do not load the held-out development gate.",
    )
    parser.add_argument(
        "--memorization-blocks",
        type=int,
        default=0,
        help="Train and evaluate the same deterministic subset as a capacity test.",
    )
    parser.add_argument(
        "--memorization-opportunity-fraction",
        type=float,
        default=0.5,
        help="Fraction of capacity blocks forced to have a reachable top-K repair.",
    )
    parser.add_argument(
        "--require-capacity-gate",
        action="store_true",
        help="Exit nonzero unless all declared memorization thresholds pass.",
    )
    parser.add_argument("--min-candidate-accuracy", type=float, default=0.95)
    parser.add_argument("--min-hard-candidate-accuracy", type=float, default=0.90)
    parser.add_argument("--min-first-miss-repair-rate", type=float, default=0.90)
    parser.add_argument("--min-oracle-gap-recovered", type=float, default=0.60)
    parser.add_argument(
        "--max-first-token-drop",
        type=float,
        default=0.001,
        help="Validation KEEP_BASE threshold constraint.",
    )
    parser.add_argument(
        "--max-domain-drop",
        type=float,
        default=0.05,
        help="Maximum allowed accepted-length loss in any validation domain.",
    )
    parser.add_argument(
        "--evidence-tier",
        choices=["capacity_probe", "development"],
        default="development",
    )
    return parser.parse_args()


class RecordDataset(Dataset[dict[str, Any]]):
    """Cheap view over already integrity-checked canonical records."""

    def __init__(
        self, records: Iterable[dict[str, Any]], metadata: dict[str, Any]
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
    data_metadata: dict[str, Any], target: Path
) -> list[dict[str, Any]]:
    """Bind candidate embeddings to the checkpoint used for collection."""

    if int(data_metadata.get("format_version", 1)) < 2:
        return []
    expected_records = data_metadata.get("provenance", {}).get("target_files")
    if not isinstance(expected_records, list):
        raise RuntimeError("protocol-v2 data lacks target file fingerprints")
    expected = {str(record["path"]): record for record in expected_records}
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


def assert_prompt_disjoint_splits(
    named_datasets: dict[str, RecordDataset | None],
) -> None:
    prompt_sets = {
        name: {str(record["sample_id"]) for record in dataset.records}
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
    """Sample a repeatable mixture of ordinary and repair-opportunity blocks."""

    if not 0.0 <= opportunity_fraction <= 1.0:
        raise ValueError("memorization opportunity fraction must be in [0, 1]")
    if count > len(records):
        raise ValueError(
            f"requested {count} memorization blocks from only {len(records)}"
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
            repairable = bool((topk[first_miss] == gold[first_miss]).any())
        else:
            repairable = False
        (opportunities if repairable else ordinary).append(record)
    generator.shuffle(opportunities)
    generator.shuffle(ordinary)
    desired_opportunities = min(
        len(opportunities), round(count * opportunity_fraction)
    )
    desired_ordinary = min(len(ordinary), count - desired_opportunities)
    selected = (
        opportunities[:desired_opportunities] + ordinary[:desired_ordinary]
    )
    if len(selected) < count:
        selected_ids = {id(record) for record in selected}
        remainder = [
            record for record in records if id(record) not in selected_ids
        ]
        generator.shuffle(remainder)
        selected.extend(remainder[: count - len(selected)])
    generator.shuffle(selected)
    return selected


def realized_prefix(
    path: Tensor, candidate_ids: Tensor, gold_ids: Tensor
) -> Tensor:
    selected_ids = candidate_ids.gather(
        -1, path.unsqueeze(-1)
    ).squeeze(-1)
    return accepted_draft_prefix_lengths(selected_ids == gold_ids)


def greedy_decode(log_probs: Tensor) -> Tensor:
    batch, length, candidates, _ = log_probs.shape
    path = torch.empty(
        batch, length, dtype=torch.long, device=log_probs.device
    )
    batch_indices = torch.arange(batch, device=log_probs.device)
    previous = torch.zeros(
        batch, dtype=torch.long, device=log_probs.device
    )
    for position in range(length):
        current = log_probs[
            batch_indices, position, previous
        ].argmax(dim=-1)
        path[:, position] = current
        previous = current.clamp(0, candidates - 1)
    return path


def _prompt_balanced_mean(
    records: list[dict[str, Any]], key: str
) -> float:
    grouped: dict[str, list[float]] = defaultdict(list)
    for record in records:
        grouped[str(record["sample_id"])].append(float(record[key]))
    return sum(sum(values) / len(values) for values in grouped.values()) / len(
        grouped
    )


def _method_summary(
    records: list[dict[str, Any]], method: str
) -> dict[str, float]:
    accepted = [
        float(record["accepted_draft_tokens"][method]) for record in records
    ]
    first = [
        float(record["first_token_correct"][method]) for record in records
    ]
    prompt_records = [
        {
            "sample_id": record["sample_id"],
            "value": record["accepted_draft_tokens"][method],
        }
        for record in records
    ]
    return {
        "mean_accepted_draft_tokens": sum(accepted) / len(accepted),
        "mean_verification_advance": sum(accepted) / len(accepted) + 1.0,
        "mean_accepted_draft_tokens_prompt_balanced": (
            _prompt_balanced_mean(prompt_records, "value")
        ),
        "mean_verification_advance_prompt_balanced": (
            _prompt_balanced_mean(prompt_records, "value") + 1.0
        ),
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


@torch.inference_mode()
def evaluate(
    model: CandidateLatticeSelector,
    loader: DataLoader,
    target_embedding: Tensor,
    device: torch.device,
    *,
    candidate_k: int,
    include_examples: bool,
) -> dict[str, Any]:
    model.eval()
    example_records: list[dict[str, Any]] = []
    loss_sums: dict[str, float] = defaultdict(float)
    loss_examples = 0
    active_correct = 0
    active_total = 0
    hard_correct = 0
    hard_total = 0
    coverage_brier_sum = 0.0
    base_correct_brier_sum = 0.0
    calibration_count = 0

    for cpu_batch in loader:
        batch = to_device(cpu_batch, device)
        candidate_embeddings = target_embedding[batch["candidate_ids"]]
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            output = model(
                batch["hidden"],
                batch["candidate_ids"],
                candidate_embeddings,
                batch["candidate_logits"],
                batch["base_logsumexp"],
                batch["anchor_ids"],
            )
            loss_output = candidate_selector_loss(
                output,
                batch["candidate_ids"],
                batch["gold_ids"],
                batch["gold_candidate_indices"],
                batch["gold_in_lattice"],
                weighting="dpace",
            )
        batch_size, length, _ = batch["candidate_ids"].shape
        loss_examples += batch_size
        loss_sums["loss"] += float(loss_output.loss) * batch_size
        loss_sums["candidate_nll"] += (
            float(loss_output.candidate_nll) * batch_size
        )
        loss_sums["in_lattice_bce"] += (
            float(loss_output.in_lattice_bce) * batch_size
        )
        loss_sums["base_correct_bce"] += (
            float(loss_output.base_correct_bce) * batch_size
        )

        teacher_logits = teacher_forced_logits(
            output.edge_scores,
            batch["gold_candidate_indices"],
            batch["gold_in_lattice"],
        )
        active = prefix_candidate_mask(batch["gold_in_lattice"])
        candidate_prediction = teacher_logits.argmax(dim=-1)
        candidate_correct = (
            candidate_prediction == batch["gold_candidate_indices"]
        )
        active_correct += int((candidate_correct & active).sum())
        active_total += int(active.sum())
        hard = active & (batch["gold_candidate_indices"] > 0)
        hard_correct += int((candidate_correct & hard).sum())
        hard_total += int(hard.sum())

        coverage_probability = output.in_lattice_logits.float().sigmoid()
        base_probability = output.base_correct_logits.float().sigmoid()
        base_correct = (
            batch["candidate_ids"][..., 0] == batch["gold_ids"]
        )
        coverage_brier_sum += float(
            (
                coverage_probability - batch["gold_in_lattice"].float()
            )
            .square()
            .sum()
        )
        base_correct_brier_sum += float(
            (base_probability - base_correct.float()).square().sum()
        )
        calibration_count += batch_size * length

        absolute_log_probs = output.log_probs + F.logsigmoid(
            output.in_lattice_logits.float()
        )[:, :, None, None]
        paths = {
            "base": torch.zeros_like(batch["gold_candidate_indices"]),
            "unary": output.unary_scores.argmax(dim=-1),
            "greedy": greedy_decode(absolute_log_probs),
            "map": viterbi_decode(output.edge_scores).path,
            "survival": survival_decode(absolute_log_probs).path,
        }
        realized = {
            name: realized_prefix(
                path, batch["candidate_ids"], batch["gold_ids"]
            )
            for name, path in paths.items()
        }
        oracle = accepted_draft_prefix_lengths(batch["gold_in_lattice"])
        first_correct = {
            name: (
                batch["candidate_ids"][:, 0]
                .gather(-1, path[:, :1])
                .squeeze(1)
                == batch["gold_ids"][:, 0]
            )
            for name, path in paths.items()
        }
        margins = {
            name: first_divergence_margin(absolute_log_probs, path)
            for name, path in paths.items()
            if name != "base"
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
                int(batch["gold_candidate_indices"][item, base_prefix]) + 1
                if has_failure
                and bool(batch["gold_in_lattice"][item, base_prefix])
                else candidate_k + 1
            )
            record: dict[str, Any] = {
                "sample_id": sample_id,
                "domain": domain,
                "accepted_draft_tokens": {
                    name: int(values[item]) for name, values in realized.items()
                },
                "first_token_correct": {
                    name: bool(values[item])
                    for name, values in first_correct.items()
                },
                "oracle_accepted_draft_tokens": int(oracle[item]),
                "base_first_miss_position": base_prefix if has_failure else None,
                "base_first_miss_gold_rank": first_miss_rank,
                "keep_base_margin": {
                    name: (
                        float(values[item])
                        if math.isfinite(float(values[item]))
                        else -1e30
                    )
                    for name, values in margins.items()
                },
            }
            if include_examples:
                record["candidate_path_indices"] = {
                    name: path[item].detach().cpu().tolist()
                    for name, path in paths.items()
                }
            example_records.append(record)

    report: dict[str, Any] = {
        "loss": {
            key: value / loss_examples for key, value in loss_sums.items()
        },
        "candidate_classification": {
            "accuracy": active_correct / active_total,
            "correct": active_correct,
            "positions": active_total,
            "non_top1_accuracy": (
                hard_correct / hard_total if hard_total else None
            ),
            "non_top1_correct": hard_correct,
            "non_top1_positions": hard_total,
        },
        "calibration": {
            "in_lattice_brier": coverage_brier_sum / calibration_count,
            "base_correct_brier": (
                base_correct_brier_sum / calibration_count
            ),
        },
        "blocks": len(example_records),
        "prompts": len(
            {str(record["sample_id"]) for record in example_records}
        ),
    }
    methods = ["base", "unary", "greedy", "map", "survival"]
    for method in methods:
        report[method] = _method_summary(example_records, method)
    oracle_records = [
        {
            "sample_id": record["sample_id"],
            "accepted_draft_tokens": {"oracle": record[
                "oracle_accepted_draft_tokens"
            ]},
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
            method: _method_summary(subset, method) for method in methods
        }

    diagnostics = {}
    for method in methods[1:]:
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
            learned_prefix = record["accepted_draft_tokens"][method]
            rank = record["base_first_miss_gold_rank"]
            if rank <= candidate_k:
                repair_opportunities += 1
                if learned_prefix > base_prefix:
                    repairs += 1
                    repair_ranks[_rank_bucket(rank, candidate_k)] += 1
            if learned_prefix > base_prefix:
                improved += 1
            elif learned_prefix < base_prefix:
                harmed += 1
            path = record.get("candidate_path_indices", {}).get(method)
            if path is not None:
                different = [
                    index for index, candidate in enumerate(path) if candidate
                ]
                if different:
                    changed += 1
                    if different[0] <= base_prefix:
                        reachable_changes += 1
                    else:
                        suffix_only_changes += 1
        diagnostics[method] = {
            "path_changed_blocks": changed if include_examples else None,
            "reachable_changed_blocks": (
                reachable_changes if include_examples else None
            ),
            "suffix_only_changed_blocks": (
                suffix_only_changes if include_examples else None
            ),
            "improved_blocks": improved,
            "harmed_blocks": harmed,
            "first_miss_repair_opportunities": repair_opportunities,
            "first_miss_repairs": repairs,
            "first_miss_repair_rate_given_k": (
                repairs / repair_opportunities
                if repair_opportunities
                else None
            ),
            "successful_repair_gold_rank": dict(repair_ranks),
        }
    report["path_diagnostics"] = diagnostics
    base_eal = report["base"][
        "mean_accepted_draft_tokens_prompt_balanced"
    ]
    oracle_eal = report["oracle"][
        "mean_accepted_draft_tokens_prompt_balanced"
    ]
    for method in methods[1:]:
        learned_eal = report[method][
            "mean_accepted_draft_tokens_prompt_balanced"
        ]
        denominator = oracle_eal - base_eal
        report["path_diagnostics"][method]["oracle_gap_recovered"] = (
            (learned_eal - base_eal) / denominator
            if denominator > 0
            else None
        )
    if include_examples:
        report["examples"] = example_records
    return report


def summarize_keep_base(
    examples: list[dict[str, Any]],
    *,
    method: str,
    threshold: float,
) -> dict[str, Any]:
    selected_records = []
    learned_blocks = 0
    for record in examples:
        use_learned = (
            float(record["keep_base_margin"][method]) >= threshold
        )
        if use_learned:
            learned_blocks += 1
        selected_records.append(
            {
                "sample_id": record["sample_id"],
                "domain": record["domain"],
                "accepted_draft_tokens": {
                    "keep_base": record["accepted_draft_tokens"][
                        method if use_learned else "base"
                    ]
                },
                "first_token_correct": {
                    "keep_base": record["first_token_correct"][
                        method if use_learned else "base"
                    ]
                },
            }
        )
    summary: dict[str, Any] = _method_summary(
        selected_records, "keep_base"
    )
    summary["learned_blocks"] = learned_blocks
    summary["learned_fraction"] = learned_blocks / len(selected_records)
    summary["by_domain"] = {}
    for domain in sorted(
        {str(record["domain"]) for record in selected_records}
    ):
        subset = [
            record
            for record in selected_records
            if record["domain"] == domain
        ]
        summary["by_domain"][domain] = _method_summary(
            subset, "keep_base"
        )
    return summary


def tune_keep_base_threshold(
    validation: dict[str, Any],
    *,
    method: str,
    max_first_token_drop: float,
    max_domain_drop: float,
) -> tuple[float, dict[str, Any]]:
    """Choose the safest validation threshold among equal-EAL solutions."""

    examples = validation["examples"]
    finite_margins = sorted(
        {
            float(record["keep_base_margin"][method])
            for record in examples
            if math.isfinite(
                float(record["keep_base_margin"][method])
            )
        }
    )
    thresholds = [float("-inf")]
    thresholds.extend(
        math.nextafter(value, float("inf")) for value in finite_margins
    )
    thresholds.append(float("inf"))
    base_first = validation["base"]["first_token_accuracy"]
    base_by_domain = {
        domain: metrics["base"]["mean_accepted_draft_tokens"]
        for domain, metrics in validation["by_domain"].items()
    }
    feasible: list[tuple[tuple[float, float, float], float, dict[str, Any]]] = []
    for threshold in thresholds:
        summary = summarize_keep_base(
            examples, method=method, threshold=threshold
        )
        first_ok = (
            summary["first_token_accuracy"]
            >= base_first - max_first_token_drop
        )
        domain_deltas = [
            summary["by_domain"][domain][
                "mean_accepted_draft_tokens"
            ]
            - base_value
            for domain, base_value in base_by_domain.items()
        ]
        domain_ok = min(domain_deltas) >= -max_domain_drop
        if first_ok and domain_ok:
            # Prefer EAL, then minimum-domain behavior, then more abstention.
            key = (
                summary[
                    "mean_accepted_draft_tokens_prompt_balanced"
                ],
                min(domain_deltas),
                threshold,
            )
            feasible.append((key, threshold, summary))
    if not feasible:
        raise RuntimeError("KEEP_BASE threshold search found no feasible point")
    _, threshold, summary = max(feasible, key=lambda item: item[0])
    return threshold, summary


def capacity_gate_report(
    evaluation: dict[str, Any], args: argparse.Namespace
) -> dict[str, Any]:
    classification = evaluation["candidate_classification"]
    diagnostics = evaluation["path_diagnostics"][PRIMARY_DECODER]
    values = {
        "candidate_accuracy": classification["accuracy"],
        "hard_candidate_accuracy": classification["non_top1_accuracy"],
        "first_miss_repair_rate": diagnostics[
            "first_miss_repair_rate_given_k"
        ],
        "oracle_gap_recovered": diagnostics["oracle_gap_recovered"],
    }
    thresholds = {
        "candidate_accuracy": args.min_candidate_accuracy,
        "hard_candidate_accuracy": args.min_hard_candidate_accuracy,
        "first_miss_repair_rate": args.min_first_miss_repair_rate,
        "oracle_gap_recovered": args.min_oracle_gap_recovered,
    }
    checks = {
        name: value is not None and value >= thresholds[name]
        for name, value in values.items()
    }
    return {
        "passed": all(checks.values()),
        "values": values,
        "thresholds": thresholds,
        "checks": checks,
    }


def main() -> None:
    args = parse_args()
    if args.require_capacity_gate and not args.memorization_blocks:
        raise ValueError("capacity gate requires --memorization-blocks")
    if args.memorization_blocks and args.evidence_tier != "capacity_probe":
        raise ValueError("memorization results must be labeled capacity_probe")
    if not torch.cuda.is_available():
        raise RuntimeError("candidate selector training requires CUDA")
    seed_everything(args.seed)
    device = torch.device("cuda:0")
    args.output.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()

    metadata_path = args.data / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
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
        train_dataset = RecordDataset(by_split[args.train_split], metadata)
        validation_dataset = RecordDataset(
            by_split[args.validation_split], metadata
        )
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
        split_protocol = "prompt_disjoint_development"

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

    target_embedding = load_target_embedding(args.target).to(
        device=device, dtype=torch.bfloat16
    )
    hidden_size = int(target_embedding.shape[1])
    vocab_size = int(target_embedding.shape[0])
    block_length = int(train_dataset.records[0]["gold_ids"].numel())
    model = CandidateLatticeSelector(
        hidden_size=hidden_size,
        vocab_size=vocab_size,
        max_positions=block_length,
        max_candidates=args.candidate_k,
        model_dim=args.model_dim,
        token_dim=args.token_dim,
        transition_dim=args.transition_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        scope=args.scope,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    parameter_count = sum(
        parameter.numel() for parameter in model.parameters()
    )
    history = []
    best_score = (float("-inf"), float("-inf"))
    best_path = args.output / "best.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        totals: dict[str, float] = defaultdict(float)
        examples_seen = 0
        for cpu_batch in train_loader:
            batch = to_device(cpu_batch, device)
            batch_size = int(batch["hidden"].shape[0])
            optimizer.zero_grad(set_to_none=True)
            candidate_embeddings = target_embedding[
                batch["candidate_ids"]
            ]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output = model(
                    batch["hidden"],
                    batch["candidate_ids"],
                    candidate_embeddings,
                    batch["candidate_logits"],
                    batch["base_logsumexp"],
                    batch["anchor_ids"],
                )
                losses = candidate_selector_loss(
                    output,
                    batch["candidate_ids"],
                    batch["gold_ids"],
                    batch["gold_candidate_indices"],
                    batch["gold_in_lattice"],
                    weighting=args.loss_weighting,
                    rank_weight_power=args.rank_weight_power,
                    in_lattice_loss_weight=args.in_lattice_loss_weight,
                    base_correct_loss_weight=args.base_correct_loss_weight,
                )
            losses.loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), args.gradient_clip
            )
            optimizer.step()
            examples_seen += batch_size
            totals["loss"] += float(losses.loss.detach()) * batch_size
            totals["candidate_nll"] += (
                float(losses.candidate_nll.detach()) * batch_size
            )
            totals["in_lattice_bce"] += (
                float(losses.in_lattice_bce.detach()) * batch_size
            )
            totals["base_correct_bce"] += (
                float(losses.base_correct_bce.detach()) * batch_size
            )

        validation = evaluate(
            model,
            validation_loader,
            target_embedding,
            device,
            candidate_k=args.candidate_k,
            include_examples=False,
        )
        train_loss = {
            key: value / examples_seen for key, value in totals.items()
        }
        selection_score = (
            validation[PRIMARY_DECODER][
                "mean_accepted_draft_tokens_prompt_balanced"
            ],
            validation["candidate_classification"]["accuracy"],
        )
        epoch_record = {
            "epoch": epoch,
            "train": train_loss,
            "validation": {
                "candidate_accuracy": validation[
                    "candidate_classification"
                ]["accuracy"],
                "non_top1_accuracy": validation[
                    "candidate_classification"
                ]["non_top1_accuracy"],
                "base_eal": validation["base"][
                    "mean_accepted_draft_tokens_prompt_balanced"
                ],
                "survival_eal": validation["survival"][
                    "mean_accepted_draft_tokens_prompt_balanced"
                ],
                "oracle_eal": validation["oracle"][
                    "mean_accepted_draft_tokens_prompt_balanced"
                ],
                "first_miss_repair_rate": validation[
                    "path_diagnostics"
                ]["survival"]["first_miss_repair_rate_given_k"],
                "harmed_blocks": validation["path_diagnostics"][
                    "survival"
                ]["harmed_blocks"],
            },
            "base_scale": float(model.base_scale.detach()),
            "transition_scale": float(model.transition_scale.detach()),
        }
        history.append(epoch_record)
        print(json.dumps(epoch_record, ensure_ascii=False), flush=True)
        if selection_score > best_score:
            best_score = selection_score
            torch.save(
                {
                    "model": model.state_dict(),
                    "args": vars(args),
                    "epoch": epoch,
                    "selection_score": selection_score,
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
        candidate_k=args.candidate_k,
        include_examples=True,
    )
    keep_base_threshold, validation_keep_base = (
        tune_keep_base_threshold(
            final_validation,
            method=PRIMARY_DECODER,
            max_first_token_drop=args.max_first_token_drop,
            max_domain_drop=args.max_domain_drop,
        )
    )
    final_validation["keep_base"] = validation_keep_base
    final_validation["keep_base"]["threshold"] = (
        keep_base_threshold
        if math.isfinite(keep_base_threshold)
        else ("all_learned" if keep_base_threshold < 0 else "all_base")
    )

    final_gate = None
    if gate_loader is not None:
        final_gate = evaluate(
            model,
            gate_loader,
            target_embedding,
            device,
            candidate_k=args.candidate_k,
            include_examples=True,
        )
        final_gate["keep_base"] = summarize_keep_base(
            final_gate["examples"],
            method=PRIMARY_DECODER,
            threshold=keep_base_threshold,
        )
        final_gate["keep_base"]["threshold_from_validation"] = (
            final_validation["keep_base"]["threshold"]
        )

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
            candidate_k=args.candidate_k,
            include_examples=False,
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
        "trainer_sha256": sha256_file(Path(__file__)),
        "head_source_sha256": sha256_file(
            PROJECT / "src" / "sph" / "candidate_lattice_selector.py"
        ),
        "verified_target_embedding_files": verified_target_files,
        "dflash_commit": git_revision(PROJECT / "third_party" / "dflash"),
        "domino_commit": git_revision(PROJECT / "third_party" / "Domino"),
    }
    report = {
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
        "array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "hostname": platform.node(),
        "device": torch.cuda.get_device_name(device),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "config": vars(args),
        "scope": args.scope,
        "primary_decoder": PRIMARY_DECODER,
        "evidence_tier": args.evidence_tier,
        "split_protocol": split_protocol,
        "train_blocks": len(train_dataset),
        "validation_blocks": len(validation_dataset),
        "gate_blocks": len(gate_dataset) if gate_dataset is not None else 0,
        "parameter_count": parameter_count,
        "selected_epoch": int(checkpoint["epoch"]),
        "seconds": time.perf_counter() - start,
        "final_train_diagnostic": final_train,
        "final_validation": final_validation,
        "final_gate": final_gate,
        "capacity_gate": capacity_gate,
        "history": history,
        "provenance": provenance,
    }
    metrics_path = args.output / "metrics.json"
    metrics_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    concise = {
        "metrics": str(metrics_path),
        "scope": args.scope,
        "selected_epoch": report["selected_epoch"],
        "parameter_count": parameter_count,
        "validation_base_eal": final_validation["base"][
            "mean_accepted_draft_tokens_prompt_balanced"
        ],
        "validation_survival_eal": final_validation["survival"][
            "mean_accepted_draft_tokens_prompt_balanced"
        ],
        "validation_keep_base_eal": final_validation["keep_base"][
            "mean_accepted_draft_tokens_prompt_balanced"
        ],
        "gate_keep_base_eal": (
            final_gate["keep_base"][
                "mean_accepted_draft_tokens_prompt_balanced"
            ]
            if final_gate is not None
            else None
        ),
        "capacity_gate": capacity_gate,
    }
    print(json.dumps(concise, ensure_ascii=False, indent=2), flush=True)
    if args.require_capacity_gate and not capacity_gate["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
