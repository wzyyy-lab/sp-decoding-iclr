#!/usr/bin/env python3
"""Analyze top-K coverage and oracle prefix acceptance for canonical blocks."""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

from sph.candidate_ceiling import (
    accepted_draft_prefix_lengths,
    first_top1_miss_gold_rank,
    gold_in_candidates,
    prefix_coverage,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plot", type=Path)
    parser.add_argument("--k", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64])
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def load_compact_records(input_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    metadata = json.loads((input_path / "metadata.json").read_text())
    compact: list[dict[str, Any]] = []
    shards = sorted(input_path.glob("shard-*.pt"))
    if not shards:
        raise FileNotFoundError(f"no shard-*.pt files in {input_path}")
    for shard in shards:
        records = torch.load(shard, map_location="cpu", weights_only=False)
        for record in records:
            compact.append(
                {
                    "sample_id": record["sample_id"],
                    "domain": record["domain"],
                    "split": record["split"],
                    "topk_ids": record["base_topk_ids"].to(torch.int64),
                    "topk_logits": record["base_topk_logits"].to(torch.float32),
                    "logsumexp": record["base_logsumexp"].to(torch.float32),
                    "gold_ids": record["gold_ids"].to(torch.int64),
                }
            )
        del records
    return metadata, compact


def mean(values: torch.Tensor) -> float:
    return float(values.to(torch.float64).mean().item())


def cluster_bootstrap_ci(
    values: torch.Tensor,
    sample_ids: list[str],
    *,
    draws: int,
    seed: int,
) -> list[float] | None:
    if draws < 1:
        return None
    grouped: dict[str, list[float]] = defaultdict(list)
    for sample_id, value in zip(sample_ids, values.tolist(), strict=True):
        grouped[sample_id].append(float(value))
    cluster_means = [sum(items) / len(items) for items in grouped.values()]
    if not cluster_means:
        return None
    rng = random.Random(seed)
    estimates = []
    for _ in range(draws):
        sampled = [rng.choice(cluster_means) for _ in cluster_means]
        estimates.append(sum(sampled) / len(sampled))
    estimates.sort()
    low = estimates[int(0.025 * (draws - 1))]
    high = estimates[int(0.975 * (draws - 1))]
    return [low, high]


def summarize_subset(
    records: list[dict[str, Any]],
    k: int,
    *,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    topk_ids = torch.stack([record["topk_ids"] for record in records])
    topk_logits = torch.stack([record["topk_logits"] for record in records])
    logsumexp = torch.stack([record["logsumexp"] for record in records])
    gold_ids = torch.stack([record["gold_ids"] for record in records])
    sample_ids = [record["sample_id"] for record in records]

    coverage = gold_in_candidates(topk_ids, gold_ids, k)
    covered_prefix = prefix_coverage(coverage)
    accepted = accepted_draft_prefix_lengths(coverage)
    verification_advance = accepted + 1
    conditional_logits = topk_logits[..., :k]
    conditional_probs = torch.softmax(conditional_logits, dim=-1)
    conditional_entropy = -(
        conditional_probs * torch.log(conditional_probs.clamp_min(1e-30))
    ).sum(dim=-1)
    retained_mass = torch.exp(
        torch.logsumexp(conditional_logits, dim=-1) - logsumexp
    )
    if k >= 2:
        top1_top2_margin = topk_logits[..., 0] - topk_logits[..., 1]
        mean_margin = mean(top1_top2_margin)
    else:
        mean_margin = None

    first_miss_rank = first_top1_miss_gold_rank(topk_ids, gold_ids)
    nonzero_first_miss = first_miss_rank[first_miss_rank > 0]
    result = {
        "blocks": len(records),
        "samples": len(set(sample_ids)),
        "mean_accepted_draft_tokens": mean(accepted),
        "mean_verification_advance": mean(verification_advance),
        "verification_advance_ci95_cluster_bootstrap": cluster_bootstrap_ci(
            verification_advance,
            sample_ids,
            draws=bootstrap_samples,
            seed=seed,
        ),
        "full_block_coverage": mean(covered_prefix[..., -1]),
        "position_gold_recall": [float(x) for x in coverage.float().mean(dim=0)],
        "prefix_coverage": [float(x) for x in covered_prefix.float().mean(dim=0)],
        "mean_topk_conditional_entropy_nats": mean(conditional_entropy),
        "mean_retained_probability_mass": mean(retained_mass),
        "mean_top1_top2_logit_margin": mean_margin,
        "first_top1_miss": {
            "blocks_with_miss": int(nonzero_first_miss.numel()),
            "mean_gold_rank": (
                mean(nonzero_first_miss) if nonzero_first_miss.numel() else None
            ),
            "gold_outside_saved_topk_fraction": (
                mean(nonzero_first_miss > topk_ids.shape[-1])
                if nonzero_first_miss.numel()
                else None
            ),
        },
    }
    return result


def render_plot(report: dict[str, Any], output: Path) -> None:
    import matplotlib.pyplot as plt

    k_values = [int(k) for k in report["overall"]]
    advances = [
        report["overall"][str(k)]["mean_verification_advance"] for k in k_values
    ]
    full_blocks = [
        report["overall"][str(k)]["full_block_coverage"] for k in k_values
    ]
    figure, axes = plt.subplots(1, 2, figsize=(9.0, 3.6))
    axes[0].plot(k_values, advances, marker="o")
    axes[0].set_xscale("log", base=2)
    axes[0].set_xticks(k_values, labels=[str(k) for k in k_values])
    axes[0].set_xlabel("candidate K")
    axes[0].set_ylabel("oracle verification advance")
    axes[0].grid(alpha=0.25)
    axes[1].plot(k_values, full_blocks, marker="o")
    axes[1].set_xscale("log", base=2)
    axes[1].set_xticks(k_values, labels=[str(k) for k in k_values])
    axes[1].set_xlabel("candidate K")
    axes[1].set_ylabel("full-block candidate coverage")
    axes[1].grid(alpha=0.25)
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    metadata, records = load_compact_records(args.input)
    saved_k = int(metadata["top_k"])
    k_values = sorted(set(args.k))
    invalid = [k for k in k_values if k < 1 or k > saved_k]
    if invalid:
        raise ValueError(f"requested K values outside saved top-{saved_k}: {invalid}")

    report: dict[str, Any] = {
        "input": str(args.input.resolve()),
        "metadata": metadata,
        "metric_convention": {
            "accepted_draft_tokens": "matching candidate positions after the known anchor",
            "verification_advance": "accepted_draft_tokens + 1, matching DFlash/Domino acceptance_lengths",
            "bootstrap_unit": "sample_id (all anchors from a prompt resampled together)",
            "entropy": "entropy after renormalizing within the retained top-K",
        },
        "overall": {},
        "by_domain": {},
        "by_split": {},
    }
    for k in k_values:
        report["overall"][str(k)] = summarize_subset(
            records,
            k,
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed + k,
        )
    for field, destination in [("domain", "by_domain"), ("split", "by_split")]:
        values = sorted({record[field] for record in records})
        for value in values:
            subset = [record for record in records if record[field] == value]
            report[destination][value] = {}
            for k in k_values:
                report[destination][value][str(k)] = summarize_subset(
                    subset,
                    k,
                    bootstrap_samples=args.bootstrap_samples,
                    seed=args.seed + k + sum(ord(char) for char in value),
                )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if args.plot is not None:
        render_plot(report, args.plot)
    print(json.dumps(report["overall"], indent=2))


if __name__ == "__main__":
    main()
