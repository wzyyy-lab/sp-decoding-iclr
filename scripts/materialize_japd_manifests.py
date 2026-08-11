#!/usr/bin/env python3
"""J002: freeze label-independent JAPD fit/select/diagnostic manifests."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import random
from typing import Any

import torch

from sph.japd import candidate_gold_ranks, clean_support
from sph.japd_data import load_rollout_records, record_key, stratified_prompt_split


CAPACITY_DOMAIN_COUNTS = {"chat": 171, "code": 171, "math": 170}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-rollout", type=Path, required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split-seed", type=int, default=20260810)
    parser.add_argument("--diagnostic-seed", type=int, default=20260811)
    return parser.parse_args()


def select_prompt_groups(
    records: list[dict[str, Any]],
    fit_prompts: set[str],
    *,
    seed: int,
) -> tuple[set[str], set[str]]:
    domains: dict[str, str] = {}
    for record in records:
        sample_id = str(record["sample_id"])
        if sample_id in fit_prompts:
            domains[sample_id] = str(record["domain"])
    by_domain: dict[str, list[str]] = defaultdict(list)
    for sample_id, domain in domains.items():
        by_domain[domain].append(sample_id)
    rng = random.Random(seed)
    capacity: set[str] = set()
    diagnostic: set[str] = set()
    for domain in sorted(CAPACITY_DOMAIN_COUNTS):
        prompts = sorted(by_domain[domain])
        rng.shuffle(prompts)
        count = CAPACITY_DOMAIN_COUNTS[domain]
        if len(prompts) < 2 * count:
            raise RuntimeError(f"not enough {domain} fit prompts for two 512 gates")
        capacity.update(prompts[:count])
        diagnostic.update(prompts[count : 2 * count])
    if capacity & diagnostic:
        raise AssertionError("capacity and full-fit diagnostic prompts overlap")
    if len(capacity) != 512 or len(diagnostic) != 512:
        raise AssertionError("JAPD gates must each contain 512 prompts")
    return capacity, diagnostic


def first_record_per_prompt(
    records: list[dict[str, Any]], prompts: set[str]
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        sample_id = str(record["sample_id"])
        if sample_id in prompts:
            grouped[sample_id].append(record)
    selected = []
    for sample_id in sorted(prompts):
        rows = sorted(
            grouped[sample_id],
            key=lambda row: (
                int(row["anchor_offset"]),
                int(row["context_length"]),
            ),
        )
        if not rows:
            raise RuntimeError(f"prompt has no rollout block: {sample_id}")
        selected.append(rows[0])
    return selected


def strict_multi_repair_count(records: list[dict[str, Any]]) -> int:
    count = 0
    for record in records:
        ids = record["base_topk_ids"].long().unsqueeze(0)
        gold = record["gold_ids"].long().unsqueeze(0)
        ranks = candidate_gold_ranks(ids, gold)
        target_matches = record["target_top1_ids"].long().eq(
            record["gold_ids"].long()
        ).unsqueeze(0)
        support, _ = clean_support(ranks, target_matches)
        count += int((support & ranks.ne(0)).sum().item() >= 2)
    return count


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists():
        raise FileExistsError(f"manifest output already exists: {args.output}")
    metadata, records = load_rollout_records(
        args.train_rollout, split=args.split
    )
    splits = stratified_prompt_split(records, seed=args.split_seed)
    capacity_prompts, full_fit_prompts = select_prompt_groups(
        records, splits["fit"], seed=args.diagnostic_seed
    )
    capacity_records = first_record_per_prompt(records, capacity_prompts)
    strict_multi = strict_multi_repair_count(capacity_records)
    if strict_multi < 256:
        raise RuntimeError(
            f"capacity manifest has only {strict_multi} strict multi-repair blocks"
        )
    prompt_domains: dict[str, str] = {}
    for record in records:
        prompt_domains[str(record["sample_id"])] = str(record["domain"])
    report = {
        "format": "japd_manifest_v1",
        "complete": True,
        "source_rollout": str(args.train_rollout.resolve()),
        "source_format": metadata.get("format"),
        "source_split": args.split,
        "selection_fields": [
            "sample_id",
            "domain",
            "anchor_offset",
            "context_length",
        ],
        "label_fields_used_for_selection": [],
        "split_seed": args.split_seed,
        "diagnostic_seed": args.diagnostic_seed,
        "prompt_splits": {
            name: sorted(values) for name, values in splits.items()
        },
        "prompt_split_counts": {
            name: len(values) for name, values in splits.items()
        },
        "prompt_split_domain_counts": {
            name: dict(
                sorted(Counter(prompt_domains[value] for value in values).items())
            )
            for name, values in splits.items()
        },
        "capacity": {
            "prompts": sorted(capacity_prompts),
            "records": [
                {
                    "sample_id": record_key(record)[0],
                    "anchor_offset": record_key(record)[1],
                    "context_length": record_key(record)[2],
                    "domain": str(record["domain"]),
                }
                for record in capacity_records
            ],
            "domain_counts": dict(
                sorted(Counter(str(row["domain"]) for row in capacity_records).items())
            ),
            "strict_multi_repair_blocks_diagnostic_only": strict_multi,
        },
        "full_fit_diagnostic": {
            "prompts": sorted(full_fit_prompts),
            "domain_counts": dict(
                sorted(Counter(prompt_domains[value] for value in full_fit_prompts).items())
            ),
        },
        "disjoint": {
            "fit_select": not bool(splits["fit"] & splits["select"]),
            "fit_diagnostic": not bool(splits["fit"] & splits["diagnostic"]),
            "select_diagnostic": not bool(splits["select"] & splits["diagnostic"]),
            "capacity_full_fit_diagnostic": not bool(
                capacity_prompts & full_fit_prompts
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({
        "output": str(args.output),
        "prompt_split_counts": report["prompt_split_counts"],
        "capacity_strict_multi": strict_multi,
        "disjoint": report["disjoint"],
    }, ensure_ascii=False), flush=True)
    return report


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
