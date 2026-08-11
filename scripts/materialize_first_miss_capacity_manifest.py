#!/usr/bin/env python3
"""Freeze the exact FMAS capacity subset and its action composition."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sph.data import CanonicalBlockDataset
from sph.first_miss_capacity import build_capacity_manifest

try:
    from train_global_direct_selector import deterministic_capacity_subset
except ModuleNotFoundError:  # Imported as ``scripts.*`` in CPU tests.
    from scripts.train_global_direct_selector import (
        deterministic_capacity_subset,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--blocks", type=int, default=512)
    parser.add_argument("--candidate-k", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--opportunity-fraction", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    collection = CanonicalBlockDataset(args.data)
    records = [
        record
        for record in collection.records
        if str(record["split"]) == args.train_split
    ]
    selected = deterministic_capacity_subset(
        records,
        count=args.blocks,
        seed=args.seed,
        opportunity_fraction=args.opportunity_fraction,
        candidate_k=args.candidate_k,
    )
    manifest = build_capacity_manifest(
        selected,
        source_metadata_path=args.data / "metadata.json",
        candidate_k=args.candidate_k,
        seed=args.seed,
        opportunity_fraction=args.opportunity_fraction,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.output)
    print(
        json.dumps(
            {
                "manifest": str(args.output),
                "blocks": manifest["blocks"],
                "subset_sha256": manifest["subset_sha256"],
                "composition": manifest["composition"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
