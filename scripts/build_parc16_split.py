#!/usr/bin/env python3
"""Build the frozen prompt-disjoint PARC-16 train/validation/held-out split."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable


DOMAINS = ("chat", "code", "math")
REQUIRED_COUNTS = {
    "train": {"chat": 30_000, "code": 30_000, "math": 30_000},
    "validation": {"chat": 1_667, "code": 1_666, "math": 1_667},
    "heldout": {"chat": 1_666, "code": 1_667, "math": 1_667},
}
CANDIDATE_COUNTS = {
    "train": {domain: 80_000 for domain in DOMAINS},
    "validation": {domain: 5_000 for domain in DOMAINS},
    "heldout": {domain: 5_000 for domain in DOMAINS},
}
SOURCE_PER_DOMAIN = 90_000
EXPECTED_TOTAL = SOURCE_PER_DOMAIN * len(DOMAINS)
DEFAULT_SEED = 20_260_810
DEFAULT_PARTS = 16


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--parts", type=int, default=DEFAULT_PARTS)
    return parser.parse_args()


def read_manifest(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    sample_ids: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                sample_id = str(record["sample_id"])
                domain = str(record["domain"])
                prompt = str(record["prompt"])
            except (json.JSONDecodeError, KeyError, TypeError) as error:
                raise RuntimeError(f"malformed source row {line_number}") from error
            if sample_id in sample_ids:
                raise RuntimeError(f"duplicate sample_id {sample_id}")
            if domain not in DOMAINS:
                raise RuntimeError(f"unsupported domain {domain!r}")
            if not prompt:
                raise RuntimeError(f"empty prompt for {sample_id}")
            sample_ids.add(sample_id)
            records.append(record)
    if len(records) != EXPECTED_TOTAL:
        raise RuntimeError(
            f"PARC split requires exactly {EXPECTED_TOTAL} prompts, found {len(records)}"
        )
    return records


def stable_rank(seed: int, sample_id: str) -> bytes:
    return hashlib.sha256(f"{seed}\0{sample_id}".encode("utf-8")).digest()


def assign_splits(
    records: list[dict[str, Any]], seed: int
) -> dict[str, list[dict[str, Any]]]:
    if len(records) != EXPECTED_TOTAL:
        raise RuntimeError(
            f"PARC split requires exactly {EXPECTED_TOTAL} prompts, found {len(records)}"
        )
    by_domain: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_domain[str(record["domain"])].append(record)
    if set(by_domain) != set(DOMAINS):
        raise RuntimeError("source domains differ from the frozen PARC contract")

    assigned: dict[str, list[dict[str, Any]]] = {
        "train": [],
        "validation": [],
        "heldout": [],
    }
    for domain in DOMAINS:
        ordered = sorted(
            by_domain[domain],
            key=lambda row: (stable_rank(seed, str(row["sample_id"])), str(row["sample_id"])),
        )
        if len(ordered) != SOURCE_PER_DOMAIN:
            raise RuntimeError(
                f"PARC reserve source requires {SOURCE_PER_DOMAIN} {domain} prompts"
            )
        cursor = 0
        for split in ("train", "validation", "heldout"):
            candidate_count = CANDIDATE_COUNTS[split][domain]
            rows = ordered[cursor : cursor + candidate_count]
            cursor += candidate_count
            if len(rows) != candidate_count:
                raise RuntimeError(f"insufficient {split}/{domain} reserve candidates")
            for selection_order, row in enumerate(rows):
                assigned[split].append(
                    {
                        **row,
                        "split": split,
                        "selection_order": selection_order,
                        "required_usable_count": REQUIRED_COUNTS[split][domain],
                        "candidate_count": candidate_count,
                    }
                )
        if cursor != len(ordered):
            raise AssertionError("PARC reserve partition did not consume its domain")

    for split in assigned:
        assigned[split].sort(
            key=lambda row: (str(row["domain"]), stable_rank(seed, str(row["sample_id"])))
        )
    for split, rows in assigned.items():
        expected = sum(CANDIDATE_COUNTS[split].values())
        if len(rows) != expected:
            raise AssertionError(f"{split} candidate cardinality drifted")
    id_sets = {
        split: {str(row["sample_id"]) for row in rows}
        for split, rows in assigned.items()
    }
    if (
        id_sets["train"] & id_sets["validation"]
        or id_sets["train"] & id_sets["heldout"]
        or id_sets["validation"] & id_sets["heldout"]
    ):
        raise AssertionError("PARC prompt splits overlap")
    return assigned


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def partition_quota(total: int, part: int, parts: int) -> int:
    return total // parts + int(part < total % parts)


def materialize(args: argparse.Namespace) -> dict[str, Any]:
    if args.parts != DEFAULT_PARTS:
        raise ValueError(f"formal PARC reserve split requires {DEFAULT_PARTS} parts")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"refusing nonempty output {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    records = read_manifest(args.source)
    assigned = assign_splits(records, args.seed)

    for split, rows in assigned.items():
        write_jsonl(args.output / f"{split}_candidates.jsonl", rows)
    all_rows = [row for split in ("train", "validation", "heldout") for row in assigned[split]]
    write_jsonl(args.output / "all.jsonl", all_rows)

    parts_dir = args.output / "train_validation_parts"
    parts_dir.mkdir()
    parts: list[list[dict[str, Any]]] = [[] for _ in range(args.parts)]
    part_required = [0 for _ in range(args.parts)]
    for split in ("train", "validation"):
        for domain in DOMAINS:
            group = sorted(
                (
                    row
                    for row in assigned[split]
                    if str(row["domain"]) == domain
                ),
                key=lambda row: int(row["selection_order"]),
            )
            for index in range(args.parts):
                local = group[index :: args.parts]
                required = partition_quota(
                    REQUIRED_COUNTS[split][domain], index, args.parts
                )
                if len(local) <= required:
                    raise RuntimeError(
                        f"part {index} has no reserve for {split}/{domain}"
                    )
                part_required[index] += required
                for local_order, row in enumerate(local):
                    parts[index].append(
                        {
                            **row,
                            "part_index": index,
                            "part_selection_order": local_order,
                            "part_required_usable_count": required,
                            "part_candidate_count": len(local),
                        }
                    )
    for rows in parts:
        rows.sort(
            key=lambda row: (
                str(row["split"]),
                str(row["domain"]),
                int(row["part_selection_order"]),
            )
        )
    for index, rows in enumerate(parts):
        write_jsonl(parts_dir / f"part-{index:03d}.jsonl", rows)

    candidate_counts = {
        split: dict(sorted(Counter(str(row["domain"]) for row in rows).items()))
        for split, rows in assigned.items()
    }
    metadata = {
        "format": "parc16_prompt_reserve_split_v2",
        "source": str(args.source.resolve()),
        "seed": args.seed,
        "label_generation_started": False,
        "labels_or_eligibility_generated": False,
        "required_usable_counts": REQUIRED_COUNTS,
        "candidate_counts": {
            split: len(rows) for split, rows in assigned.items()
        },
        "candidate_domain_counts": candidate_counts,
        "parts": [
            {
                "path": f"train_validation_parts/part-{index:03d}.jsonl",
                "candidate_prompts": len(rows),
                "required_usable_prompts": part_required[index],
                "candidate_split_counts": dict(
                    sorted(Counter(str(row["split"]) for row in rows).items())
                ),
            }
            for index, rows in enumerate(parts)
        ],
        "candidate_prompt_overlap": {
            "train_validation": 0,
            "train_heldout": 0,
            "validation_heldout": 0,
        },
    }
    (args.output / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return metadata


def main() -> None:
    print(json.dumps(materialize(parse_args()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
