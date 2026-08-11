#!/usr/bin/env python3
"""Build a decontaminated, sharded Open-PerfectBlend training manifest."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import heapq
import json
import os
from pathlib import Path
import subprocess
from typing import Any, Iterable

import pyarrow.parquet as parquet

try:
    from scripts.build_phase3_manifests import (
        NgramOverlapIndex,
        first_human_turn,
        normalized_prompt_hash,
        prompt_tokens,
        sha256_file,
    )
except ModuleNotFoundError:
    from build_phase3_manifests import (
        NgramOverlapIndex,
        first_human_turn,
        normalized_prompt_hash,
        prompt_tokens,
        sha256_file,
    )


PROJECT = Path(__file__).resolve().parents[1]
MATH_SOURCES = {
    "HuggingFaceH4/orca-math-word-problems-200k",
    "meta-math/MetaMathQA",
    "microsoft/orca-math-word-problems-200k",
}
CODE_SOURCES = {
    "theblackcat102/evol-codealpaca-v1",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument(
        "--exclude-manifest",
        type=Path,
        action="append",
        default=[],
    )
    parser.add_argument("--math-count", type=int, default=33334)
    parser.add_argument("--code-count", type=int, default=33333)
    parser.add_argument("--chat-count", type=int, default=33333)
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--minimum-characters", type=int, default=16)
    parser.add_argument("--maximum-characters", type=int, default=8000)
    parser.add_argument("--overlap-ngram-size", type=int, default=8)
    parser.add_argument("--overlap-threshold", type=float, default=0.5)
    parser.add_argument("--parts", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--parts-dir", type=Path, required=True)
    parser.add_argument("--metadata-output", type=Path, required=True)
    return parser.parse_args()


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
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_write_jsonl(
    path: Path, records: Iterable[dict[str, Any]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def source_domain(source: str) -> str:
    if source in MATH_SOURCES:
        return "math"
    if source in CODE_SOURCES:
        return "code"
    return "chat"


def stable_rank(seed: int, namespace: str, material: str) -> int:
    digest = hashlib.sha256(
        f"{seed}\0{namespace}\0{material}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest, "big")


def read_forbidden_prompts(
    paths: list[Path],
) -> tuple[set[str], list[str]]:
    hashes: set[str] = set()
    prompts: list[str] = []
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                prompt = str(json.loads(line)["prompt"]).strip()
                if not prompt:
                    continue
                prompt_hash = normalized_prompt_hash(prompt)
                if prompt_hash not in hashes:
                    hashes.add(prompt_hash)
                    prompts.append(prompt)
    return hashes, prompts


def parquet_rows(paths: list[Path]) -> Iterable[dict[str, Any]]:
    for path in paths:
        file = parquet.ParquetFile(path)
        for batch in file.iter_batches(
            columns=["conversations", "source"],
            batch_size=4096,
        ):
            yield from batch.to_pylist()


def make_record(
    *,
    prompt: str,
    source: str,
    prompt_hash: str,
) -> dict[str, Any]:
    return {
        "schema_version": 4,
        "sample_id": f"open-perfectblend:{prompt_hash}",
        "domain": source_domain(source),
        "source": f"open-perfectblend/{source}",
        "prompt": prompt,
        "prompt_sha256": hashlib.sha256(
            prompt.encode("utf-8")
        ).hexdigest(),
        "normalized_prompt_sha256": prompt_hash,
        "split": "train",
    }


def main() -> None:
    args = parse_args()
    outputs = [args.output, args.metadata_output]
    if any(path.exists() for path in outputs):
        raise FileExistsError("refusing to overwrite an existing manifest")
    if args.parts_dir.exists() and any(args.parts_dir.iterdir()):
        raise FileExistsError("refusing to mix with existing manifest parts")
    requested = {
        "math": args.math_count,
        "code": args.code_count,
        "chat": args.chat_count,
    }
    if min(*requested.values(), args.parts) < 1:
        raise ValueError("counts and --parts must be positive")
    if not 0.0 < args.overlap_threshold <= 1.0:
        raise ValueError("--overlap-threshold must be in (0, 1]")
    if not 0 < args.minimum_characters <= args.maximum_characters:
        raise ValueError("invalid prompt character bounds")

    data_paths = sorted((args.data_dir / "data").glob("train-*.parquet"))
    if not data_paths:
        raise FileNotFoundError(
            f"no train parquet shards below {args.data_dir}"
        )
    forbidden_hashes, forbidden_prompts = read_forbidden_prompts(
        args.exclude_manifest
    )
    overlap_index = NgramOverlapIndex(
        forbidden_prompts, args.overlap_ngram_size
    )

    # Each heap retains the smallest deterministic hashes.  The root is the
    # currently worst retained row because ranks are stored negated.
    heaps: dict[
        str, list[tuple[int, str, dict[str, Any]]]
    ] = {domain: [] for domain in requested}
    seen_prompt_hashes: set[str] = set()
    statistics: dict[str, int] = defaultdict(int)
    source_input_counts: dict[str, int] = defaultdict(int)
    source_selected_counts: dict[str, int] = defaultdict(int)
    for row in parquet_rows(data_paths):
        statistics["input_rows"] += 1
        source = str(row.get("source", "unknown"))
        source_input_counts[source] += 1
        prompt = first_human_turn(row)
        if prompt is None:
            statistics["missing_human_prompt"] += 1
            continue
        prompt = prompt.strip()
        if not args.minimum_characters <= len(prompt) <= args.maximum_characters:
            statistics["too_short_or_long"] += 1
            continue
        prompt_hash = normalized_prompt_hash(prompt)
        if prompt_hash in seen_prompt_hashes:
            statistics["duplicate_prompt"] += 1
            continue
        seen_prompt_hashes.add(prompt_hash)
        if prompt_hash in forbidden_hashes:
            statistics["exact_forbidden_overlap"] += 1
            continue
        # Avoid spending n-gram work on prompts that cannot share even one
        # full forbidden n-gram.
        if len(prompt_tokens(prompt)) >= args.overlap_ngram_size:
            if (
                overlap_index.maximum_jaccard(prompt)
                >= args.overlap_threshold
            ):
                statistics["ngram_forbidden_overlap"] += 1
                continue
        domain = source_domain(source)
        record = make_record(
            prompt=prompt,
            source=source,
            prompt_hash=prompt_hash,
        )
        rank = stable_rank(args.seed, f"select/{domain}", prompt_hash)
        entry = (-rank, record["sample_id"], record)
        heap = heaps[domain]
        if len(heap) < requested[domain]:
            heapq.heappush(heap, entry)
        elif rank < -heap[0][0]:
            heapq.heapreplace(heap, entry)
        statistics["eligible_rows"] += 1

    selected: list[dict[str, Any]] = []
    for domain, count in requested.items():
        if len(heaps[domain]) != count:
            raise RuntimeError(
                f"{domain} requested {count} prompts but retained "
                f"{len(heaps[domain])}"
            )
        domain_records = [entry[2] for entry in heaps[domain]]
        selected.extend(domain_records)
        for record in domain_records:
            source_selected_counts[str(record["source"])] += 1
    selected.sort(
        key=lambda record: stable_rank(
            args.seed, "manifest-order", str(record["sample_id"])
        )
    )
    if len(
        {record["normalized_prompt_sha256"] for record in selected}
    ) != len(selected):
        raise AssertionError("selected manifest contains duplicate prompts")

    atomic_write_jsonl(args.output, selected)
    args.parts_dir.mkdir(parents=True, exist_ok=True)
    part_records = [selected[index:: args.parts] for index in range(args.parts)]
    part_descriptions = []
    for index, records in enumerate(part_records):
        part_path = args.parts_dir / f"part-{index:03d}.jsonl"
        atomic_write_jsonl(part_path, records)
        part_descriptions.append(
            {
                "index": index,
                "path": str(part_path.resolve()),
                "records": len(records),
                "domain_counts": dict(
                    sorted(
                        {
                            domain: sum(
                                record["domain"] == domain
                                for record in records
                            )
                            for domain in requested
                        }.items()
                    )
                ),
                "sha256": sha256_file(part_path),
            }
        )

    metadata = {
        "schema_version": 4,
        "evidence_tier": "training_manifest",
        "dataset": "mlabonne/open-perfectblend",
        "selection": {
            "seed": args.seed,
            "requested_domain_counts": requested,
            "actual_domain_counts": {
                domain: sum(
                    record["domain"] == domain for record in selected
                )
                for domain in requested
            },
            "source_domain_policy": {
                "math": sorted(MATH_SOURCES),
                "code": sorted(CODE_SOURCES),
                "chat": "all remaining Open-PerfectBlend sources",
            },
            "minimum_characters": args.minimum_characters,
            "maximum_characters": args.maximum_characters,
            "overlap_ngram_size": args.overlap_ngram_size,
            "overlap_threshold": args.overlap_threshold,
            "split": "train_only",
        },
        "statistics": dict(sorted(statistics.items())),
        "source_input_counts": dict(sorted(source_input_counts.items())),
        "source_selected_counts": dict(
            sorted(source_selected_counts.items())
        ),
        "output": {
            "path": str(args.output.resolve()),
            "records": len(selected),
            "bytes": args.output.stat().st_size,
            "sha256": sha256_file(args.output),
        },
        "parts": part_descriptions,
        "provenance": {
            "project_commit": git_revision(PROJECT),
            "project_dirty": git_is_dirty(PROJECT),
            "builder_sha256": sha256_file(Path(__file__)),
            "phase3_builder_sha256": sha256_file(
                PROJECT / "scripts" / "build_phase3_manifests.py"
            ),
            "data_files": [
                {
                    "path": str(path.resolve()),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
                for path in data_paths
            ],
            "excluded_manifests": [
                {
                    "path": str(path.resolve()),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
                for path in args.exclude_manifest
            ],
        },
    }
    atomic_write_json(args.metadata_output, metadata)
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
