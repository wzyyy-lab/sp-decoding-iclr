#!/usr/bin/env python3
"""Materialize one physically isolated split from a canonical collection."""

from __future__ import annotations

import argparse
from collections import defaultdict
import copy
import json
import os
from pathlib import Path
import platform
import time
from typing import Any

import torch

try:
    from materialize_canonical_prompt_subset import (
        SubsetShardWriter,
        atomic_write_json,
        load_verified_shard,
        prompt_set_sha256,
        sha256_file,
        validate_source_metadata,
        write_selected_manifest,
    )
except ModuleNotFoundError:  # Imported as ``scripts.*`` in CPU tests.
    from scripts.materialize_canonical_prompt_subset import (
        SubsetShardWriter,
        atomic_write_json,
        load_verified_shard,
        prompt_set_sha256,
        sha256_file,
        validate_source_metadata,
        write_selected_manifest,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--expected-prompts", type=int, required=True)
    parser.add_argument("--expected-blocks", type=int, required=True)
    parser.add_argument("--shard-blocks", type=int, default=256)
    return parser.parse_args()


def read_source_manifest(path: Path) -> dict[str, dict[str, Any]]:
    """Load a unique prompt manifest indexed by sample id."""

    records: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                sample_id = str(record["sample_id"])
                str(record["domain"])
                str(record["split"])
            except (json.JSONDecodeError, KeyError, TypeError) as error:
                raise RuntimeError(
                    f"malformed manifest {path}:{line_number}"
                ) from error
            if sample_id in records:
                raise RuntimeError(
                    f"duplicate prompt in source manifest: {sample_id}"
                )
            records[sample_id] = record
    if not records:
        raise RuntimeError("source manifest contains no prompts")
    return records


def materialize(args: argparse.Namespace) -> dict[str, Any]:
    if not args.split:
        raise ValueError("--split cannot be empty")
    if min(args.expected_prompts, args.expected_blocks) < 1:
        raise ValueError("expected prompt and block counts must be positive")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(
            f"refusing nonempty output collection: {args.output}"
        )
    args.output.mkdir(parents=True, exist_ok=True)

    metadata_path = args.source / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    validate_source_metadata([args.source], [metadata])
    manifest_path = Path(metadata["manifest"])
    manifest_records = read_source_manifest(manifest_path)
    declared_split_ids = {
        sample_id
        for sample_id, record in manifest_records.items()
        if str(record["split"]) == args.split
    }
    if not declared_split_ids:
        raise RuntimeError(f"source manifest has no split {args.split!r}")

    incomplete = {
        **copy.deepcopy(metadata),
        "collection_complete": False,
        "created_unix": time.time(),
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "hostname": platform.node(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "requested_split": args.split,
    }
    incomplete_path = args.output / "INCOMPLETE.json"
    atomic_write_json(incomplete_path, incomplete)

    writer = SubsetShardWriter(args.output, args.shard_blocks)
    observed_source_ids: set[str] = set()
    observed_source_blocks = 0
    source_split_counts: dict[str, int] = defaultdict(int)
    start = time.perf_counter()
    entries = {
        str(entry["path"]): entry for entry in metadata["shards"]
    }
    for shard in sorted(args.source.glob("shard-*.pt")):
        entry = entries[shard.name]
        records = load_verified_shard(shard, entry)
        if len(records) != int(entry["blocks"]):
            raise RuntimeError(f"source shard block count differs: {shard}")
        observed_source_blocks += len(records)
        for record in records:
            sample_id = str(record["sample_id"])
            observed_source_ids.add(sample_id)
            manifest_record = manifest_records.get(sample_id)
            if manifest_record is None:
                raise RuntimeError(
                    f"source record is absent from manifest: {sample_id}"
                )
            record_domain = str(record["domain"])
            record_split = str(record["split"])
            if record_domain != str(manifest_record["domain"]):
                raise RuntimeError(
                    f"source record domain differs from manifest: {sample_id}"
                )
            if record_split != str(manifest_record["split"]):
                raise RuntimeError(
                    f"source record split differs from manifest: {sample_id}"
                )
            source_split_counts[f"{record_domain}/{record_split}"] += 1
            if record_split == args.split:
                writer.add(record)
    writer.flush()

    if observed_source_blocks != int(metadata["num_blocks"]):
        raise RuntimeError("source total block count differs from metadata")
    if dict(sorted(source_split_counts.items())) != metadata.get(
        "block_counts_by_domain_split"
    ):
        raise RuntimeError("source domain/split counts differ from metadata")
    if not writer.sample_ids <= declared_split_ids:
        raise RuntimeError("materialized prompt set escapes the requested split")
    if len(writer.sample_ids) != args.expected_prompts:
        raise RuntimeError(
            f"expected {args.expected_prompts} materialized prompts, "
            f"found {len(writer.sample_ids)}"
        )
    if writer.total_blocks != args.expected_blocks:
        raise RuntimeError(
            f"expected {args.expected_blocks} materialized blocks, "
            f"found {writer.total_blocks}"
        )

    selected_manifest = args.output / "selected_manifest.jsonl"
    selected_manifest_sha256 = write_selected_manifest(
        [manifest_path], writer.sample_ids, selected_manifest
    )
    selected_prompt_sha256 = prompt_set_sha256(writer.sample_ids)
    provenance = copy.deepcopy(metadata["provenance"])
    provenance["manifest_sha256"] = selected_manifest_sha256
    provenance["split_materialization"] = {
        "script_sha256": sha256_file(Path(__file__)),
        "split": args.split,
        "selected_prompt_set_sha256": selected_prompt_sha256,
        "selected_manifest_sha256": selected_manifest_sha256,
        "source_collection": {
            "path": str(args.source.resolve()),
            "metadata_sha256": sha256_file(metadata_path),
            "num_blocks": observed_source_blocks,
            "num_observed_prompts": len(observed_source_ids),
        },
        "source_manifest": {
            "path": str(manifest_path.resolve()),
            "sha256": sha256_file(manifest_path),
            "declared_split_prompts": len(declared_split_ids),
            "materialized_split_prompts": len(writer.sample_ids),
        },
    }
    final = {
        **copy.deepcopy(metadata),
        "collection_complete": True,
        "created_unix": incomplete["created_unix"],
        "job_id": incomplete["job_id"],
        "hostname": incomplete["hostname"],
        "python": incomplete["python"],
        "torch": incomplete["torch"],
        "device": "cpu_split_materialization",
        "manifest": str(selected_manifest.resolve()),
        "num_manifest_samples": len(writer.sample_ids),
        "num_collected_samples": len(writer.sample_ids),
        "num_blocks": writer.total_blocks,
        "block_counts_by_domain_split": dict(sorted(writer.counts.items())),
        "shards": writer.shards,
        "collection_seconds": time.perf_counter() - start,
        "provenance": provenance,
    }
    atomic_write_json(args.output / "metadata.json", final)
    incomplete_path.unlink()
    print(json.dumps(final, ensure_ascii=False, indent=2), flush=True)
    return final


def main() -> None:
    try:
        materialize(parse_args())
    except Exception as error:
        print(
            json.dumps(
                {"status": "artifact_error", "error": str(error)},
                ensure_ascii=False,
            ),
            flush=True,
        )
        raise SystemExit(2) from error


if __name__ == "__main__":
    main()
