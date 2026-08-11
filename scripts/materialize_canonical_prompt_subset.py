#!/usr/bin/env python3
"""Materialize an exact hash-ranked prompt subset from canonical shards."""

from __future__ import annotations

import argparse
from collections import defaultdict
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import re
import time
from typing import Any, Iterable

import torch


COLLECTION_LINE = re.compile(
    r"^\[\d+/\d+\] (?P<sample_id>.+): "
    r"(?P<blocks>\d+) blocks in "
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, action="append", required=True)
    parser.add_argument(
        "--collection-log", type=Path, action="append", required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-prompts", type=int, required=True)
    parser.add_argument("--seed", type=int, default=20_260_730)
    parser.add_argument("--shard-blocks", type=int, default=256)
    parser.add_argument("--expected-source-prompts", type=int)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def prompt_set_sha256(sample_ids: Iterable[str]) -> str:
    return hashlib.sha256(
        "\n".join(sorted(sample_ids)).encode("utf-8")
    ).hexdigest()


def read_manifest_domains(paths: Iterable[Path]) -> dict[str, str]:
    domains: dict[str, str] = {}
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                    sample_id = str(record["sample_id"])
                    domain = str(record["domain"])
                except (json.JSONDecodeError, KeyError, TypeError) as error:
                    raise RuntimeError(
                        f"malformed manifest {path}:{line_number}"
                    ) from error
                previous = domains.setdefault(sample_id, domain)
                if previous != domain:
                    raise RuntimeError(
                        f"prompt {sample_id!r} occurs in multiple domains"
                    )
    if not domains:
        raise RuntimeError("source manifests contain no prompts")
    return domains


def read_collected_prompt_ids(paths: Iterable[Path]) -> set[str]:
    collected: set[str] = set()
    observed_lines = 0
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                match = COLLECTION_LINE.match(line.rstrip("\n"))
                if match is None:
                    continue
                observed_lines += 1
                if int(match.group("blocks")) > 0:
                    collected.add(match.group("sample_id"))
    if observed_lines == 0 or not collected:
        raise RuntimeError("collection logs contain no sample completion lines")
    return collected


def write_selected_manifest(
    sources: Iterable[Path],
    selected_ids: set[str],
    output: Path,
) -> str:
    """Write the exact selected manifest in original part/line order."""

    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    written: set[str] = set()
    with temporary.open("w", encoding="utf-8") as destination:
        for source in sources:
            with source.open(encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    if not line.strip():
                        continue
                    try:
                        record = json.loads(line)
                        sample_id = str(record["sample_id"])
                    except (json.JSONDecodeError, KeyError, TypeError) as error:
                        raise RuntimeError(
                            f"malformed manifest {source}:{line_number}"
                        ) from error
                    if sample_id not in selected_ids:
                        continue
                    if sample_id in written:
                        raise RuntimeError(
                            f"selected prompt is duplicated in manifests: "
                            f"{sample_id}"
                        )
                    destination.write(line.rstrip("\n") + "\n")
                    written.add(sample_id)
    if written != selected_ids:
        missing = selected_ids - written
        raise RuntimeError(
            "selected manifest is incomplete: "
            f"{sorted(missing)[:3]}"
        )
    os.replace(temporary, output)
    return sha256_file(output)


def select_prompt_ids(
    prompt_domains: dict[str, str],
    collected_ids: set[str],
    *,
    max_prompts: int,
    seed: int,
) -> set[str]:
    if max_prompts < 1:
        raise ValueError("max_prompts must be positive")
    missing = collected_ids - prompt_domains.keys()
    if missing:
        raise RuntimeError(
            "collected prompts are absent from manifests: "
            f"{sorted(missing)[:3]}"
        )
    if max_prompts > len(collected_ids):
        raise ValueError(
            f"requested {max_prompts} prompts from {len(collected_ids)}"
        )

    def rank(sample_id: str) -> bytes:
        return hashlib.sha256(
            f"{seed}\0{sample_id}".encode("utf-8")
        ).digest()

    return set(sorted(collected_ids, key=rank)[:max_prompts])


def validate_source_metadata(
    sources: list[Path], metadata: list[dict[str, Any]]
) -> None:
    if len(sources) != len(metadata):
        raise AssertionError("source/metadata cardinality mismatch")
    base = metadata[0]
    keys = (
        "format_version",
        "block_size",
        "draft_positions",
        "top_k",
        "anchors_per_sample",
        "continuation_tokens",
        "attention_implementation",
        "dtype",
        "target_layer_ids",
    )
    for source, candidate in zip(sources, metadata, strict=True):
        if not candidate.get("collection_complete", False):
            raise RuntimeError(f"source collection is incomplete: {source}")
        if (source / "INCOMPLETE.json").exists():
            raise RuntimeError(f"source has an INCOMPLETE marker: {source}")
        for key in keys:
            if candidate.get(key) != base.get(key):
                raise RuntimeError(
                    f"source metadata differs for {key}: {source}"
                )
        for fingerprint in ("target_files", "draft_files"):
            if candidate.get("provenance", {}).get(fingerprint) != base.get(
                "provenance", {}
            ).get(fingerprint):
                raise RuntimeError(
                    f"source checkpoint fingerprint differs: {source}"
                )
        entries = candidate.get("shards")
        if not isinstance(entries, list) or not entries:
            raise RuntimeError(f"source has no shard manifest: {source}")
        expected = [str(entry["path"]) for entry in entries]
        actual = [path.name for path in sorted(source.glob("shard-*.pt"))]
        if actual != expected:
            raise RuntimeError(f"source shard set differs: {source}")
        for entry in entries:
            path = source / str(entry["path"])
            if path.stat().st_size != int(entry["bytes"]):
                raise RuntimeError(f"source shard size differs: {path}")


def load_verified_shard(
    path: Path, entry: dict[str, Any]
) -> list[dict[str, Any]]:
    """Hash and decode one shard from a single filesystem read."""

    payload = path.read_bytes()
    if len(payload) != int(entry["bytes"]):
        raise RuntimeError(f"source shard size differs: {path}")
    observed_sha256 = hashlib.sha256(payload).hexdigest()
    if observed_sha256 != str(entry["sha256"]):
        raise RuntimeError(f"source shard SHA256 differs: {path}")
    try:
        records = torch.load(
            io.BytesIO(payload), map_location="cpu", weights_only=False
        )
    except Exception as error:
        raise RuntimeError(f"cannot decode source shard: {path}") from error
    if not isinstance(records, list):
        raise RuntimeError(f"source shard payload is not a record list: {path}")
    return records


class SubsetShardWriter:
    def __init__(self, output: Path, blocks_per_shard: int) -> None:
        if blocks_per_shard < 1:
            raise ValueError("blocks_per_shard must be positive")
        self.output = output
        self.blocks_per_shard = blocks_per_shard
        self.records: list[dict[str, Any]] = []
        self.shard_index = 0
        self.total_blocks = 0
        self.sample_ids: set[str] = set()
        self.counts: dict[str, int] = defaultdict(int)
        self.shards: list[dict[str, Any]] = []

    def add(self, record: dict[str, Any]) -> None:
        self.records.append(record)
        self.sample_ids.add(str(record["sample_id"]))
        self.counts[f"{record['domain']}/{record['split']}"] += 1
        if len(self.records) >= self.blocks_per_shard:
            self.flush()

    def flush(self) -> None:
        if not self.records:
            return
        path = self.output / f"shard-{self.shard_index:05d}.pt"
        temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
        torch.save(self.records, temporary)
        os.replace(temporary, path)
        self.shards.append(
            {
                "path": path.name,
                "blocks": len(self.records),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
        self.total_blocks += len(self.records)
        print(
            f"wrote {path} ({len(self.records)} blocks; "
            f"{len(self.sample_ids)} prompts seen)",
            flush=True,
        )
        self.records = []
        self.shard_index += 1


def materialize(args: argparse.Namespace) -> dict[str, Any]:
    if len(args.source) != len(args.collection_log):
        raise ValueError("repeat --source and --collection-log equally")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(
            f"refusing nonempty output collection: {args.output}"
        )
    args.output.mkdir(parents=True, exist_ok=True)
    source_metadata = [
        json.loads((source / "metadata.json").read_text(encoding="utf-8"))
        for source in args.source
    ]
    validate_source_metadata(args.source, source_metadata)
    manifests = [Path(metadata["manifest"]) for metadata in source_metadata]
    prompt_domains = read_manifest_domains(manifests)
    collected_ids = read_collected_prompt_ids(args.collection_log)
    declared_collected = sum(
        int(metadata["num_collected_samples"])
        for metadata in source_metadata
    )
    if len(collected_ids) != declared_collected:
        raise RuntimeError(
            "collection-log prompt count differs from source metadata: "
            f"{len(collected_ids)} != {declared_collected}"
        )
    if (
        args.expected_source_prompts is not None
        and len(collected_ids) != args.expected_source_prompts
    ):
        raise RuntimeError(
            f"expected {args.expected_source_prompts} source prompts, "
            f"found {len(collected_ids)}"
        )
    selected_ids = select_prompt_ids(
        prompt_domains,
        collected_ids,
        max_prompts=args.max_prompts,
        seed=args.seed,
    )

    base = copy.deepcopy(source_metadata[0])
    provenance = copy.deepcopy(base["provenance"])
    incomplete = {
        **base,
        "collection_complete": False,
        "created_unix": time.time(),
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "hostname": platform.node(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "num_manifest_samples": len(selected_ids),
    }
    incomplete_path = args.output / "INCOMPLETE.json"
    atomic_write_json(incomplete_path, incomplete)
    selected_manifest = args.output / "selected_manifest.jsonl"
    selected_manifest_sha256 = write_selected_manifest(
        manifests, selected_ids, selected_manifest
    )

    writer = SubsetShardWriter(args.output, args.shard_blocks)
    start = time.perf_counter()
    total_source_blocks = 0
    observed_source_ids: set[str] = set()
    for source, metadata in zip(
        args.source, source_metadata, strict=True
    ):
        declared_by_path = {
            str(entry["path"]): entry
            for entry in metadata["shards"]
        }
        source_blocks = 0
        for shard in sorted(source.glob("shard-*.pt")):
            entry = declared_by_path[shard.name]
            records = load_verified_shard(shard, entry)
            if len(records) != int(entry["blocks"]):
                raise RuntimeError(f"source shard block count differs: {shard}")
            source_blocks += len(records)
            for record in records:
                sample_id = str(record["sample_id"])
                observed_source_ids.add(sample_id)
                if sample_id not in collected_ids:
                    raise RuntimeError(
                        f"source record is absent from collection logs: {sample_id}"
                    )
                if str(record["domain"]) != prompt_domains[sample_id]:
                    raise RuntimeError(
                        f"source record domain differs from manifest: {sample_id}"
                    )
                if sample_id in selected_ids:
                    writer.add(record)
        if source_blocks != int(metadata["num_blocks"]):
            raise RuntimeError(f"source total block count differs: {source}")
        total_source_blocks += source_blocks
        print(
            f"scanned {source}: {source_blocks} blocks; "
            f"selected prompts observed={len(writer.sample_ids)}",
            flush=True,
        )
    writer.flush()
    if observed_source_ids != collected_ids:
        missing = collected_ids - observed_source_ids
        extra = observed_source_ids - collected_ids
        raise RuntimeError(
            "source record prompt set differs from collection logs; "
            f"missing={sorted(missing)[:3]}, extra={sorted(extra)[:3]}"
        )
    if writer.sample_ids != selected_ids:
        missing = selected_ids - writer.sample_ids
        extra = writer.sample_ids - selected_ids
        raise RuntimeError(
            f"materialized prompt set mismatch; missing={sorted(missing)[:3]}, "
            f"extra={sorted(extra)[:3]}"
        )

    provenance["subset_materialization"] = {
        "script_sha256": sha256_file(Path(__file__)),
        "selection": "sha256(seed + NUL + sample_id)",
        "selection_seed": args.seed,
        "selected_prompt_set_sha256": prompt_set_sha256(selected_ids),
        "selected_manifest_sha256": selected_manifest_sha256,
        "source_prompt_count": len(collected_ids),
        "source_block_count": total_source_blocks,
        "source_collections": [
            {
                "path": str(source.resolve()),
                "metadata_sha256": sha256_file(source / "metadata.json"),
                "job_id": metadata.get("job_id"),
            }
            for source, metadata in zip(
                args.source, source_metadata, strict=True
            )
        ],
        "source_collection_logs": [
            {
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
            }
            for path in args.collection_log
        ],
        "source_manifests": [
            {
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
            }
            for path in manifests
        ],
    }
    provenance["manifest_sha256"] = selected_manifest_sha256
    final = {
        **base,
        "collection_complete": True,
        "created_unix": incomplete["created_unix"],
        "job_id": incomplete["job_id"],
        "hostname": incomplete["hostname"],
        "python": incomplete["python"],
        "torch": incomplete["torch"],
        "device": "cpu_subset_materialization",
        "manifest": str(selected_manifest.resolve()),
        "num_manifest_samples": len(selected_ids),
        "num_blocks": writer.total_blocks,
        "num_collected_samples": len(writer.sample_ids),
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
