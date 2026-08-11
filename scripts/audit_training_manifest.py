#!/usr/bin/env python3
"""Independently audit a sharded, decontaminated training manifest.

This deliberately does not import the manifest builder: normalization,
hashing, source/domain checks, sharding, and overlap checks are recomputed
from the serialized artifacts so that a shared builder bug is less likely to
make both generation and validation pass.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import unicodedata
from typing import Any, Iterable


MATH_SOURCES = {
    "open-perfectblend/HuggingFaceH4/orca-math-word-problems-200k",
    "open-perfectblend/meta-math/MetaMathQA",
    "open-perfectblend/microsoft/orca-math-word-problems-200k",
}
CODE_SOURCES = {
    "open-perfectblend/theblackcat102/evol-codealpaca-v1",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--parts-dir", type=Path, required=True)
    parser.add_argument(
        "--exclude-manifest",
        type=Path,
        action="append",
        default=[],
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"{path}:{line_number}: invalid JSON"
                ) from error


def normalize_prompt(prompt: str) -> str:
    text = unicodedata.normalize("NFKC", prompt).lower()
    return " ".join(text.split())


def prompt_hash(prompt: str) -> str:
    return hashlib.sha256(
        normalize_prompt(prompt).encode("utf-8")
    ).hexdigest()


def prompt_tokens(prompt: str) -> list[str]:
    return re.findall(r"\w+|[^\w\s]", normalize_prompt(prompt))


def prompt_ngrams(
    prompt: str, size: int
) -> frozenset[tuple[str, ...]]:
    tokens = prompt_tokens(prompt)
    if len(tokens) < size:
        return frozenset()
    return frozenset(
        tuple(tokens[index : index + size])
        for index in range(len(tokens) - size + 1)
    )


def expected_domain(source: str) -> str:
    if source in MATH_SOURCES:
        return "math"
    if source in CODE_SOURCES:
        return "code"
    return "chat"


def fail_if(condition: bool, message: str) -> None:
    if condition:
        raise AssertionError(message)


def main() -> None:
    args = parse_args()
    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    selection = metadata["selection"]
    minimum_characters = int(selection["minimum_characters"])
    maximum_characters = int(selection["maximum_characters"])
    ngram_size = int(selection["overlap_ngram_size"])
    overlap_threshold = float(selection["overlap_threshold"])

    forbidden_hashes: set[str] = set()
    forbidden_grams: list[frozenset[tuple[str, ...]]] = []
    postings: dict[tuple[str, ...], list[int]] = defaultdict(list)
    for path in args.exclude_manifest:
        for record in iter_jsonl(path):
            prompt = str(record["prompt"]).strip()
            normalized_hash = prompt_hash(prompt)
            forbidden_hashes.add(normalized_hash)
            grams = prompt_ngrams(prompt, ngram_size)
            if grams:
                index = len(forbidden_grams)
                forbidden_grams.append(grams)
                for gram in grams:
                    postings[gram].append(index)

    ids: set[str] = set()
    normalized_hashes: set[str] = set()
    domains: Counter[str] = Counter()
    sources: Counter[str] = Counter()
    records = 0
    maximum_overlap = 0.0
    maximum_overlap_sample: str | None = None
    for record in iter_jsonl(args.manifest):
        records += 1
        prompt = str(record["prompt"])
        stripped = prompt.strip()
        fail_if(prompt != stripped, f"unstripped prompt: record {records}")
        fail_if(
            not minimum_characters <= len(prompt) <= maximum_characters,
            f"prompt length outside policy: record {records}",
        )
        exact_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        normalized_hash = prompt_hash(prompt)
        sample_id = str(record["sample_id"])
        source = str(record["source"])
        domain = str(record["domain"])
        fail_if(
            record.get("prompt_sha256") != exact_hash,
            f"prompt_sha256 mismatch: {sample_id}",
        )
        fail_if(
            record.get("normalized_prompt_sha256") != normalized_hash,
            f"normalized_prompt_sha256 mismatch: {sample_id}",
        )
        fail_if(
            sample_id != f"open-perfectblend:{normalized_hash}",
            f"sample_id mismatch: {sample_id}",
        )
        fail_if(sample_id in ids, f"duplicate sample_id: {sample_id}")
        fail_if(
            normalized_hash in normalized_hashes,
            f"duplicate normalized prompt: {sample_id}",
        )
        fail_if(
            normalized_hash in forbidden_hashes,
            f"exact excluded-manifest overlap: {sample_id}",
        )
        fail_if(
            domain != expected_domain(source),
            f"source/domain mismatch: {source} -> {domain}",
        )
        fail_if(
            record.get("split") != "train",
            f"non-training split: {sample_id}",
        )

        grams = prompt_ngrams(prompt, ngram_size)
        candidate_indices: set[int] = set()
        for gram in grams:
            candidate_indices.update(postings.get(gram, ()))
        for index in candidate_indices:
            reference = forbidden_grams[index]
            overlap = len(grams & reference) / len(grams | reference)
            if overlap > maximum_overlap:
                maximum_overlap = overlap
                maximum_overlap_sample = sample_id
            fail_if(
                overlap >= overlap_threshold,
                f"n-gram excluded-manifest overlap {overlap:.6f}: "
                f"{sample_id}",
            )

        ids.add(sample_id)
        normalized_hashes.add(normalized_hash)
        domains[domain] += 1
        sources[source] += 1

    output_metadata = metadata["output"]
    fail_if(records != output_metadata["records"], "record count mismatch")
    fail_if(
        args.manifest.stat().st_size != output_metadata["bytes"],
        "manifest byte count mismatch",
    )
    fail_if(
        sha256_file(args.manifest) != output_metadata["sha256"],
        "manifest SHA-256 mismatch",
    )
    fail_if(
        dict(domains) != selection["actual_domain_counts"],
        "domain counts disagree with metadata",
    )
    fail_if(
        dict(sorted(sources.items()))
        != metadata["source_selected_counts"],
        "source counts disagree with metadata",
    )

    part_ids: set[str] = set()
    part_summaries: list[dict[str, Any]] = []
    for description in metadata["parts"]:
        index = int(description["index"])
        path = args.parts_dir / f"part-{index:03d}.jsonl"
        fail_if(not path.is_file(), f"missing part: {path}")
        local_ids: set[str] = set()
        local_domains: Counter[str] = Counter()
        for record in iter_jsonl(path):
            sample_id = str(record["sample_id"])
            fail_if(
                sample_id not in ids,
                f"part contains unknown record: {sample_id}",
            )
            fail_if(
                sample_id in part_ids,
                f"record repeated across parts: {sample_id}",
            )
            part_ids.add(sample_id)
            local_ids.add(sample_id)
            local_domains[str(record["domain"])] += 1
        fail_if(
            len(local_ids) != description["records"],
            f"part {index} record count mismatch",
        )
        fail_if(
            dict(sorted(local_domains.items()))
            != description["domain_counts"],
            f"part {index} domain counts mismatch",
        )
        fail_if(
            sha256_file(path) != description["sha256"],
            f"part {index} SHA-256 mismatch",
        )
        part_summaries.append(
            {
                "index": index,
                "records": len(local_ids),
                "domain_counts": dict(sorted(local_domains.items())),
                "sha256": description["sha256"],
            }
        )

    unexpected_parts = sorted(
        path.name
        for path in args.parts_dir.glob("part-*.jsonl")
        if int(path.stem.split("-")[-1])
        not in {int(item["index"]) for item in metadata["parts"]}
    )
    fail_if(bool(unexpected_parts), f"unexpected parts: {unexpected_parts}")
    fail_if(part_ids != ids, "parts are not an exact partition of manifest")

    report = {
        "status": "passed",
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": output_metadata["sha256"],
        "records": records,
        "unique_sample_ids": len(ids),
        "unique_normalized_prompts": len(normalized_hashes),
        "domain_counts": dict(sorted(domains.items())),
        "source_counts": dict(sorted(sources.items())),
        "excluded_normalized_prompts": len(forbidden_hashes),
        "maximum_excluded_ngram_jaccard": maximum_overlap,
        "maximum_overlap_sample": maximum_overlap_sample,
        "parts": part_summaries,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
