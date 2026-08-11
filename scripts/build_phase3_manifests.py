#!/usr/bin/env python3
"""Build disjoint development and reserved-test manifests for Phase 3.

Training/validation prompts come only from training corpora.  The reserved test
uses benchmark test prompts plus a disjoint held-out ShareGPT slice for chat.
Exact and high-overlap training prompts are removed against the full benchmark
pool, not merely against the selected test subset.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


PROJECT = Path(__file__).resolve().parents[1]
MATH_SUFFIX = (
    "\nPlease reason step by step, and put your final answer within \\boxed{}."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gsm8k-train", type=Path, required=True)
    parser.add_argument("--code-alpaca-train", type=Path, required=True)
    parser.add_argument("--sharegpt-train", type=Path, required=True)
    parser.add_argument("--gsm8k-test", type=Path, required=True)
    parser.add_argument("--math500-test", type=Path, required=True)
    parser.add_argument("--humaneval-test", type=Path, required=True)
    parser.add_argument("--mbpp-test", type=Path, required=True)
    parser.add_argument("--mtbench-test", type=Path, required=True)
    parser.add_argument(
        "--exclude-manifest",
        type=Path,
        action="append",
        default=[],
        help="Previously observed prompt manifest; repeat as needed.",
    )
    parser.add_argument("--math-train", type=int, default=667)
    parser.add_argument("--code-train", type=int, default=667)
    parser.add_argument("--chat-train", type=int, default=666)
    parser.add_argument("--validation-select-per-domain", type=int, default=50)
    parser.add_argument("--validation-gate-per-domain", type=int, default=50)
    parser.add_argument("--test-per-domain", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--overlap-ngram-size", type=int, default=8)
    parser.add_argument("--overlap-threshold", type=float, default=0.5)
    parser.add_argument("--development-output", type=Path, required=True)
    parser.add_argument("--reserved-test-output", type=Path, required=True)
    parser.add_argument("--metadata-output", type=Path, required=True)
    return parser.parse_args()


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


def iter_rows(path: Path) -> Iterable[dict[str, Any]]:
    if path.suffix == ".jsonl":
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)
        return
    if path.suffix == ".parquet":
        import pyarrow.parquet as parquet

        for batch in parquet.ParquetFile(path).iter_batches():
            yield from batch.to_pylist()
        return
    if path.suffix == ".arrow":
        import pyarrow as arrow
        import pyarrow.ipc as ipc

        with arrow.memory_map(str(path), "r") as source:
            for batch in ipc.open_stream(source):
                yield from batch.to_pylist()
        return
    raise ValueError(f"unsupported dataset format: {path}")


def load_rows(path: Path) -> list[dict[str, Any]]:
    return list(iter_rows(path))


def normalize_prompt(prompt: str) -> str:
    normalized = unicodedata.normalize("NFKC", prompt).lower()
    return " ".join(normalized.split())


def prompt_tokens(prompt: str) -> list[str]:
    return re.findall(r"\w+|[^\w\s]", normalize_prompt(prompt))


def prompt_ngrams(prompt: str, size: int) -> frozenset[tuple[str, ...]]:
    tokens = prompt_tokens(prompt)
    if not tokens:
        return frozenset()
    if len(tokens) < size:
        return frozenset({tuple(tokens)})
    return frozenset(
        tuple(tokens[index : index + size])
        for index in range(len(tokens) - size + 1)
    )


class NgramOverlapIndex:
    def __init__(self, prompts: Iterable[str], size: int) -> None:
        if size < 1:
            raise ValueError("ngram size must be positive")
        self.size = size
        self.gram_sets = [prompt_ngrams(prompt, size) for prompt in prompts]
        postings: dict[tuple[str, ...], list[int]] = defaultdict(list)
        for prompt_index, grams in enumerate(self.gram_sets):
            for gram in grams:
                postings[gram].append(prompt_index)
        self.postings = postings

    def maximum_jaccard(self, prompt: str) -> float:
        grams = prompt_ngrams(prompt, self.size)
        if not grams:
            return 0.0
        candidate_indices: set[int] = set()
        for gram in grams:
            candidate_indices.update(self.postings.get(gram, ()))
        maximum = 0.0
        for index in candidate_indices:
            reference = self.gram_sets[index]
            union = len(grams | reference)
            if union:
                maximum = max(maximum, len(grams & reference) / union)
        return maximum


def exact_prompt_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def normalized_prompt_hash(prompt: str) -> str:
    return hashlib.sha256(normalize_prompt(prompt).encode("utf-8")).hexdigest()


def stable_key(seed: int, namespace: str, record: dict[str, Any]) -> str:
    material = f"{seed}|{namespace}|{record['sample_id']}"
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def stable_select(
    records: Iterable[dict[str, Any]],
    count: int,
    *,
    seed: int,
    namespace: str,
) -> list[dict[str, Any]]:
    ordered = sorted(records, key=lambda item: stable_key(seed, namespace, item))
    if len(ordered) < count:
        raise ValueError(
            f"{namespace} needs {count} prompts but only {len(ordered)} remain"
        )
    return ordered[:count]


def make_record(
    *,
    source: str,
    raw_id: Any,
    domain: str,
    prompt: str,
) -> dict[str, Any]:
    prompt = prompt.strip()
    if not prompt:
        raise ValueError(f"empty prompt from {source}")
    identity = str(raw_id) if raw_id is not None else exact_prompt_hash(prompt)[:20]
    return {
        "schema_version": 3,
        "sample_id": f"{source}:{identity}",
        "domain": domain,
        "source": source,
        "prompt": prompt,
        "prompt_sha256": exact_prompt_hash(prompt),
        "normalized_prompt_sha256": normalized_prompt_hash(prompt),
    }


def deduplicate(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    seen_ids: set[str] = set()
    seen_prompts: set[str] = set()
    for record in records:
        prompt_hash = record["normalized_prompt_sha256"]
        if record["sample_id"] in seen_ids or prompt_hash in seen_prompts:
            continue
        output.append(record)
        seen_ids.add(record["sample_id"])
        seen_prompts.add(prompt_hash)
    return output


def first_human_turn(row: dict[str, Any]) -> str | None:
    for turn in row.get("conversations", []):
        # Hugging Face's arrow.json extension is exposed as a JSON string by
        # raw PyArrow and as a dict by datasets.Dataset; support both without
        # requiring the heavier datasets materialization path.
        if isinstance(turn, str):
            turn = json.loads(turn)
        if not isinstance(turn, dict):
            continue
        if turn.get("from") in {"human", "user"}:
            value = turn.get("value")
            if isinstance(value, str) and value.strip():
                return value.strip()
    return None


def build_source_records(args: argparse.Namespace) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    result["gsm8k_train"] = [
        make_record(
            source="gsm8k_train",
            raw_id=None,
            domain="math",
            prompt=row["question"] + MATH_SUFFIX,
        )
        for row in load_rows(args.gsm8k_train)
    ]
    result["code_alpaca_train"] = []
    for row in load_rows(args.code_alpaca_train):
        instruction = str(row.get("instruction", "")).strip()
        extra_input = str(row.get("input", "")).strip()
        prompt = instruction
        if extra_input:
            prompt += f"\n\nAdditional input:\n{extra_input}"
        if prompt:
            result["code_alpaca_train"].append(
                make_record(
                    source="code_alpaca_train",
                    raw_id=None,
                    domain="code",
                    prompt=prompt,
                )
            )
    result["sharegpt_train"] = []
    # Stream the 700+ MiB nested-conversation Arrow file so assistant turns do
    # not all become Python objects at once.
    for row in iter_rows(args.sharegpt_train):
        prompt = first_human_turn(row)
        if prompt is not None:
            result["sharegpt_train"].append(
                make_record(
                    source="sharegpt",
                    raw_id=row.get("id"),
                    domain="chat",
                    prompt=prompt,
                )
            )

    result["gsm8k_test"] = [
        make_record(
            source="gsm8k_test",
            raw_id=None,
            domain="math",
            prompt=row["question"] + MATH_SUFFIX,
        )
        for row in load_rows(args.gsm8k_test)
    ]
    result["math500_test"] = [
        make_record(
            source="math500_test",
            raw_id=row.get("unique_id") or row.get("id"),
            domain="math",
            prompt=row["problem"] + MATH_SUFFIX,
        )
        for row in load_rows(args.math500_test)
    ]
    result["humaneval_test"] = [
        make_record(
            source="humaneval_test",
            raw_id=row.get("task_id"),
            domain="code",
            prompt=(
                "Write a solution to the following problem and make sure that it "
                f"passes the tests:\n```python\n{row['prompt']}\n```"
            ),
        )
        for row in load_rows(args.humaneval_test)
    ]
    result["mbpp_test"] = [
        make_record(
            source="mbpp_test",
            raw_id=row.get("task_id"),
            domain="code",
            prompt=row["prompt"],
        )
        for row in load_rows(args.mbpp_test)
    ]
    result["mtbench_test"] = []
    for row in load_rows(args.mtbench_test):
        prompt_field = row["prompt"]
        prompt = prompt_field[0] if isinstance(prompt_field, list) else prompt_field
        result["mtbench_test"].append(
            make_record(
                source="mtbench_test",
                raw_id=row.get("question_id"),
                domain="chat",
                prompt=prompt,
            )
        )
    return {name: deduplicate(records) for name, records in result.items()}


def read_excluded_hashes(paths: list[Path]) -> set[str]:
    hashes: set[str] = set()
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    hashes.add(normalized_prompt_hash(json.loads(line)["prompt"]))
    return hashes


def remove_observed(
    records: Iterable[dict[str, Any]], observed_hashes: set[str]
) -> list[dict[str, Any]]:
    return [
        record
        for record in records
        if record["normalized_prompt_sha256"] not in observed_hashes
    ]


def decontaminate(
    records: list[dict[str, Any]],
    *,
    forbidden_hashes: set[str],
    overlap_index: NgramOverlapIndex,
    overlap_threshold: float,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    kept = []
    statistics = {
        "input": len(records),
        "too_short_or_long": 0,
        "exact_overlap": 0,
        "ngram_overlap": 0,
        "kept": 0,
    }
    for record in records:
        prompt = record["prompt"]
        if not 16 <= len(prompt) <= 8000:
            statistics["too_short_or_long"] += 1
            continue
        if record["normalized_prompt_sha256"] in forbidden_hashes:
            statistics["exact_overlap"] += 1
            continue
        if overlap_index.maximum_jaccard(prompt) >= overlap_threshold:
            statistics["ngram_overlap"] += 1
            continue
        kept.append(record)
    statistics["kept"] = len(kept)
    return kept, statistics


def assign_split(
    records: Iterable[dict[str, Any]], split: str
) -> list[dict[str, Any]]:
    return [{**record, "split": split} for record in records]


def write_jsonl_atomic(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def count_records(records: Iterable[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for record in records:
        counts[f"{record['domain']}/{record['split']}/{record['source']}"] += 1
    return dict(sorted(counts.items()))


def main() -> None:
    args = parse_args()
    outputs = [
        args.development_output,
        args.reserved_test_output,
        args.metadata_output,
    ]
    if any(path.exists() for path in outputs):
        raise FileExistsError("refusing to overwrite an existing Phase 3 manifest")
    if min(
        args.math_train,
        args.code_train,
        args.chat_train,
        args.validation_select_per_domain,
        args.validation_gate_per_domain,
        args.test_per_domain,
        args.overlap_ngram_size,
    ) < 1:
        raise ValueError("all requested counts and ngram size must be positive")
    if not 0.0 < args.overlap_threshold <= 1.0:
        raise ValueError("overlap threshold must be in (0, 1]")

    provenance = {
        "project_commit": git_revision(PROJECT),
        "project_dirty_at_start": git_is_dirty(PROJECT),
        "builder_sha256": sha256_file(Path(__file__)),
        "input_files": {
            name: {
                "path": str(path.resolve()),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for name, path in {
                "gsm8k_train": args.gsm8k_train,
                "code_alpaca_train": args.code_alpaca_train,
                "sharegpt_train": args.sharegpt_train,
                "gsm8k_test": args.gsm8k_test,
                "math500_test": args.math500_test,
                "humaneval_test": args.humaneval_test,
                "mbpp_test": args.mbpp_test,
                "mtbench_test": args.mtbench_test,
                **{
                    f"exclude_manifest_{index}": path
                    for index, path in enumerate(args.exclude_manifest)
                },
            }.items()
        },
    }
    sources = build_source_records(args)
    observed_hashes = read_excluded_hashes(args.exclude_manifest)

    benchmark_names = [
        "gsm8k_test",
        "math500_test",
        "humaneval_test",
        "mbpp_test",
        "mtbench_test",
    ]
    all_benchmarks = [
        record for name in benchmark_names for record in sources[name]
    ]
    benchmark_index = NgramOverlapIndex(
        [record["prompt"] for record in all_benchmarks],
        args.overlap_ngram_size,
    )
    benchmark_hashes = {
        record["normalized_prompt_sha256"] for record in all_benchmarks
    }

    math_left = args.test_per_domain // 2
    math_right = args.test_per_domain - math_left
    code_left = args.test_per_domain // 2
    code_right = args.test_per_domain - code_left
    reserved_math = stable_select(
        remove_observed(sources["gsm8k_test"], observed_hashes),
        math_left,
        seed=args.seed,
        namespace="reserved/gsm8k",
    ) + stable_select(
        remove_observed(sources["math500_test"], observed_hashes),
        math_right,
        seed=args.seed,
        namespace="reserved/math500",
    )
    reserved_code = stable_select(
        remove_observed(sources["humaneval_test"], observed_hashes),
        code_left,
        seed=args.seed,
        namespace="reserved/humaneval",
    ) + stable_select(
        remove_observed(sources["mbpp_test"], observed_hashes),
        code_right,
        seed=args.seed,
        namespace="reserved/mbpp",
    )
    clean_sharegpt, sharegpt_pre_stats = decontaminate(
        sources["sharegpt_train"],
        forbidden_hashes=benchmark_hashes | observed_hashes,
        overlap_index=benchmark_index,
        overlap_threshold=args.overlap_threshold,
    )
    available_mtbench = remove_observed(sources["mtbench_test"], observed_hashes)
    mtbench_count = min(len(available_mtbench), args.test_per_domain)
    reserved_chat_benchmark = stable_select(
        available_mtbench,
        mtbench_count,
        seed=args.seed,
        namespace="reserved/mtbench",
    )
    heldout_chat_count = args.test_per_domain - mtbench_count
    reserved_chat_heldout = stable_select(
        clean_sharegpt,
        heldout_chat_count,
        seed=args.seed,
        namespace="reserved/sharegpt",
    )
    reserved_chat_hashes = {
        record["normalized_prompt_sha256"] for record in reserved_chat_heldout
    }
    reserved_test = assign_split(
        reserved_math + reserved_code + reserved_chat_benchmark + reserved_chat_heldout,
        "test",
    )

    full_forbidden_records = all_benchmarks + reserved_chat_heldout
    full_overlap_index = NgramOverlapIndex(
        [record["prompt"] for record in full_forbidden_records],
        args.overlap_ngram_size,
    )
    full_forbidden_hashes = (
        benchmark_hashes | observed_hashes | reserved_chat_hashes
    )
    train_candidates: dict[str, list[dict[str, Any]]] = {}
    decontamination: dict[str, dict[str, int]] = {
        "sharegpt_pre_reservation": sharegpt_pre_stats
    }
    for domain, source_name in {
        "math": "gsm8k_train",
        "code": "code_alpaca_train",
        "chat": "sharegpt_train",
    }.items():
        candidates, statistics = decontaminate(
            sources[source_name],
            forbidden_hashes=full_forbidden_hashes,
            overlap_index=full_overlap_index,
            overlap_threshold=args.overlap_threshold,
        )
        train_candidates[domain] = candidates
        decontamination[source_name] = statistics

    requested_train = {
        "math": args.math_train,
        "code": args.code_train,
        "chat": args.chat_train,
    }
    development: list[dict[str, Any]] = []
    for domain in ["math", "code", "chat"]:
        ordered = sorted(
            train_candidates[domain],
            key=lambda item: stable_key(
                args.seed, f"development/{domain}", item
            ),
        )
        needed = (
            args.validation_gate_per_domain
            + args.validation_select_per_domain
            + requested_train[domain]
        )
        if len(ordered) < needed:
            raise ValueError(
                f"{domain} needs {needed} development prompts but only "
                f"{len(ordered)} remain"
            )
        # The gate is reserved before the selection split, and both are
        # reserved before training. Future larger training tiers therefore
        # remain nested without changing either held-out development set.
        validation_gate = ordered[: args.validation_gate_per_domain]
        select_start = args.validation_gate_per_domain
        validation_select = ordered[
            select_start : select_start + args.validation_select_per_domain
        ]
        train_start = select_start + args.validation_select_per_domain
        training = ordered[
            train_start : train_start + requested_train[domain]
        ]
        development.extend(assign_split(training, "train"))
        development.extend(assign_split(validation_select, "validation_select"))
        development.extend(assign_split(validation_gate, "validation_gate"))

    development = sorted(
        development,
        key=lambda item: (item["split"], item["domain"], item["sample_id"]),
    )
    reserved_test = sorted(
        reserved_test,
        key=lambda item: (item["domain"], item["source"], item["sample_id"]),
    )
    development_hashes = {
        record["normalized_prompt_sha256"] for record in development
    }
    test_hashes = {
        record["normalized_prompt_sha256"] for record in reserved_test
    }
    if len(development_hashes) != len(development):
        raise RuntimeError("development manifest contains duplicate prompts")
    if len(test_hashes) != len(reserved_test):
        raise RuntimeError("reserved test manifest contains duplicate prompts")
    if development_hashes & test_hashes:
        raise RuntimeError("development and reserved test prompts overlap")
    all_ids = [record["sample_id"] for record in development + reserved_test]
    if len(all_ids) != len(set(all_ids)):
        raise RuntimeError("sample IDs are not globally unique")

    write_jsonl_atomic(args.development_output, development)
    write_jsonl_atomic(args.reserved_test_output, reserved_test)
    metadata = {
        "schema_version": 3,
        "evidence_tier": "protocol_manifest",
        "formal_test_status": "reserved_unobserved",
        "selection_policy": {
            "seed": args.seed,
            "development_train_counts": requested_train,
            "validation_select_per_domain": args.validation_select_per_domain,
            "validation_gate_per_domain": args.validation_gate_per_domain,
            "test_per_domain": args.test_per_domain,
            "training_sources": {
                "math": "GSM8K train",
                "code": "CodeAlpaca-20K train",
                "chat": "ShareGPT first user turns",
            },
            "reserved_test_sources": {
                "math": "balanced GSM8K test and MATH-500",
                "code": "balanced HumanEval and MBPP test",
                "chat": (
                    "all unobserved MT-Bench prompts up to the requested count, "
                    "then disjoint ShareGPT heldout prompts"
                ),
            },
            "split_unit": "prompt; all future anchors inherit the prompt split",
            "validation_gate_never_used_for_checkpoint_selection": True,
            "validation_gate_reserved_before_selection_and_training": True,
            "learning_curve_training_prefix_is_nested": True,
            "overlap_ngram_size": args.overlap_ngram_size,
            "overlap_jaccard_rejection_threshold": args.overlap_threshold,
            "exact_overlap_normalization": "Unicode NFKC + lowercase + whitespace collapse",
        },
        "counts": {
            "development": count_records(development),
            "reserved_test": count_records(reserved_test),
            "development_total": len(development),
            "reserved_test_total": len(reserved_test),
            "previously_observed_unique_prompts": len(observed_hashes),
        },
        "decontamination": decontamination,
        "outputs": {
            "development": {
                "path": str(args.development_output.resolve()),
                "sha256": sha256_file(args.development_output),
            },
            "reserved_test": {
                "path": str(args.reserved_test_output.resolve()),
                "sha256": sha256_file(args.reserved_test_output),
            },
        },
        "provenance": provenance,
    }
    write_json_atomic(args.metadata_output, metadata)
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
