#!/usr/bin/env python3
"""Build a deterministic math/code/chat prompt manifest from local assets."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any, Iterable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets-root", type=Path, required=True)
    parser.add_argument("--per-domain", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_parquet(path: Path) -> list[dict[str, Any]]:
    import pyarrow.parquet as parquet

    return parquet.read_table(path).to_pylist()


def stable_id(source: str, raw_id: Any, prompt: str) -> str:
    if raw_id is None:
        raw_id = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]
    return f"{source}:{raw_id}"


def choose_balanced(
    sources: list[tuple[str, list[dict[str, Any]]]],
    total: int,
    rng: random.Random,
) -> list[tuple[str, dict[str, Any]]]:
    shuffled: list[tuple[str, list[dict[str, Any]]]] = []
    for source, rows in sources:
        rows = list(rows)
        rng.shuffle(rows)
        shuffled.append((source, rows))

    selected: list[tuple[str, dict[str, Any]]] = []
    cursors = [0 for _ in shuffled]
    while len(selected) < total:
        made_progress = False
        for index, (source, rows) in enumerate(shuffled):
            if cursors[index] >= len(rows) or len(selected) >= total:
                continue
            selected.append((source, rows[cursors[index]]))
            cursors[index] += 1
            made_progress = True
        if not made_progress:
            break
    return selected


def assign_splits(records: list[dict[str, Any]]) -> None:
    by_domain: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_domain.setdefault(record["domain"], []).append(record)
    for domain_records in by_domain.values():
        count = len(domain_records)
        train_end = int(0.8 * count)
        validation_end = train_end + int(0.1 * count)
        for index, record in enumerate(domain_records):
            if index < train_end:
                record["split"] = "train"
            elif index < validation_end:
                record["split"] = "validation"
            else:
                record["split"] = "test"


def main() -> None:
    args = parse_args()
    if args.per_domain < 1:
        raise ValueError("--per-domain must be positive")
    datasets_root = args.assets_root / "datasets"
    rng = random.Random(args.seed)

    gsm8k = load_parquet(datasets_root / "gsm8k/main/test-00000-of-00001.parquet")
    math500 = load_jsonl(datasets_root / "MATH-500/test.jsonl")
    humaneval = load_parquet(
        datasets_root / "openai_humaneval/openai_humaneval/test-00000-of-00001.parquet"
    )
    mbpp = load_parquet(datasets_root / "mbpp/sanitized/test-00000-of-00001.parquet")
    mt_bench = load_jsonl(datasets_root / "mt_bench_prompts/raw/question.jsonl")

    records: list[dict[str, Any]] = []
    math_rows = choose_balanced(
        [("gsm8k", gsm8k), ("math500", math500)], args.per_domain, rng
    )
    for source, row in math_rows:
        raw_prompt = row["question"] if source == "gsm8k" else row["problem"]
        prompt = (
            raw_prompt
            + "\nPlease reason step by step, and put your final answer within \\boxed{}."
        )
        raw_id = row.get("unique_id") or row.get("id")
        records.append(
            {
                "sample_id": stable_id(source, raw_id, prompt),
                "domain": "math",
                "source": source,
                "prompt": prompt,
            }
        )

    code_rows = choose_balanced(
        [("humaneval", humaneval), ("mbpp", mbpp)], args.per_domain, rng
    )
    for source, row in code_rows:
        if source == "humaneval":
            raw_prompt = row["prompt"]
            prompt = (
                "Write a solution to the following problem and make sure that it "
                f"passes the tests:\n```python\n{raw_prompt}\n```"
            )
            raw_id = row.get("task_id")
        else:
            prompt = row["prompt"]
            raw_id = row.get("task_id")
        records.append(
            {
                "sample_id": stable_id(source, raw_id, prompt),
                "domain": "code",
                "source": source,
                "prompt": prompt,
            }
        )

    chat_rows = choose_balanced([("mt_bench", mt_bench)], args.per_domain, rng)
    for source, row in chat_rows:
        prompt_field = row["prompt"]
        prompt = prompt_field[0] if isinstance(prompt_field, list) else prompt_field
        records.append(
            {
                "sample_id": stable_id(source, row.get("question_id"), prompt),
                "domain": "chat",
                "source": source,
                "prompt": prompt,
            }
        )

    assign_splits(records)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + f".{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    os.replace(temporary, args.output)
    counts: dict[str, int] = {}
    for record in records:
        key = f"{record['domain']}/{record['split']}"
        counts[key] = counts.get(key, 0) + 1
    print(json.dumps({"output": str(args.output), "counts": counts}, indent=2))


if __name__ == "__main__":
    main()
