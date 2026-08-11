#!/usr/bin/env python3
"""Run a fixed prompt-disjoint R056 workload against one SGLang server."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics
import time
from typing import Any
from urllib import request

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-rollout", type=Path, required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--mode", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", default="validation_select")
    parser.add_argument("--max-prompts", type=int)
    parser.add_argument(
        "--sample-ids",
        nargs="+",
        help="Diagnostic-only exact prompt IDs; preserves the supplied order.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--timeout", type=float, default=300.0)
    return parser.parse_args()


def load_prompts(root: Path, split: str) -> list[dict[str, Any]]:
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    if not bool(metadata.get("collection_complete", False)):
        raise RuntimeError("source rollout is incomplete")
    first: dict[str, dict[str, Any]] = {}
    for shard in sorted(root.glob("shard-*.pt")):
        for row in torch.load(shard, map_location="cpu", weights_only=False):
            if str(row["split"]) != split:
                continue
            sample_id = str(row["sample_id"])
            current = first.get(sample_id)
            if current is None or int(row["anchor_offset"]) < int(
                current["anchor_offset"]
            ):
                first[sample_id] = row
    prompts = [
        {
            "sample_id": sample_id,
            "domain": str(row["domain"]),
            "anchor_offset": int(row["anchor_offset"]),
            "input_ids": row["context_ids_before_anchor"].long().tolist(),
        }
        for sample_id, row in first.items()
    ]
    prompts.sort(key=lambda row: (row["domain"], row["sample_id"]))
    if not prompts:
        raise ValueError(f"no prompts for split={split!r}")
    if any(row["anchor_offset"] != 0 for row in prompts):
        raise RuntimeError("R056 requires the original prompt at anchor offset zero")
    return prompts


def balanced_subset(prompts: list[dict[str, Any]], maximum: int | None):
    if maximum is None or maximum >= len(prompts):
        return prompts
    if maximum < 1:
        raise ValueError("max-prompts must be positive")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for prompt in prompts:
        grouped[prompt["domain"]].append(prompt)
    domains = sorted(grouped)
    selected: list[dict[str, Any]] = []
    cursor = 0
    while len(selected) < maximum:
        changed = False
        for domain in domains:
            if cursor < len(grouped[domain]):
                selected.append(grouped[domain][cursor])
                changed = True
                if len(selected) == maximum:
                    break
        if not changed:
            break
        cursor += 1
    return selected


def post_json(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    encoded = json.dumps(payload).encode("utf-8")
    req = request.Request(
        url,
        data=encoded,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read())


def main() -> None:
    args = parse_args()
    all_prompts = load_prompts(args.source_rollout, args.split)
    if args.sample_ids:
        if args.max_prompts is not None:
            raise ValueError("--sample-ids and --max-prompts are mutually exclusive")
        by_id = {str(prompt["sample_id"]): prompt for prompt in all_prompts}
        missing = [sample_id for sample_id in args.sample_ids if sample_id not in by_id]
        if missing:
            raise ValueError(f"unknown sample IDs: {missing}")
        prompts = [by_id[sample_id] for sample_id in args.sample_ids]
    else:
        prompts = balanced_subset(all_prompts, args.max_prompts)
    # Server startup already runs its own health-generation warmup.  This separate
    # non-benchmark prompt exercises the exact requested output length and kernels.
    post_json(
        f"{args.base_url.rstrip('/')}/generate",
        {
            "text": "R056 benchmark warmup: continue with deterministic tokens.",
            "rid": f"r056-warmup-{args.mode}",
            "sampling_params": {
                "temperature": 0.0,
                "max_new_tokens": args.max_new_tokens,
                "ignore_eos": True,
            },
            "log_metrics": False,
        },
        args.timeout,
    )

    records: list[dict[str, Any]] = []
    run_start = time.perf_counter()
    for index, prompt in enumerate(prompts):
        payload = {
            "input_ids": prompt["input_ids"],
            "rid": prompt["sample_id"],
            "sampling_params": {
                "temperature": 0.0,
                "max_new_tokens": args.max_new_tokens,
                "ignore_eos": True,
            },
        }
        start = time.perf_counter()
        response = post_json(
            f"{args.base_url.rstrip('/')}/generate", payload, args.timeout
        )
        client_seconds = time.perf_counter() - start
        meta = response.get("meta_info", {})
        output_ids = [int(value) for value in response.get("output_ids", [])]
        if len(output_ids) != args.max_new_tokens:
            raise RuntimeError(
                f"{prompt['sample_id']} returned {len(output_ids)} tokens, "
                f"expected {args.max_new_tokens}"
            )
        records.append(
            {
                "sample_id": prompt["sample_id"],
                "domain": prompt["domain"],
                "prompt_tokens": len(prompt["input_ids"]),
                "output_ids": output_ids,
                "client_seconds": client_seconds,
                "server_e2e_seconds": float(meta["e2e_latency"]),
                "completion_tokens": int(meta["completion_tokens"]),
                "cached_tokens": int(meta.get("cached_tokens", 0)),
                "spec_verify_ct": int(meta.get("spec_verify_ct", 0)),
                "spec_accept_token_num": int(meta.get("spec_accept_token_num", 0)),
                "spec_draft_token_num": int(meta.get("spec_draft_token_num", 0)),
                "spec_accept_length": (
                    None
                    if meta.get("spec_accept_length") is None
                    else float(meta["spec_accept_length"])
                ),
            }
        )
        if (index + 1) % 10 == 0 or index + 1 == len(prompts):
            print(
                f"R056 {args.mode}: {index + 1}/{len(prompts)} prompts",
                flush=True,
            )

    wall_seconds = time.perf_counter() - run_start
    total_tokens = sum(row["completion_tokens"] for row in records)
    total_server = sum(row["server_e2e_seconds"] for row in records)
    total_verify = sum(row["spec_verify_ct"] for row in records)
    total_accepted = sum(row["spec_accept_token_num"] for row in records)
    report = {
        "format": "r056_sglang_run_v1",
        "mode": args.mode,
        "source_rollout": str(args.source_rollout.resolve()),
        "split": args.split,
        "num_prompts": len(records),
        "max_new_tokens": args.max_new_tokens,
        "wall_seconds": wall_seconds,
        "total_completion_tokens": total_tokens,
        "sum_server_e2e_seconds": total_server,
        "aggregate_server_tokens_per_second": total_tokens / total_server,
        "aggregate_client_tokens_per_second": total_tokens / wall_seconds,
        "median_server_e2e_seconds": statistics.median(
            row["server_e2e_seconds"] for row in records
        ),
        "total_spec_verify_ct": total_verify,
        "dynamic_accepted_draft_eal": (
            None if total_verify == 0 else total_accepted / total_verify
        ),
        "dynamic_output_advance": (
            None if total_verify == 0 else total_tokens / total_verify
        ),
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "records"}, indent=2))


if __name__ == "__main__":
    main()
