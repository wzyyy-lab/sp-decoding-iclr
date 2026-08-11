#!/usr/bin/env python3
"""Materialize exact DFlash Top-K lattices from cached Domino hidden states.

This converts ``domino_same_anchor_hidden_v1`` fixed-anchor caches into the
minimal ``gfpr_rollout_v1`` record contract used by the lightweight candidate
head.  Candidate logits are recomputed from the *stored Domino-path hidden
states* in one BF16 vocabulary GEMM, so they do not inherit the forward-shape
drift of a separately collected DFlash cache.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import time
from typing import Any, Iterable

import torch
from torch import Tensor
from torch.nn import functional as F

from materialize_domino_same_anchor import MinimalShardWriter
from train_domino_cached_head import load_tensor_from_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--domino-rollout", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--domino-draft", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--topk", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--shard-blocks", type=int, default=512)
    parser.add_argument("--max-prompts", type=int)
    return parser.parse_args()


def _accepted_length(tokens: Tensor, gold: Tensor) -> int:
    mismatch = tokens.long().ne(gold.long()).nonzero(as_tuple=False)
    return int(mismatch[0, 0]) if mismatch.numel() else int(gold.numel())


def _prompt_balanced(values: dict[str, list[int]]) -> float:
    if not values:
        raise ValueError("prompt-balanced metric requires records")
    return sum(sum(group) / len(group) for group in values.values()) / len(values)


def _batches(values: list[dict[str, Any]], size: int) -> Iterable[list[dict[str, Any]]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def _validate_metadata(metadata: dict[str, Any], args: argparse.Namespace) -> int:
    if metadata.get("format") != "domino_same_anchor_hidden_v1" or not metadata.get(
        "collection_complete", False
    ):
        raise RuntimeError("input is not a complete same-anchor Domino cache")
    expected = {
        "target": args.target.resolve(),
        "domino_draft": args.domino_draft.resolve(),
    }
    for field, expected_path in expected.items():
        stored = metadata.get(field)
        if stored is None or Path(stored).resolve() != expected_path:
            raise ValueError(f"{field}={stored!r} differs from {expected_path}")
    positions = int(metadata.get("draft_positions", 0))
    if not 1 <= positions <= 16:
        raise ValueError(f"unsupported draft_positions={positions}")
    return positions


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("candidate materialization requires CUDA")
    if args.output.exists():
        raise FileExistsError(f"refusing to reuse output {args.output}")
    if args.topk < 2 or args.batch_size < 1 or args.shard_blocks < 1:
        raise ValueError("topk, batch size, and shard size must be positive")
    if args.max_prompts is not None and args.max_prompts < 1:
        raise ValueError("max-prompts must be positive")

    metadata = json.loads(
        (args.domino_rollout / "metadata.json").read_text(encoding="utf-8")
    )
    positions = _validate_metadata(metadata, args)
    work = args.output.with_name(
        f"{args.output.name}.incomplete-{os.environ.get('SLURM_JOB_ID', os.getpid())}"
    )
    if work.exists():
        raise FileExistsError(f"refusing to reuse incomplete output {work}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    work.mkdir()

    target_weight = load_tensor_from_checkpoint(
        args.target, "model.embed_tokens.weight"
    ).to("cuda:0", torch.bfloat16)
    if args.topk > int(target_weight.shape[0]):
        raise ValueError("topk exceeds target vocabulary")

    selected_prompts: set[str] = set()
    writer = MinimalShardWriter(work, args.shard_blocks)
    counts: Counter[str] = Counter()
    accepted_by_prompt: dict[str, list[int]] = defaultdict(list)
    started = time.perf_counter()
    for shard_path in sorted(args.domino_rollout.glob("shard-*.pt")):
        records = torch.load(shard_path, map_location="cpu", weights_only=False)
        if args.max_prompts is not None:
            filtered: list[dict[str, Any]] = []
            for record in records:
                sample_id = str(record["sample_id"])
                if sample_id not in selected_prompts:
                    if len(selected_prompts) >= args.max_prompts:
                        continue
                    selected_prompts.add(sample_id)
                filtered.append(record)
            records = filtered
        else:
            selected_prompts.update(str(record["sample_id"]) for record in records)

        for batch in _batches(records, args.batch_size):
            hidden = torch.stack(
                [record["parallel_hidden"].to(torch.bfloat16) for record in batch]
            )
            if hidden.ndim != 3 or hidden.shape[1] != positions:
                raise ValueError("cached hidden tensor has an inconsistent horizon")
            logits = F.linear(hidden.to("cuda:0", non_blocking=True), target_weight)
            top_logits, top_ids = logits.float().topk(args.topk, dim=-1)
            top_logits = top_logits.to("cpu", torch.float16)
            top_ids = top_ids.to("cpu", torch.int32)
            for index, source in enumerate(batch):
                gold = source["gold_ids"].long()
                policy = source["released_onpolicy_ids"].long()
                if gold.shape != (positions,) or policy.shape != gold.shape:
                    raise ValueError("cached gold/policy horizon is inconsistent")
                accepted = _accepted_length(policy, gold)
                stored_accepted = int(source["released_accepted_length"])
                if accepted != stored_accepted:
                    raise RuntimeError(
                        f"accepted length mismatch for {source['sample_id']}: "
                        f"{accepted} != {stored_accepted}"
                    )
                sample_id = str(source["sample_id"])
                domain = str(source["domain"])
                split = str(source["split"])
                writer.add(
                    {
                        "sample_id": sample_id,
                        "domain": domain,
                        "source": str(source["source"]),
                        "split": split,
                        "mode": "fixed",
                        "policy_version": "released-v0",
                        "anchor_offset": int(source["anchor_offset"]),
                        "anchor_token_id": int(source["anchor_token_id"]),
                        "gold_ids": gold.to(torch.int32),
                        "parallel_hidden": hidden[index],
                        "base_topk_ids": top_ids[index],
                        "base_topk_logits": top_logits[index],
                        "policy_ids": policy.to(torch.int32),
                        "accepted_length": accepted,
                        "next_anchor_offset": int(source["anchor_offset"])
                        + accepted
                        + 1,
                        "bonus_token_id": -1,
                    }
                )
                counts[f"{domain}/{split}"] += 1
                accepted_by_prompt[sample_id].append(accepted)
            del logits, top_logits, top_ids
        print(
            f"[{shard_path.name}] blocks={writer.total + len(writer.buffer)} "
            f"prompts={len(selected_prompts)}",
            flush=True,
        )
        if args.max_prompts is not None and len(selected_prompts) >= args.max_prompts:
            # Continue only while the current prompt may spill into a later shard.
            # Same-anchor materialization keeps complete prompts contiguous, so the
            # selected prefix is complete at this shard boundary.
            break

    writer.flush()
    output_metadata = {
        "format": "gfpr_rollout_v1",
        "collection_complete": True,
        "mode": "fixed",
        "policy_version": "released-v0",
        "position_zero_scale": 0.0,
        "adaptation": None,
        "source_domino_rollout": str(args.domino_rollout.resolve()),
        "target": str(args.target.resolve()),
        "domino_draft": str(args.domino_draft.resolve()),
        "source_format": metadata["format"],
        "samples": len(accepted_by_prompt),
        "blocks": writer.total,
        "positions": positions,
        "topk": args.topk,
        "counts_by_domain_split": dict(sorted(counts.items())),
        "prompt_balanced_eal_by_split": {
            "train": _prompt_balanced(accepted_by_prompt)
        },
        "seconds": time.perf_counter() - started,
        "shards": writer.shards,
    }
    (work / "metadata.json").write_text(
        json.dumps(output_metadata, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(work, args.output)
    print(json.dumps(output_metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
