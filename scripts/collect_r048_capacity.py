#!/usr/bin/env python3
"""Collect deployment-shaped Fast-K64 early states and unsplit target labels."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import time
from typing import Any

import torch
from transformers import AutoModel, AutoModelForCausalLM

from sph.fast_r048 import (
    candidate_union_with_proposal,
    fast_candidate_domino_decode,
    repair_earliest_frontier,
)
from sph.gfpr import accepted_lengths
from sph.r048_layer_split import clone_dynamic_cache, early_decision_prepass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-rollout", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--domino-draft", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--max-prompts", type=int, default=64)
    parser.add_argument("--candidate-topk", type=int, default=64)
    parser.add_argument("--early-layers", type=int, default=4)
    parser.add_argument("--shard-blocks", type=int, default=128)
    return parser.parse_args()


def load_source(root: Path, split: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    if not bool(metadata.get("collection_complete", False)):
        raise RuntimeError("source rollout is incomplete")
    records: list[dict[str, Any]] = []
    for shard in sorted(root.glob("shard-*.pt")):
        records.extend(
            row
            for row in torch.load(shard, map_location="cpu", weights_only=False)
            if str(row["split"]) == split
        )
    if not records:
        raise ValueError(f"no source records for split={split!r}")
    return metadata, records


def select_balanced_prompts(records: list[dict[str, Any]], maximum: int) -> list[str]:
    if maximum < 1:
        raise ValueError("max prompts must be positive")
    prompt_domain: dict[str, str] = {}
    for row in records:
        sample_id = str(row["sample_id"])
        domain = str(row["domain"])
        if sample_id in prompt_domain and prompt_domain[sample_id] != domain:
            raise ValueError("prompt domain changes across blocks")
        prompt_domain[sample_id] = domain
    grouped: dict[str, list[str]] = defaultdict(list)
    for sample_id, domain in sorted(prompt_domain.items()):
        grouped[domain].append(sample_id)
    selected: list[str] = []
    domains = sorted(grouped)
    while len(selected) < min(maximum, len(prompt_domain)):
        changed = False
        for domain in domains:
            if grouped[domain]:
                selected.append(grouped[domain].pop(0))
                changed = True
                if len(selected) == maximum:
                    break
        if not changed:
            break
    return selected


def prompt_balanced(sample_ids: list[str], values: list[int]) -> float:
    grouped: dict[str, list[int]] = defaultdict(list)
    for sample_id, value in zip(sample_ids, values, strict=True):
        grouped[sample_id].append(int(value))
    return sum(sum(group) / len(group) for group in grouped.values()) / len(grouped)


def authoritative_frontier_contract(
    proposal: torch.Tensor,
    verifier_top1: torch.Tensor,
    candidate_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, Any]:
    """Return the clean-verifier frontier, its strict valid prefix, and repair.

    The valid teacher region is contiguous through the first rejected row.  In
    particular it must never be constructed as a pointwise comparison against
    a stored canonical continuation: after the first verifier mismatch, later
    coincidental matches are suffix rows and are not causal supervision.
    """

    if proposal.ndim != 2 or verifier_top1.shape != proposal.shape:
        raise ValueError("proposal and verifier top1 must share [batch, positions]")
    if candidate_ids.shape[:2] != proposal.shape:
        raise ValueError("candidate lattice differs from proposal")
    accepted = accepted_lengths(proposal, verifier_top1)
    positions = torch.arange(proposal.shape[1], device=proposal.device).view(1, -1)
    valid = positions.le(accepted[:, None])
    repair = repair_earliest_frontier(
        proposal,
        verifier_top1,
        candidate_ids=candidate_ids,
    )
    return accepted, valid, repair


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to reuse {args.output}")
    if not torch.cuda.is_available():
        raise RuntimeError("R048 capacity collection requires CUDA")
    source_metadata, all_records = load_source(args.source_rollout, args.split)
    if Path(str(source_metadata["target"])).resolve() != args.target.resolve():
        raise ValueError("source target provenance differs")
    if Path(str(source_metadata["domino_draft"])).resolve() != args.domino_draft.resolve():
        raise ValueError("source Domino provenance differs")
    if str(source_metadata.get("mode")) != "fixed":
        raise ValueError("capacity source must use fixed anchors")
    selected_ids = set(select_balanced_prompts(all_records, args.max_prompts))
    records = [row for row in all_records if str(row["sample_id"]) in selected_ids]
    if len(selected_ids) != args.max_prompts:
        raise ValueError("source has fewer prompts than requested capacity set")

    device = torch.device("cuda:0")
    target = AutoModelForCausalLM.from_pretrained(
        str(args.target),
        local_files_only=True,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map="cuda:0",
    ).eval()
    domino = AutoModel.from_pretrained(
        str(args.domino_draft),
        trust_remote_code=True,
        local_files_only=True,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map="cuda:0",
    ).eval()
    target.requires_grad_(False)
    domino.requires_grad_(False)
    target_weight = target.model.embed_tokens.weight

    work = args.output.with_name(
        f"{args.output.name}.incomplete-{os.environ.get('SLURM_JOB_ID', os.getpid())}"
    )
    work.mkdir(parents=True)
    shard: list[dict[str, Any]] = []
    shard_metadata: list[dict[str, Any]] = []
    collected = 0
    sample_ids: list[str] = []
    baseline_lengths: list[int] = []
    oracle_lengths: list[int] = []
    target_gold_mismatches = 0
    valid_rows = 0
    frontier_rows = 0
    frontier_gold_available = 0
    started = time.perf_counter()

    def flush() -> None:
        nonlocal shard
        if not shard:
            return
        path = work / f"shard-{len(shard_metadata):05d}.pt"
        torch.save(shard, path)
        shard_metadata.append({"path": path.name, "blocks": len(shard)})
        shard = []

    for index, row in enumerate(records, start=1):
        context = row["context_ids_before_anchor"].long().to(device)[None]
        prefix_length = int(context.shape[1])
        prefix = target.model(context, use_cache=True, return_dict=True)
        prefix_cache = prefix.past_key_values
        hidden = row["parallel_hidden"].to(device, torch.bfloat16)[None]
        anchor = torch.tensor(
            [int(row["anchor_token_id"])], dtype=torch.long, device=device
        )
        gold = row["gold_ids"].long().to(device)[None]
        proposal = fast_candidate_domino_decode(
            domino=domino,
            target_weight=target_weight,
            anchors=anchor,
            hidden=hidden,
            candidate_topk=args.candidate_topk,
        )
        support = candidate_union_with_proposal(
            proposal.candidate_ids,
            proposal.token_ids,
            support_size=args.candidate_topk,
        )
        if not torch.equal(support, proposal.candidate_ids):
            raise RuntimeError("Fast-K decoder did not retain its own proposal")
        verifier_ids = torch.cat([anchor[:, None], proposal.token_ids], dim=1)

        decision_cache = clone_dynamic_cache(prefix_cache, config=target.config)
        early_states = early_decision_prepass(
            target=target,
            cache=decision_cache,
            input_ids=verifier_ids[:, :16],
            prefix_length=prefix_length,
            early_layers=args.early_layers,
        )
        teacher_cache = clone_dynamic_cache(prefix_cache, config=target.config)
        verifier = target(
            verifier_ids,
            past_key_values=teacher_cache,
            use_cache=True,
            return_dict=True,
        )
        teacher_logits = verifier.logits[:, :16].float()
        candidate_teacher = teacher_logits.gather(-1, support)
        target_top1 = teacher_logits.argmax(dim=-1)
        accepted, valid, oracle_repair = authoritative_frontier_contract(
            proposal.token_ids,
            target_top1,
            support,
        )
        target_available = support.eq(target_top1.unsqueeze(-1)).any(dim=-1)

        if bool(oracle_repair.repair_available[0]):
            repaired_ids = torch.cat(
                [anchor[:, None], oracle_repair.token_ids], dim=1
            )
            oracle_cache = clone_dynamic_cache(prefix_cache, config=target.config)
            repaired_verifier = target(
                repaired_ids,
                past_key_values=oracle_cache,
                use_cache=True,
                return_dict=True,
            )
            repaired_top1 = repaired_verifier.logits[:, :16].float().argmax(dim=-1)
            oracle_accepted = accepted_lengths(
                oracle_repair.token_ids,
                repaired_top1,
            )
        else:
            repaired_top1 = target_top1
            oracle_accepted = accepted

        target_gold_mismatches += int(target_top1.ne(gold).sum())
        valid_rows += int(valid.sum())
        if int(accepted[0]) < 16:
            frontier_rows += 1
            frontier_gold_available += int(
                target_available[0, int(accepted[0])]
            )
        sample_id = str(row["sample_id"])
        sample_ids.append(sample_id)
        baseline_lengths.append(int(accepted[0]))
        oracle_lengths.append(int(oracle_accepted[0]))
        shard.append(
            {
                "sample_id": sample_id,
                "domain": str(row["domain"]),
                "split": str(row["split"]),
                "anchor_offset": int(row["anchor_offset"]),
                "anchor_token_id": int(row["anchor_token_id"]),
                "canonical_gold_ids": gold[0].cpu().to(torch.int32),
                "proposal_ids": proposal.token_ids[0].cpu().to(torch.int32),
                "candidate_ids": support[0].cpu().to(torch.int32),
                "candidate_scores": proposal.candidate_scores[0].cpu().to(torch.bfloat16),
                "target_candidate_logits": candidate_teacher[0].cpu().to(torch.float32),
                "target_top1_ids": target_top1[0].cpu().to(torch.int32),
                "valid_teacher_mask": valid[0].cpu(),
                "early_states": early_states[0].cpu().to(torch.bfloat16),
                "accepted_length": int(accepted[0]),
                "oracle_accepted_length": int(oracle_accepted[0]),
                "oracle_repaired_ids": oracle_repair.token_ids[0].cpu().to(torch.int32),
                "oracle_target_top1_ids": repaired_top1[0].cpu().to(torch.int32),
                "frontier_target_available": bool(
                    int(accepted[0]) < 16
                    and target_available[0, int(accepted[0])]
                ),
            }
        )
        collected += 1
        if len(shard) >= args.shard_blocks:
            flush()
        if index % 32 == 0 or index == len(records):
            print(f"collected {index}/{len(records)}", flush=True)
    flush()

    baseline_eal = prompt_balanced(sample_ids, baseline_lengths)
    oracle_eal = prompt_balanced(sample_ids, oracle_lengths)
    metadata = {
        "format": "r048_capacity_v2",
        "collection_complete": True,
        "capacity_only": True,
        "source_rollout": str(args.source_rollout.resolve()),
        "target": str(args.target.resolve()),
        "domino_draft": str(args.domino_draft.resolve()),
        "split": args.split,
        "candidate_topk": args.candidate_topk,
        "early_layers": args.early_layers,
        "prompts": len(selected_ids),
        "blocks": collected,
        "baseline_eal_prompt_balanced": baseline_eal,
        "oracle_eal_prompt_balanced": oracle_eal,
        "oracle_gain": oracle_eal - baseline_eal,
        "target_canonical_mismatches_all_rows": target_gold_mismatches,
        "valid_teacher_rows": valid_rows,
        "frontier_rows": frontier_rows,
        "frontier_target_available": frontier_gold_available,
        "frontier_target_coverage": frontier_gold_available / max(frontier_rows, 1),
        "authority": "clean unsplit full target verifier",
        "decision_features": "disposable prefix-cache fork, target layers 0-3, anchor+proposal[:15]",
        "seconds": time.perf_counter() - started,
        "shards": shard_metadata,
    }
    (work / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    work.rename(args.output)
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
