#!/usr/bin/env python3
"""Evaluate fixed-depth iterative DFlash refinement before Domino rollout."""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModel, AutoModelForCausalLM

from collect_canonical_blocks import extract_context_feature
from diagnose_domino_bias_scale import (
    accepted_length,
    domino_scaled_onpolicy_ids,
)
from materialize_domino_same_anchor import (
    load_records,
    prompt_balanced_mean,
    validate_domino_contract,
)
from sph.data import validate_stored_canonical_contexts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--canonical", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--domino-draft", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", default="validation_select")
    parser.add_argument("--passes", nargs="+", type=int, default=[1, 2, 3, 4])
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--attn-implementation", default="sdpa")
    return parser.parse_args()


@torch.inference_mode()
def parallel_pass(
    *,
    draft: Any,
    target: Any,
    target_hidden: torch.Tensor,
    context_length: int,
    anchor: torch.Tensor,
    previous: torch.Tensor | None,
    horizon: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    block_ids = torch.full(
        (1, int(draft.block_size)),
        int(draft.mask_token_id),
        dtype=torch.long,
        device=target.device,
    )
    block_ids[0, 0] = anchor
    if previous is not None:
        if previous.shape != (horizon,):
            raise ValueError("previous proposal has the wrong horizon")
        block_ids[0, 1 : 1 + horizon] = previous
    position_ids = torch.arange(
        context_length + int(draft.block_size), device=target.device
    ).unsqueeze(0)
    hidden = draft(
        target_hidden=target_hidden[:, :context_length],
        noise_embedding=target.model.embed_tokens(block_ids),
        position_ids=position_ids,
        past_key_values=None,
        use_cache=False,
        is_causal=False,
    )[:, :horizon]
    logits = target.lm_head(hidden)
    return hidden, logits


def summarize(
    records: list[dict[str, Any]], keys: list[str], horizon: int
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key in keys:
        by_prompt: dict[str, list[int]] = defaultdict(list)
        by_domain: dict[str, list[int]] = defaultdict(list)
        values: list[int] = []
        for record in records:
            value = int(record[key])
            values.append(value)
            by_prompt[str(record["sample_id"])].append(value)
            by_domain[str(record["domain"])].append(value)
        result[key] = {
            "blocks": len(values),
            "mean_accepted_draft_tokens_round_weighted": sum(values) / len(values),
            "mean_accepted_draft_tokens_prompt_balanced": prompt_balanced_mean(
                by_prompt
            ),
            "full_horizon_acceptance": sum(value == horizon for value in values)
            / len(values),
            "by_domain_round_weighted": {
                domain: sum(domain_values) / len(domain_values)
                for domain, domain_values in sorted(by_domain.items())
            },
        }
    return result


def paired_prompt_bootstrap(
    records: list[dict[str, Any]],
    left: str,
    right: str,
    *,
    draws: int,
    seed: int,
) -> dict[str, float]:
    by_prompt: dict[str, list[float]] = defaultdict(list)
    for record in records:
        by_prompt[str(record["sample_id"])].append(
            float(record[left]) - float(record[right])
        )
    prompt_deltas = [sum(values) / len(values) for values in by_prompt.values()]
    point = sum(prompt_deltas) / len(prompt_deltas)
    rng = random.Random(seed)
    estimates = sorted(
        sum(rng.choice(prompt_deltas) for _ in prompt_deltas) / len(prompt_deltas)
        for _ in range(draws)
    )
    return {
        "point": point,
        "ci95_low": estimates[int(0.025 * (draws - 1))],
        "ci95_high": estimates[int(0.975 * (draws - 1))],
    }


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    if not args.passes or min(args.passes) < 1:
        raise ValueError("passes must contain positive integers")
    if not torch.cuda.is_available():
        raise RuntimeError("iterative refinement evaluation requires CUDA")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    metadata, source_records = load_records(args.canonical, {args.split})
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in source_records:
        grouped[str(record["sample_id"])].append(record)
    sample_ids = sorted(grouped)
    if args.max_samples is not None:
        sample_ids = sample_ids[: args.max_samples]

    torch.cuda.set_device(0)
    target = AutoModelForCausalLM.from_pretrained(
        str(args.target),
        local_files_only=True,
        dtype=torch.bfloat16,
        attn_implementation=args.attn_implementation,
        device_map="cuda:0",
    ).eval()
    draft = AutoModel.from_pretrained(
        str(args.domino_draft),
        trust_remote_code=True,
        local_files_only=True,
        dtype=torch.bfloat16,
        attn_implementation=args.attn_implementation,
        device_map="cuda:0",
    ).eval()
    horizon = int(metadata["draft_positions"])
    validate_domino_contract(draft, horizon)
    max_passes = max(args.passes)
    block_results: list[dict[str, Any]] = []
    started = time.perf_counter()

    for sample_index, sample_id in enumerate(sample_ids):
        records = sorted(grouped[sample_id], key=lambda item: int(item["anchor_offset"]))
        longest_context = validate_stored_canonical_contexts(records, sample_id)
        target_output = target.model(
            longest_context.unsqueeze(0).to(target.device),
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
        )
        context_features = extract_context_feature(
            target_output.hidden_states, list(draft.target_layer_ids)
        )
        del target_output

        for record in records:
            context_length = int(record["context_ids_before_anchor"].numel())
            anchor = torch.tensor(
                int(record["anchor_token_id"]),
                dtype=torch.long,
                device=target.device,
            )
            gold = record["gold_ids"].long().to(target.device)[:horizon]
            previous: torch.Tensor | None = None
            result: dict[str, Any] = {
                "sample_id": sample_id,
                "domain": str(record["domain"]),
                "anchor_offset": int(record["anchor_offset"]),
            }
            converged_at: int | None = None
            for pass_index in range(1, max_passes + 1):
                hidden, base_logits = parallel_pass(
                    draft=draft,
                    target=target,
                    target_hidden=context_features,
                    context_length=context_length,
                    anchor=anchor,
                    previous=previous,
                    horizon=horizon,
                )
                proposal = domino_scaled_onpolicy_ids(
                    draft,
                    target,
                    anchor,
                    hidden,
                    base_logits,
                    [1.0],
                )[0]
                result[f"pass_{pass_index}"] = accepted_length(proposal, gold)
                if previous is not None and torch.equal(proposal, previous):
                    converged_at = pass_index
                previous = proposal
            result["converged_at"] = converged_at
            block_results.append(result)
        print(
            f"[{sample_index + 1}/{len(sample_ids)}] {sample_id}: {len(records)} anchors",
            flush=True,
        )

    keys = [f"pass_{value}" for value in sorted(set(args.passes))]
    summary = summarize(block_results, keys, horizon)
    baseline_key = "pass_1"
    paired = {
        key: paired_prompt_bootstrap(
            block_results,
            key,
            baseline_key,
            draws=args.bootstrap_samples,
            seed=args.seed + 1009 * index,
        )
        for index, key in enumerate(keys)
        if key != baseline_key
    }
    report = {
        "status": "completed",
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "split": args.split,
        "samples": len(sample_ids),
        "blocks": len(block_results),
        "horizon": horizon,
        "summary": summary,
        "paired_vs_pass_1": paired,
        "converged_blocks_by_pass": {
            str(pass_index): sum(
                record["converged_at"] == pass_index for record in block_results
            )
            for pass_index in range(2, max_passes + 1)
        },
        "seconds": time.perf_counter() - started,
        "block_results": block_results,
    }
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: report[key] for key in ["summary", "paired_vs_pass_1"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
