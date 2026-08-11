#!/usr/bin/env python3
"""Rerank Domino beam paths with a reusable prefix of the target model."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoModel, AutoModelForCausalLM

from evaluate_domino_beam_search import beam_paths
from train_domino_cached_head import (
    acceptance_lengths,
    load_records,
    summarize_lengths,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--canonical", type=Path, required=True)
    parser.add_argument("--source-canonical", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--domino-draft", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", default="validation_select")
    parser.add_argument("--early-layers", type=int, required=True)
    parser.add_argument("--candidate-topk", type=int, default=16)
    parser.add_argument("--beam-width", type=int, default=16)
    parser.add_argument(
        "--target-path-batch-size",
        type=int,
        default=0,
        help="Maximum target paths per forward; 0 evaluates the full beam at once.",
    )
    parser.add_argument("--score-gammas", type=float, nargs="+", default=[0.5, 0.75])
    parser.add_argument(
        "--fusion-weights", type=float, nargs="+", default=[0.25, 0.5, 1.0, 2.0]
    )
    return parser.parse_args()


def load_contexts(
    root: Path,
    split: str,
    required: set[tuple[str, int]],
) -> dict[tuple[str, int], torch.Tensor]:
    contexts: dict[tuple[str, int], torch.Tensor] = {}
    for shard in sorted(root.glob("shard-*.pt")):
        shard_records = torch.load(shard, map_location="cpu", weights_only=False)
        for record in shard_records:
            if record["split"] != split:
                continue
            key = (str(record["sample_id"]), int(record["anchor_offset"]))
            if key in required:
                contexts[key] = record["context_ids_before_anchor"].long().clone()
        del shard_records
    missing = required.difference(contexts)
    if missing:
        preview = sorted(missing)[:3]
        raise KeyError(f"missing {len(missing)} source contexts, first={preview}")
    return contexts


def truncate_target(
    target: torch.nn.Module,
    early_layers: int,
) -> torch.nn.Module:
    layers = target.model.layers
    if not 1 <= early_layers <= len(layers):
        raise ValueError(
            f"early-layers must be within [1, {len(layers)}], got {early_layers}"
        )
    target.model.layers = nn.ModuleList(list(layers[:early_layers]))
    del layers
    gc.collect()
    torch.cuda.empty_cache()
    return target


@torch.inference_mode()
def early_candidate_log_probs(
    *,
    target: torch.nn.Module,
    paths: torch.Tensor,
    context: torch.Tensor,
    anchor: torch.Tensor,
    path_batch_size: int = 0,
) -> torch.Tensor:
    path_count, horizon = paths.shape
    prefix = torch.cat([context.to(paths.device), anchor.reshape(1)])
    batch_size = path_count if path_batch_size == 0 else min(path_batch_size, path_count)
    prediction_chunks: list[torch.Tensor] = []
    for start in range(0, path_count, batch_size):
        path_chunk = paths[start : start + batch_size]
        input_ids = torch.cat(
            [prefix[None].expand(path_chunk.shape[0], -1), path_chunk], dim=-1
        )
        hidden = target.model(
            input_ids=input_ids,
            use_cache=False,
            return_dict=True,
        ).last_hidden_state
        prediction_chunks.append(
            hidden[:, prefix.numel() - 1 : prefix.numel() - 1 + horizon]
        )
    prediction_hidden = torch.cat(prediction_chunks, dim=0)
    target_weight = target.model.embed_tokens.weight
    edge_log_probs: list[torch.Tensor] = []
    for position in range(horizon):
        candidates = torch.unique(paths[:, position], sorted=False)
        logits = F.linear(
            prediction_hidden[:, position], target_weight[candidates]
        ).float()
        local_log_probs = F.log_softmax(logits, dim=-1)
        candidate_index = paths[:, position, None].eq(candidates[None]).long().argmax(
            dim=-1
        )
        edge_log_probs.append(
            local_log_probs.gather(1, candidate_index[:, None]).squeeze(1)
        )
    return torch.stack(edge_log_probs, dim=-1)


def path_scores(
    *,
    domino_edges: torch.Tensor,
    early_edges: torch.Tensor,
    gammas: Sequence[float],
    fusion_weights: Sequence[float],
) -> dict[str, torch.Tensor]:
    horizon = domino_edges.shape[1]
    axis = torch.arange(horizon, device=domino_edges.device)
    scores: dict[str, torch.Tensor] = {}
    for gamma in gammas:
        weights = domino_edges.new_tensor(float(gamma)).pow(axis)
        domino_score = (domino_edges * weights[None]).sum(dim=-1)
        early_score = (early_edges * weights[None]).sum(dim=-1)
        scores[f"domino_gamma_{gamma:g}"] = domino_score
        scores[f"early_gamma_{gamma:g}"] = early_score
        domino_scale = domino_score.std(unbiased=False).clamp_min(1e-5)
        early_scale = early_score.std(unbiased=False).clamp_min(1e-5)
        domino_z = (domino_score - domino_score.mean()) / domino_scale
        early_z = (early_score - early_score.mean()) / early_scale
        for fusion_weight in fusion_weights:
            scores[
                f"fusion_g{gamma:g}_w{fusion_weight:g}"
            ] = domino_z + float(fusion_weight) * early_z
    early_survival = early_edges.cumsum(dim=-1).exp().sum(dim=-1)
    scores["early_survival"] = early_survival
    return scores


def greedy_prefix_path_index(paths: torch.Tensor, edge_scores: torch.Tensor) -> int:
    """Follow the scorer's greedy child while retaining only matching paths."""

    if paths.shape != edge_scores.shape or paths.ndim != 2:
        raise ValueError("paths and edge scores must have matching [paths, horizon] shape")
    active = torch.arange(paths.shape[0], device=paths.device)
    for position in range(paths.shape[1]):
        winner = active[edge_scores[active, position].argmax()]
        chosen_token = paths[winner, position]
        active = active[paths[active, position].eq(chosen_token)]
        if active.numel() == 1:
            return int(active[0])
    return int(active[0])


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    if not torch.cuda.is_available():
        raise RuntimeError("early-exit reranking requires CUDA")
    if any(not 0.0 < gamma <= 1.0 for gamma in args.score_gammas):
        raise ValueError("score gammas must be in (0, 1]")
    if any(weight < 0.0 for weight in args.fusion_weights):
        raise ValueError("fusion weights must be nonnegative")
    if args.target_path_batch_size < 0:
        raise ValueError("target-path-batch-size must be nonnegative")

    records = load_records(args.canonical, args.split, None)
    required = {
        (str(record["sample_id"]), int(record["anchor_offset"]))
        for record in records
    }
    contexts = load_contexts(args.source_canonical, args.split, required)

    target = AutoModelForCausalLM.from_pretrained(
        str(args.target),
        local_files_only=True,
        dtype=torch.bfloat16,
        device_map="cuda:0",
    ).eval()
    truncate_target(target, args.early_layers)
    target.requires_grad_(False)
    target_weight = target.model.embed_tokens.weight
    domino = AutoModel.from_pretrained(
        str(args.domino_draft),
        trust_remote_code=True,
        local_files_only=True,
        dtype=torch.bfloat16,
        device_map="cuda:0",
    ).eval()
    domino.requires_grad_(False)

    sample_ids: list[str] = []
    domains: list[str] = []
    released_lengths: list[int] = []
    selected_lengths: dict[str, list[int]] = {}
    oracle_lengths: list[int] = []
    domino_greedy_mismatch_blocks = 0
    domino_greedy_mismatch_tokens = 0
    horizon = int(records[0]["gold_ids"].numel())
    for index, record in enumerate(records, start=1):
        anchor = torch.tensor(
            [int(record["anchor_token_id"])], device="cuda:0", dtype=torch.long
        )
        gold = record["gold_ids"].to(device="cuda:0", dtype=torch.long)
        hidden = record["parallel_hidden"].to(
            device="cuda:0", dtype=torch.bfloat16
        )[None]
        released = record["released_onpolicy_ids"].to(
            device="cuda:0", dtype=torch.long
        )[None]
        paths, domino_edges, _ = beam_paths(
            domino=domino,
            target_weight=target_weight,
            anchor=anchor,
            hidden=hidden,
            released_ids=released,
            horizon=horizon,
            candidate_topk=args.candidate_topk,
            beam_width=args.beam_width,
            normalization="full_vocab",
            prune_gamma=1.0,
        )
        key = (str(record["sample_id"]), int(record["anchor_offset"]))
        early_edges = early_candidate_log_probs(
            target=target,
            paths=paths,
            context=contexts[key],
            anchor=anchor,
            path_batch_size=args.target_path_batch_size,
        )
        scores = path_scores(
            domino_edges=domino_edges,
            early_edges=early_edges,
            gammas=args.score_gammas,
            fusion_weights=args.fusion_weights,
        )
        lengths = acceptance_lengths(paths, gold[None].expand_as(paths))
        for name, score in scores.items():
            selected_lengths.setdefault(name, []).append(
                int(lengths[score.argmax()])
            )
        selected_lengths.setdefault("early_greedy_prefix", []).append(
            int(lengths[greedy_prefix_path_index(paths, early_edges)])
        )
        domino_greedy_index = greedy_prefix_path_index(paths, domino_edges)
        domino_mismatches = int(paths[domino_greedy_index].ne(released[0]).sum())
        domino_greedy_mismatch_blocks += int(domino_mismatches > 0)
        domino_greedy_mismatch_tokens += domino_mismatches
        selected_lengths.setdefault("domino_greedy_prefix", []).append(
            int(lengths[domino_greedy_index])
        )
        oracle_lengths.append(int(lengths.max()))
        released_lengths.append(int(record["released_accepted_length"]))
        sample_ids.append(key[0])
        domains.append(str(record["domain"]))
        if index % 100 == 0:
            print(f"evaluated {index}/{len(records)} blocks", flush=True)

    methods = {
        "released": released_lengths,
        **selected_lengths,
        "beam_path_oracle": oracle_lengths,
    }
    summaries = {
        name: summarize_lengths(sample_ids, domains, lengths, horizon)
        for name, lengths in methods.items()
    }
    baseline = summaries["released"]["overall"][
        "mean_accepted_draft_tokens_prompt_balanced"
    ]
    compact = {
        name: {
            "eal": summary["overall"][
                "mean_accepted_draft_tokens_prompt_balanced"
            ],
            "delta_vs_domino": summary["overall"][
                "mean_accepted_draft_tokens_prompt_balanced"
            ]
            - baseline,
            "overall": summary["overall"],
            "by_domain": summary["by_domain"],
        }
        for name, summary in summaries.items()
    }
    result: dict[str, Any] = {
        "status": "completed",
        "split": args.split,
        "blocks": len(records),
        "prompts": len(set(sample_ids)),
        "early_target_layers": args.early_layers,
        "target_total_layers": int(target.config.num_hidden_layers),
        "beam_width": args.beam_width,
        "candidate_topk": args.candidate_topk,
        "beam_normalization": "full_vocab",
        "beam_prune_gamma": 1.0,
        "early_score_normalization": "per_position_candidate_union",
        "verification_design": "target_prefix_multi_path_then_single_chain_completion",
        "target_prefix_path_batch": (
            args.beam_width
            if args.target_path_batch_size == 0
            else min(args.beam_width, args.target_path_batch_size)
        ),
        "constraint_change": "uses target-side multi-path compute in the early layers",
        "evaluation_scope": "offline path-selection feasibility; remaining-layer reuse not yet implemented",
        "domino_greedy_numeric_diagnostic": {
            "mismatch_blocks_vs_cached_released": domino_greedy_mismatch_blocks,
            "mismatch_tokens_vs_cached_released": domino_greedy_mismatch_tokens,
            "interpretation": "cross-run BF16 eager-score diagnostic only; cached released is the baseline",
        },
        "methods": compact,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
