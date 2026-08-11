#!/usr/bin/env python3
"""Evaluate single-chain search on a draft-only Domino candidate lattice."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModel

from train_domino_cached_head import (
    CachedDominoDataset,
    acceptance_lengths,
    collate,
    load_records,
    load_tensor_from_checkpoint,
    summarize_lengths,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--canonical", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--domino-draft", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", default="validation_select")
    parser.add_argument("--candidate-topk", type=int, default=16)
    parser.add_argument(
        "--beam-widths", type=int, nargs="+", default=[1, 4, 16, 64, 256]
    )
    parser.add_argument(
        "--score-gammas", type=float, nargs="+", default=[0.25, 0.5, 0.75, 1.0]
    )
    parser.add_argument(
        "--normalization",
        choices=("candidate", "full_vocab"),
        default="candidate",
        help=(
            "Normalize Domino edge probabilities over only the proposal set or "
            "over the complete vocabulary. full_vocab is slower but preserves "
            "the probability model used by released Domino."
        ),
    )
    parser.add_argument(
        "--prune-gamma",
        type=float,
        default=1.0,
        help="Discount applied during beam pruning; 1.0 is ordinary MAP pruning.",
    )
    return parser.parse_args()


@torch.inference_mode()
def beam_paths(
    *,
    domino: torch.nn.Module,
    target_weight: torch.Tensor,
    anchor: torch.Tensor,
    hidden: torch.Tensor,
    released_ids: torch.Tensor,
    horizon: int,
    candidate_topk: int,
    beam_width: int,
    normalization: str = "candidate",
    prune_gamma: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return final paths, per-edge log probabilities, and MAP scores."""

    if anchor.shape != (1,) or hidden.shape[0] != 1:
        raise ValueError("beam decoder currently requires batch size one")
    if released_ids.shape != (1, horizon):
        raise ValueError("released path must have shape [1, horizon]")
    if not 1 <= horizon <= hidden.shape[1]:
        raise ValueError("invalid beam horizon")
    if candidate_topk < 1 or beam_width < 1:
        raise ValueError("candidate_topk and beam_width must be positive")
    if normalization not in {"candidate", "full_vocab"}:
        raise ValueError(f"unsupported normalization: {normalization}")
    if not 0.0 < prune_gamma <= 1.0:
        raise ValueError("prune_gamma must be in (0, 1]")

    base_logits = F.linear(hidden[:, :horizon], target_weight)
    # Search the union of the parallel-backbone Top-K and released Domino's
    # greedy token.  This is prefix-independent, cheap to score for every beam,
    # and has a substantially higher oracle ceiling than corrected Top-K alone.
    first_candidates = torch.unique(
        torch.cat(
            [
                base_logits[0, 0].topk(candidate_topk).indices,
                released_ids[0, :1],
            ]
        ),
        sorted=False,
    )
    if normalization == "full_vocab":
        first_log_probs = F.log_softmax(base_logits[0, 0].float(), dim=-1)[
            first_candidates
        ]
    else:
        first_log_probs = F.log_softmax(
            base_logits[0, 0, first_candidates].float(), dim=-1
        )
    first_count = min(int(first_candidates.numel()), beam_width)
    first_values, first_indices = first_log_probs.topk(first_count)
    released_first = int(first_candidates.eq(released_ids[0, 0]).nonzero()[0, 0])
    if not bool(first_indices.eq(released_first).any()):
        first_indices[-1] = released_first
        first_values[-1] = first_log_probs[released_first]
    first_ids = first_candidates[first_indices]
    paths = first_ids[:, None]
    edge_log_probs = first_values[:, None]
    map_scores = first_values
    search_scores = first_values

    anchor_ids = anchor.expand(first_count)[:, None]
    prefix_ids = torch.cat([anchor_ids, first_ids[:, None]], dim=-1)
    _, state = domino.prefix_gru(F.embedding(prefix_ids, target_weight))

    for position in range(1, horizon):
        active = int(paths.shape[0])
        z_i = hidden[:, position : position + 1].expand(active, -1, -1)
        state_i = state.transpose(0, 1)
        if getattr(domino, "use_bias_norm", False):
            state_i = domino.bias_norm(state_i)
        joined = torch.cat([z_i, state_i], dim=-1)
        projected = domino.embed_proj[1](domino.embed_proj[0](joined))
        candidates = torch.unique(
            torch.cat(
                [
                    base_logits[0, position].topk(candidate_topk).indices,
                    released_ids[0, position : position + 1],
                ]
            ),
            sorted=False,
        )
        if normalization == "full_vocab":
            correction = F.linear(projected[:, 0], domino.embed_proj[2].weight)
            # Match released Domino numerics: base and correction are summed in
            # BF16, then promoted only for the probability normalization.
            corrected_logits = base_logits[0, position][None] + correction
            local_log_probs = F.log_softmax(
                corrected_logits.float(), dim=-1
            )[:, candidates]
        else:
            correction = F.linear(
                projected, domino.embed_proj[2].weight[candidates]
            )
            candidate_logits = (
                base_logits[0, position, candidates][None, None, :] + correction
            )
            local_log_probs = F.log_softmax(
                candidate_logits[:, 0].float(), dim=-1
            )
        local_values = local_log_probs
        local_ids = candidates[None, :].expand(active, -1)
        branch_count = int(candidates.numel())
        expanded_map_scores = map_scores[:, None] + local_values
        position_weight = float(prune_gamma) ** position
        expanded_search_scores = (
            search_scores[:, None] + position_weight * local_values
        )
        keep = min(beam_width, int(expanded_search_scores.numel()))
        search_scores, flat_indices = expanded_search_scores.flatten().topk(keep)
        released_parent_matches = paths.eq(
            released_ids[0, :position][None]
        ).all(dim=-1)
        if not bool(released_parent_matches.any()):
            raise RuntimeError("released prefix disappeared from a protected beam")
        released_parent = int(released_parent_matches.nonzero()[0, 0])
        released_child_matches = candidates.eq(released_ids[0, position])
        if not bool(released_child_matches.any()):
            raise RuntimeError("released token disappeared from the candidate union")
        released_child = int(released_child_matches.nonzero()[0, 0])
        released_flat = released_parent * branch_count + released_child
        if not bool(flat_indices.eq(released_flat).any()):
            flat_indices[-1] = released_flat
            search_scores[-1] = expanded_search_scores[
                released_parent, released_child
            ]
        parent = torch.div(flat_indices, branch_count, rounding_mode="floor")
        child = flat_indices.remainder(branch_count)
        map_scores = expanded_map_scores[parent, child]
        chosen_ids = local_ids[parent, child]
        chosen_log_probs = local_values[parent, child]
        paths = torch.cat([paths[parent], chosen_ids[:, None]], dim=-1)
        edge_log_probs = torch.cat(
            [edge_log_probs[parent], chosen_log_probs[:, None]], dim=-1
        )
        state = state[:, parent, :]
        if position + 1 < horizon:
            _, state = domino.prefix_gru(
                F.embedding(chosen_ids[:, None], target_weight), state
            )
    return paths, edge_log_probs, map_scores


def select_path_indices(
    edge_log_probs: torch.Tensor, map_scores: torch.Tensor, gammas: list[float]
) -> dict[str, int]:
    result = {"map": int(map_scores.argmax())}
    axis = torch.arange(edge_log_probs.shape[1], device=edge_log_probs.device)
    for gamma in gammas:
        weights = torch.pow(
            edge_log_probs.new_tensor(float(gamma)), axis
        )
        score = (edge_log_probs * weights[None]).sum(dim=-1)
        result[f"gamma_{gamma:g}"] = int(score.argmax())
    survival = edge_log_probs.cumsum(dim=-1).exp().sum(dim=-1)
    result["survival"] = int(survival.argmax())
    return result


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    if not torch.cuda.is_available():
        raise RuntimeError("Domino beam evaluation requires CUDA")
    if any(width < 1 for width in args.beam_widths):
        raise ValueError("beam widths must be positive")
    if any(gamma <= 0 for gamma in args.score_gammas):
        raise ValueError("score gammas must be positive")

    records = load_records(args.canonical, args.split, None)
    loader = DataLoader(
        CachedDominoDataset(records),
        batch_size=1,
        shuffle=False,
        pin_memory=True,
        collate_fn=collate,
    )
    target_weight = load_tensor_from_checkpoint(
        args.target, "model.embed_tokens.weight"
    ).to(device="cuda:0", dtype=torch.bfloat16)
    domino = AutoModel.from_pretrained(
        str(args.domino_draft),
        trust_remote_code=True,
        local_files_only=True,
        dtype=torch.bfloat16,
        device_map="cuda:0",
    ).eval()

    sample_ids: list[str] = []
    domains: list[str] = []
    released_lengths: list[int] = []
    by_width: dict[int, dict[str, list[int]]] = {}
    greedy_token_mismatches = 0
    horizon = int(records[0]["gold_ids"].numel())

    for width in sorted(set(args.beam_widths)):
        selection_names = [
            "map",
            *(f"gamma_{gamma:g}" for gamma in args.score_gammas),
            "survival",
        ]
        method_names = [
            *selection_names,
            *(f"{name}_or_released_oracle" for name in selection_names),
            "path_oracle",
        ]
        by_width[width] = {name: [] for name in method_names}

    for batch_index, batch in enumerate(loader, start=1):
        anchor = batch["anchors"].to("cuda:0", non_blocking=True)
        gold = batch["gold"].to("cuda:0", non_blocking=True)
        hidden = batch["hidden"].to("cuda:0", non_blocking=True)
        cached_ids = batch["cached_released_ids"].to("cuda:0", non_blocking=True)
        cached_length = int(batch["cached_released_lengths"][0])
        for width in sorted(by_width):
            paths, edge_log_probs, map_scores = beam_paths(
                domino=domino,
                target_weight=target_weight,
                anchor=anchor,
                hidden=hidden,
                released_ids=cached_ids,
                horizon=horizon,
                candidate_topk=args.candidate_topk,
                beam_width=width,
                normalization=args.normalization,
                prune_gamma=args.prune_gamma,
            )
            if width == 1:
                greedy_token_mismatches += int(paths[0].ne(cached_ids[0]).sum())
            lengths = acceptance_lengths(paths, gold.expand(paths.shape[0], -1))
            selected = select_path_indices(
                edge_log_probs, map_scores, list(args.score_gammas)
            )
            for name, index in selected.items():
                selected_length = int(lengths[index])
                by_width[width][name].append(selected_length)
                by_width[width][f"{name}_or_released_oracle"].append(
                    max(selected_length, cached_length)
                )
            by_width[width]["path_oracle"].append(int(lengths.max()))
        sample_ids.extend(batch["sample_ids"])
        domains.extend(batch["domains"])
        released_lengths.append(cached_length)
        if batch_index % 100 == 0:
            print(f"evaluated {batch_index}/{len(records)} blocks", flush=True)

    released = summarize_lengths(sample_ids, domains, released_lengths, horizon)
    baseline = released["overall"]["mean_accepted_draft_tokens_prompt_balanced"]
    result: dict[str, object] = {
        "status": "completed",
        "split": args.split,
        "blocks": len(records),
        "prompts": len(set(sample_ids)),
        "candidate_topk": args.candidate_topk,
        "candidate_source": "parallel_topk_union_released_token",
        "path_score": (
            "full_vocab_domino_log_probability"
            if args.normalization == "full_vocab"
            else "candidate_truncated_domino_log_probability"
        ),
        "normalization": args.normalization,
        "prune_gamma": args.prune_gamma,
        "released_domino": released,
        "greedy_token_mismatches_if_width1": greedy_token_mismatches,
        "beam_widths": {},
    }
    width_reports: dict[str, object] = {}
    for width, methods in by_width.items():
        method_reports: dict[str, object] = {}
        for name, lengths in methods.items():
            summary = summarize_lengths(sample_ids, domains, lengths, horizon)
            eal = summary["overall"]["mean_accepted_draft_tokens_prompt_balanced"]
            method_reports[name] = {
                "eal": eal,
                "delta_vs_domino": eal - baseline,
                "overall": summary["overall"],
                "by_domain": summary["by_domain"],
            }
        width_reports[str(width)] = method_reports
    result["beam_widths"] = width_reports
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
