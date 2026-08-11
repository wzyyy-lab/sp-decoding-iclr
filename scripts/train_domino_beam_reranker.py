#!/usr/bin/env python3
"""Train a draft-only acceptance-aware reranker over Domino beam paths."""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Sequence

import torch
from transformers import AutoModel

from evaluate_domino_beam_search import beam_paths
from sph.domino_beam_reranker import (
    DominoBeamPathReranker,
    DominoBeamSetReranker,
    acceptance_aware_reranker_loss,
    oracle_path_ranking_loss,
)
from sph.fbpf import cosine_warmup_learning_rate
from train_domino_cached_head import (
    acceptance_lengths,
    load_records,
    load_tensor_from_checkpoint,
    summarize_lengths,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-canonical",
        type=Path,
        nargs="+",
        required=True,
        help="One or more disjoint Domino canonical training collections.",
    )
    parser.add_argument("--eval-canonical", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--domino-draft", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--eval-split", default="validation_select")
    parser.add_argument("--max-train-prompts", type=int, default=9_999)
    parser.add_argument("--train-blocks-per-prompt", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--gradient-accumulation-blocks", type=int, default=8)
    parser.add_argument("--candidate-topk", type=int, default=16)
    parser.add_argument("--beam-width", type=int, default=64)
    parser.add_argument(
        "--normalization", choices=("candidate", "full_vocab"), default="candidate"
    )
    parser.add_argument(
        "--reranker-kind", choices=("independent", "set"), default="independent"
    )
    parser.add_argument("--model-dim", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--set-layers", type=int, default=2)
    parser.add_argument(
        "--target-feature-dim",
        type=int,
        default=256,
        help="Number of target embedding dimensions per token; 0 uses all dimensions.",
    )
    parser.add_argument("--base-gamma", type=float, default=0.75)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--warmup-ratio", type=float, default=0.04)
    parser.add_argument("--list-temperature", type=float, default=0.5)
    parser.add_argument("--list-weight", type=float, default=1.0)
    parser.add_argument("--frontier-weight", type=float, default=0.25)
    parser.add_argument("--ranking-regret-weight", type=float, default=0.25)
    parser.add_argument("--eval-every-steps", type=int, default=250)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def prompt_balanced_records(
    records: Sequence[dict[str, Any]],
    *,
    max_prompts: int | None,
    blocks_per_prompt: int,
) -> list[dict[str, Any]]:
    if blocks_per_prompt < 1:
        raise ValueError("train-blocks-per-prompt must be positive")
    groups: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()
    seen_blocks: set[tuple[str, int]] = set()
    for record in records:
        sample_id = str(record["sample_id"])
        block_key = (sample_id, int(record["anchor_offset"]))
        if block_key in seen_blocks:
            continue
        seen_blocks.add(block_key)
        groups.setdefault(sample_id, []).append(record)
    eligible = OrderedDict(
        (sample_id, sample_records)
        for sample_id, sample_records in groups.items()
        if len(sample_records) >= blocks_per_prompt
    )
    skipped = len(groups) - len(eligible)
    if skipped:
        print(
            f"skipped {skipped} prompts with fewer than {blocks_per_prompt} blocks",
            flush=True,
        )
    groups = eligible
    if max_prompts is not None:
        if len(groups) < max_prompts:
            raise ValueError(
                f"requested {max_prompts} prompts but loaded only {len(groups)}"
            )
        groups = OrderedDict(list(groups.items())[:max_prompts])
    selected: list[dict[str, Any]] = []
    for sample_records in groups.values():
        ordered = sorted(sample_records, key=lambda item: int(item["anchor_offset"]))
        if blocks_per_prompt == 1:
            indices = [0]
        else:
            indices = [
                round(index * (len(ordered) - 1) / (blocks_per_prompt - 1))
                for index in range(blocks_per_prompt)
            ]
        selected.extend(ordered[index] for index in indices)
    return selected


def block_tensors(record: dict[str, Any]) -> tuple[torch.Tensor, ...]:
    anchor = torch.tensor(
        [int(record["anchor_token_id"])], dtype=torch.long, device="cuda:0"
    )
    gold = record["gold_ids"].to(device="cuda:0", dtype=torch.long)[None]
    hidden = record["parallel_hidden"].to(
        device="cuda:0", dtype=torch.bfloat16
    )[None]
    released = record["released_onpolicy_ids"].to(
        device="cuda:0", dtype=torch.long
    )[None]
    return anchor, gold, hidden, released


def make_paths(
    *,
    domino: torch.nn.Module,
    target_weight: torch.Tensor,
    record: dict[str, Any],
    candidate_topk: int,
    beam_width: int,
    normalization: str,
) -> tuple[torch.Tensor, ...]:
    anchor, gold, hidden, released = block_tensors(record)
    paths, edge_log_probs, map_scores = beam_paths(
        domino=domino,
        target_weight=target_weight,
        anchor=anchor,
        hidden=hidden,
        released_ids=released,
        horizon=int(gold.shape[1]),
        candidate_topk=candidate_topk,
        beam_width=beam_width,
        normalization=normalization,
    )
    # beam_paths runs under inference_mode; ordinary clones can safely feed a
    # differentiable reranker while the beam itself remains a fixed proposal.
    return (
        paths.clone(),
        edge_log_probs.clone(),
        map_scores.clone(),
        anchor[0],
        gold[0],
        hidden[0],
        released[0],
    )


def reranker_forward(
    *,
    reranker: DominoBeamPathReranker | DominoBeamSetReranker,
    domino: torch.nn.Module,
    target_weight: torch.Tensor,
    paths: torch.Tensor,
    edge_log_probs: torch.Tensor,
    anchor: torch.Tensor,
    hidden: torch.Tensor,
    released: torch.Tensor,
    target_feature_dim: int,
):
    domino_token_features = domino.embed_proj[2].weight[paths].detach()
    target_token_features = target_weight[paths].detach()
    if target_feature_dim > 0:
        target_token_features = target_token_features[..., :target_feature_dim]
    token_features = torch.cat(
        [domino_token_features, target_token_features], dim=-1
    )
    prefix_ids = torch.cat(
        [anchor.expand(paths.shape[0])[:, None], paths[:, :-1]], dim=-1
    )
    causal_states, _ = domino.prefix_gru(
        torch.nn.functional.embedding(prefix_ids, target_weight)
    )
    released_indicator = paths.eq(released[None])
    return reranker(
        hidden=hidden,
        causal_states=causal_states.detach(),
        token_features=token_features,
        edge_log_probs=edge_log_probs,
        released_indicator=released_indicator,
    )


@torch.inference_mode()
def evaluate(
    *,
    reranker: DominoBeamPathReranker | DominoBeamSetReranker,
    domino: torch.nn.Module,
    target_weight: torch.Tensor,
    records: Sequence[dict[str, Any]],
    candidate_topk: int,
    beam_width: int,
    normalization: str,
    target_feature_dim: int,
) -> dict[str, Any]:
    reranker.eval()
    sample_ids: list[str] = []
    domains: list[str] = []
    released_lengths: list[int] = []
    selected_lengths: list[int] = []
    survival_lengths: list[int] = []
    gamma_lengths: list[int] = []
    oracle_lengths: list[int] = []
    horizon = int(records[0]["gold_ids"].numel())
    gamma_weights = torch.pow(
        torch.tensor(0.75, device="cuda:0"),
        torch.arange(horizon, device="cuda:0"),
    )
    for index, record in enumerate(records, start=1):
        paths, edge_log_probs, map_scores, anchor, gold, hidden, released = make_paths(
            domino=domino,
            target_weight=target_weight,
            record=record,
            candidate_topk=candidate_topk,
            beam_width=beam_width,
            normalization=normalization,
        )
        output = reranker_forward(
            reranker=reranker,
            domino=domino,
            target_weight=target_weight,
            paths=paths,
            edge_log_probs=edge_log_probs,
            anchor=anchor,
            hidden=hidden,
            released=released,
            target_feature_dim=target_feature_dim,
        )
        lengths = acceptance_lengths(paths, gold[None].expand_as(paths))
        selected_lengths.append(int(lengths[output.selection_scores.argmax()]))
        base_survival = edge_log_probs.cumsum(dim=-1).exp().sum(dim=-1)
        survival_lengths.append(int(lengths[base_survival.argmax()]))
        gamma_score = (edge_log_probs * gamma_weights[None]).sum(dim=-1)
        gamma_lengths.append(int(lengths[gamma_score.argmax()]))
        oracle_lengths.append(int(lengths.max()))
        released_lengths.append(int(record["released_accepted_length"]))
        sample_ids.append(str(record["sample_id"]))
        domains.append(str(record["domain"]))
        if index % 200 == 0:
            print(f"evaluation {index}/{len(records)}", flush=True)
    methods = {
        "released": released_lengths,
        "beam_survival": survival_lengths,
        "beam_gamma_0.75": gamma_lengths,
        "reranker": selected_lengths,
        "beam_path_oracle": oracle_lengths,
    }
    return {
        name: summarize_lengths(sample_ids, domains, lengths, horizon)
        for name, lengths in methods.items()
    }


def eal(summary: dict[str, Any], method: str) -> float:
    return float(
        summary[method]["overall"]["mean_accepted_draft_tokens_prompt_balanced"]
    )


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to reuse output {args.output}")
    if not torch.cuda.is_available():
        raise RuntimeError("beam-reranker training requires CUDA")
    if args.gradient_accumulation_blocks < 1 or args.eval_every_steps < 1:
        raise ValueError("accumulation and evaluation intervals must be positive")
    args.output.mkdir(parents=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    raw_train: list[dict[str, Any]] = []
    for canonical in args.train_canonical:
        raw_train.extend(load_records(canonical, args.train_split, None))
    train_records = prompt_balanced_records(
        raw_train,
        max_prompts=(None if args.max_train_prompts <= 0 else args.max_train_prompts),
        blocks_per_prompt=args.train_blocks_per_prompt,
    )
    del raw_train
    eval_records = load_records(args.eval_canonical, args.eval_split, None)
    horizon = int(eval_records[0]["gold_ids"].numel())
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
    domino.requires_grad_(False)
    if args.target_feature_dim < 0 or args.target_feature_dim > target_weight.shape[1]:
        raise ValueError("target-feature-dim exceeds the target embedding width")
    target_feature_dim = (
        int(target_weight.shape[1])
        if args.target_feature_dim == 0
        else args.target_feature_dim
    )
    token_feature_size = (
        int(domino.embed_proj[2].weight.shape[1]) + target_feature_dim
    )
    common_model_args = {
        "hidden_size": int(domino.config.hidden_size),
        "causal_state_size": int(domino.gru_hidden_dim),
        "token_feature_size": token_feature_size,
        "horizon": horizon,
        "model_dim": args.model_dim,
        "num_heads": args.num_heads,
    }
    if args.reranker_kind == "set":
        reranker = DominoBeamSetReranker(
            **common_model_args,
            position_layers=args.num_layers,
            set_layers=args.set_layers,
            base_gamma=args.base_gamma,
        ).to("cuda:0")
    else:
        reranker = DominoBeamPathReranker(
            **common_model_args,
            num_layers=args.num_layers,
        ).to("cuda:0")
    optimizer = torch.optim.AdamW(
        reranker.parameters(), lr=args.learning_rate, weight_decay=0.01
    )
    total_steps = args.epochs * math.ceil(
        len(train_records) / args.gradient_accumulation_blocks
    )
    initial = evaluate(
        reranker=reranker,
        domino=domino,
        target_weight=target_weight,
        records=eval_records,
        candidate_topk=args.candidate_topk,
        beam_width=args.beam_width,
        normalization=args.normalization,
        target_feature_dim=target_feature_dim,
    )
    best_eal = eal(initial, "reranker")
    best_step = 0
    best_state = {
        name: tensor.detach().cpu().clone()
        for name, tensor in reranker.state_dict().items()
    }
    checkpoint = args.output / "best_reranker.pt"
    torch.save(
        {
            "model_state_dict": best_state,
            "args": vars(args),
            "best_step": best_step,
            "selected": initial,
        },
        checkpoint,
    )
    history: list[dict[str, Any]] = [
        {"global_step": 0, "validation": initial}
    ]
    print(
        json.dumps(
            {
                "initial_eal": best_eal,
                "released_eal": eal(initial, "released"),
                "beam_oracle_eal": eal(initial, "beam_path_oracle"),
            },
            indent=2,
        ),
        flush=True,
    )

    generator = random.Random(args.seed)
    global_step = 0
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        order = list(range(len(train_records)))
        generator.shuffle(order)
        reranker.train()
        optimizer.zero_grad(set_to_none=True)
        accum_metrics: dict[str, float] = {}
        for order_position, record_index in enumerate(order):
            micro = order_position % args.gradient_accumulation_blocks
            accumulation_target = min(
                args.gradient_accumulation_blocks,
                len(order) - (order_position - micro),
            )
            record = train_records[record_index]
            paths, edge_log_probs, _, anchor, gold, hidden, released = make_paths(
                domino=domino,
                target_weight=target_weight,
                record=record,
                candidate_topk=args.candidate_topk,
                beam_width=args.beam_width,
                normalization=args.normalization,
            )
            output = reranker_forward(
                reranker=reranker,
                domino=domino,
                target_weight=target_weight,
                paths=paths,
                edge_log_probs=edge_log_probs,
                anchor=anchor,
                hidden=hidden,
                released=released,
                target_feature_dim=target_feature_dim,
            )
            if args.reranker_kind == "set":
                loss, parts = oracle_path_ranking_loss(
                    output,
                    paths,
                    gold,
                    temperature=args.list_temperature,
                    regret_weight=args.ranking_regret_weight,
                )
            else:
                loss, parts = acceptance_aware_reranker_loss(
                    output,
                    paths,
                    gold,
                    list_temperature=args.list_temperature,
                    list_weight=args.list_weight,
                    frontier_weight=args.frontier_weight,
                )
            (loss / accumulation_target).backward()
            for name, value in parts.items():
                accum_metrics[name] = accum_metrics.get(name, 0.0) + float(value)
            if micro + 1 < accumulation_target:
                continue
            lr = cosine_warmup_learning_rate(
                global_step,
                total_steps=total_steps,
                peak=args.learning_rate,
                warmup_ratio=args.warmup_ratio,
            )
            for group in optimizer.param_groups:
                group["lr"] = lr
            grad_norm = float(torch.nn.utils.clip_grad_norm_(reranker.parameters(), 1.0))
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            if global_step % 25 == 0:
                mean_parts = {
                    name: value / accumulation_target
                    for name, value in accum_metrics.items()
                }
                print(
                    f"step={global_step}/{total_steps} loss={float(loss):.6f} "
                    f"grad_norm={grad_norm:.4f} lr={lr:.3e} parts={mean_parts}",
                    flush=True,
                )
            accum_metrics = {}
            should_evaluate = (
                global_step % args.eval_every_steps == 0
                or global_step == total_steps
            )
            if not should_evaluate:
                continue
            validation = evaluate(
                reranker=reranker,
                domino=domino,
                target_weight=target_weight,
                records=eval_records,
                candidate_topk=args.candidate_topk,
                beam_width=args.beam_width,
                normalization=args.normalization,
                target_feature_dim=target_feature_dim,
            )
            current_eal = eal(validation, "reranker")
            history.append(
                {
                    "epoch": epoch,
                    "global_step": global_step,
                    "validation": validation,
                }
            )
            print(
                json.dumps(
                    {
                        "step_validation": global_step,
                        "reranker_eal": current_eal,
                        "delta_vs_released": current_eal
                        - eal(validation, "released"),
                    },
                    indent=2,
                ),
                flush=True,
            )
            if current_eal > best_eal:
                best_eal = current_eal
                best_step = global_step
                best_state = {
                    name: tensor.detach().cpu().clone()
                    for name, tensor in reranker.state_dict().items()
                }
                torch.save(
                    {
                        "model_state_dict": best_state,
                        "args": vars(args),
                        "best_step": best_step,
                        "selected": validation,
                    },
                    checkpoint,
                )
            reranker.train()

    reranker.load_state_dict(best_state, strict=True)
    selected = evaluate(
        reranker=reranker,
        domino=domino,
        target_weight=target_weight,
        records=eval_records,
        candidate_topk=args.candidate_topk,
        beam_width=args.beam_width,
        normalization=args.normalization,
        target_feature_dim=target_feature_dim,
    )
    torch.save(
        {
            "model_state_dict": best_state,
            "args": vars(args),
            "best_step": best_step,
            "selected": selected,
        },
        checkpoint,
    )
    report = {
        "status": "completed",
        "train_prompts": len({str(record["sample_id"]) for record in train_records}),
        "train_blocks": len(train_records),
        "eval_blocks": len(eval_records),
        "beam_width": args.beam_width,
        "candidate_topk": args.candidate_topk,
        "candidate_source": "parallel_topk_union_released_token",
        "proposal_score": (
            "full_vocab_domino_log_probability"
            if args.normalization == "full_vocab"
            else "candidate_truncated_domino_log_probability"
        ),
        "reranker_kind": args.reranker_kind,
        "target_feature_dim": target_feature_dim,
        "trainable_parameters": sum(
            parameter.numel() for parameter in reranker.parameters()
        ),
        "best_step": best_step,
        "selected": selected,
        "selected_eal": eal(selected, "reranker"),
        "selected_delta_vs_released": eal(selected, "reranker")
        - eal(selected, "released"),
        "history": history,
        "seconds": time.perf_counter() - started,
        "checkpoint": str(checkpoint.resolve()),
    }
    (args.output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
