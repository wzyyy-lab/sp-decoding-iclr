#!/usr/bin/env python3
"""Train a candidate-level tuned exit head on an intermediate target layer."""

from __future__ import annotations

import argparse
import gc
import json
import math
import random
import time
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoModel, AutoModelForCausalLM

from evaluate_domino_beam_search import beam_paths
from evaluate_domino_early_exit_reranker import greedy_prefix_path_index
from sph.fbpf import cosine_warmup_learning_rate
from sph.target_tuned_exit import TargetTunedExitHead
from train_domino_beam_reranker import prompt_balanced_records
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
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--eval-split", default="validation_select")
    parser.add_argument("--early-layers", type=int, required=True)
    parser.add_argument("--rank", type=int, default=512)
    parser.add_argument("--candidate-topk", type=int, default=16)
    parser.add_argument("--beam-width", type=int, default=16)
    parser.add_argument("--max-train-prompts", type=int, default=1985)
    parser.add_argument("--train-blocks-per-prompt", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--gradient-accumulation-blocks", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--warmup-ratio", type=float, default=0.04)
    parser.add_argument("--position-discount", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def load_source_features(
    root: Path,
    splits: set[str],
    required: set[tuple[str, int]],
) -> dict[tuple[str, int], dict[str, torch.Tensor]]:
    result: dict[tuple[str, int], dict[str, torch.Tensor]] = {}
    for shard in sorted(root.glob("shard-*.pt")):
        shard_records = torch.load(shard, map_location="cpu", weights_only=False)
        for record in shard_records:
            if str(record["split"]) not in splits:
                continue
            key = (str(record["sample_id"]), int(record["anchor_offset"]))
            if key not in required:
                continue
            result[key] = {
                "context": record["context_ids_before_anchor"].long().clone(),
            }
        del shard_records
    missing = required.difference(result)
    if missing:
        raise KeyError(f"missing {len(missing)} source records, first={sorted(missing)[:3]}")
    return result


def truncate_target_for_tuned_exit(
    target: torch.nn.Module,
    early_layers: int,
) -> tuple[torch.Tensor, float]:
    layers = target.model.layers
    if not 1 <= early_layers <= len(layers):
        raise ValueError("invalid early target layer count")
    target.model.layers = nn.ModuleList(list(layers[:early_layers]))
    final_norm = target.model.norm
    norm_weight = final_norm.weight.detach().float().clone()
    norm_epsilon = float(final_norm.variance_epsilon)
    target.model.norm = nn.Identity()
    del layers, final_norm
    gc.collect()
    torch.cuda.empty_cache()
    return norm_weight, norm_epsilon


@torch.inference_mode()
def clean_prediction_hidden(
    *,
    target: torch.nn.Module,
    context: torch.Tensor,
    anchor: torch.Tensor,
    gold: torch.Tensor,
) -> torch.Tensor:
    prefix = torch.cat([context.to(gold.device), anchor.reshape(1)])
    input_ids = torch.cat([prefix, gold], dim=0)[None]
    hidden = target.model(
        input_ids=input_ids, use_cache=False, return_dict=True
    ).last_hidden_state[0]
    return hidden[prefix.numel() - 1 : prefix.numel() - 1 + gold.numel()].clone()


def candidate_set(
    base_topk_ids: torch.Tensor,
    released: torch.Tensor,
    position: int,
    candidate_topk: int,
) -> torch.Tensor:
    return torch.unique(
        torch.cat(
            [
                base_topk_ids[position, :candidate_topk],
                released[position : position + 1],
            ]
        ),
        sorted=False,
    )


def clean_prefix_loss(
    *,
    head: TargetTunedExitHead,
    hidden: torch.Tensor,
    target_weight: torch.Tensor,
    base_topk_ids: torch.Tensor,
    released: torch.Tensor,
    gold: torch.Tensor,
    candidate_topk: int,
    position_discount: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    losses: list[torch.Tensor] = []
    weights: list[float] = []
    correct = 0
    for position in range(gold.numel()):
        candidates = candidate_set(
            base_topk_ids, released, position, candidate_topk
        )
        label_matches = candidates.eq(gold[position])
        if not bool(label_matches.any()):
            break
        label = label_matches.long().argmax()[None]
        logits = head.score(hidden[position : position + 1], target_weight[candidates])
        losses.append(F.cross_entropy(logits, label))
        weights.append(float(position_discount) ** position)
        correct += int(logits.detach().argmax(dim=-1).eq(label).item())
    if not losses:
        return head.channel_scale.sum() * 0.0, {
            "supervised_positions": 0.0,
            "candidate_accuracy": 0.0,
        }
    weight_tensor = torch.tensor(weights, device=losses[0].device)
    stacked = torch.stack(losses)
    loss = (stacked * weight_tensor).sum() / weight_tensor.sum()
    return loss, {
        "supervised_positions": float(len(losses)),
        "candidate_accuracy": float(correct / len(losses)),
    }


@torch.inference_mode()
def tuned_path_edges(
    *,
    target: torch.nn.Module,
    head: TargetTunedExitHead,
    target_weight: torch.Tensor,
    paths: torch.Tensor,
    context: torch.Tensor,
    anchor: torch.Tensor,
    base_topk_ids: torch.Tensor,
    released: torch.Tensor,
    candidate_topk: int,
) -> torch.Tensor:
    path_count, horizon = paths.shape
    prefix = torch.cat([context.to(paths.device), anchor.reshape(1)])
    input_ids = torch.cat(
        [prefix[None].expand(path_count, -1), paths], dim=-1
    )
    hidden = target.model(
        input_ids=input_ids, use_cache=False, return_dict=True
    ).last_hidden_state
    prediction_hidden = hidden[:, prefix.numel() - 1 : prefix.numel() - 1 + horizon]
    edges: list[torch.Tensor] = []
    for position in range(horizon):
        candidates = candidate_set(
            base_topk_ids, released, position, candidate_topk
        )
        logits = head.score(prediction_hidden[:, position], target_weight[candidates])
        log_probs = F.log_softmax(logits, dim=-1)
        candidate_index = paths[:, position, None].eq(candidates[None]).long().argmax(
            dim=-1
        )
        edges.append(log_probs.gather(1, candidate_index[:, None]).squeeze(1))
    return torch.stack(edges, dim=-1)


def method_scores(
    domino_edges: torch.Tensor,
    tuned_edges: torch.Tensor,
) -> dict[str, torch.Tensor]:
    result: dict[str, torch.Tensor] = {}
    axis = torch.arange(domino_edges.shape[1], device=domino_edges.device)
    for gamma in (0.75, 1.0):
        weights = domino_edges.new_tensor(gamma).pow(axis)
        domino_score = (domino_edges * weights[None]).sum(dim=-1)
        tuned_score = (tuned_edges * weights[None]).sum(dim=-1)
        result[f"tuned_gamma_{gamma:g}"] = tuned_score
        domino_z = (domino_score - domino_score.mean()) / domino_score.std(
            unbiased=False
        ).clamp_min(1e-5)
        tuned_z = (tuned_score - tuned_score.mean()) / tuned_score.std(
            unbiased=False
        ).clamp_min(1e-5)
        for fusion_weight in (0.25, 0.5, 1.0):
            result[f"fusion_g{gamma:g}_w{fusion_weight:g}"] = (
                domino_z + fusion_weight * tuned_z
            )
    return result


@torch.inference_mode()
def evaluate(
    *,
    target: torch.nn.Module,
    domino: torch.nn.Module,
    head: TargetTunedExitHead,
    target_weight: torch.Tensor,
    records: Sequence[dict[str, Any]],
    source_features: dict[tuple[str, int], dict[str, torch.Tensor]],
    candidate_topk: int,
    beam_width: int,
) -> dict[str, Any]:
    head.eval()
    sample_ids: list[str] = []
    domains: list[str] = []
    released_lengths: list[int] = []
    selected_lengths: dict[str, list[int]] = {}
    oracle_lengths: list[int] = []
    horizon = int(records[0]["gold_ids"].numel())
    for index, record in enumerate(records, start=1):
        key = (str(record["sample_id"]), int(record["anchor_offset"]))
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
            candidate_topk=candidate_topk,
            beam_width=beam_width,
            normalization="full_vocab",
            prune_gamma=1.0,
        )
        base_topk_ids = F.linear(hidden[0], target_weight).topk(
            candidate_topk, dim=-1
        ).indices
        tuned_edges = tuned_path_edges(
            target=target,
            head=head,
            target_weight=target_weight,
            paths=paths,
            context=source_features[key]["context"],
            anchor=anchor,
            base_topk_ids=base_topk_ids,
            released=released[0],
            candidate_topk=candidate_topk,
        )
        lengths = acceptance_lengths(paths, gold[None].expand_as(paths))
        selected_lengths.setdefault("tuned_greedy_prefix", []).append(
            int(lengths[greedy_prefix_path_index(paths, tuned_edges)])
        )
        for name, score in method_scores(domino_edges, tuned_edges).items():
            selected_lengths.setdefault(name, []).append(int(lengths[score.argmax()]))
        released_lengths.append(int(record["released_accepted_length"]))
        oracle_lengths.append(int(lengths.max()))
        sample_ids.append(key[0])
        domains.append(str(record["domain"]))
        if index % 200 == 0:
            print(f"evaluation {index}/{len(records)}", flush=True)
    methods = {
        "released": released_lengths,
        **selected_lengths,
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


def best_learned_method(summary: dict[str, Any]) -> tuple[str, float]:
    allowed = [
        name for name in summary if name not in {"released", "beam_path_oracle"}
    ]
    return max(((name, eal(summary, name)) for name in allowed), key=lambda item: item[1])


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to reuse output {args.output}")
    if not torch.cuda.is_available():
        raise RuntimeError("tuned target exit training requires CUDA")
    if not 0.0 < args.position_discount <= 1.0:
        raise ValueError("position-discount must be in (0, 1]")
    args.output.mkdir(parents=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    raw_train = load_records(args.canonical, args.train_split, None)
    train_records = prompt_balanced_records(
        raw_train,
        max_prompts=args.max_train_prompts,
        blocks_per_prompt=args.train_blocks_per_prompt,
    )
    del raw_train
    eval_records = load_records(args.canonical, args.eval_split, None)
    required = {
        (str(record["sample_id"]), int(record["anchor_offset"]))
        for record in [*train_records, *eval_records]
    }
    source_features = load_source_features(
        args.source_canonical,
        {args.train_split, args.eval_split},
        required,
    )

    target = AutoModelForCausalLM.from_pretrained(
        str(args.target),
        local_files_only=True,
        dtype=torch.bfloat16,
        device_map="cuda:0",
    ).eval()
    norm_weight, norm_epsilon = truncate_target_for_tuned_exit(
        target, args.early_layers
    )
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
    head = TargetTunedExitHead(
        hidden_size=int(target.config.hidden_size),
        rank=args.rank,
        final_norm_weight=norm_weight,
        rms_epsilon=norm_epsilon,
    ).to("cuda:0")
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.learning_rate, weight_decay=0.01)
    total_steps = args.epochs * math.ceil(
        len(train_records) / args.gradient_accumulation_blocks
    )

    initial = evaluate(
        target=target,
        domino=domino,
        head=head,
        target_weight=target_weight,
        records=eval_records,
        source_features=source_features,
        candidate_topk=args.candidate_topk,
        beam_width=args.beam_width,
    )
    best_method, best_eal = best_learned_method(initial)
    best_epoch = 0
    best_validation = initial
    best_state = {
        name: tensor.detach().cpu().clone() for name, tensor in head.state_dict().items()
    }
    history: list[dict[str, Any]] = [
        {"epoch": 0, "validation": initial, "best_method": best_method}
    ]
    print(
        json.dumps(
            {
                "initial_best_method": best_method,
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
        head.train()
        optimizer.zero_grad(set_to_none=True)
        metric_positions = 0.0
        metric_correct_mass = 0.0
        for order_position, record_index in enumerate(order):
            record = train_records[record_index]
            key = (str(record["sample_id"]), int(record["anchor_offset"]))
            anchor = torch.tensor(
                [int(record["anchor_token_id"])], device="cuda:0", dtype=torch.long
            )
            gold = record["gold_ids"].to(device="cuda:0", dtype=torch.long)
            released = record["released_onpolicy_ids"].to(
                device="cuda:0", dtype=torch.long
            )
            clean_hidden = clean_prediction_hidden(
                target=target,
                context=source_features[key]["context"],
                anchor=anchor,
                gold=gold,
            ).detach().clone()
            with torch.inference_mode():
                parallel_hidden = record["parallel_hidden"].to(
                    device="cuda:0", dtype=torch.bfloat16
                )
                base_topk_ids = F.linear(parallel_hidden, target_weight).topk(
                    args.candidate_topk, dim=-1
                ).indices
            loss, metrics = clean_prefix_loss(
                head=head,
                hidden=clean_hidden,
                target_weight=target_weight,
                base_topk_ids=base_topk_ids,
                released=released,
                gold=gold,
                candidate_topk=args.candidate_topk,
                position_discount=args.position_discount,
            )
            accumulation_target = min(
                args.gradient_accumulation_blocks,
                len(order) - (
                    order_position - order_position % args.gradient_accumulation_blocks
                ),
            )
            (loss / accumulation_target).backward()
            metric_positions += metrics["supervised_positions"]
            metric_correct_mass += (
                metrics["candidate_accuracy"] * metrics["supervised_positions"]
            )
            if (order_position % args.gradient_accumulation_blocks) + 1 < accumulation_target:
                continue
            lr = cosine_warmup_learning_rate(
                global_step,
                total_steps=total_steps,
                peak=args.learning_rate,
                warmup_ratio=args.warmup_ratio,
            )
            for group in optimizer.param_groups:
                group["lr"] = lr
            grad_norm = float(torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0))
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            if global_step % 50 == 0:
                accuracy = metric_correct_mass / max(metric_positions, 1.0)
                print(
                    f"step={global_step}/{total_steps} loss={float(loss):.6f} "
                    f"candidate_accuracy={accuracy:.4f} grad_norm={grad_norm:.4f} "
                    f"lr={lr:.3e}",
                    flush=True,
                )
                metric_positions = 0.0
                metric_correct_mass = 0.0

        validation = evaluate(
            target=target,
            domino=domino,
            head=head,
            target_weight=target_weight,
            records=eval_records,
            source_features=source_features,
            candidate_topk=args.candidate_topk,
            beam_width=args.beam_width,
        )
        current_method, current_eal = best_learned_method(validation)
        history.append(
            {"epoch": epoch, "validation": validation, "best_method": current_method}
        )
        print(
            json.dumps(
                {
                    "epoch": epoch,
                    "best_method": current_method,
                    "best_eal": current_eal,
                    "delta_vs_released": current_eal - eal(validation, "released"),
                },
                indent=2,
            ),
            flush=True,
        )
        if current_eal > best_eal:
            best_eal = current_eal
            best_method = current_method
            best_epoch = epoch
            best_validation = validation
            best_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in head.state_dict().items()
            }

    head.load_state_dict(best_state, strict=True)
    # Each candidate checkpoint was already evaluated deterministically at the
    # end of its epoch.  Reuse that exact summary instead of spending another
    # full target-path pass solely to reproduce identical numbers.
    selected = best_validation
    checkpoint = args.output / "best_tuned_exit.pt"
    torch.save(
        {
            "model_state_dict": best_state,
            "args": vars(args),
            "best_epoch": best_epoch,
            "best_method": best_method,
            "selected": selected,
        },
        checkpoint,
    )
    report = {
        "status": "completed",
        "early_layers": args.early_layers,
        "rank": args.rank,
        "train_prompts": len({str(record["sample_id"]) for record in train_records}),
        "train_blocks": len(train_records),
        "eval_blocks": len(eval_records),
        "best_epoch": best_epoch,
        "best_method": best_method,
        "selected_eal": eal(selected, best_method),
        "selected_delta_vs_released": eal(selected, best_method)
        - eal(selected, "released"),
        "selected": selected,
        "history": history,
        "seconds": time.perf_counter() - started,
        "checkpoint": str(checkpoint.resolve()),
        "verification_design": "target_prefix_multi_path_then_single_chain_completion",
    }
    (args.output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
