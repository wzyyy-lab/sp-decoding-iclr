#!/usr/bin/env python3
"""Pretrain the causal Top-K decoder on the large cached DFlash corpus."""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from train_domino_cached_head import (
    CachedDominoDataset,
    acceptance_lengths,
    cosine_schedule,
    load_records,
    load_tensor_from_checkpoint,
    summarize_lengths,
)
from train_domino_causal_lattice_decoder import (
    CausalLatticeDecoder,
    training_loss,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-canonical", nargs="+", type=Path, required=True)
    parser.add_argument("--eval-canonical", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate-topk", type=int, default=4)
    parser.add_argument("--model-dim", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--lattice-layers", type=int, default=1)
    parser.add_argument("--decoder-layers", type=int, default=2)
    parser.add_argument(
        "--objective", choices=["decay_ce", "breaker_margin"], default="breaker_margin"
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-ratio", type=float, default=0.04)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--loss-decay-gamma", type=float, default=7.0)
    parser.add_argument("--prefix-weight", type=float, default=0.5)
    parser.add_argument("--margin-temperature", type=float, default=1.0)
    parser.add_argument("--margin-offset", type=float, default=0.0)
    parser.add_argument("--residual-penalty-weight", type=float, default=1e-4)
    parser.add_argument("--eval-every-steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-train-blocks", type=int)
    parser.add_argument("--max-eval-blocks", type=int)
    return parser.parse_args()


def base_collate(records: list[dict[str, Any]], topk: int) -> dict[str, Any]:
    return {
        "sample_ids": [str(record["sample_id"]) for record in records],
        "domains": [str(record["domain"]) for record in records],
        "anchors": torch.tensor(
            [int(record["anchor_token_id"]) for record in records], dtype=torch.long
        ),
        "gold": torch.stack([record["gold_ids"].long() for record in records]),
        "hidden": torch.stack(
            [record["parallel_hidden"].to(torch.bfloat16) for record in records]
        ),
        "candidate_ids": torch.stack(
            [record["base_topk_ids"][:, :topk].long() for record in records]
        ),
        "candidate_logits": torch.stack(
            [record["base_topk_logits"][:, :topk].float() for record in records]
        ),
        "full_logsumexp": torch.stack(
            [record["base_logsumexp"].float() for record in records]
        ),
        "prompt_balance_weights": torch.tensor(
            [float(record["_prompt_balance_weight"]) for record in records],
            dtype=torch.float32,
        ),
    }


def decoder_forward(
    *,
    decoder: CausalLatticeDecoder,
    target_weight: torch.Tensor,
    batch: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    anchors = batch["anchors"].to("cuda:0", non_blocking=True)
    gold = batch["gold"].to("cuda:0", non_blocking=True)
    hidden = batch["hidden"].to("cuda:0", non_blocking=True)
    ids = batch["candidate_ids"].to("cuda:0", non_blocking=True)
    logits = batch["candidate_logits"].to("cuda:0", non_blocking=True)
    full_logsumexp = batch["full_logsumexp"].to("cuda:0", non_blocking=True)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        memory, local_hidden = decoder.encode_lattice(
            hidden=hidden,
            candidate_embeddings=F.embedding(ids, target_weight),
            candidate_logits=logits,
            full_logsumexp=full_logsumexp,
        )
        prefix_ids = torch.cat([anchors[:, None], gold[:, :-1]], dim=-1)
        prefix_states = decoder.encode_prefix(
            prefix_embeddings=F.embedding(prefix_ids, target_weight), memory=memory
        )
        action_features = torch.zeros(
            (*ids.shape, 2), dtype=torch.float32, device=ids.device
        )
        action_features[:, :, 0, 0] = 1.0
        scores, residual = decoder.score_candidates(
            prefix_states=prefix_states,
            local_hidden=local_hidden,
            candidate_embeddings=F.embedding(ids, target_weight),
            fixed_candidate_logits=logits,
            fixed_logsumexp=full_logsumexp,
            action_features=action_features,
        )
    return scores, residual, ids, gold


@torch.inference_mode()
def evaluate(
    *,
    decoder: CausalLatticeDecoder,
    target_weight: torch.Tensor,
    loader: DataLoader,
) -> dict[str, Any]:
    decoder.eval()
    sample_ids: list[str] = []
    domains: list[str] = []
    lengths: list[int] = []
    baseline_lengths: list[int] = []
    horizon = 0
    for batch in loader:
        scores, _, ids, gold = decoder_forward(
            decoder=decoder, target_weight=target_weight, batch=batch
        )
        proposals = ids.gather(-1, scores.argmax(dim=-1, keepdim=True)).squeeze(-1)
        baseline = ids[:, :, 0]
        lengths.extend(int(value) for value in acceptance_lengths(proposals, gold).cpu())
        baseline_lengths.extend(
            int(value) for value in acceptance_lengths(baseline, gold).cpu()
        )
        sample_ids.extend(batch["sample_ids"])
        domains.extend(batch["domains"])
        horizon = int(gold.shape[1])
    selected = summarize_lengths(sample_ids, domains, lengths, horizon)
    baseline = summarize_lengths(sample_ids, domains, baseline_lengths, horizon)
    return {
        "selected": selected,
        "baseline": baseline,
        "sample_ids": sample_ids,
        "domains": domains,
        "lengths": lengths,
        "baseline_lengths": baseline_lengths,
    }


def prompt_balanced_eal(result: dict[str, Any], key: str) -> float:
    return float(
        result[key]["overall"]["mean_accepted_draft_tokens_prompt_balanced"]
    )


def save_decoder_checkpoint(
    *,
    path: Path,
    state: dict[str, torch.Tensor],
    best_step: int,
    args: argparse.Namespace,
) -> None:
    torch.save(
        {
            "decoder_state_dict": state,
            "best_step": best_step,
            "candidate_topk": args.candidate_topk,
            "model_dim": args.model_dim,
            "num_heads": args.num_heads,
            "lattice_layers": args.lattice_layers,
            "decoder_layers": args.decoder_layers,
        },
        path,
    )


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("causal decoder pretraining requires CUDA")
    if args.output.exists():
        raise FileExistsError(f"refusing to reuse output {args.output}")
    args.output.mkdir(parents=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.set_device(0)

    train_records: list[dict[str, Any]] = []
    for root in args.train_canonical:
        train_records.extend(load_records(root, "train", None))
    if args.max_train_blocks is not None:
        train_records = train_records[: args.max_train_blocks]
    eval_records = load_records(
        args.eval_canonical, "validation_select", args.max_eval_blocks
    )
    prompt_counts = Counter(str(record["sample_id"]) for record in train_records)
    for record in train_records:
        record["_prompt_balance_weight"] = 1.0 / prompt_counts[
            str(record["sample_id"])
        ]
    for record in eval_records:
        record["_prompt_balance_weight"] = 1.0

    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        CachedDominoDataset(train_records),
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        pin_memory=True,
        collate_fn=lambda records: base_collate(records, args.candidate_topk),
    )
    eval_loader = DataLoader(
        CachedDominoDataset(eval_records),
        batch_size=args.eval_batch_size,
        shuffle=False,
        pin_memory=True,
        collate_fn=lambda records: base_collate(records, args.candidate_topk),
    )
    target_weight = load_tensor_from_checkpoint(
        args.target, "model.embed_tokens.weight"
    ).to(device="cuda:0", dtype=torch.bfloat16)
    target_weight.requires_grad_(False)
    horizon = int(train_records[0]["gold_ids"].numel())
    hidden_size = int(train_records[0]["parallel_hidden"].shape[-1])
    decoder = CausalLatticeDecoder(
        hidden_size=hidden_size,
        positions=horizon,
        candidates=args.candidate_topk,
        model_dim=args.model_dim,
        num_heads=args.num_heads,
        lattice_layers=args.lattice_layers,
        decoder_layers=args.decoder_layers,
    ).to("cuda:0")
    optimizer = torch.optim.AdamW(
        decoder.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
    )
    total_steps = args.epochs * len(train_loader)
    warmup_steps = int(round(args.warmup_ratio * total_steps))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: cosine_schedule(step, total_steps, warmup_steps)
    )

    initial = evaluate(decoder=decoder, target_weight=target_weight, loader=eval_loader)
    initial_eal = prompt_balanced_eal(initial, "baseline")
    if prompt_balanced_eal(initial, "selected") != initial_eal:
        raise RuntimeError("zero residual did not exactly reproduce cached DFlash")
    print(json.dumps({"baseline_eal": initial_eal}, indent=2), flush=True)
    best_state = {
        name: value.detach().cpu().clone() for name, value in decoder.state_dict().items()
    }
    best_eval = initial
    best_step = 0
    history: list[dict[str, Any]] = []
    global_step = 0
    started = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        decoder.train()
        totals: dict[str, float] = defaultdict(float)
        batches = 0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            scores, residual, ids, gold = decoder_forward(
                decoder=decoder, target_weight=target_weight, batch=batch
            )
            task_loss, diagnostics = training_loss(
                scores=scores,
                candidate_ids=ids,
                gold=gold,
                objective=args.objective,
                gamma=args.loss_decay_gamma,
                prefix_weight=args.prefix_weight,
                margin_temperature=args.margin_temperature,
                margin_offset=args.margin_offset,
                block_weights=batch["prompt_balance_weights"].to(
                    "cuda:0", non_blocking=True
                ),
            )
            residual_penalty = residual.square().mean()
            loss = task_loss + args.residual_penalty_weight * residual_penalty
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at step {global_step}")
            loss.backward()
            grad_norm = float(
                torch.nn.utils.clip_grad_norm_(decoder.parameters(), args.max_grad_norm)
            )
            optimizer.step()
            scheduler.step()
            global_step += 1
            batches += 1
            totals["loss"] += float(loss.detach())
            totals["task_loss"] += float(task_loss.detach())
            totals["residual_rms"] += float(residual.detach().square().mean().sqrt())
            totals["grad_norm"] += grad_norm
            for key, value in diagnostics.items():
                totals[key] += value
            if global_step % 100 == 0:
                print(
                    f"step={global_step}/{total_steps} epoch={epoch} "
                    f"loss={float(loss.detach()):.6f} "
                    f"residual_rms={float(residual.detach().square().mean().sqrt()):.4f} "
                    f"lr={scheduler.get_last_lr()[0]:.3e}",
                    flush=True,
                )
            if args.eval_every_steps > 0 and global_step % args.eval_every_steps == 0:
                current = evaluate(
                    decoder=decoder, target_weight=target_weight, loader=eval_loader
                )
                current_eal = prompt_balanced_eal(current, "selected")
                record = {
                    "epoch": epoch,
                    "global_step": global_step,
                    "selected_eal": current_eal,
                    "delta_vs_dflash": current_eal - initial_eal,
                }
                print(json.dumps({"step_validation": record}, indent=2), flush=True)
                history.append(record)
                if current_eal > prompt_balanced_eal(best_eval, "selected"):
                    best_state = {
                        name: value.detach().cpu().clone()
                        for name, value in decoder.state_dict().items()
                    }
                    best_eval = current
                    best_step = global_step
                    save_decoder_checkpoint(
                        path=args.output / "best_decoder.pt",
                        state=best_state,
                        best_step=best_step,
                        args=args,
                    )
                decoder.train()

        current = evaluate(
            decoder=decoder, target_weight=target_weight, loader=eval_loader
        )
        current_eal = prompt_balanced_eal(current, "selected")
        record = {
            "epoch": epoch,
            "global_step": global_step,
            "selected_eal": current_eal,
            "delta_vs_dflash": current_eal - initial_eal,
            "train": {key: value / batches for key, value in totals.items()},
        }
        print(json.dumps(record, indent=2), flush=True)
        history.append(record)
        if current_eal > prompt_balanced_eal(best_eval, "selected"):
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in decoder.state_dict().items()
            }
            best_eval = current
            best_step = global_step
            save_decoder_checkpoint(
                path=args.output / "best_decoder.pt",
                state=best_state,
                best_step=best_step,
                args=args,
            )

    checkpoint = args.output / "best_decoder.pt"
    save_decoder_checkpoint(
        path=checkpoint, state=best_state, best_step=best_step, args=args
    )
    selected_eal = prompt_balanced_eal(best_eval, "selected")
    report = {
        "status": "completed",
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "train_blocks": len(train_records),
        "train_prompts": len(prompt_counts),
        "validation_blocks": len(eval_records),
        "objective": args.objective,
        "baseline_eal": initial_eal,
        "best_step": best_step,
        "selected_eal": selected_eal,
        "selected_delta_vs_dflash": selected_eal - initial_eal,
        "selected": {
            "overall": best_eval["selected"]["overall"],
            "by_domain": best_eval["selected"]["by_domain"],
        },
        "history": history,
        "trainable_parameters": sum(p.numel() for p in decoder.parameters()),
        "seconds": time.perf_counter() - started,
        "checkpoint": str(checkpoint.resolve()),
    }
    (args.output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
