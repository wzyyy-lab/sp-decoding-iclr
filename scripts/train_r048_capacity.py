#!/usr/bin/env python3
"""Train the fixed 180K R048 tuned lens as a same-set capacity falsifier."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
import time
from typing import Any

import torch

from sph.fast_r048 import R048TunedLens
from sph.r048_capacity import r048_capacity_loss, select_zero_harm_threshold
from train_domino_cached_head import MasterAdamW, load_tensor_from_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection", type=Path, required=True)
    parser.add_argument("--domino-draft", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--frontier-weight", type=float, default=4.0)
    parser.add_argument("--keep-weight", type=float, default=1.0)
    parser.add_argument("--keep-margin", type=float, default=0.05)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--minimum-recovery", type=float, default=0.90)
    parser.add_argument("--required-prompts", type=int)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def load_collection(root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    if not bool(metadata.get("collection_complete", False)):
        raise RuntimeError("R048 capacity collection is incomplete")
    if not bool(metadata.get("capacity_only", False)):
        raise ValueError("R048-B trainer requires an explicitly capacity-only set")
    records: list[dict[str, Any]] = []
    for shard in sorted(root.glob("shard-*.pt")):
        records.extend(torch.load(shard, map_location="cpu", weights_only=False))
    if len(records) != int(metadata["blocks"]):
        raise ValueError("capacity metadata block count differs from shards")
    return metadata, records


def stack_batch(records: list[dict[str, Any]], indices: list[int], device: torch.device) -> dict[str, torch.Tensor]:
    rows = [records[index] for index in indices]
    return {
        "early": torch.stack([row["early_states"] for row in rows]).to(device, torch.bfloat16),
        "candidate_ids": torch.stack([row["candidate_ids"].long() for row in rows]).to(device),
        "base_scores": torch.stack([row["candidate_scores"] for row in rows]).to(device, torch.bfloat16),
        "target": torch.stack([row["target_candidate_logits"] for row in rows]).to(device, torch.float32),
        "proposal": torch.stack([row["proposal_ids"].long() for row in rows]).to(device),
        "verifier_top1": torch.stack([row["target_top1_ids"].long() for row in rows]).to(device),
        "valid": torch.stack([row["valid_teacher_mask"].bool() for row in rows]).to(device),
        "accepted": torch.tensor([int(row["accepted_length"]) for row in rows], device=device),
        "oracle": torch.tensor([int(row["oracle_accepted_length"]) for row in rows], device=device),
    }


@torch.inference_mode()
def evaluate(head: R048TunedLens, records: list[dict[str, Any]], device: torch.device) -> dict[str, Any]:
    sample_ids = [str(row["sample_id"]) for row in records]
    proposals: list[torch.Tensor] = []
    verifier_top1: list[torch.Tensor] = []
    candidates: list[torch.Tensor] = []
    scores: list[torch.Tensor] = []
    oracle_lengths: list[int] = []
    baseline_lengths: list[int] = []
    for start in range(0, len(records), 64):
        indices = list(range(start, min(start + 64, len(records))))
        batch = stack_batch(records, indices, device)
        delta = head(batch["early"], batch["candidate_ids"])
        proposals.append(batch["proposal"].cpu())
        verifier_top1.append(batch["verifier_top1"].cpu())
        candidates.append(batch["candidate_ids"].cpu())
        scores.append((batch["base_scores"].float() + delta.float()).cpu())
        oracle_lengths.extend(int(value) for value in batch["oracle"].cpu())
        baseline_lengths.extend(int(value) for value in batch["accepted"].cpu())
    return select_zero_harm_threshold(
        sample_ids=sample_ids,
        proposal=torch.cat(proposals),
        verifier_top1=torch.cat(verifier_top1),
        candidate_ids=torch.cat(candidates),
        adjusted_scores=torch.cat(scores),
        baseline_lengths=torch.tensor(baseline_lengths),
        oracle_lengths=torch.tensor(oracle_lengths),
    )


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to reuse {args.output}")
    if not torch.cuda.is_available():
        raise RuntimeError("R048 capacity training requires CUDA")
    if not 1 <= args.max_steps <= 200 or args.batch_size < 1 or args.eval_every < 1:
        raise ValueError("invalid training schedule")
    metadata, records = load_collection(args.collection)
    if str(metadata.get("format")) != "r048_capacity_v2":
        raise ValueError("R048-B requires clean-verifier capacity_v2 records")
    if str(metadata.get("authority")) != "clean unsplit full target verifier":
        raise ValueError("R048-B collection does not declare the required verifier authority")
    if int(metadata["candidate_topk"]) != 64 or int(metadata["early_layers"]) != 4:
        raise ValueError("R048-B is frozen to K64 and L4")
    if int(metadata["prompts"]) > 64:
        raise ValueError("R048-B capacity set exceeds 64 prompts")
    if (
        args.required_prompts is not None
        and int(metadata["prompts"]) != args.required_prompts
    ):
        raise ValueError("R048-B collection does not match the required prompt count")
    if Path(str(metadata["domino_draft"])).resolve() != args.domino_draft.resolve():
        raise ValueError("capacity collection and basis checkpoint differ")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda:0")
    basis = load_tensor_from_checkpoint(
        args.domino_draft, "embed_proj.2.weight"
    ).to(device, torch.bfloat16)
    head = R048TunedLens(
        hidden_width=2560,
        rank=64,
        candidate_basis=basis,
    ).to(device)
    if head.trainable_parameter_count != 180_224:
        raise RuntimeError("R048-B head must have exactly 180,224 parameters")
    named = [(name, parameter) for name, parameter in head.named_parameters()]
    optimizer = MasterAdamW(named, args.learning_rate)
    optimizer.optimizer = torch.optim.AdamW(
        optimizer.masters,
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=1e-4,
    )

    args.output.mkdir(parents=True)
    history: list[dict[str, Any]] = []
    started = time.perf_counter()
    baseline = evaluate(head, records, device)
    if (
        float(baseline["eal_prompt_balanced"])
        != float(baseline["baseline_eal_prompt_balanced"])
        or int(baseline["changed_blocks"]) != 0
        or int(baseline["harmful_blocks"]) != 0
    ):
        raise RuntimeError("zero-initialized R048 head did not exactly retain Fast-K64")
    history.append({"step": 0, "evaluation": baseline})
    best = history[0]
    best_state = {name: tensor.detach().cpu().clone() for name, tensor in head.state_dict().items()}
    order = list(range(len(records)))
    random.shuffle(order)
    cursor = 0

    for step in range(1, args.max_steps + 1):
        if cursor + args.batch_size > len(order):
            random.shuffle(order)
            cursor = 0
        indices = order[cursor : cursor + args.batch_size]
        cursor += args.batch_size
        batch = stack_batch(records, indices, device)
        optimizer.zero_grad()
        delta = head(batch["early"], batch["candidate_ids"])
        loss_output = r048_capacity_loss(
            base_scores=batch["base_scores"],
            lens_delta=delta,
            candidate_ids=batch["candidate_ids"],
            proposal=batch["proposal"],
            target_candidate_logits=batch["target"],
            valid_teacher_mask=batch["valid"],
            accepted=batch["accepted"],
            oracle_accepted=batch["oracle"],
            temperature=args.temperature,
            frontier_weight=args.frontier_weight,
            keep_weight=args.keep_weight,
            keep_margin=args.keep_margin,
        )
        if not bool(torch.isfinite(loss_output.loss)):
            raise FloatingPointError("non-finite R048 capacity loss")
        loss_output.loss.backward()
        grad_norm = optimizer.step(args.max_grad_norm)
        if not math.isfinite(grad_norm):
            raise FloatingPointError("non-finite R048 capacity gradient norm")

        if step % args.eval_every == 0 or step == args.max_steps:
            evaluation = evaluate(head, records, device)
            row = {
                "step": step,
                "train_loss": float(loss_output.loss.detach()),
                "train_kl": float(loss_output.kl_loss),
                "train_keep": float(loss_output.keep_loss),
                "grad_norm": grad_norm,
                "evaluation": evaluation,
            }
            history.append(row)
            if (
                int(evaluation["harmful_blocks"]) == 0
                and float(evaluation["eal_prompt_balanced"])
                > float(best["evaluation"]["eal_prompt_balanced"])
            ):
                best = row
                best_state = {
                    name: tensor.detach().cpu().clone()
                    for name, tensor in head.state_dict().items()
                }
            print(json.dumps(row), flush=True)

    best_eval = best["evaluation"]
    passed = (
        float(best_eval["oracle_gain_recovery"]) >= args.minimum_recovery
        and int(best_eval["harmful_blocks"]) == 0
    )
    checkpoint = {
        "format": "r048_tuned_lens_v1",
        "step": int(best["step"]),
        "state_dict": best_state,
        "threshold": float(best_eval["threshold"]),
        "config": {
            "candidate_topk": 64,
            "early_layers": 4,
            "hidden_width": 2560,
            "rank": 64,
            "parameters": 180_224,
        },
        "provenance": {
            "collection": str(args.collection.resolve()),
            "domino_draft": str(args.domino_draft.resolve()),
        },
    }
    torch.save(checkpoint, args.output / "best.pt")
    report = {
        "status": "completed",
        "capacity_only": True,
        "collection": str(args.collection.resolve()),
        "prompts": int(metadata["prompts"]),
        "blocks": len(records),
        "trainable_parameters": head.trainable_parameter_count,
        "baseline": baseline,
        "best_step": int(best["step"]),
        "best": best_eval,
        "minimum_recovery": args.minimum_recovery,
        "passed": passed,
        "seconds": time.perf_counter() - started,
        "history": history,
    }
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2), flush=True)
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
