#!/usr/bin/env python3
"""Constrained FBPF adaptation of the exact released Domino backbone."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModel, AutoModelForCausalLM

from sph.domino_fbpf_runtime import (
    DominoFunctionalInputs,
    FunctionalDominoTeacherForward,
)
from sph.domino_joint_runtime import select_even_prompt_blocks
from sph.fbpf import (
    FBPF_EXPECTED_TRAINABLE_PARAMETERS,
    BatchLinearization,
    TransactionState,
    TransactionalAdamW,
    cosine_warmup_learning_rate,
    count_lora_parameters,
    inject_fbpf_lora,
    named_lora_parameters,
)
from sph.fbpf_runtime import (
    FunctionalDFlashBatch,
    copy_flat_lora_,
    evaluate_flat_transaction,
    flatten_current_lora,
)
from train_domino_backbone_lora import (
    evaluate,
    load_prompt_groups,
    load_released_cache,
    materialize_prompt_inputs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--domino-draft", type=Path, required=True)
    parser.add_argument("--train-canonical", type=Path, required=True)
    parser.add_argument("--eval-canonical", type=Path, required=True)
    parser.add_argument("--eval-domino-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--eval-split", default="validation_select")
    parser.add_argument("--max-train-prompts", type=int, default=8_000)
    parser.add_argument("--max-eval-prompts", type=int)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--peak-learning-rate", type=float, default=1e-4)
    parser.add_argument("--warmup-ratio", type=float, default=0.04)
    parser.add_argument("--eval-every-steps", type=int, default=250)
    parser.add_argument("--macro-prompts", type=int, default=1)
    parser.add_argument("--preserve-reference-margin", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--attn-implementation", default="sdpa")
    return parser.parse_args()


def eal(report: dict[str, Any]) -> float:
    return float(
        report["overall"]["mean_accepted_draft_tokens_prompt_balanced"]
    )


def aggregate_linearizations(
    evaluations: list[BatchLinearization],
    *,
    need_task: bool,
    need_vjp: bool,
) -> BatchLinearization:
    """Average independent-prompt task gradients and join their constraints."""

    if not evaluations:
        raise ValueError("a macro transaction requires at least one prompt")
    for item in evaluations:
        maximum = item.max_all_position_constraint
        if math.isnan(maximum) or maximum == math.inf:
            raise FloatingPointError("non-finite macro constraint maximum")
        if item.constraint_values.numel() and not bool(
            torch.isfinite(item.constraint_values).all().item()
        ):
            raise FloatingPointError("non-finite macro constraint value")
    values = torch.cat([item.constraint_values for item in evaluations])
    if need_task:
        task_gradients = [item.task_gradient for item in evaluations]
        if any(item is None for item in task_gradients):
            raise RuntimeError("macro transaction is missing a task gradient")
        task_gradient = torch.stack(
            [item for item in task_gradients if item is not None]
        ).mean(dim=0)
    else:
        task_gradient = None
    if need_vjp and values.numel():
        gradient_parts = [
            item.constraint_gradients
            for item in evaluations
            if item.constraint_values.numel()
        ]
        if any(item is None for item in gradient_parts):
            raise RuntimeError("macro transaction is missing constraint gradients")
        constraint_gradients = torch.cat(
            [item for item in gradient_parts if item is not None], dim=0
        )
    else:
        constraint_gradients = None
    row_ids: list[int] = []
    for prompt_index, item in enumerate(evaluations):
        row_ids.extend(prompt_index * 4 + row_id for row_id in item.row_ids)
    return BatchLinearization(
        constraint_values=values,
        constraint_gradients=constraint_gradients,
        max_all_position_constraint=max(
            item.max_all_position_constraint for item in evaluations
        ),
        task_gradient=task_gradient,
        row_ids=tuple(row_ids),
    )


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Domino FBPF adaptation requires CUDA")
    if args.output.exists():
        raise FileExistsError(f"refusing to reuse output {args.output}")
    if args.macro_prompts < 1:
        raise ValueError("macro-prompts must be positive")
    args.output.mkdir(parents=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.set_device(0)

    train_groups_all = load_prompt_groups(
        args.train_canonical,
        split=args.train_split,
        max_prompts=args.max_train_prompts,
    )
    train_groups = [select_even_prompt_blocks(group) for group in train_groups_all]
    eval_groups = load_prompt_groups(
        args.eval_canonical,
        split=args.eval_split,
        max_prompts=args.max_eval_prompts,
    )
    released_cache = load_released_cache(
        args.eval_domino_cache, split=args.eval_split
    )

    target = AutoModelForCausalLM.from_pretrained(
        str(args.target),
        trust_remote_code=True,
        local_files_only=True,
        dtype=torch.bfloat16,
        attn_implementation=args.attn_implementation,
        device_map="cuda:0",
    ).eval()
    domino = AutoModel.from_pretrained(
        str(args.domino_draft),
        trust_remote_code=True,
        local_files_only=True,
        dtype=torch.bfloat16,
        attn_implementation=args.attn_implementation,
        device_map="cuda:0",
    ).eval()
    for model in (target, domino):
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    injected = inject_fbpf_lora(domino, training_seed=args.seed)
    if count_lora_parameters(domino) != FBPF_EXPECTED_TRAINABLE_PARAMETERS:
        raise RuntimeError("unexpected Domino LoRA parameter count")
    if sum(parameter.numel() for parameter in domino.parameters() if parameter.requires_grad) != FBPF_EXPECTED_TRAINABLE_PARAMETERS:
        raise RuntimeError("parameters outside the FBPF LoRA scope are trainable")
    target_weight = target.lm_head.weight.detach()

    baseline = evaluate(
        target=target,
        domino=domino,
        target_weight=target_weight,
        prompt_groups=eval_groups,
        released_cache=released_cache,
    )
    if baseline["released_token_mismatches"] or baseline["released_length_mismatches"]:
        raise RuntimeError("zero-LoRA evaluator differs from released Domino")
    baseline_eal = eal(baseline)
    print(json.dumps({"baseline": baseline}, indent=2), flush=True)

    layout, released_theta = flatten_current_lora(domino)
    state = TransactionState.initialize(released_theta)
    engine = TransactionalAdamW()
    steps_per_epoch = (len(train_groups) + args.macro_prompts - 1) // args.macro_prompts
    total_steps = args.epochs * steps_per_epoch
    best_theta = state.theta.detach().clone()
    best_step = 0
    best_eval = baseline
    history: list[dict[str, Any]] = []
    status_counts: Counter[str] = Counter()
    alpha_counts: Counter[str] = Counter()
    restoration_batches = 0
    generator = random.Random(args.seed)
    started = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        order = list(range(len(train_groups)))
        generator.shuffle(order)
        for macro_start in range(0, len(order), args.macro_prompts):
            macro_indices = order[macro_start : macro_start + args.macro_prompts]
            transactions: list[
                tuple[FunctionalDominoTeacherForward, FunctionalDFlashBatch]
            ] = []
            for group_index in macro_indices:
                records = train_groups[group_index]
                materialized = materialize_prompt_inputs(
                    target=target, domino=domino, records=records
                )
                functional_inputs = DominoFunctionalInputs(
                    target_hidden=materialized["target_hidden"],
                    noise_embedding=materialized["noise_embedding"],
                    context_lengths=materialized["context_lengths"],
                    anchors=materialized["anchors"],
                    gold=materialized["gold"],
                )
                forward = FunctionalDominoTeacherForward(
                    domino=domino,
                    target_weight=target_weight,
                    inputs=functional_inputs,
                    layout=layout,
                )
                with torch.no_grad():
                    released_logits = forward(released_theta).detach()
                transaction_batch = FunctionalDFlashBatch(
                    target_hidden=functional_inputs.target_hidden,
                    noise_embedding=functional_inputs.noise_embedding,
                    position_ids=torch.empty(
                        0, dtype=torch.long, device=functional_inputs.gold.device
                    ),
                    gold=functional_inputs.gold,
                    base_logits=released_logits,
                )
                transactions.append((forward, transaction_batch))

            def transaction_evaluate(
                theta: torch.Tensor, need_task: bool, need_vjp: bool
            ):
                evaluations = [
                    evaluate_flat_transaction(
                        theta=theta,
                        forward_logits=forward,
                        batch=transaction_batch,
                        arm="D",
                        need_task=need_task,
                        need_vjp=need_vjp,
                        # Released BF16 Domino has accepted exact-logit ties.
                        # The per-position requirement follows argmax's
                        # lowest-id tie rule.
                        argmax_tie_aware_constraints=True,
                        preserve_reference_margin=args.preserve_reference_margin,
                    )
                    for forward, transaction_batch in transactions
                ]
                return aggregate_linearizations(
                    evaluations, need_task=need_task, need_vjp=need_vjp
                )

            learning_rate = cosine_warmup_learning_rate(
                state.k_outer,
                total_steps=total_steps,
                peak=args.peak_learning_rate,
                warmup_ratio=args.warmup_ratio,
            )
            result = engine.step(
                state, transaction_evaluate, learning_rate=learning_rate
            )
            state = result.state
            if result.aborted:
                raise RuntimeError(f"FBPF transaction aborted: {result.status}")
            status_counts[result.status] += 1
            restoration_batches += int(result.restored)
            if result.attempted_alphas:
                alpha_counts[f"{result.attempted_alphas[-1]:g}"] += 1

            global_step = state.k_outer
            if global_step % 25 == 0:
                print(
                    f"step={global_step}/{total_steps} committed={state.t_adam} "
                    f"status={dict(status_counts)} alphas={dict(alpha_counts)} "
                    f"lr={learning_rate:.3e}",
                    flush=True,
                )
            if args.eval_every_steps > 0 and global_step % args.eval_every_steps == 0:
                copy_flat_lora_(domino, layout, state.theta)
                current = evaluate(
                    target=target,
                    domino=domino,
                    target_weight=target_weight,
                    prompt_groups=eval_groups,
                    released_cache=released_cache,
                )
                record = {
                    "epoch": epoch,
                    "global_step": global_step,
                    "committed_steps": state.t_adam,
                    "validation": current,
                    "status_counts": dict(status_counts),
                    "alpha_counts": dict(alpha_counts),
                }
                history.append(record)
                print(json.dumps({"step_validation": record}, indent=2), flush=True)
                if eal(current) > eal(best_eval):
                    best_theta = state.theta.detach().clone()
                    best_step = global_step
                    best_eval = current

    if state.t_adam == 0 or not bool(
        torch.linalg.vector_norm(state.theta - released_theta).item() > 0.0
    ):
        raise RuntimeError("FBPF completed without a committed nonzero update")

    if not history or history[-1]["global_step"] != state.k_outer:
        copy_flat_lora_(domino, layout, state.theta)
        current = evaluate(
            target=target,
            domino=domino,
            target_weight=target_weight,
            prompt_groups=eval_groups,
            released_cache=released_cache,
        )
        history.append(
            {
                "epoch": args.epochs,
                "global_step": state.k_outer,
                "committed_steps": state.t_adam,
                "validation": current,
            }
        )
        if eal(current) > eal(best_eval):
            best_theta = state.theta.detach().clone()
            best_step = state.k_outer
            best_eval = current

    copy_flat_lora_(domino, layout, best_theta)
    selected = evaluate(
        target=target,
        domino=domino,
        target_weight=target_weight,
        prompt_groups=eval_groups,
        released_cache=released_cache,
    )
    checkpoint = args.output / "best_lora.pt"
    torch.save(
        {
            "lora_state_dict": {
                name: parameter.detach().cpu().clone()
                for name, parameter in named_lora_parameters(domino)
            },
            "flat_theta": best_theta.detach().cpu(),
            "best_step": best_step,
            "injected_modules": list(injected),
        },
        checkpoint,
    )
    report = {
        "status": "completed",
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "method": "exact_domino_fbpf",
        "seed": args.seed,
        "train_prompts": len(train_groups),
        "macro_prompts": args.macro_prompts,
        "eval_prompts": len(eval_groups),
        "trainable_parameters": FBPF_EXPECTED_TRAINABLE_PARAMETERS,
        "baseline_eal": baseline_eal,
        "best_step": best_step,
        "best_validation": best_eval,
        "selected": selected,
        "selected_delta_vs_released": eal(selected) - baseline_eal,
        "status_counts": dict(status_counts),
        "alpha_counts": dict(alpha_counts),
        "restoration_batches": restoration_batches,
        "preserve_reference_margin": args.preserve_reference_margin,
        "committed_steps": state.t_adam,
        "history": history,
        "seconds": time.perf_counter() - started,
        "checkpoint": str(checkpoint.resolve()),
    }
    (args.output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
