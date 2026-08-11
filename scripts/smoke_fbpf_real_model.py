#!/usr/bin/env python3
"""Run the single authorized real-model FBPF synthetic GPU smoke.

The fixture contains no scientific labels.  It materializes target context
features once, constructs four released-DFlash protected prefixes, then checks
one ordinary D-PACE LoRA step and one exact FBPF transactional step.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoModel, AutoModelForCausalLM

from sph.fbpf import (
    FBPF_EXPECTED_TRAINABLE_PARAMETERS,
    TransactionState,
    TransactionalAdamW,
    arm_loss,
    cosine_warmup_learning_rate,
    count_lora_parameters,
    inject_fbpf_lora,
    margin_state,
)
from sph.fbpf_runtime import (
    FunctionalDFlashBatch,
    FunctionalDFlashForward,
    engineering_gold_with_protected_prefix,
    evaluate_flat_transaction,
    flatten_current_lora,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--draft", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--context-length", type=int, default=32)
    parser.add_argument("--a-steps", type=int, default=1)
    parser.add_argument("--d-steps", type=int, default=1)
    parser.add_argument("--peak-learning-rate", type=float, default=1e-4)
    parser.add_argument("--attn-implementation", default="sdpa")
    return parser.parse_args()


def extract_context_feature(
    hidden_states: tuple[torch.Tensor, ...], layer_ids: list[int]
) -> torch.Tensor:
    return torch.cat([hidden_states[layer_id + 1] for layer_id in layer_ids], dim=-1)


@torch.no_grad()
def materialize_fixture(
    *,
    target: nn.Module,
    draft: nn.Module,
    seed: int,
    context_length: int,
) -> tuple[FunctionalDFlashBatch, torch.Tensor]:
    if context_length < 2:
        raise ValueError("context length must be at least two")
    generator = torch.Generator(device="cpu").manual_seed(2_026_080_800 + seed)
    vocabulary = int(target.config.vocab_size)
    # Synthetic engineering tokens deliberately avoid special-token handling.
    context_ids = torch.randint(
        100,
        vocabulary - 256,
        (4, context_length),
        generator=generator,
        dtype=torch.long,
    ).to("cuda:0")
    anchors = torch.randint(
        100,
        vocabulary - 256,
        (4,),
        generator=generator,
        dtype=torch.long,
    ).to("cuda:0")
    target_outputs = target.model(
        context_ids,
        use_cache=False,
        output_hidden_states=True,
        return_dict=True,
    )
    target_hidden = extract_context_feature(
        target_outputs.hidden_states, list(draft.target_layer_ids)
    ).detach()
    block_size = int(draft.block_size)
    block_ids = torch.full(
        (4, block_size),
        int(draft.mask_token_id),
        dtype=torch.long,
        device="cuda:0",
    )
    block_ids[:, 0] = anchors
    noise_embedding = target.model.embed_tokens(block_ids).detach()
    position_ids = torch.arange(
        context_length + block_size, device="cuda:0", dtype=torch.long
    ).view(1, -1).expand(4, -1)
    parallel_hidden = draft(
        target_hidden=target_hidden,
        noise_embedding=noise_embedding,
        position_ids=position_ids,
        past_key_values=None,
        use_cache=False,
        is_causal=False,
    )[:, 1 - block_size :, :]
    target_weight = target.lm_head.weight.detach()
    base_logits = F.linear(parallel_hidden, target_weight).float()
    gold = engineering_gold_with_protected_prefix(
        base_logits, mismatch_position=1
    )
    batch = FunctionalDFlashBatch(
        target_hidden=target_hidden,
        noise_embedding=noise_embedding,
        position_ids=position_ids,
        gold=gold,
        base_logits=base_logits.detach(),
    )
    return batch, target_weight


def scalar_loss(
    forward: FunctionalDFlashForward,
    theta: torch.Tensor,
    batch: FunctionalDFlashBatch,
    arm: str,
) -> float:
    with torch.no_grad():
        logits = forward(theta)
        return float(
            arm_loss(arm, logits, batch.base_logits, batch.gold).total.item()
        )


def run_ordinary_arm(
    *,
    forward: FunctionalDFlashForward,
    initial_theta: torch.Tensor,
    batch: FunctionalDFlashBatch,
    steps: int,
    peak_learning_rate: float,
) -> dict[str, Any]:
    theta = nn.Parameter(initial_theta.detach().clone())
    optimizer = torch.optim.AdamW(
        [theta],
        lr=peak_learning_rate,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.0,
    )
    losses: list[float] = []
    started = time.perf_counter()
    for step in range(steps):
        optimizer.param_groups[0]["lr"] = cosine_warmup_learning_rate(
            step, peak=peak_learning_rate
        )
        optimizer.zero_grad(set_to_none=True)
        logits = forward(theta)
        loss = arm_loss("A", logits, batch.base_logits, batch.gold).total
        if not bool(torch.isfinite(loss).item()):
            raise FloatingPointError("non-finite ordinary D-PACE loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_((theta,), 1.0)
        optimizer.step()
        losses.append(float(loss.detach()))
    torch.cuda.synchronize()
    return {
        "steps": steps,
        "losses": losses,
        "final_loss": scalar_loss(forward, theta, batch, "A"),
        "theta_delta_norm": float(
            torch.linalg.vector_norm(theta.detach() - initial_theta).item()
        ),
        "seconds": time.perf_counter() - started,
    }


def run_fbpf_arm(
    *,
    forward: FunctionalDFlashForward,
    initial_theta: torch.Tensor,
    batch: FunctionalDFlashBatch,
    steps: int,
    peak_learning_rate: float,
) -> dict[str, Any]:
    engine = TransactionalAdamW()
    state = TransactionState.initialize(initial_theta)
    history: list[dict[str, Any]] = []

    def evaluate(
        theta: torch.Tensor, need_task: bool, need_vjp: bool
    ):
        return evaluate_flat_transaction(
            theta=theta,
            forward_logits=forward,
            batch=batch,
            arm="D",
            need_task=need_task,
            need_vjp=need_vjp,
        )

    started = time.perf_counter()
    for step in range(steps):
        learning_rate = cosine_warmup_learning_rate(
            state.k_outer, peak=peak_learning_rate
        )
        result = engine.step(state, evaluate, learning_rate=learning_rate)
        state = result.state
        history.append(
            {
                "step": step + 1,
                "status": result.status,
                "restored": result.restored,
                "restoration_cycles": result.restoration_cycles,
                "attempted_alphas": list(result.attempted_alphas),
                "k_outer": state.k_outer,
                "t_adam": state.t_adam,
            }
        )
        if result.aborted:
            raise RuntimeError(f"FBPF transaction aborted: {result.status}")
    final_linearization = evaluate(state.theta, False, False)
    if not final_linearization.feasible():
        raise RuntimeError("FBPF smoke ended outside the exact feasible set")
    torch.cuda.synchronize()
    return {
        "steps": steps,
        "history": history,
        "final_loss": scalar_loss(forward, state.theta, batch, "D"),
        "theta_delta_norm": float(
            torch.linalg.vector_norm(state.theta - initial_theta).item()
        ),
        "max_constraint": final_linearization.max_all_position_constraint,
        "seconds": time.perf_counter() - started,
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("real-model FBPF smoke requires CUDA")
    if args.output.exists():
        raise FileExistsError(f"refusing to reuse output {args.output}")
    args.output.mkdir(parents=True)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.set_device(0)

    target = AutoModelForCausalLM.from_pretrained(
        str(args.target),
        trust_remote_code=True,
        local_files_only=True,
        dtype=torch.bfloat16,
        attn_implementation=args.attn_implementation,
        device_map="cuda:0",
    ).eval()
    draft = AutoModel.from_pretrained(
        str(args.draft),
        trust_remote_code=True,
        local_files_only=True,
        dtype=torch.bfloat16,
        attn_implementation=args.attn_implementation,
        device_map="cuda:0",
    ).eval()
    for model in (target, draft):
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    injected = inject_fbpf_lora(draft, training_seed=args.seed)
    trainable_parameters = count_lora_parameters(draft)
    if trainable_parameters != FBPF_EXPECTED_TRAINABLE_PARAMETERS:
        raise RuntimeError(
            f"unexpected LoRA size {trainable_parameters}; "
            f"expected {FBPF_EXPECTED_TRAINABLE_PARAMETERS}"
        )

    batch, target_weight = materialize_fixture(
        target=target,
        draft=draft,
        seed=args.seed,
        context_length=args.context_length,
    )
    layout, initial_theta = flatten_current_lora(draft)
    forward = FunctionalDFlashForward(
        draft=draft,
        target_weight=target_weight,
        batch=batch,
        layout=layout,
    )
    with torch.no_grad():
        functional_logits = forward(initial_theta)
    identity_max_abs = float(
        (functional_logits - batch.base_logits).abs().max().item()
    )
    if identity_max_abs != 0.0:
        raise RuntimeError(
            f"zero LoRA functional path changed released logits by {identity_max_abs}"
        )
    base_state = margin_state(batch.base_logits, batch.gold)
    protected_counts = [int(value) for value in base_state.first_mismatch.tolist()]
    if len(protected_counts) != 4 or any(value < 1 for value in protected_counts):
        raise RuntimeError("synthetic smoke must have K=4 nonempty protected prefixes")

    torch.cuda.reset_peak_memory_stats()
    ordinary = run_ordinary_arm(
        forward=forward,
        initial_theta=initial_theta,
        batch=batch,
        steps=args.a_steps,
        peak_learning_rate=args.peak_learning_rate,
    )
    fbpf = run_fbpf_arm(
        forward=forward,
        initial_theta=initial_theta,
        batch=batch,
        steps=args.d_steps,
        peak_learning_rate=args.peak_learning_rate,
    )
    peak_allocated = torch.cuda.max_memory_allocated() / 2**30
    peak_reserved = torch.cuda.max_memory_reserved() / 2**30
    report = {
        "status": "passed",
        "job_id": __import__("os").environ.get("SLURM_JOB_ID"),
        "seed": args.seed,
        "attention_implementation": args.attn_implementation,
        "injected_modules": list(injected),
        "trainable_parameters": trainable_parameters,
        "identity_max_abs": identity_max_abs,
        "protected_counts": protected_counts,
        "ordinary_a": ordinary,
        "fbpf_d": fbpf,
        "peak_allocated_gib": peak_allocated,
        "peak_reserved_gib": peak_reserved,
    }
    if not math.isfinite(ordinary["final_loss"]) or not math.isfinite(
        fbpf["final_loss"]
    ):
        raise FloatingPointError("non-finite final smoke loss")
    (args.output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
