#!/usr/bin/env python3
"""GPU mechanics receipts for the frozen PCLD-16R head."""

from __future__ import annotations

import argparse
import copy
import inspect
import json
from pathlib import Path
from typing import Any, Iterable

import torch
from torch import Tensor, nn

from sph.japd_data import load_rollout_records
from sph.pcld import (
    EXPECTED_PARAMETER_COUNT,
    PCLD16Head,
    assert_frozen_architecture,
    pcld_per_block_loss,
)
from sph.pcld_data import (
    attach_pcld_sidecar,
    calibrate_epsilon_from_records,
    collate_pcld_records,
    compute_latent_scale,
    filter_effective_records,
    load_pcld_sidecar,
    pcld_forward_inputs,
    select_balanced_smoke_records,
    validate_sidecar_receipt,
    validate_sidecar_source,
)
from train_pcld16 import load_target_lm_head_weight, move_batch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollout", type=Path, required=True)
    parser.add_argument("--sidecar", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--records", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--require-cuda", action="store_true")
    return parser.parse_args()


def squared_gradient_norm(parameters: Iterable[nn.Parameter]) -> float:
    total = torch.zeros((), dtype=torch.float64)
    for parameter in parameters:
        if parameter.grad is not None:
            total += parameter.grad.detach().double().square().sum().cpu()
    return float(total.sqrt())


def named_parameter_groups(
    model: PCLD16Head,
) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    residual: list[nn.Parameter] = []
    upstream: list[nn.Parameter] = []
    for name, parameter in model.named_parameters():
        if name.startswith("residual_projection."):
            residual.append(parameter)
        else:
            upstream.append(parameter)
    return residual, upstream


def loss_for_batch(
    model: PCLD16Head,
    batch: dict[str, Any],
    lm_head_weight: Tensor,
    latent_scale: Tensor,
) -> tuple[Tensor, Any]:
    output = model(**pcld_forward_inputs(batch, lm_head_weight))
    loss_output = pcld_per_block_loss(
        output,
        batch["candidate_ids"],
        batch["gold_ids"],
        batch["target_residual"],
        batch["target_candidate_logits"],
        batch["target_top1_ids"],
        batch["stable_rows"],
        latent_scale,
        alpha=1.0,
    )
    return loss_output.per_block_loss.mean(), output


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite PCLD mechanics report {args.output}")
    if args.records < 1 or args.records > 32:
        raise ValueError("mechanics records must lie in [1,32]")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if args.require_cuda and device.type != "cuda":
        raise RuntimeError("PCLD mechanics requires CUDA")
    if device.type == "cuda":
        torch.cuda.set_device(0)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    _, raw_records = load_rollout_records(args.rollout, split=args.split)
    records = select_balanced_smoke_records(raw_records, count=32)
    sidecar_metadata, sidecar = load_pcld_sidecar(args.sidecar)
    validate_sidecar_source(
        sidecar_metadata,
        rollout=args.rollout,
        target=args.target,
        split=args.split,
        group="smoke32",
    )
    receipt = validate_sidecar_receipt(
        args.sidecar, sidecar_metadata, require_manual_records=32
    )
    records = attach_pcld_sidecar(records, sidecar, require_exact_keys=True)
    epsilon_num = calibrate_epsilon_from_records(records)
    records = filter_effective_records(records, epsilon_num)
    latent_scale_cpu, latent_rows = compute_latent_scale(records, epsilon_num)
    batch = move_batch(
        collate_pcld_records(
            records[: args.records],
            epsilon_num=epsilon_num,
            require_effective=True,
        ),
        device,
    )
    lm_head_weight_cpu, serialized_key = load_target_lm_head_weight(args.target)
    lm_head_weight = lm_head_weight_cpu.to(device=device, dtype=torch.bfloat16)
    latent_scale = latent_scale_cpu.to(device)

    model = PCLD16Head(scope="global").to(device)
    assert_frozen_architecture(model)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    with torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=device.type == "cuda",
    ):
        zero_output = model(**pcld_forward_inputs(batch, lm_head_weight))
    zero_score_identity = torch.equal(
        zero_output.scores, batch["candidate_logits"].float()
    )
    zero_rank_identity = torch.equal(
        zero_output.scores.argmax(dim=-1),
        torch.zeros_like(batch["gold_ids"], dtype=torch.long),
    )

    online_inputs = pcld_forward_inputs(batch, lm_head_weight)
    perturbed_inputs = dict(online_inputs)
    perturbed_rows = online_inputs["candidate_lm_rows"].clone()
    perturbation = torch.linspace(
        -0.05,
        0.05,
        perturbed_rows.shape[-1],
        device=device,
        dtype=perturbed_rows.dtype,
    )
    perturbed_rows[:, 15, 0] = perturbed_rows[:, 15, 0] + perturbation
    perturbed_inputs["candidate_lm_rows"] = perturbed_rows
    with torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=device.type == "cuda",
    ):
        remote_zero_output = model(**perturbed_inputs)
    remote_global_state_delta = float(
        (
            remote_zero_output.global_states[:, 0].float()
            - zero_output.global_states[:, 0].float()
        )
        .abs()
        .max()
        .item()
    )
    remote_zero_score_delta = float(
        (remote_zero_output.scores - zero_output.scores).abs().max().item()
    )

    nonzero_model = copy.deepcopy(model)
    with torch.no_grad():
        nonzero_model.residual_projection.weight.zero_()
        nonzero_model.residual_projection.bias.zero_()
        nonzero_model.residual_projection.weight[0, 0] = 0.125
    with torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=device.type == "cuda",
    ):
        nonzero_base = nonzero_model(**online_inputs)
        nonzero_remote = nonzero_model(**perturbed_inputs)
    remote_nonzero_score_delta = float(
        (
            nonzero_remote.scores[:, 0].float()
            - nonzero_base.scores[:, 0].float()
        )
        .abs()
        .max()
        .item()
    )

    gradient_model = copy.deepcopy(model)
    residual_parameters, upstream_parameters = named_parameter_groups(gradient_model)
    optimizer = torch.optim.AdamW(gradient_model.parameters(), lr=3e-4, weight_decay=1e-2)
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=device.type == "cuda",
    ):
        step0_loss, _ = loss_for_batch(
            gradient_model, batch, lm_head_weight, latent_scale
        )
    step0_loss.backward()
    step0_residual_gradient = squared_gradient_norm(residual_parameters)
    step0_upstream_gradient = squared_gradient_norm(upstream_parameters)
    optimizer.step()
    projection_after_one_update = float(
        gradient_model.residual_projection.weight.detach().float().abs().max().item()
    )

    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=device.type == "cuda",
    ):
        step1_loss, _ = loss_for_batch(
            gradient_model, batch, lm_head_weight, latent_scale
        )
    step1_loss.backward()
    step1_upstream_gradient = squared_gradient_norm(upstream_parameters)

    try:
        PCLD16Head(scope="causal")
    except ValueError:
        causal_scope_rejected = True
    else:
        causal_scope_rejected = False
    forward_parameters = set(inspect.signature(PCLD16Head.forward).parameters)
    forbidden_forward = {
        name
        for name in forward_parameters
        if any(token in name for token in ("gold", "target", "teacher", "selected"))
    }

    checks = {
        "sidecar_manual_parity_32": int(
            sidecar_metadata.get("manual_parity_records", -1)
        )
        == 32
        and sidecar_metadata.get("manual_parity_passed") is True
        and receipt.get("manual_parity_passed") is True,
        "sidecar_row_geometry_full16": sidecar_metadata.get("teacher_geometry")
        == "context+anchor+gold[0:15] -> rows anchor..gold14",
        "residual_cancellation_within_fp32_tolerance": float(
            sidecar_metadata.get("max_residual_cancellation_error", float("inf"))
        )
        <= 1e-3,
        "parameter_count_exact": parameter_count == EXPECTED_PARAMETER_COUNT,
        "score_shape_full16_k16": list(zero_output.scores.shape)
        == [args.records, 16, 16],
        "proposal_shape_one_chain": list(
            model.proposal_ids(batch["candidate_ids"], zero_output).shape
        )
        == [args.records, 16],
        "production_encoder_mask_none": model._encoder_mask(device) is None,
        "production_query_mask_none": model._query_mask(device) is None,
        "causal_scope_rejected": causal_scope_rejected,
        "production_forward_has_no_target_fields": not forbidden_forward,
        "zero_score_identity": zero_score_identity,
        "zero_selected_rank_identity": zero_rank_identity,
        "remote_changes_global_state": remote_global_state_delta > 0.0,
        "zero_u_keeps_scores_fixed": remote_zero_score_delta == 0.0,
        "nonzero_u_exposes_remote_score_effect": remote_nonzero_score_delta > 0.0,
        "step0_residual_gradient_finite_nonzero": math_is_finite_positive(
            step0_residual_gradient
        ),
        "step0_upstream_gradient_zero": step0_upstream_gradient == 0.0,
        "one_update_changes_u": projection_after_one_update > 0.0,
        "step1_upstream_gradient_finite_nonzero": math_is_finite_positive(
            step1_upstream_gradient
        ),
    }
    report = {
        "format": "pcld16_mechanics_v1",
        "device": {
            "type": str(device),
            "name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        },
        "records": args.records,
        "parameter_count": parameter_count,
        "serialized_lm_head_key": serialized_key,
        "epsilon_num": epsilon_num,
        "latent_scale_rows": latent_rows,
        "sidecar_receipt": receipt,
        "measurements": {
            "remote_global_state_delta": remote_global_state_delta,
            "remote_zero_score_delta": remote_zero_score_delta,
            "remote_nonzero_score_delta": remote_nonzero_score_delta,
            "step0_loss": float(step0_loss.detach()),
            "step1_loss": float(step1_loss.detach()),
            "step0_residual_gradient": step0_residual_gradient,
            "step0_upstream_gradient": step0_upstream_gradient,
            "projection_after_one_update": projection_after_one_update,
            "step1_upstream_gradient": step1_upstream_gradient,
        },
        "checks": checks,
        "passed": all(checks.values()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
    if not report["passed"]:
        raise RuntimeError(
            f"PCLD mechanics failed: {[name for name, ok in checks.items() if not ok]}"
        )
    return report


def math_is_finite_positive(value: float) -> bool:
    return value > 0.0 and value < float("inf")


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
