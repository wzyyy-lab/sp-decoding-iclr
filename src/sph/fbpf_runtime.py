"""Real-model runtime bridge for the FBPF optimizer.

The mathematical optimizer in :mod:`sph.fbpf` is deliberately model agnostic.
This module supplies the small amount of glue needed to evaluate a flat LoRA
parameter vector through a DFlash model without mutating the live module.  It
is shared by the synthetic GPU gate and the prospective training entry point.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.func import functional_call

from .fbpf import (
    FBPF_EPSILON_TIE,
    FBPF_TAU_FEASIBLE,
    BatchLinearization,
    FlatParameterLayout,
    arm_loss,
    constraint_batched_vjp,
    margin_state,
    named_lora_parameters,
)


@dataclass(frozen=True)
class FunctionalDFlashBatch:
    """Materialized inputs reused by every candidate in one outer step."""

    target_hidden: Tensor
    noise_embedding: Tensor
    position_ids: Tensor
    gold: Tensor
    base_logits: Tensor

    def __post_init__(self) -> None:
        if self.gold.ndim != 2 or self.gold.shape[0] != 4:
            raise ValueError("FBPF requires exactly four [4, L] gold rows")
        if self.base_logits.shape[:2] != self.gold.shape:
            raise ValueError("base logits and gold geometry differ")
        if self.noise_embedding.shape[0] != 4:
            raise ValueError("noise embeddings must contain four rows")
        if self.target_hidden.shape[0] != 4:
            raise ValueError("target features must contain four rows")


class FunctionalDFlashForward:
    """Run DFlash with a functional flat LoRA vector.

    Only LoRA parameters are overridden.  Frozen base parameters and buffers
    stay resident in the loaded model, so backtracking candidates neither copy
    nor mutate the roughly billion-parameter released checkpoint.
    """

    def __init__(
        self,
        *,
        draft: nn.Module,
        target_weight: Tensor,
        batch: FunctionalDFlashBatch,
        layout: FlatParameterLayout | None = None,
        extra_forward_kwargs: Mapping[str, object] | None = None,
    ) -> None:
        self.draft = draft
        self.target_weight = target_weight.detach()
        self.batch = batch
        self.layout = (
            FlatParameterLayout.from_named_parameters(named_lora_parameters(draft))
            if layout is None
            else layout
        )
        self.extra_forward_kwargs = dict(extra_forward_kwargs or {})
        if self.layout.total_numel == 0:
            raise ValueError("DFlash model has no injected LoRA parameters")

    def __call__(self, theta: Tensor) -> Tensor:
        if theta.ndim != 1 or theta.numel() != self.layout.total_numel:
            raise ValueError("flat LoRA vector has the wrong geometry")
        parameters = self.layout.unflatten(theta)
        kwargs: dict[str, object] = {
            "target_hidden": self.batch.target_hidden,
            "noise_embedding": self.batch.noise_embedding,
            "position_ids": self.batch.position_ids,
            "past_key_values": None,
            "use_cache": False,
            "is_causal": False,
            **self.extra_forward_kwargs,
        }
        hidden = functional_call(
            self.draft,
            parameters,
            args=(),
            kwargs=kwargs,
            tie_weights=False,
            strict=False,
        )
        positions = self.batch.gold.shape[1]
        hidden = hidden[:, -positions:, :]
        # The target head is frozen but remains differentiable with respect to
        # DFlash hidden states.  Float32 logits implement the frozen margin and
        # decision contract without changing the bf16 model execution path.
        return F.linear(hidden, self.target_weight).float()


def flatten_current_lora(model: nn.Module) -> tuple[FlatParameterLayout, Tensor]:
    named = named_lora_parameters(model)
    layout = FlatParameterLayout.from_named_parameters(named)
    values = {name: parameter.detach() for name, parameter in named}
    return layout, layout.flatten(values).to(dtype=torch.float32)


@torch.no_grad()
def copy_flat_lora_(
    model: nn.Module, layout: FlatParameterLayout, theta: Tensor
) -> None:
    parameters = dict(named_lora_parameters(model))
    if tuple(parameters) != layout.names:
        raise ValueError("live LoRA layout differs from the flat checkpoint")
    values = layout.unflatten(theta)
    for name in layout.names:
        parameters[name].copy_(values[name].to(device=parameters[name].device))


def engineering_gold_with_protected_prefix(
    base_logits: Tensor,
    *,
    mismatch_position: int = 1,
    epsilon_tie: float = FBPF_EPSILON_TIE,
    tau_f: float = FBPF_TAU_FEASIBLE,
) -> Tensor:
    """Construct non-scientific labels that force four protected prefixes.

    This is used only by the synthetic engineering gate.  Gold equals released
    DFlash before ``mismatch_position`` and is changed to the runner-up exactly
    at that position, making every row's protected set nonempty while retaining
    an explicit positive-margin feasibility check.
    """

    if base_logits.ndim != 3 or base_logits.shape[0] != 4:
        raise ValueError("engineering logits must have shape [4, L, V]")
    if not 1 <= mismatch_position < base_logits.shape[1]:
        raise ValueError("mismatch_position must leave a nonempty protected prefix")
    detached = base_logits.detach().float()
    released = detached.argmax(dim=-1)
    competitor_scores = detached.clone()
    competitor_scores.scatter_(-1, released.unsqueeze(-1), -torch.inf)
    competitor = competitor_scores.argmax(dim=-1)
    gold = released.clone()
    gold[:, mismatch_position] = competitor[:, mismatch_position]
    state = margin_state(base_logits, gold)
    expected = torch.full_like(state.first_mismatch, mismatch_position)
    if not torch.equal(state.first_mismatch, expected):
        raise RuntimeError("engineering fixture did not create the intended first miss")
    protected = state.gold_margin[:, :mismatch_position]
    required_margin = float(epsilon_tie - tau_f)
    if not bool(torch.all(protected >= required_margin).item()):
        raise RuntimeError(
            "engineering fixture has a protected released margin below the "
            f"required {required_margin:g}"
        )
    return gold


def evaluate_flat_transaction(
    *,
    theta: Tensor,
    forward_logits: Callable[[Tensor], Tensor],
    batch: FunctionalDFlashBatch,
    arm: str,
    need_task: bool,
    need_vjp: bool,
    epsilon_tie: float = FBPF_EPSILON_TIE,
    argmax_tie_aware_constraints: bool = False,
    preserve_reference_margin: bool = False,
) -> BatchLinearization:
    """Build one exact FBPF linearization at ``theta``.

    Task and constraint gradients come from the same freshly recomputed model
    state.  Nonlinear backtracking candidates request neither and therefore do
    not retain an autograd graph.
    """

    requires_grad = need_task or need_vjp
    local_theta = theta.detach().to(dtype=torch.float32).requires_grad_(requires_grad)
    with torch.set_grad_enabled(requires_grad):
        logits = forward_logits(local_theta)
        losses = arm_loss(
            arm,
            logits,
            batch.base_logits,
            batch.gold,
            epsilon_tie=epsilon_tie,
            argmax_tie_aware_constraints=argmax_tie_aware_constraints,
            preserve_reference_margin=preserve_reference_margin,
        )
        constraints = losses.constraints

        task_gradient: Tensor | None = None
        if need_task:
            task_gradient = torch.autograd.grad(
                losses.total,
                local_theta,
                retain_graph=need_vjp,
                create_graph=False,
            )[0].detach().float()

        constraint_gradients: Tensor | None = None
        if constraints is not None and need_vjp:
            rows = constraint_batched_vjp(
                constraints,
                (local_theta,),
                retain_graph=False,
                create_graph=False,
            )[0]
            constraint_gradients = rows.reshape(rows.shape[0], -1).detach().double()

    if constraints is None:
        values = torch.empty(0, dtype=torch.float64, device=theta.device)
        maximum = -math.inf
        row_ids: tuple[int, ...] = ()
    else:
        values = constraints.block_max.detach().double()
        row_ids = tuple(int(value) for value in constraints.row_ids.tolist())
        if bool(constraints.protected_mask.any()):
            maximum = float(
                constraints.per_position.detach()[constraints.protected_mask]
                .max()
                .item()
            )
        else:
            maximum = -math.inf

    for name, value in (
        ("task_gradient", task_gradient),
        ("constraint_gradients", constraint_gradients),
    ):
        if value is not None and not bool(torch.isfinite(value).all().item()):
            raise FloatingPointError(f"non-finite {name} in FBPF linearization")
    return BatchLinearization(
        constraint_values=values,
        constraint_gradients=constraint_gradients,
        max_all_position_constraint=maximum,
        task_gradient=task_gradient,
        row_ids=row_ids,
    )


__all__ = [
    "FunctionalDFlashBatch",
    "FunctionalDFlashForward",
    "copy_flat_lora_",
    "engineering_gold_with_protected_prefix",
    "evaluate_flat_transaction",
    "flatten_current_lora",
]
