"""Core FBPF-DFlash primitives for the prospective-v2 route.

This module intentionally contains no data loading, model checkpoint loading, or
GPU-specific execution.  G0 authorizes only local implementation and CPU/mock
tests.  The real-model smoke and every scientific stage remain gated elsewhere.
"""

from __future__ import annotations

import hashlib
import math
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable, Iterable, Iterator, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn


FBPF_LORA_RANK = 16
FBPF_LORA_ALPHA = 16.0
FBPF_LORA_DROPOUT = 0.0
FBPF_LORA_INIT_SEED_BASE = 2_026_080_600
FBPF_EXPECTED_TRAINABLE_PARAMETERS = 1_835_008
FBPF_EPSILON_TIE = 1e-4
FBPF_TAU_FEASIBLE = 1e-5


def expected_lora_module_paths() -> tuple[str, ...]:
    """Return the frozen, lexicographically sorted native-LoRA module paths."""

    paths: list[str] = []
    for layer in (3, 4):
        for module in ("q_proj", "k_proj", "v_proj", "o_proj"):
            paths.append(f"layers.{layer}.self_attn.{module}")
        for module in ("gate_proj", "up_proj", "down_proj"):
            paths.append(f"layers.{layer}.mlp.{module}")
    return tuple(sorted(paths))


def lora_initialization_seed(training_seed: int) -> int:
    if isinstance(training_seed, bool) or not isinstance(training_seed, int):
        raise TypeError("training_seed must be an integer")
    if training_seed < 0:
        raise ValueError("training_seed must be non-negative")
    return FBPF_LORA_INIT_SEED_BASE + training_seed


def expected_dflash_lora_parameter_count(
    *,
    hidden_size: int = 2_560,
    intermediate_size: int = 9_728,
    attention_heads: int = 32,
    key_value_heads: int = 8,
    head_dim: int = 128,
    rank: int = FBPF_LORA_RANK,
    adapted_layers: int = 2,
) -> int:
    """Configuration-level count for q/k/v/o/gate/up/down LoRA pairs."""

    q_width = attention_heads * head_dim
    kv_width = key_value_heads * head_dim
    per_layer = rank * (
        (hidden_size + q_width)
        + 2 * (hidden_size + kv_width)
        + (q_width + hidden_size)
        + 3 * (hidden_size + intermediate_size)
    )
    return adapted_layers * per_layer


def _resolve_parent(root: nn.Module, path: str) -> tuple[nn.Module, str]:
    parent_path, separator, child_name = path.rpartition(".")
    if not separator:
        return root, path
    return root.get_submodule(parent_path), child_name


def _tensor_sha256(tensor: Tensor) -> str:
    contiguous = tensor.detach().contiguous().cpu()
    payload = contiguous.view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(payload).hexdigest()


class LoRALinear(nn.Module):
    """A minimal, auditable LoRA wrapper with float32 master parameters.

    The base linear is frozen.  The LoRA branch is evaluated in float32 and its
    delta is cast to the base output dtype immediately before the addition.
    Initialization consumes only the supplied CPU generator.
    """

    def __init__(
        self,
        base: nn.Linear,
        *,
        rank: int = FBPF_LORA_RANK,
        alpha: float = FBPF_LORA_ALPHA,
        generator: torch.Generator,
    ) -> None:
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError(f"base must be nn.Linear, got {type(base)!r}")
        if rank <= 0:
            raise ValueError("rank must be positive")
        if generator.device.type != "cpu":
            raise ValueError("the frozen LoRA initialization generator must be CPU")

        self.base = base
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scale = self.alpha / self.rank
        self.enabled = True

        for parameter in self.base.parameters():
            parameter.requires_grad_(False)

        # Draw on CPU so initialization is identical across execution devices.
        a_cpu = torch.empty(
            (self.rank, self.base.in_features), dtype=torch.float32, device="cpu"
        )
        nn.init.kaiming_uniform_(a_cpu, a=math.sqrt(5), generator=generator)
        b_cpu = torch.zeros(
            (self.base.out_features, self.rank), dtype=torch.float32, device="cpu"
        )
        device = self.base.weight.device
        self.lora_A = nn.Parameter(a_cpu.to(device=device))
        self.lora_B = nn.Parameter(b_cpu.to(device=device))

    def forward(self, inputs: Tensor) -> Tensor:
        base_output = self.base(inputs)
        if not self.enabled:
            return base_output
        # The FBPF contract requires a float32 adapter branch even when a
        # caller accidentally surrounds the released bf16 graph with autocast.
        # Disabling autocast locally makes that numerical boundary explicit.
        with torch.autocast(device_type=inputs.device.type, enabled=False):
            hidden = F.linear(inputs.to(dtype=torch.float32), self.lora_A)
            delta = F.linear(hidden, self.lora_B) * self.scale
        return base_output + delta.to(dtype=base_output.dtype)

    def merged_weight(self) -> Tensor:
        """Return the frozen float32-accumulated, base-dtype merged weight."""

        delta = self.scale * (self.lora_B.detach() @ self.lora_A.detach())
        return (self.base.weight.detach().float() + delta).to(
            dtype=self.base.weight.dtype
        )

    def parameter_hashes(self) -> dict[str, str]:
        return {"A": _tensor_sha256(self.lora_A), "B": _tensor_sha256(self.lora_B)}

    def extra_repr(self) -> str:
        return (
            f"in_features={self.base.in_features}, "
            f"out_features={self.base.out_features}, rank={self.rank}, "
            f"alpha={self.alpha}, scale={self.scale}"
        )


def iter_lora_modules(model: nn.Module) -> Iterator[tuple[str, LoRALinear]]:
    for name, module in sorted(model.named_modules(), key=lambda item: item[0]):
        if isinstance(module, LoRALinear):
            yield name, module


def inject_fbpf_lora(
    model: nn.Module,
    *,
    training_seed: int,
    module_paths: Sequence[str] | None = None,
    rank: int = FBPF_LORA_RANK,
    alpha: float = FBPF_LORA_ALPHA,
) -> tuple[str, ...]:
    """Inject native LoRA modules using one isolated, deterministic CPU RNG."""

    paths = tuple(
        sorted(expected_lora_module_paths() if module_paths is None else module_paths)
    )
    if len(paths) != len(set(paths)):
        raise ValueError("module_paths must be unique")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(lora_initialization_seed(training_seed))

    for path in paths:
        parent, child_name = _resolve_parent(model, path)
        base = getattr(parent, child_name)
        if isinstance(base, LoRALinear):
            raise ValueError(f"module is already LoRA-wrapped: {path}")
        if not isinstance(base, nn.Linear):
            raise TypeError(f"frozen LoRA path {path!r} is not nn.Linear")
        setattr(
            parent,
            child_name,
            LoRALinear(base, rank=rank, alpha=alpha, generator=generator),
        )
    return paths


def named_lora_parameters(model: nn.Module) -> tuple[tuple[str, nn.Parameter], ...]:
    parameters: list[tuple[str, nn.Parameter]] = []
    for module_name, module in iter_lora_modules(model):
        parameters.append((f"{module_name}.lora_A", module.lora_A))
        parameters.append((f"{module_name}.lora_B", module.lora_B))
    return tuple(parameters)


def count_lora_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for _, parameter in named_lora_parameters(model))


def lora_parameter_hashes(model: nn.Module) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for module_name, module in iter_lora_modules(model):
        for suffix, digest in module.parameter_hashes().items():
            hashes[f"{module_name}.{suffix}"] = digest
    return hashes


@contextmanager
def lora_disabled(model: nn.Module) -> Iterator[None]:
    modules = tuple(module for _, module in iter_lora_modules(model))
    previous = tuple(module.enabled for module in modules)
    try:
        for module in modules:
            module.enabled = False
        yield
    finally:
        for module, enabled in zip(modules, previous, strict=True):
            module.enabled = enabled


def merge_fbpf_lora_(model: nn.Module) -> tuple[str, ...]:
    """Merge every adapter into bf16/base-dtype weights and remove wrappers."""

    modules = tuple(iter_lora_modules(model))
    for path, wrapper in modules:
        with torch.no_grad():
            wrapper.base.weight.copy_(wrapper.merged_weight())
        parent, child_name = _resolve_parent(model, path)
        setattr(parent, child_name, wrapper.base)
    return tuple(path for path, _ in modules)


@dataclass(frozen=True)
class FlatParameterLayout:
    """Stable flatten/unflatten layout for float32 LoRA master parameters."""

    names: tuple[str, ...]
    shapes: tuple[torch.Size, ...]
    numels: tuple[int, ...]

    @classmethod
    def from_named_parameters(
        cls, named_parameters: Iterable[tuple[str, Tensor]]
    ) -> "FlatParameterLayout":
        items = tuple(named_parameters)
        names = tuple(name for name, _ in items)
        if names != tuple(sorted(names)):
            raise ValueError("named parameters must be supplied in sorted order")
        if len(names) != len(set(names)):
            raise ValueError("parameter names must be unique")
        return cls(
            names=names,
            shapes=tuple(parameter.shape for _, parameter in items),
            numels=tuple(parameter.numel() for _, parameter in items),
        )

    @property
    def total_numel(self) -> int:
        return sum(self.numels)

    def flatten(self, tensors: Mapping[str, Tensor]) -> Tensor:
        pieces: list[Tensor] = []
        for name, shape in zip(self.names, self.shapes, strict=True):
            tensor = tensors[name]
            if tensor.shape != shape:
                raise ValueError(f"shape mismatch for {name}: {tensor.shape} != {shape}")
            pieces.append(tensor.reshape(-1))
        if not pieces:
            return torch.empty(0, dtype=torch.float32)
        return torch.cat(pieces)

    def unflatten(self, vector: Tensor) -> dict[str, Tensor]:
        if vector.ndim != 1 or vector.numel() != self.total_numel:
            raise ValueError(
                f"expected a flat vector of length {self.total_numel}, got {vector.shape}"
            )
        result: dict[str, Tensor] = {}
        offset = 0
        for name, shape, numel in zip(
            self.names, self.shapes, self.numels, strict=True
        ):
            result[name] = vector[offset : offset + numel].view(shape)
            offset += numel
        return result


def dpace_loss(
    logits: Tensor,
    gold: Tensor,
    *,
    alpha: float = 0.5,
    reduction_divisor: int | None = None,
) -> Tensor:
    """Full, uncensored D-PACE loss with detached suffix-value weights."""

    if logits.ndim != 3:
        raise ValueError("logits must have shape [batch, positions, vocabulary]")
    if gold.shape != logits.shape[:2]:
        raise ValueError("gold must have shape [batch, positions]")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must lie in [0, 1]")
    batch_size = logits.shape[0]
    divisor = batch_size if reduction_divisor is None else reduction_divisor
    if divisor <= 0:
        raise ValueError("reduction_divisor must be positive")

    per_token = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), gold.reshape(-1), reduction="none"
    ).view_as(gold)
    with torch.no_grad():
        q = torch.exp(-per_token)
        smooth = (1.0 - alpha) * q + alpha
        prefix = torch.cumprod(smooth, dim=-1)
        weights = torch.flip(
            torch.cumsum(torch.flip(prefix, dims=(-1,)), dim=-1), dims=(-1,)
        )
    return (per_token * weights).sum() / float(divisor)


@dataclass(frozen=True)
class MarginState:
    logits: Tensor
    gold: Tensor
    predictions: Tensor
    non_gold_winner: Tensor
    gold_margin: Tensor
    first_mismatch: Tensor


def margin_state(logits: Tensor, gold: Tensor) -> MarginState:
    """Compute detached decisions and differentiable float32 gold margins."""

    if logits.ndim != 3 or gold.shape != logits.shape[:2]:
        raise ValueError("expected logits [B,L,V] and gold [B,L]")
    logits32 = logits.float()
    gold_long = gold.to(dtype=torch.long)
    with torch.no_grad():
        predictions = torch.argmax(logits32.detach(), dim=-1)
        competitor_scores = logits32.detach().clone()
        competitor_scores.scatter_(-1, gold_long.unsqueeze(-1), -torch.inf)
        non_gold_winner = torch.argmax(competitor_scores, dim=-1)
        positions = torch.arange(logits.shape[1], device=logits.device).view(1, -1)
        sentinel = torch.full_like(positions, logits.shape[1]).expand_as(gold_long)
        mismatch_positions = torch.where(
            predictions.ne(gold_long), positions.expand_as(gold_long), sentinel
        )
        first_mismatch = mismatch_positions.min(dim=-1).values

    gold_logits = logits32.gather(-1, gold_long.unsqueeze(-1)).squeeze(-1)
    competitor_logits = logits32.gather(
        -1, non_gold_winner.unsqueeze(-1)
    ).squeeze(-1)
    return MarginState(
        logits=logits32,
        gold=gold_long,
        predictions=predictions,
        non_gold_winner=non_gold_winner,
        gold_margin=gold_logits - competitor_logits,
        first_mismatch=first_mismatch,
    )


def protected_prefix_mask(base_first_mismatch: Tensor, positions: int) -> Tensor:
    if base_first_mismatch.ndim != 1:
        raise ValueError("base_first_mismatch must have shape [batch]")
    indices = torch.arange(positions, device=base_first_mismatch.device).view(1, -1)
    return indices < base_first_mismatch.detach().view(-1, 1)


def frontier_hinge_loss(
    state: MarginState,
    frontier: Tensor,
    *,
    epsilon_tie: float = FBPF_EPSILON_TIE,
) -> Tensor:
    """Mean first-miss hinge; an empty frontier has a differentiable zero."""

    frontier = frontier.detach().to(dtype=torch.long)
    valid = frontier < state.gold_margin.shape[1]
    if not bool(valid.any()):
        return state.gold_margin.sum() * 0.0
    rows = torch.arange(state.gold_margin.shape[0], device=state.logits.device)[valid]
    selected = state.gold_margin[rows, frontier[valid]]
    return torch.relu(torch.as_tensor(epsilon_tie, device=selected.device) - selected).mean()


@dataclass(frozen=True)
class ConstraintState:
    """Exact all-position constraints and detached block-max tie masks."""

    per_position: Tensor
    protected_mask: Tensor
    row_ids: Tensor
    block_max: Tensor
    tie_mask: Tensor

    @property
    def row_count(self) -> int:
        return int(self.row_ids.numel())

    def max_constraint(self) -> float:
        if self.row_count == 0:
            return -math.inf
        return float(self.block_max.detach().max().item())

    def feasible(self, tau_f: float = FBPF_TAU_FEASIBLE) -> bool:
        if not bool(self.protected_mask.any()):
            return True
        values = self.per_position.detach()[self.protected_mask]
        return bool(torch.all(values <= tau_f).item())


def build_constraint_state(
    adapted: MarginState,
    base_first_mismatch: Tensor,
    *,
    epsilon_tie: float = FBPF_EPSILON_TIE,
    argmax_tie_aware: bool = False,
    reference_gold_margin: Tensor | None = None,
) -> ConstraintState:
    mask = protected_prefix_mask(base_first_mismatch, adapted.gold_margin.shape[1])
    epsilon = torch.as_tensor(
        epsilon_tie, dtype=adapted.gold_margin.dtype, device=adapted.gold_margin.device
    )
    if argmax_tie_aware:
        # torch.argmax resolves exact ties to the lowest token id.  A zero
        # margin is therefore safe exactly when gold has the lower id; when a
        # non-gold winner has the lower id, retain a strictly positive margin.
        required_margin = torch.where(
            adapted.non_gold_winner < adapted.gold,
            epsilon,
            torch.zeros((), dtype=epsilon.dtype, device=epsilon.device),
        )
    else:
        required_margin = epsilon
    if reference_gold_margin is not None:
        if reference_gold_margin.shape != adapted.gold_margin.shape:
            raise ValueError("reference gold margins do not match adapted margins")
        required_margin = torch.maximum(
            required_margin,
            reference_gold_margin.detach().to(
                dtype=adapted.gold_margin.dtype,
                device=adapted.gold_margin.device,
            ),
        )
    constraints = required_margin - adapted.gold_margin
    row_ids = torch.nonzero(mask.any(dim=-1), as_tuple=False).flatten().detach()
    maxima: list[Tensor] = []
    ties: list[Tensor] = []
    for row in row_ids.tolist():
        row_values = constraints[row].masked_fill(~mask[row], -torch.inf)
        maximum = row_values.max()
        maxima.append(maximum)
        ties.append(((constraints[row] == maximum) & mask[row]).detach())
    if maxima:
        block_max = torch.stack(maxima)
        tie_mask = torch.stack(ties)
    else:
        block_max = constraints.new_empty((0,))
        tie_mask = torch.empty(
            (0, constraints.shape[1]), dtype=torch.bool, device=constraints.device
        )
    return ConstraintState(
        per_position=constraints,
        protected_mask=mask.detach(),
        row_ids=row_ids,
        block_max=block_max,
        tie_mask=tie_mask,
    )


def constraint_batched_vjp(
    constraints: ConstraintState,
    parameters: Sequence[Tensor],
    *,
    retain_graph: bool = False,
    create_graph: bool = False,
) -> tuple[Tensor, ...]:
    """Return one uniformly tie-averaged VJP row per nonempty block."""

    if not parameters:
        raise ValueError("at least one parameter is required")
    row_count = constraints.row_count
    if row_count == 0:
        return tuple(
            parameter.new_empty((0,) + tuple(parameter.shape))
            for parameter in parameters
        )

    grad_outputs = torch.zeros(
        (row_count,) + tuple(constraints.per_position.shape),
        dtype=constraints.per_position.dtype,
        device=constraints.per_position.device,
    )
    for output_row, block_row in enumerate(constraints.row_ids.tolist()):
        tie = constraints.tie_mask[output_row]
        grad_outputs[output_row, block_row, tie] = 1.0 / int(tie.sum().item())

    gradients = torch.autograd.grad(
        outputs=constraints.per_position,
        inputs=tuple(parameters),
        grad_outputs=grad_outputs,
        retain_graph=retain_graph,
        create_graph=create_graph,
        allow_unused=True,
        is_grads_batched=True,
    )
    materialized: list[Tensor] = []
    for parameter, gradient in zip(parameters, gradients, strict=True):
        if gradient is None:
            gradient = parameter.new_zeros((row_count,) + tuple(parameter.shape))
        materialized.append(gradient)
    return tuple(materialized)


def flatten_batched_vjp_rows(gradients: Sequence[Tensor]) -> Tensor:
    if not gradients:
        raise ValueError("at least one batched gradient tensor is required")
    row_count = gradients[0].shape[0]
    if any(gradient.shape[0] != row_count for gradient in gradients):
        raise ValueError("all batched gradients must have the same row count")
    if row_count == 0:
        total = sum(math.prod(gradient.shape[1:]) for gradient in gradients)
        return gradients[0].new_empty((0, total), dtype=torch.float64)
    return torch.cat(
        [gradient.reshape(row_count, -1) for gradient in gradients], dim=1
    ).to(dtype=torch.float64)


@dataclass(frozen=True)
class ArmLoss:
    total: Tensor
    dpace: Tensor
    frontier: Tensor
    adapted: MarginState
    base: MarginState | None
    constraints: ConstraintState | None


def arm_loss(
    arm: str,
    adapted_logits: Tensor,
    base_logits: Tensor | None,
    gold: Tensor,
    *,
    dpace_alpha: float = 0.5,
    frontier_scale: float = 1.0,
    epsilon_tie: float = FBPF_EPSILON_TIE,
    argmax_tie_aware_constraints: bool = False,
    preserve_reference_margin: bool = False,
) -> ArmLoss:
    """Compute one of the frozen A/B/C/D task objectives."""

    normalized_arm = arm.upper()
    if normalized_arm not in {"A", "B", "C", "D"}:
        raise ValueError("arm must be one of A, B, C, D")
    if adapted_logits.shape[0] != 4:
        raise ValueError("the frozen outer tensor batch size is exactly four")

    adapted = margin_state(adapted_logits, gold)
    task = dpace_loss(
        adapted_logits,
        gold,
        alpha=dpace_alpha,
        reduction_divisor=4,
    )
    base: MarginState | None = None
    constraints: ConstraintState | None = None
    if normalized_arm in {"B", "D"}:
        if base_logits is None:
            raise ValueError(f"arm {normalized_arm} requires frozen base logits")
        base = margin_state(base_logits.detach(), gold)
        constraints = build_constraint_state(
            adapted,
            base.first_mismatch,
            epsilon_tie=epsilon_tie,
            argmax_tie_aware=argmax_tie_aware_constraints,
            reference_gold_margin=(
                base.gold_margin if preserve_reference_margin else None
            ),
        )

    if normalized_arm == "B":
        assert base is not None
        frontier = frontier_hinge_loss(
            adapted, base.first_mismatch, epsilon_tie=epsilon_tie
        )
    elif normalized_arm in {"C", "D"}:
        frontier = frontier_hinge_loss(
            adapted, adapted.first_mismatch, epsilon_tie=epsilon_tie
        )
    else:
        frontier = task * 0.0
    return ArmLoss(
        total=task + frontier_scale * frontier,
        dpace=task,
        frontier=frontier,
        adapted=adapted,
        base=base,
        constraints=constraints,
    )


def cosine_warmup_learning_rate(
    k_outer: int,
    *,
    total_steps: int = 8_000,
    peak: float = 1e-4,
    warmup_ratio: float = 0.04,
) -> float:
    """One-indexed 4% linear warmup followed by cosine decay to zero."""

    if not 0 <= k_outer < total_steps:
        raise ValueError("k_outer must be in [0, total_steps)")
    warmup_steps = int(round(total_steps * warmup_ratio))
    step = k_outer + 1
    if step <= warmup_steps:
        return peak * step / warmup_steps
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    return peak * 0.5 * (1.0 + math.cos(math.pi * progress))


@dataclass(frozen=True)
class TransactionState:
    theta: Tensor
    first_moment: Tensor
    second_moment: Tensor
    k_outer: int = 0
    t_adam: int = 0

    @classmethod
    def initialize(cls, theta: Tensor) -> "TransactionState":
        if theta.ndim != 1:
            raise ValueError("theta must be a flat vector")
        theta32 = theta.detach().to(dtype=torch.float32).clone()
        return cls(
            theta=theta32,
            first_moment=torch.zeros_like(theta32),
            second_moment=torch.zeros_like(theta32),
        )


@dataclass(frozen=True)
class BatchLinearization:
    """Complete exact constraint evaluation at one candidate parameter vector."""

    constraint_values: Tensor
    constraint_gradients: Tensor | None
    max_all_position_constraint: float
    task_gradient: Tensor | None = None
    row_ids: tuple[int, ...] = ()

    def feasible(self, tau_f: float = FBPF_TAU_FEASIBLE) -> bool:
        return self.max_all_position_constraint <= tau_f

    def violation(self, tau_f: float = FBPF_TAU_FEASIBLE) -> float:
        return max(0.0, self.max_all_position_constraint - tau_f)


def _validate_linearization(
    evaluation: BatchLinearization,
    parameter_count: int,
    *,
    need_task: bool,
    need_vjp: bool,
) -> None:
    if evaluation.constraint_values.ndim != 1:
        raise ValueError("constraint_values must be a vector")
    rows = evaluation.constraint_values.numel()
    if evaluation.row_ids and len(evaluation.row_ids) != rows:
        raise ValueError("row_ids must align with constraint_values")
    if evaluation.constraint_gradients is None:
        if rows and need_vjp:
            raise ValueError("constraint gradients are required for nonempty rows")
    elif evaluation.constraint_gradients.shape != (rows, parameter_count):
        raise ValueError("constraint gradient matrix has the wrong shape")
    if need_task:
        if evaluation.task_gradient is None:
            raise ValueError("task gradient is required")
        if evaluation.task_gradient.shape != (parameter_count,):
            raise ValueError("task gradient has the wrong shape")


def _stable_descending_order(values: Tensor, row_ids: tuple[int, ...]) -> list[int]:
    ids = row_ids or tuple(range(values.numel()))
    return sorted(range(values.numel()), key=lambda i: (-float(values[i]), ids[i]))


class ProjectionBudgetExhausted(RuntimeError):
    """Raised when the frozen cyclic projection budget cannot find a direction."""


def project_task_direction(
    direction: Tensor,
    constraint_values: Tensor,
    constraint_gradients: Tensor,
    *,
    row_ids: tuple[int, ...] = (),
    max_sweeps: int = 4,
    tau_linear: float = 1e-7,
    norm_epsilon: float = 1e-12,
) -> Tensor:
    """Frozen cyclic linearized projection with stable block-row tie order."""

    projected = direction.to(dtype=torch.float64).clone()
    values = constraint_values.to(dtype=torch.float64)
    gradients = constraint_gradients.to(dtype=torch.float64)
    if gradients.shape != (values.numel(), projected.numel()):
        raise ValueError("constraint gradients do not match direction")
    for _ in range(max_sweeps):
        sweep_residual = values + gradients @ projected
        for row in _stable_descending_order(sweep_residual, row_ids):
            gradient = gradients[row]
            residual = values[row] + torch.dot(gradient, projected)
            if float(residual) > tau_linear:
                projected = projected - residual * gradient / (
                    torch.dot(gradient, gradient) + norm_epsilon
                )
    final_residual = values + gradients @ projected
    if bool(torch.any(final_residual > tau_linear).item()):
        raise ProjectionBudgetExhausted(
            "linearized prefix-feasibility projection exhausted its frozen "
            f"{max_sweeps}-sweep budget"
        )
    return projected


def _restoration_direction(
    evaluation: BatchLinearization,
    parameter_count: int,
    *,
    tau_restore: float,
    norm_epsilon: float,
) -> Tensor:
    direction = torch.zeros(
        parameter_count,
        dtype=torch.float64,
        device=evaluation.constraint_values.device,
    )
    if evaluation.constraint_values.numel() == 0:
        return direction
    assert evaluation.constraint_gradients is not None
    values = evaluation.constraint_values.to(dtype=torch.float64)
    gradients = evaluation.constraint_gradients.to(dtype=torch.float64)
    for row in _stable_descending_order(values, evaluation.row_ids):
        gradient = gradients[row]
        residual = values[row] + tau_restore + torch.dot(gradient, direction)
        if float(residual) > 0.0:
            direction = direction - residual * gradient / (
                torch.dot(gradient, gradient) + norm_epsilon
            )
    return direction


@dataclass(frozen=True)
class TransactionResult:
    state: TransactionState
    status: str
    restored: bool
    restoration_cycles: int
    attempted_alphas: tuple[float, ...]
    aborted: bool


class TransactionalAdamW:
    """Pure transactional AdamW engine for constrained B/D arms.

    ``evaluate(theta, need_task, need_vjp)`` must recompute exact all-position
    constraints at ``theta``.  The boolean flags request the task gradient and
    complete blockwise VJP matrix independently; nonlinear candidates require
    neither, while each relinearization requires the VJP.
    """

    def __init__(
        self,
        *,
        beta1: float = 0.9,
        beta2: float = 0.95,
        epsilon: float = 1e-8,
        gradient_clip: float = 1.0,
        tau_f: float = FBPF_TAU_FEASIBLE,
        tau_linear: float = 1e-7,
        tau_restore: float = 1e-4,
        minimum_violation_decrease: float = 1e-7,
        max_projection_sweeps: int = 4,
        max_restoration_cycles: int = 8,
        candidate_alphas: Sequence[float] = (
            1.0,
            0.5,
            0.25,
            0.125,
            0.0625,
            0.03125,
            0.015625,
            0.0078125,
        ),
        norm_epsilon: float = 1e-12,
    ) -> None:
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.gradient_clip = gradient_clip
        self.tau_f = tau_f
        self.tau_linear = tau_linear
        self.tau_restore = tau_restore
        self.minimum_violation_decrease = minimum_violation_decrease
        self.max_projection_sweeps = max_projection_sweeps
        self.max_restoration_cycles = max_restoration_cycles
        self.candidate_alphas = tuple(float(alpha) for alpha in candidate_alphas)
        self.norm_epsilon = norm_epsilon

    def step(
        self,
        state: TransactionState,
        evaluate: Callable[[Tensor, bool, bool], BatchLinearization],
        *,
        learning_rate: float,
    ) -> TransactionResult:
        if state.theta.ndim != 1:
            raise ValueError("state theta must be flat")
        parameter_count = state.theta.numel()
        batch_start = TransactionState(
            theta=state.theta.detach().clone(),
            first_moment=state.first_moment.detach().clone(),
            second_moment=state.second_moment.detach().clone(),
            k_outer=state.k_outer,
            t_adam=state.t_adam,
        )
        current_theta = batch_start.theta
        evaluation = evaluate(current_theta, True, True)
        _validate_linearization(
            evaluation, parameter_count, need_task=True, need_vjp=True
        )
        restored = False
        restoration_cycles = 0

        if not evaluation.feasible(self.tau_f):
            restored = True
            previous_violation = evaluation.violation(self.tau_f)
            for cycle in range(1, self.max_restoration_cycles + 1):
                restoration_cycles = cycle
                direction = _restoration_direction(
                    evaluation,
                    parameter_count,
                    tau_restore=self.tau_restore,
                    norm_epsilon=self.norm_epsilon,
                )
                candidate_theta = current_theta + direction.to(
                    dtype=current_theta.dtype, device=current_theta.device
                )
                candidate_evaluation = evaluate(candidate_theta, False, False)
                _validate_linearization(
                    candidate_evaluation,
                    parameter_count,
                    need_task=False,
                    need_vjp=False,
                )
                new_violation = candidate_evaluation.violation(self.tau_f)
                if previous_violation - new_violation < self.minimum_violation_decrease:
                    failed = TransactionState(
                        theta=batch_start.theta,
                        first_moment=batch_start.first_moment,
                        second_moment=batch_start.second_moment,
                        k_outer=batch_start.k_outer + 1,
                        t_adam=batch_start.t_adam,
                    )
                    return TransactionResult(
                        state=failed,
                        status="restoration_failed",
                        restored=False,
                        restoration_cycles=restoration_cycles,
                        attempted_alphas=(),
                        aborted=True,
                    )
                current_theta = candidate_theta
                evaluation = candidate_evaluation
                previous_violation = new_violation
                if evaluation.feasible(self.tau_f):
                    break
                # A new VJP is required for the next relinearized cycle.
                evaluation = evaluate(current_theta, False, True)
                _validate_linearization(
                    evaluation, parameter_count, need_task=False, need_vjp=True
                )
            else:
                failed = TransactionState(
                    theta=batch_start.theta,
                    first_moment=batch_start.first_moment,
                    second_moment=batch_start.second_moment,
                    k_outer=batch_start.k_outer + 1,
                    t_adam=batch_start.t_adam,
                )
                return TransactionResult(
                    state=failed,
                    status="restoration_failed",
                    restored=False,
                    restoration_cycles=restoration_cycles,
                    attempted_alphas=(),
                    aborted=True,
                )

            # Contractual post-restoration recomputation of every task quantity.
            evaluation = evaluate(current_theta, True, True)
            _validate_linearization(
                evaluation, parameter_count, need_task=True, need_vjp=True
            )
            if not evaluation.feasible(self.tau_f):
                failed = TransactionState(
                    theta=batch_start.theta,
                    first_moment=batch_start.first_moment,
                    second_moment=batch_start.second_moment,
                    k_outer=batch_start.k_outer + 1,
                    t_adam=batch_start.t_adam,
                )
                return TransactionResult(
                    state=failed,
                    status="restoration_failed",
                    restored=False,
                    restoration_cycles=restoration_cycles,
                    attempted_alphas=(),
                    aborted=True,
                )

        assert evaluation.task_gradient is not None
        gradient = evaluation.task_gradient.detach().to(dtype=torch.float32)
        gradient_norm = torch.linalg.vector_norm(gradient)
        if float(gradient_norm) > self.gradient_clip:
            gradient = gradient * (self.gradient_clip / float(gradient_norm))

        shadow_first = self.beta1 * batch_start.first_moment + (1.0 - self.beta1) * gradient
        shadow_second = self.beta2 * batch_start.second_moment + (
            1.0 - self.beta2
        ) * gradient.square()
        next_t = batch_start.t_adam + 1
        first_hat = shadow_first / (1.0 - self.beta1**next_t)
        second_hat = shadow_second / (1.0 - self.beta2**next_t)
        raw_direction = -learning_rate * first_hat / (
            torch.sqrt(second_hat) + self.epsilon
        )

        if evaluation.constraint_values.numel():
            assert evaluation.constraint_gradients is not None
            direction = project_task_direction(
                raw_direction,
                evaluation.constraint_values,
                evaluation.constraint_gradients,
                row_ids=evaluation.row_ids,
                max_sweeps=self.max_projection_sweeps,
                tau_linear=self.tau_linear,
                norm_epsilon=self.norm_epsilon,
            )
        else:
            direction = raw_direction.to(dtype=torch.float64)

        attempted: list[float] = []
        for alpha in self.candidate_alphas:
            attempted.append(alpha)
            candidate_theta = current_theta + alpha * direction.to(
                dtype=current_theta.dtype, device=current_theta.device
            )
            candidate_evaluation = evaluate(candidate_theta, False, False)
            _validate_linearization(
                candidate_evaluation,
                parameter_count,
                need_task=False,
                need_vjp=False,
            )
            if candidate_evaluation.feasible(self.tau_f):
                committed = TransactionState(
                    theta=candidate_theta.detach().clone(),
                    first_moment=shadow_first.detach().clone(),
                    second_moment=shadow_second.detach().clone(),
                    k_outer=batch_start.k_outer + 1,
                    t_adam=next_t,
                )
                return TransactionResult(
                    state=committed,
                    status="committed",
                    restored=restored,
                    restoration_cycles=restoration_cycles,
                    attempted_alphas=tuple(attempted),
                    aborted=False,
                )

        skipped = TransactionState(
            theta=current_theta.detach().clone(),
            first_moment=batch_start.first_moment,
            second_moment=batch_start.second_moment,
            k_outer=batch_start.k_outer + 1,
            t_adam=batch_start.t_adam,
        )
        return TransactionResult(
            state=skipped,
            status="task_skipped",
            restored=restored,
            restoration_cycles=restoration_cycles,
            attempted_alphas=tuple(attempted),
            aborted=False,
        )
