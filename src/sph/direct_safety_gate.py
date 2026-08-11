"""Producer-reusing binary safety gate for a frozen Direct selector.

This module contains only gold-free model semantics and small mathematical
helpers.  It does not load canonical datasets, checkpoints, or validation
artifacts.  The staged experiment protocol lives in
``sph.direct_safety_protocol``.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Any, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from sph.first_miss_action_selector import realized_prefix_lengths
from sph.global_direct_selector import GlobalDirectOutput


BLOCK_LENGTH = 15
CANDIDATE_COUNT = 16
NODE_DIMENSION = 64
POSITION_FEATURE_DIMENSION = 200
BLOCK_FEATURE_DIMENSION = 257
SCALAR_COMPARATOR_DIMENSION = 21
SIDECAR_PARAMETER_COUNT = 38_674


@dataclass
class DirectSafetyFeatureOutput:
    """Gold-free features derived from one frozen Direct forward pass."""

    position_features: Tensor
    direct_path: Tensor
    change_mask: Tensor


@dataclass
class DirectSafetyOutput:
    """One block-level KEEP/APPLY score and its frozen Direct path."""

    scores: Tensor
    apply_direct: Tensor
    direct_path: Tensor
    change_mask: Tensor


@dataclass
class GainWeightedHingeOutput:
    """Utility-aligned binary hinge and independently decoded regret."""

    loss: Tensor
    per_block_loss: Tensor
    weighted_per_block_loss: Tensor
    apply_direct: Tensor
    decoded_regret: Tensor
    bound_slack: Tensor


@dataclass
class BinaryOutcomeOutput:
    """Exact DFlash/Direct accepted-prefix outcomes for synthetic tensors."""

    base_lengths: Tensor
    direct_lengths: Tensor
    normalized_gains: Tensor
    oracle_apply_direct: Tensor


def _require_finite(name: str, value: Tensor) -> None:
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must be finite")


def freeze_direct_producer(producer: nn.Module) -> nn.Module:
    """Freeze a Direct producer without changing its parameter values."""

    producer.requires_grad_(False)
    producer.eval()
    return producer


@torch.no_grad()
def frozen_direct_forward_with_states(
    producer: nn.Module,
    *args: Any,
    **kwargs: Any,
) -> tuple[GlobalDirectOutput, Tensor]:
    """Run a frozen Direct producer and capture pre-projection node states.

    A transient forward-pre-hook observes the input to the existing residual
    projection.  It does not replace a module or change the producer state
    dictionary.  The producer must already be explicitly frozen and in eval
    mode; callers cannot silently rely on this helper to repair a mutable
    training configuration.
    """

    if producer.training:
        raise ValueError("Direct producer must be in evaluation mode")
    trainable = [name for name, value in producer.named_parameters() if value.requires_grad]
    if trainable:
        raise ValueError(
            "Direct producer must be frozen before state capture: "
            f"{trainable[:3]}"
        )
    projection = getattr(producer, "residual_projection", None)
    if not isinstance(projection, nn.Linear):
        raise ValueError("Direct producer lacks the expected residual projection")

    captured: list[Tensor] = []

    def capture_input(_module: nn.Module, inputs: tuple[Tensor, ...]) -> None:
        if len(inputs) != 1 or not isinstance(inputs[0], Tensor):
            raise RuntimeError("unexpected residual-projection input")
        captured.append(inputs[0].detach())

    handle = projection.register_forward_pre_hook(capture_input)
    try:
        output = producer(*args, **kwargs)
    finally:
        handle.remove()
    if not isinstance(output, GlobalDirectOutput):
        raise TypeError("Direct producer returned an unexpected output type")
    if len(captured) != 1:
        raise RuntimeError(
            "Direct producer must invoke its residual projection exactly once"
        )
    if captured[0].requires_grad:
        raise RuntimeError("captured Direct states were not detached")
    return output, captured[0]


def direct_safety_position_features(
    node_states: Tensor,
    direct_output: GlobalDirectOutput,
    candidate_logits: Tensor,
    base_logsumexp: Tensor,
) -> DirectSafetyFeatureOutput:
    """Construct the exact 200-dimensional PROS-Gate position features."""

    if node_states.ndim != 4:
        raise ValueError("node_states must have shape [B, 15, 16, 64]")
    batch, length, candidates, node_dim = node_states.shape
    expected = (batch, length, candidates)
    if (length, candidates, node_dim) != (
        BLOCK_LENGTH,
        CANDIDATE_COUNT,
        NODE_DIMENSION,
    ):
        raise ValueError("node states differ from the frozen 15x16x64 contract")
    for name, value in (
        ("Direct scores", direct_output.scores),
        ("Direct residual scores", direct_output.residual_scores),
        ("DFlash log probabilities", direct_output.base_log_probs),
        ("candidate logits", candidate_logits),
    ):
        if value.shape != expected:
            raise ValueError(f"{name} must have shape [B, 15, 16]")
    if base_logsumexp.shape != (batch, length):
        raise ValueError("base_logsumexp must have shape [B, 15]")

    states = node_states.detach().float()
    scores = direct_output.scores.detach().float()
    residual = direct_output.residual_scores.detach().float()
    base_log_probs = direct_output.base_log_probs.detach().float()
    logits = candidate_logits.detach().float()
    full_lse = base_logsumexp.detach().float()
    for name, value in (
        ("node states", states),
        ("Direct scores", scores),
        ("Direct residual scores", residual),
        ("DFlash log probabilities", base_log_probs),
        ("candidate logits", logits),
        ("base logsumexp", full_lse),
    ):
        _require_finite(name, value)

    expected_base = logits - full_lse.unsqueeze(-1)
    if not torch.equal(base_log_probs, expected_base):
        raise ValueError(
            "DFlash log probabilities differ from logits-base_logsumexp"
        )
    if not torch.allclose(
        scores,
        base_log_probs + residual,
        rtol=0.0,
        atol=1e-6,
    ):
        raise ValueError("Direct scores differ from base plus residual scores")

    direct_path = scores.argmax(dim=-1)
    gather_state = direct_path[..., None, None].expand(
        batch, length, 1, node_dim
    )
    chosen_states = states.gather(2, gather_state).squeeze(2)
    base_states = states[:, :, 0]
    state_difference = chosen_states - base_states

    gather_score = direct_path.unsqueeze(-1)
    chosen_scores = scores.gather(-1, gather_score).squeeze(-1)
    chosen_residual = residual.gather(-1, gather_score).squeeze(-1)
    chosen_base = base_log_probs.gather(-1, gather_score).squeeze(-1)
    total_margin = chosen_scores - scores[:, :, 0]
    residual_margin = chosen_residual - residual[:, :, 0]
    base_difference = chosen_base - base_log_probs[:, :, 0]

    log_q = torch.log_softmax(logits, dim=-1)
    q = log_q.exp()
    entropy = -(q * log_q).sum(dim=-1) / math.log(CANDIDATE_COUNT)
    retained_mass = torch.tanh(
        (torch.logsumexp(logits, dim=-1) - full_lse) / 2.0
    )
    change_mask = direct_path.ne(0)
    rank_feature = direct_path.float() / float(CANDIDATE_COUNT - 1)
    position_feature = (
        torch.arange(length, device=states.device, dtype=torch.float32)
        / float(length - 1)
    )[None].expand(batch, -1)
    scalar_features = torch.stack(
        [
            total_margin,
            residual_margin,
            base_difference,
            rank_feature,
            position_feature,
            change_mask.float(),
            entropy,
            retained_mass,
        ],
        dim=-1,
    )
    position_features = torch.cat(
        [chosen_states, base_states, state_difference, scalar_features],
        dim=-1,
    )
    if position_features.shape != (
        batch,
        BLOCK_LENGTH,
        POSITION_FEATURE_DIMENSION,
    ):
        raise RuntimeError("constructed position features have an invalid shape")
    _require_finite("position features", position_features)
    return DirectSafetyFeatureOutput(
        position_features=position_features,
        direct_path=direct_path,
        change_mask=change_mask,
    )


def scalar_comparator_features(
    position_features: Tensor,
    change_mask: Tensor,
) -> Tensor:
    """Construct the frozen 21-dimensional scalar comparator features.

    The order is binding: normalized change count; sum/mean/min/max of the
    Direct-total, Direct-residual, and DFlash margins on changed positions;
    mean/min/max entropy; mean/min/max retained mass; first-position change;
    and first-position Direct-total margin.  Changed-position summaries are
    exactly zero when a block contains no Direct change.
    """

    if position_features.ndim != 3:
        raise ValueError("position_features must have shape [B, 15, 200]")
    batch, length, dimensions = position_features.shape
    if (length, dimensions) != (
        BLOCK_LENGTH,
        POSITION_FEATURE_DIMENSION,
    ):
        raise ValueError("position features differ from the 15x200 contract")
    if change_mask.shape != (batch, length) or change_mask.dtype != torch.bool:
        raise ValueError("change_mask must be boolean with shape [B, 15]")

    values = position_features.detach().float()
    _require_finite("position features", values)
    scalars = values[..., 192:]
    if bool(((scalars[..., 5] != 0.0) & (scalars[..., 5] != 1.0)).any()):
        raise ValueError("scalar change feature must be exactly zero or one")
    if not torch.equal(change_mask, scalars[..., 5].ne(0)):
        raise ValueError("change mask differs from the scalar change feature")

    change_float = change_mask.to(dtype=torch.float32)
    change_count = change_float.sum(dim=1)
    has_change = change_count.gt(0)
    outputs = [(change_count / float(BLOCK_LENGTH)).unsqueeze(-1)]
    negative_infinity = torch.finfo(torch.float32).min
    positive_infinity = torch.finfo(torch.float32).max
    for scalar_index in (0, 1, 2):
        margin = scalars[..., scalar_index]
        changed_sum = (margin * change_float).sum(dim=1)
        changed_mean = changed_sum / change_count.clamp_min(1.0)
        changed_min = margin.masked_fill(~change_mask, positive_infinity).amin(dim=1)
        changed_max = margin.masked_fill(~change_mask, negative_infinity).amax(dim=1)
        changed_min = torch.where(
            has_change, changed_min, torch.zeros_like(changed_min)
        )
        changed_max = torch.where(
            has_change, changed_max, torch.zeros_like(changed_max)
        )
        outputs.append(
            torch.stack(
                [changed_sum, changed_mean, changed_min, changed_max], dim=-1
            )
        )

    for scalar_index in (6, 7):
        scalar = scalars[..., scalar_index]
        outputs.append(
            torch.stack(
                [scalar.mean(dim=1), scalar.amin(dim=1), scalar.amax(dim=1)],
                dim=-1,
            )
        )
    outputs.extend(
        [
            scalars[:, 0, 5].unsqueeze(-1),
            scalars[:, 0, 0].unsqueeze(-1),
        ]
    )
    result = torch.cat(outputs, dim=-1)
    if result.shape != (batch, SCALAR_COMPARATOR_DIMENSION):
        raise RuntimeError("scalar comparator features have an invalid shape")
    _require_finite("scalar comparator features", result)
    return result


class DirectSafetySidecar(nn.Module):
    """The exact 38,674-parameter frozen-producer safety sidecar."""

    def __init__(self, *, initialization_seed: int = 0) -> None:
        super().__init__()
        construction_rng_state = torch.random.get_rng_state()
        self.initialization_seed = int(initialization_seed)
        self.position_norm = nn.LayerNorm(
            POSITION_FEATURE_DIMENSION,
            eps=1e-5,
            elementwise_affine=True,
        )
        self.position_input = nn.Linear(
            POSITION_FEATURE_DIMENSION, NODE_DIMENSION, bias=True
        )
        self.position_output = nn.Linear(
            NODE_DIMENSION, NODE_DIMENSION, bias=True
        )
        self.block_norm = nn.LayerNorm(
            BLOCK_FEATURE_DIMENSION,
            eps=1e-5,
            elementwise_affine=True,
        )
        self.block_input = nn.Linear(
            BLOCK_FEATURE_DIMENSION, NODE_DIMENSION, bias=True
        )
        self.block_output = nn.Linear(
            NODE_DIMENSION, NODE_DIMENSION, bias=True
        )
        self.final_projection = nn.Linear(
            NODE_DIMENSION, 1, bias=False
        )
        self._reset_parameters_deterministically()
        torch.random.set_rng_state(construction_rng_state)
        parameter_count = sum(value.numel() for value in self.parameters())
        if parameter_count != SIDECAR_PARAMETER_COUNT:
            raise RuntimeError(
                "sidecar parameter count differs from 38,674: "
                f"{parameter_count}"
            )

    def _named_seed(self, name: str) -> int:
        payload = f"{self.initialization_seed}:{name}".encode("utf-8")
        return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (
            2**63 - 1
        )

    def _reset_parameters_deterministically(self) -> None:
        state = torch.random.get_rng_state()
        try:
            for name, module in self.named_modules():
                if not name:
                    continue
                reset = getattr(module, "reset_parameters", None)
                if callable(reset):
                    torch.default_generator.manual_seed(self._named_seed(name))
                    reset()
            nn.init.zeros_(self.final_projection.weight)
        finally:
            torch.random.set_rng_state(state)

    def forward(
        self,
        position_features: Tensor,
        change_mask: Tensor,
        direct_path: Tensor,
    ) -> DirectSafetyOutput:
        if position_features.ndim != 3:
            raise ValueError("position_features must have shape [B, 15, 200]")
        batch, length, feature_dim = position_features.shape
        if (length, feature_dim) != (
            BLOCK_LENGTH,
            POSITION_FEATURE_DIMENSION,
        ):
            raise ValueError("position features differ from the 15x200 contract")
        if change_mask.shape != (batch, length) or change_mask.dtype != torch.bool:
            raise ValueError("change_mask must be boolean with shape [B, 15]")
        if direct_path.shape != (batch, length):
            raise ValueError("direct_path must have shape [B, 15]")
        if direct_path.dtype == torch.bool or direct_path.is_floating_point():
            raise ValueError("direct_path must use an integer dtype")
        if bool(((direct_path < 0) | (direct_path >= CANDIDATE_COUNT)).any()):
            raise ValueError("direct_path indices must lie in [0, 15]")
        if not torch.equal(change_mask, direct_path.ne(0)):
            raise ValueError("change mask differs from the Direct path")
        features = position_features.detach().float()
        _require_finite("position features", features)

        encoded = self.position_output(
            F.silu(self.position_input(self.position_norm(features)))
        )
        all_mean = encoded.mean(dim=1)
        change_float = change_mask.float()
        change_count = change_float.sum(dim=1)
        changed_mean = (
            encoded * change_float.unsqueeze(-1)
        ).sum(dim=1) / change_count.clamp_min(1.0).unsqueeze(-1)
        negative_infinity = torch.finfo(encoded.dtype).min
        masked = encoded.masked_fill(
            ~change_mask.unsqueeze(-1), negative_infinity
        )
        changed_max = masked.amax(dim=1)
        has_change = change_count.gt(0)
        changed_max = torch.where(
            has_change.unsqueeze(-1),
            changed_max,
            torch.zeros_like(changed_max),
        )
        block_features = torch.cat(
            [
                all_mean,
                changed_mean,
                changed_max,
                encoded[:, 0],
                (change_count / float(BLOCK_LENGTH)).unsqueeze(-1),
            ],
            dim=-1,
        )
        if block_features.shape != (batch, BLOCK_FEATURE_DIMENSION):
            raise RuntimeError("pooled block features have an invalid shape")
        block_state = self.block_output(
            F.silu(self.block_input(self.block_norm(block_features)))
        )
        raw_scores = self.final_projection(block_state).squeeze(-1)
        scores = raw_scores * has_change.to(raw_scores.dtype)
        _require_finite("sidecar scores", scores)
        return DirectSafetyOutput(
            scores=scores,
            apply_direct=scores.gt(0),
            direct_path=direct_path,
            change_mask=change_mask,
        )

    def forward_from_direct(
        self,
        node_states: Tensor,
        direct_output: GlobalDirectOutput,
        candidate_logits: Tensor,
        base_logsumexp: Tensor,
    ) -> DirectSafetyOutput:
        features = direct_safety_position_features(
            node_states,
            direct_output,
            candidate_logits,
            base_logsumexp,
        )
        return self(
            features.position_features,
            features.change_mask,
            features.direct_path,
        )


def normalized_signed_gain(
    base_lengths: Tensor,
    direct_lengths: Tensor,
    *,
    block_length: int = BLOCK_LENGTH,
) -> Tensor:
    """Return ``(A_D-A_B)/L`` after validating accepted-prefix lengths."""

    if base_lengths.shape != direct_lengths.shape:
        raise ValueError("base and Direct lengths must have equal shape")
    if base_lengths.ndim != 1:
        raise ValueError("accepted-prefix lengths must have shape [B]")
    if block_length < 1:
        raise ValueError("block_length must be positive")
    for name, value in (("base", base_lengths), ("Direct", direct_lengths)):
        if value.dtype == torch.bool or value.is_floating_point():
            raise ValueError(f"{name} lengths must use an integer dtype")
        if bool(((value < 0) | (value > block_length)).any()):
            raise ValueError(f"{name} length is outside [0, block_length]")
    return (direct_lengths - base_lengths).float() / float(block_length)


def gain_weighted_unit_hinge(
    scores: Tensor,
    normalized_gains: Tensor,
    *,
    example_weights: Tensor | None = None,
    reduction: str = "mean",
) -> GainWeightedHingeOutput:
    """Weightable, utility-consistent binary unit-margin hinge.

    ``mean`` is the ordinary prompt-balanced fit convention, where weights are
    multipliers whose block mean is the declared risk.  ``sum`` is reserved
    for probability-mass weights such as the capacity protocol's exact
    ``1/512`` record masses.
    """

    if scores.shape != normalized_gains.shape or scores.ndim != 1:
        raise ValueError("scores and gains must have equal shape [B]")
    if reduction not in {"mean", "sum"}:
        raise ValueError("reduction must be 'mean' or 'sum'")
    scores = scores.float()
    gains = normalized_gains.float()
    _require_finite("scores", scores)
    _require_finite("normalized gains", gains)
    if bool((gains.abs() > 1.0 + 1e-7).any()):
        raise ValueError("normalized gains must lie in [-1, 1]")
    if example_weights is None:
        weights = torch.ones_like(scores)
    else:
        if example_weights.shape != scores.shape:
            raise ValueError("example weights must have shape [B]")
        weights = example_weights.detach().float()
        _require_finite("example weights", weights)
        if bool((weights <= 0).any()):
            raise ValueError("example weights must be strictly positive")

    labels = gains.sign()
    active = gains.ne(0)
    per_block_loss = torch.where(
        active,
        gains.abs() * F.relu(1.0 - labels * scores),
        torch.zeros_like(scores),
    )
    apply_direct = scores.gt(0)
    decoded_regret = torch.where(
        gains.gt(0) & ~apply_direct,
        gains,
        torch.where(
            gains.lt(0) & apply_direct,
            -gains,
            torch.zeros_like(gains),
        ),
    )
    bound_slack = per_block_loss - decoded_regret
    if float(bound_slack.detach().min()) < -1e-6:
        raise RuntimeError("gain-weighted hinge violated decoded regret bound")
    weighted = weights * per_block_loss
    loss = weighted.mean() if reduction == "mean" else weighted.sum()
    return GainWeightedHingeOutput(
        loss=loss,
        per_block_loss=per_block_loss,
        weighted_per_block_loss=weighted,
        apply_direct=apply_direct,
        decoded_regret=decoded_regret,
        bound_slack=bound_slack,
    )


def capacity_gain_weighted_unit_hinge(
    scores: Tensor,
    normalized_gains: Tensor,
) -> GainWeightedHingeOutput:
    """Apply the exact 512-record capacity probability-mass convention."""

    if scores.ndim != 1 or scores.numel() != 512:
        raise ValueError("capacity loss requires exactly 512 records")
    masses = torch.full_like(scores.float(), 1.0 / 512.0)
    return gain_weighted_unit_hinge(
        scores,
        normalized_gains,
        example_weights=masses,
        reduction="sum",
    )


def prompt_balanced_example_weights(
    sample_ids: Sequence[str],
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Return record weights whose block mean equals a prompt-balanced mean."""

    if not sample_ids:
        raise ValueError("sample_ids cannot be empty")
    counts: dict[str, int] = {}
    for sample_id in sample_ids:
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError("sample IDs must be nonempty strings")
        counts[sample_id] = counts.get(sample_id, 0) + 1
    if dtype not in {torch.float32, torch.float64}:
        raise ValueError("prompt weights require float32 or float64")
    blocks = len(sample_ids)
    prompts = len(counts)
    return torch.tensor(
        [blocks / (prompts * counts[sample_id]) for sample_id in sample_ids],
        dtype=dtype,
        device=device,
    )


def prompt_balanced_mean(values: Tensor, sample_ids: Sequence[str]) -> Tensor:
    """Compute the exact mean of per-prompt block means."""

    if values.ndim != 1 or values.shape[0] != len(sample_ids):
        raise ValueError("values and sample IDs must describe the same blocks")
    _require_finite("prompt-balanced values", values.float())
    weights = prompt_balanced_example_weights(sample_ids, device=values.device)
    return (values.float() * weights).mean()


def binary_outcomes_from_tokens(
    direct_path: Tensor,
    candidate_ids: Tensor,
    gold_ids: Tensor,
) -> BinaryOutcomeOutput:
    """Reconstruct DFlash, Direct, and exact binary-oracle outcomes."""

    if candidate_ids.ndim != 3:
        raise ValueError("candidate_ids must have shape [B, L, K]")
    for name, value in (
        ("candidate_ids", candidate_ids),
        ("gold_ids", gold_ids),
        ("direct_path", direct_path),
    ):
        if value.dtype != torch.int64:
            raise ValueError(f"{name} must use torch.int64")
    batch, length, candidates = candidate_ids.shape
    if (length, candidates) != (BLOCK_LENGTH, CANDIDATE_COUNT):
        raise ValueError("candidate lattice differs from the 15x16 contract")
    if direct_path.shape != (batch, length):
        raise ValueError("direct_path must have shape [B, 15]")
    if gold_ids.shape != (batch, length):
        raise ValueError("gold_ids must have shape [B, 15]")
    if bool(((direct_path < 0) | (direct_path >= candidates)).any()):
        raise ValueError("direct_path indices must lie in [0, 15]")
    if not (
        direct_path.device == candidate_ids.device == gold_ids.device
    ):
        raise ValueError("token reconstruction tensors must share one device")
    base_path = torch.zeros_like(direct_path)
    base_lengths = realized_prefix_lengths(
        base_path, candidate_ids, gold_ids
    )
    direct_lengths = realized_prefix_lengths(
        direct_path, candidate_ids, gold_ids
    )
    gains = normalized_signed_gain(base_lengths, direct_lengths)
    return BinaryOutcomeOutput(
        base_lengths=base_lengths,
        direct_lengths=direct_lengths,
        normalized_gains=gains,
        oracle_apply_direct=direct_lengths.gt(base_lengths),
    )


def validated_oracle_recovery(
    method_eal: float,
    base_eal: float,
    oracle_eal: float,
    *,
    upper_tolerance: float = 1e-6,
) -> float:
    """Return an unclipped binary-oracle recovery or fail closed."""

    values = (method_eal, base_eal, oracle_eal, upper_tolerance)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("recovery inputs must be finite")
    if upper_tolerance < 0:
        raise ValueError("upper_tolerance cannot be negative")
    denominator = oracle_eal - base_eal
    if denominator <= 0:
        raise ValueError("binary-oracle recovery denominator must be positive")
    recovery = (method_eal - base_eal) / denominator
    if not 0.0 <= recovery <= 1.0 + upper_tolerance:
        raise ValueError("unclipped recovery is outside [0, 1+tolerance]")
    return recovery
