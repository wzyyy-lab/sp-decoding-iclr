"""Pure deterministic protocol helpers for PROS-Gate.

The functions here operate only on caller-supplied identities and synthetic
numeric tensors.  They intentionally contain no filesystem or dataset-loading
entry point so Gate-0 tests cannot open experiment splits by accident.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from numbers import Integral, Real
from typing import Iterable, Mapping, Sequence

import torch
from torch import Tensor


SPLIT_PROTOCOL = "pros-gate-phase3-oos-v1"
ORDER_PROTOCOL = "pros-fit-order-v1"
CAPACITY_PROTOCOL = "pros-capacity-v1"
CAPACITY_ADJUDICATION_SCHEMA = "pros-capacity-adjudication-v2"
RIDGE_FEATURE_DIMENSION = 21
RIDGE_COEFFICIENT = 1e-3
CAPACITY_COUNTS = {
    "harmful": 128,
    "changed-neutral": 128,
    "beneficial": 256,
}
STAGE_ALLOWED_SPLITS: dict[str, frozenset[str]] = {
    "cpu_synthetic": frozenset({"synthetic"}),
    "capacity": frozenset({"fit"}),
    "fit": frozenset({"fit"}),
    "checkpoint": frozenset({"checkpoint"}),
    "falsifier": frozenset({"falsifier"}),
    "manifest_identity": frozenset(
        {"fit", "checkpoint", "falsifier", "opb_train"}
    ),
}
DEFAULT_SPLIT_COUNTS: dict[str, dict[str, int]] = {
    "chat": {"fit": 523, "checkpoint": 66, "falsifier": 66},
    "code": {"fit": 531, "checkpoint": 67, "falsifier": 67},
    "math": {"fit": 533, "checkpoint": 67, "falsifier": 67},
}


@dataclass(frozen=True, order=True)
class BlockKey:
    """Immutable canonical block identity."""

    sample_id: str
    anchor_offset: int
    context_length: int

    def serialize(self) -> bytes:
        if (
            not isinstance(self.sample_id, str)
            or not self.sample_id
            or "\0" in self.sample_id
        ):
            raise ValueError("sample_id must be nonempty and contain no NUL")
        for name, value in (
            ("anchor_offset", self.anchor_offset),
            ("context_length", self.context_length),
        ):
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise ValueError(f"{name} must be a non-boolean integer")
            if int(value) < 0:
                raise ValueError(f"{name} must be nonnegative")
        return (
            self.sample_id.encode("utf-8")
            + b"\0"
            + str(int(self.anchor_offset)).encode("ascii")
            + b"\0"
            + str(int(self.context_length)).encode("ascii")
        )


@dataclass(frozen=True)
class CompletePassSchedule:
    """Whole-pass schedule under a hard optimizer-update maximum."""

    records: int
    batch_size: int
    steps_per_pass: int
    passes: int
    total_steps: int
    warmup_steps: int


@dataclass(frozen=True)
class CapacityRecord:
    """Fit-only capacity candidate with independently derived stratum."""

    block_key: BlockKey
    normalized_gain: float
    direct_changed: bool


@dataclass(frozen=True)
class SavedGateRecord:
    """One saved block record sufficient for independent gate replay."""

    block_key: BlockKey
    base_length: int
    direct_length: int
    score: float
    base_first_token_correct: bool
    direct_first_token_correct: bool


@dataclass(frozen=True)
class SavedGateEvaluation:
    """Actions, outcomes, losses, and fully denominatored replay metrics."""

    apply_direct: tuple[bool, ...]
    method_lengths: tuple[int, ...]
    oracle_lengths: tuple[int, ...]
    normalized_gains: tuple[float, ...]
    per_block_loss: tuple[float, ...]
    decoded_regret: tuple[float, ...]
    bound_slack: tuple[float, ...]
    metrics: dict[str, float | int | None]


@dataclass(frozen=True)
class WeightedRidgeModel:
    """Frozen float64 scalar comparator with fit-only normalization."""

    feature_mean: Tensor
    feature_scale: Tensor
    constant_features: Tensor
    coefficients: Tensor
    intercept: Tensor
    ridge: float

    def predict(self, features: Tensor) -> Tensor:
        if features.ndim != 2:
            raise ValueError("ridge features must have shape [N, D]")
        if features.shape[1] != self.feature_mean.numel():
            raise ValueError("ridge feature dimension differs from fitted model")
        values = features.detach().to(dtype=torch.float64, device="cpu")
        if not bool(torch.isfinite(values).all()):
            raise ValueError("ridge features must be finite")
        standardized = (values - self.feature_mean) / self.feature_scale
        standardized[:, self.constant_features] = 0.0
        return self.intercept + standardized @ self.coefficients


def _validate_sha256(name: str, value: str) -> None:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{name} must be a 64-character SHA256")
    if value != value.lower():
        raise ValueError(f"{name} must use canonical lowercase hexadecimal")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError(f"{name} must be hexadecimal") from error


def build_prompt_split(
    prompt_domains: Mapping[str, str],
    canonical_metadata_sha256: str,
    *,
    split_counts: Mapping[str, Mapping[str, int]] = DEFAULT_SPLIT_COUNTS,
) -> dict[str, str]:
    """Hash-split prompt identities without reading outcomes or examples."""

    _validate_sha256("canonical metadata hash", canonical_metadata_sha256)
    grouped: dict[str, list[str]] = {domain: [] for domain in split_counts}
    for sample_id, domain in prompt_domains.items():
        if not sample_id or "\0" in sample_id:
            raise ValueError("prompt IDs must be nonempty and contain no NUL")
        if domain not in grouped:
            raise ValueError(f"unexpected prompt domain: {domain}")
        grouped[domain].append(sample_id)

    assignments: dict[str, str] = {}
    for domain, counts in split_counts.items():
        if set(counts) != {"fit", "checkpoint", "falsifier"}:
            raise ValueError("every domain needs fit/checkpoint/falsifier counts")
        if any(
            isinstance(value, bool)
            or not isinstance(value, Integral)
            or int(value) < 0
            for value in counts.values()
        ):
            raise ValueError("split counts must be nonnegative integers")
        expected = sum(int(value) for value in counts.values())
        if len(grouped[domain]) != expected:
            raise ValueError(
                f"{domain} prompt count differs from frozen split: "
                f"{len(grouped[domain])} != {expected}"
            )

        def rank_key(sample_id: str) -> tuple[bytes, str]:
            payload = (
                SPLIT_PROTOCOL.encode("ascii")
                + b"\0"
                + domain.encode("utf-8")
                + b"\0"
                + sample_id.encode("utf-8")
                + b"\0"
                + canonical_metadata_sha256.encode("ascii")
            )
            return hashlib.sha256(payload).digest(), sample_id

        ordered = sorted(grouped[domain], key=rank_key)
        fit_end = int(counts["fit"])
        checkpoint_end = fit_end + int(counts["checkpoint"])
        for sample_id in ordered[:fit_end]:
            assignments[sample_id] = "fit"
        for sample_id in ordered[fit_end:checkpoint_end]:
            assignments[sample_id] = "checkpoint"
        for sample_id in ordered[checkpoint_end:]:
            assignments[sample_id] = "falsifier"

    if len(assignments) != len(prompt_domains):
        raise RuntimeError("prompt split lost or duplicated an identity")
    return assignments


def assert_disjoint_prompt_sets(
    named_prompt_sets: Mapping[str, Iterable[str]],
) -> None:
    """Fail on overlap without inspecting any example content."""

    materialized = {
        name: set(values) for name, values in named_prompt_sets.items()
    }
    names = list(materialized)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            overlap = materialized[left] & materialized[right]
            if overlap:
                raise ValueError(
                    f"prompt overlap between {left} and {right}: "
                    f"{sorted(overlap)[:3]}"
                )


def assert_stage_splits(stage: str, observed_splits: Iterable[str]) -> None:
    """Reject any split outside the stage's explicit least-privilege allowlist."""

    if stage not in STAGE_ALLOWED_SPLITS:
        raise ValueError(f"unknown protocol stage: {stage}")
    observed = set(observed_splits)
    if not observed:
        raise ValueError("observed split set cannot be empty")
    if any(not isinstance(value, str) or not value for value in observed):
        raise ValueError("split names must be nonempty strings")
    forbidden = observed - STAGE_ALLOWED_SPLITS[stage]
    if forbidden:
        raise PermissionError(
            f"stage {stage} cannot access split(s): {sorted(forbidden)}"
        )


def ordered_block_keys(
    block_keys: Sequence[BlockKey],
    *,
    pass_index: int,
    training_manifest_sha256: str,
) -> list[BlockKey]:
    """Return the version-independent SHA256 order for one complete pass."""

    if isinstance(pass_index, bool) or not isinstance(pass_index, Integral):
        raise ValueError("pass_index must be a non-boolean integer")
    if int(pass_index) < 0:
        raise ValueError("pass_index must be nonnegative")
    _validate_sha256("training manifest hash", training_manifest_sha256)
    if len(set(block_keys)) != len(block_keys):
        raise ValueError("block keys must be unique")

    def rank_key(block_key: BlockKey) -> tuple[bytes, bytes]:
        serialized = block_key.serialize()
        payload = (
            ORDER_PROTOCOL.encode("ascii")
            + b"\0"
            + str(int(pass_index)).encode("ascii")
            + b"\0"
            + serialized
            + b"\0"
            + training_manifest_sha256.encode("ascii")
        )
        return hashlib.sha256(payload).digest(), serialized

    return sorted(block_keys, key=rank_key)


def ordered_block_keys_sha256(block_keys: Sequence[BlockKey]) -> str:
    """Hash an ordered key sequence with length-prefix framing."""

    digest = hashlib.sha256()
    for block_key in block_keys:
        serialized = block_key.serialize()
        digest.update(len(serialized).to_bytes(8, "big"))
        digest.update(serialized)
    return digest.hexdigest()


def capacity_record_rank_sha256(
    block_key: BlockKey,
    producer_checkpoint_sha256: str,
    producer_metrics_sha256: str,
    canonical_metadata_sha256: str,
    split_manifest_sha256: str,
) -> str:
    """Return the binding capacity ranking digest for one fit block."""

    hashes = (
        ("producer checkpoint hash", producer_checkpoint_sha256),
        ("producer metrics hash", producer_metrics_sha256),
        ("canonical metadata hash", canonical_metadata_sha256),
        ("split manifest hash", split_manifest_sha256),
    )
    for name, value in hashes:
        _validate_sha256(name, value)
    serialized = block_key.serialize()
    payload = b"\0".join(
        [
            CAPACITY_PROTOCOL.encode("ascii"),
            serialized,
            producer_checkpoint_sha256.encode("ascii"),
            producer_metrics_sha256.encode("ascii"),
            canonical_metadata_sha256.encode("ascii"),
            split_manifest_sha256.encode("ascii"),
        ]
    )
    return hashlib.sha256(payload).hexdigest()


def _capacity_stratum(record: CapacityRecord) -> str | None:
    gain = record.normalized_gain
    if isinstance(gain, bool) or not isinstance(gain, Real):
        raise ValueError("capacity gain must be a real non-boolean number")
    numeric_gain = float(gain)
    if not math.isfinite(numeric_gain) or abs(numeric_gain) > 1.0 + 1e-7:
        raise ValueError("capacity gain must be finite and in [-1, 1]")
    if not isinstance(record.direct_changed, bool):
        raise ValueError("direct_changed must be boolean")
    record.block_key.serialize()
    if numeric_gain != 0.0 and not record.direct_changed:
        raise ValueError("nonzero capacity gain requires a changed Direct path")
    if numeric_gain < 0:
        return "harmful"
    if numeric_gain > 0:
        return "beneficial"
    if record.direct_changed:
        return "changed-neutral"
    return None


def select_capacity_records(
    fit_records: Sequence[CapacityRecord],
    producer_checkpoint_sha256: str,
    producer_metrics_sha256: str,
    canonical_metadata_sha256: str,
    split_manifest_sha256: str,
) -> list[CapacityRecord]:
    """Select the exact prompt-unique 256/128/128 capacity composition.

    Strata are processed in the frozen scarcity order harmful,
    changed-neutral, beneficial.  A prompt selected by an earlier stratum is
    excluded globally, and an unavailable exact composition fails closed.
    """

    if not fit_records:
        raise ValueError("fit capacity candidates cannot be empty")
    serialized_keys: list[bytes] = []
    by_stratum: dict[str, list[CapacityRecord]] = {
        name: [] for name in CAPACITY_COUNTS
    }
    for record in fit_records:
        if not isinstance(record, CapacityRecord):
            raise ValueError("fit_records must contain CapacityRecord values")
        serialized_keys.append(record.block_key.serialize())
        stratum = _capacity_stratum(record)
        if stratum is not None:
            by_stratum[stratum].append(record)
    if len(set(serialized_keys)) != len(serialized_keys):
        raise ValueError("capacity candidate block keys must be unique")

    def rank_key(record: CapacityRecord) -> tuple[bytes, bytes]:
        serialized = record.block_key.serialize()
        digest = capacity_record_rank_sha256(
            record.block_key,
            producer_checkpoint_sha256,
            producer_metrics_sha256,
            canonical_metadata_sha256,
            split_manifest_sha256,
        )
        return bytes.fromhex(digest), serialized

    selected: list[CapacityRecord] = []
    selected_prompts: set[str] = set()
    for stratum in ("harmful", "changed-neutral", "beneficial"):
        target = CAPACITY_COUNTS[stratum]
        chosen = 0
        for record in sorted(by_stratum[stratum], key=rank_key):
            sample_id = record.block_key.sample_id
            if sample_id in selected_prompts:
                continue
            selected.append(record)
            selected_prompts.add(sample_id)
            chosen += 1
            if chosen == target:
                break
        if chosen != target:
            raise RuntimeError(
                f"insufficient prompt-unique {stratum} capacity records: "
                f"{chosen} != {target}"
            )
    if len(selected) != 512 or len(selected_prompts) != 512:
        raise RuntimeError("capacity selection lost exact prompt uniqueness")
    return selected


def capacity_records_sha256(records: Sequence[CapacityRecord]) -> str:
    """Hash an ordered capacity selection including outcome-defining fields."""

    digest = hashlib.sha256()
    for record in records:
        stratum = _capacity_stratum(record)
        if stratum is None:
            raise ValueError("capacity selection contains an ineligible record")
        serialized = record.block_key.serialize()
        numeric_gain = float(record.normalized_gain)
        if numeric_gain == 0.0:
            numeric_gain = 0.0
        gain = format(numeric_gain, ".17g").encode("ascii")
        payload = b"\0".join(
            [serialized, stratum.encode("ascii"), gain, b"1" if record.direct_changed else b"0"]
        )
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def _validated_record_length(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be a non-boolean integer")
    numeric = int(value)
    if not 0 <= numeric <= 15:
        raise ValueError(f"{name} must lie in [0, 15]")
    return numeric


def reconstruct_saved_gate_evaluation(
    records: Sequence[SavedGateRecord],
    *,
    require_valid_recovery: bool = True,
) -> SavedGateEvaluation:
    """Independently replay binary gate behavior from saved block records.

    This pure Python adjudicator deliberately does not call model, training,
    feature, or loss-output code.  It reconstructs strict actions, all four
    EALs, loss/regret/slack, recovery, token outcomes, and explicit numerator /
    denominator pairs for behavioral rates.
    """

    if not isinstance(require_valid_recovery, bool):
        raise ValueError("require_valid_recovery must be boolean")
    if not records:
        raise ValueError("saved gate records cannot be empty")
    keys: list[bytes] = []
    prompt_ids: list[str] = []
    base_lengths: list[int] = []
    direct_lengths: list[int] = []
    scores: list[float] = []
    base_first: list[bool] = []
    direct_first: list[bool] = []
    for record in records:
        if not isinstance(record, SavedGateRecord):
            raise ValueError("records must contain SavedGateRecord values")
        keys.append(record.block_key.serialize())
        prompt_ids.append(record.block_key.sample_id)
        base_length = _validated_record_length("base_length", record.base_length)
        direct_length = _validated_record_length(
            "direct_length", record.direct_length
        )
        base_lengths.append(base_length)
        direct_lengths.append(direct_length)
        if isinstance(record.score, bool) or not isinstance(record.score, Real):
            raise ValueError("saved score must be a real non-boolean number")
        score = float(record.score)
        if not math.isfinite(score):
            raise ValueError("saved score must be finite")
        scores.append(score)
        if not isinstance(record.base_first_token_correct, bool) or not isinstance(
            record.direct_first_token_correct, bool
        ):
            raise ValueError("first-token outcomes must be boolean")
        if record.base_first_token_correct is not (base_length > 0):
            raise ValueError(
                "base first-token witness differs from base_length > 0"
            )
        if record.direct_first_token_correct is not (direct_length > 0):
            raise ValueError(
                "Direct first-token witness differs from direct_length > 0"
            )
        base_first.append(record.base_first_token_correct)
        direct_first.append(record.direct_first_token_correct)
    if len(set(keys)) != len(keys):
        raise ValueError("saved gate block keys must be unique")

    apply = [score > 0.0 for score in scores]
    method_lengths = [
        direct if action else base
        for base, direct, action in zip(base_lengths, direct_lengths, apply)
    ]
    oracle_apply = [
        direct > base for base, direct in zip(base_lengths, direct_lengths)
    ]
    oracle_lengths = [
        direct if action else base
        for base, direct, action in zip(base_lengths, direct_lengths, oracle_apply)
    ]
    gains = [
        (direct - base) / 15.0
        for base, direct in zip(base_lengths, direct_lengths)
    ]
    losses: list[float] = []
    regrets: list[float] = []
    slacks: list[float] = []
    for score, gain, action in zip(scores, gains, apply):
        if gain == 0.0:
            loss = 0.0
            regret = 0.0
        else:
            label = 1.0 if gain > 0.0 else -1.0
            loss = abs(gain) * max(0.0, 1.0 - label * score)
            regret = gain if gain > 0.0 and not action else 0.0
            if gain < 0.0 and action:
                regret = -gain
        losses.append(loss)
        regrets.append(regret)
        slacks.append(loss - regret)

    def prompt_balanced(values: Sequence[float | int]) -> float:
        totals: dict[str, float] = {}
        counts: dict[str, int] = {}
        for sample_id, value in zip(prompt_ids, values):
            totals[sample_id] = totals.get(sample_id, 0.0) + float(value)
            counts[sample_id] = counts.get(sample_id, 0) + 1
        return sum(totals[key] / counts[key] for key in totals) / len(totals)

    def fraction(numerator: int, denominator: int) -> float | None:
        return None if denominator == 0 else numerator / denominator

    record_count = len(records)
    prompt_count = len(set(prompt_ids))
    beneficial = [gain > 0.0 for gain in gains]
    harmful = [gain < 0.0 for gain in gains]
    neutral = [gain == 0.0 for gain in gains]
    beneficial_count = sum(beneficial)
    harmful_count = sum(harmful)
    neutral_count = sum(neutral)
    beneficial_apply = sum(b and a for b, a in zip(beneficial, apply))
    harmful_apply = sum(h and a for h, a in zip(harmful, apply))
    neutral_apply = sum(n and a for n, a in zip(neutral, apply))
    harmful_keep = harmful_count - harmful_apply
    beneficial_keep = beneficial_count - beneficial_apply
    neutral_keep = neutral_count - neutral_apply
    zero_regret_count = sum(value == 0.0 for value in regrets)
    harmed_count = sum(
        action and direct < base
        for action, base, direct in zip(apply, base_lengths, direct_lengths)
    )
    false_apply_count = sum(
        action and gain <= 0.0 for action, gain in zip(apply, gains)
    )
    base_eal = prompt_balanced(base_lengths)
    direct_eal = prompt_balanced(direct_lengths)
    method_eal = prompt_balanced(method_lengths)
    oracle_eal = prompt_balanced(oracle_lengths)
    recovery_denominator = oracle_eal - base_eal
    recovery_numerator = method_eal - base_eal
    recovery: float | None
    if not math.isfinite(recovery_denominator) or recovery_denominator <= 0.0:
        if require_valid_recovery:
            raise ValueError(
                "binary-oracle recovery denominator must be positive"
            )
        recovery = None
    else:
        recovery = recovery_numerator / recovery_denominator
        if not math.isfinite(recovery):
            if require_valid_recovery:
                raise ValueError("unclipped recovery must be finite")
            recovery = None
        elif require_valid_recovery and not 0.0 <= recovery <= 1.0 + 1e-6:
            raise ValueError("unclipped recovery is outside [0, 1+tolerance]")

    method_first = [
        direct if action else base
        for base, direct, action in zip(base_first, direct_first, apply)
    ]
    oracle_first = [
        direct if action else base
        for base, direct, action in zip(base_first, direct_first, oracle_apply)
    ]
    violation_count = sum(slack < -1e-6 for slack in slacks)
    metrics: dict[str, float | int | None] = {
        "record_count": record_count,
        "prompt_count": prompt_count,
        "base_eal": base_eal,
        "direct_eal": direct_eal,
        "method_eal": method_eal,
        "oracle_eal": oracle_eal,
        "eal_prompt_denominator": prompt_count,
        "base_accepted_token_mass": sum(base_lengths),
        "direct_accepted_token_mass": sum(direct_lengths),
        "method_accepted_token_mass": sum(method_lengths),
        "oracle_accepted_token_mass": sum(oracle_lengths),
        "prompt_weighted_gain_hinge": prompt_balanced(losses),
        "prompt_weighted_decoded_regret": prompt_balanced(regrets),
        "loss_prompt_denominator": prompt_count,
        "loss_block_sum": sum(losses),
        "decoded_regret_block_sum": sum(regrets),
        "minimum_bound_slack": min(slacks),
        "regret_bound_violation_count": violation_count,
        "recovery_numerator": recovery_numerator,
        "recovery_denominator": recovery_denominator,
        "oracle_recovery": recovery,
        "apply_count": sum(apply),
        "keep_count": record_count - sum(apply),
        "beneficial_count": beneficial_count,
        "harmful_count": harmful_count,
        "neutral_count": neutral_count,
        "outcome_fraction_denominator": record_count,
        "beneficial_fraction": beneficial_count / record_count,
        "harmful_fraction": harmful_count / record_count,
        "neutral_fraction": neutral_count / record_count,
        "beneficial_apply_count": beneficial_apply,
        "benefit_recall_numerator": beneficial_apply,
        "benefit_recall_denominator": beneficial_count,
        "benefit_recall": fraction(beneficial_apply, beneficial_count),
        "beneficial_keep_count": beneficial_keep,
        "harmful_keep_count": harmful_keep,
        "harm_avoidance_numerator": harmful_keep,
        "harm_avoidance_denominator": harmful_count,
        "harm_avoidance": fraction(harmful_keep, harmful_count),
        "harmful_apply_count": harmful_apply,
        "neutral_apply_count": neutral_apply,
        "apply_composition_denominator": sum(apply),
        "beneficial_apply_fraction": fraction(beneficial_apply, sum(apply)),
        "harmful_apply_fraction": fraction(harmful_apply, sum(apply)),
        "neutral_apply_fraction": fraction(neutral_apply, sum(apply)),
        "neutral_keep_count": neutral_keep,
        "tie_agreement_numerator": neutral_keep,
        "tie_agreement_denominator": neutral_count,
        "tie_agreement": fraction(neutral_keep, neutral_count),
        "utility_optimal_numerator": zero_regret_count,
        "utility_optimal_denominator": record_count,
        "utility_optimal_fraction": zero_regret_count / record_count,
        "harmed_numerator": harmed_count,
        "harmed_denominator": record_count,
        "harmed_fraction": harmed_count / record_count,
        "false_apply_numerator": false_apply_count,
        "false_apply_denominator": harmful_count + neutral_count,
        "false_apply_fraction": fraction(
            false_apply_count, harmful_count + neutral_count
        ),
        "base_first_token_count": sum(base_first),
        "direct_first_token_count": sum(direct_first),
        "method_first_token_count": sum(method_first),
        "oracle_first_token_count": sum(oracle_first),
        "first_token_denominator": record_count,
    }
    return SavedGateEvaluation(
        apply_direct=tuple(apply),
        method_lengths=tuple(method_lengths),
        oracle_lengths=tuple(oracle_lengths),
        normalized_gains=tuple(gains),
        per_block_loss=tuple(losses),
        decoded_regret=tuple(regrets),
        bound_slack=tuple(slacks),
        metrics=metrics,
    )


def half_up_warmup_steps(total_steps: int, ratio: float = 0.04) -> int:
    """Compute ``floor(ratio*steps + 0.5)`` exactly as preregistered."""

    if total_steps < 1:
        raise ValueError("total_steps must be positive")
    if not math.isfinite(ratio) or not 0.0 <= ratio < 1.0:
        raise ValueError("warmup ratio must be finite and in [0, 1)")
    return math.floor(ratio * total_steps + 0.5)


def complete_pass_schedule(
    records: int,
    *,
    batch_size: int,
    max_updates: int = 5_120,
) -> CompletePassSchedule:
    """Use the largest whole-pass multiple not exceeding ``max_updates``."""

    if records < 1 or batch_size < 1 or max_updates < 1:
        raise ValueError("records, batch_size, and max_updates must be positive")
    steps_per_pass = math.ceil(records / batch_size)
    passes = max_updates // steps_per_pass
    if passes < 1:
        raise ValueError("one complete pass exceeds the update budget")
    total_steps = passes * steps_per_pass
    return CompletePassSchedule(
        records=records,
        batch_size=batch_size,
        steps_per_pass=steps_per_pass,
        passes=passes,
        total_steps=total_steps,
        warmup_steps=half_up_warmup_steps(total_steps),
    )


def earliest_exact_minimum(values: Sequence[float]) -> int:
    """Return the first full-precision minimum, rejecting nonfinite values."""

    if not values:
        raise ValueError("checkpoint values cannot be empty")
    numeric = [float(value) for value in values]
    if not all(math.isfinite(value) for value in numeric):
        raise ValueError("checkpoint values must be finite")
    minimum = min(numeric)
    return numeric.index(minimum)


def capacity_gate_passes(
    metrics: Mapping[str, float | int | bool],
    epoch_zero_loss: float,
) -> bool:
    """Independently adjudicate every binding capacity-gate conjunct."""

    required_numeric = (
        "prompt_weighted_loss",
        "regret_bound_violation_count",
        "record_count",
        "prompt_count",
        "beneficial_count",
        "beneficial_apply_count",
        "harmful_count",
        "harmful_keep_count",
        "harm_avoidance_numerator",
        "harm_avoidance_denominator",
        "neutral_count",
        "utility_optimal_count",
        "base_eal",
        "method_eal",
        "oracle_eal",
        "recovery_numerator",
        "recovery_denominator",
        "oracle_recovery",
        "harmful_apply_count",
    )
    try:
        if isinstance(epoch_zero_loss, bool):
            return False
        if any(isinstance(metrics[name], bool) for name in required_numeric):
            return False
        values = {name: float(metrics[name]) for name in required_numeric}
        initial = float(epoch_zero_loss)
    except (KeyError, TypeError, ValueError):
        return False
    if not all(math.isfinite(value) for value in (*values.values(), initial)):
        return False
    if initial <= 0.0:
        return False
    if metrics.get("values_finite") is not True:
        return False
    if metrics.get("gradients_finite") is not True:
        return False
    recovery_denominator = values["oracle_eal"] - values["base_eal"]
    recovery_numerator = values["method_eal"] - values["base_eal"]
    if recovery_denominator <= 0.0:
        return False
    recovery = recovery_numerator / recovery_denominator
    if not 0.0 <= recovery <= 1.0 + 1e-6:
        return False
    for reported, reconstructed in (
        (values["recovery_numerator"], recovery_numerator),
        (values["recovery_denominator"], recovery_denominator),
        (values["oracle_recovery"], recovery),
    ):
        if not math.isclose(reported, reconstructed, rel_tol=0.0, abs_tol=1e-12):
            return False
    count_names = (
        "regret_bound_violation_count",
        "record_count",
        "prompt_count",
        "beneficial_count",
        "beneficial_apply_count",
        "harmful_count",
        "harmful_keep_count",
        "harm_avoidance_numerator",
        "harm_avoidance_denominator",
        "neutral_count",
        "utility_optimal_count",
        "harmful_apply_count",
    )
    if any(not values[name].is_integer() or values[name] < 0 for name in count_names):
        return False
    if values["prompt_weighted_loss"] < 0.0:
        return False
    return bool(
        values["regret_bound_violation_count"] == 0
        and values["prompt_weighted_loss"] <= 0.05 * initial
        and values["record_count"] == 512
        and values["prompt_count"] == 512
        and values["beneficial_count"] == 256
        and values["beneficial_apply_count"] >= 254
        and values["harmful_count"] == 128
        and values["harmful_keep_count"] >= 127
        and values["neutral_count"] == 128
        and values["utility_optimal_count"] >= 509
        and 0.95 <= recovery <= 1.0 + 1e-6
        and values["harmful_apply_count"] <= 1
        and values["harmful_keep_count"]
        == values["harm_avoidance_numerator"]
        and values["harm_avoidance_denominator"] == values["harmful_count"]
        and values["harmful_keep_count"] + values["harmful_apply_count"] == 128
        and values["beneficial_apply_count"] <= 256
        and values["utility_optimal_count"]
        == values["beneficial_apply_count"]
        + values["harmful_keep_count"]
        + values["neutral_count"]
    )


def selected_capacity_checkpoint(
    history: Sequence[Mapping[str, float | int | bool]],
) -> Mapping[str, float | int | bool]:
    """Select earliest minimum loss and require that exact row to pass."""

    if len(history) != 321:
        raise ValueError("capacity history must contain epoch zero plus 320 passes")
    epochs: list[int] = []
    for row in history:
        value = row.get("pass", -1)
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise ValueError("capacity pass indices must be integers")
        epochs.append(int(value))
    if epochs != list(range(321)):
        raise ValueError("capacity history pass indices must be 0..320")
    selected_index = earliest_exact_minimum(
        [float(row["prompt_weighted_loss"]) for row in history]
    )
    selected = history[selected_index]
    if not capacity_gate_passes(
        selected,
        float(history[0]["prompt_weighted_loss"]),
    ):
        raise RuntimeError("selected minimum-loss capacity checkpoint did not pass")
    return selected


def fit_checkpoint_selection_key(
    metrics: Mapping[str, float | int | bool],
) -> tuple[float, float, float] | None:
    """Return the frozen R082 key, or ``None`` for an ineligible checkpoint."""

    required = (
        "base_eal",
        "method_eal",
        "oracle_eal",
        "recovery_numerator",
        "recovery_denominator",
        "oracle_recovery",
        "harmed_numerator",
        "prompt_weighted_gain_hinge",
    )
    try:
        if any(isinstance(metrics[name], bool) for name in required):
            return None
        values = {name: float(metrics[name]) for name in required}
    except (KeyError, TypeError, ValueError):
        return None
    if not all(math.isfinite(value) for value in values.values()):
        return None
    if metrics.get("values_finite") is not True:
        return None
    if metrics.get("gradients_finite") is not True:
        return None
    harmed = values["harmed_numerator"]
    hinge = values["prompt_weighted_gain_hinge"]
    if not harmed.is_integer() or harmed < 0.0 or hinge < 0.0:
        return None
    denominator = values["oracle_eal"] - values["base_eal"]
    if denominator <= 0.0:
        return None
    numerator = values["method_eal"] - values["base_eal"]
    recovery = numerator / denominator
    if not 0.0 <= recovery <= 1.0 + 1e-6:
        return None
    for reported, reconstructed in (
        (values["recovery_numerator"], numerator),
        (values["recovery_denominator"], denominator),
        (values["oracle_recovery"], recovery),
    ):
        if not math.isclose(reported, reconstructed, rel_tol=0.0, abs_tol=1e-12):
            return None
    return values["method_eal"], -harmed, -hinge


def selected_fit_checkpoint(
    history: Sequence[Mapping[str, object]],
) -> Mapping[str, object]:
    """Select the earliest strict lexicographic R082 checkpoint."""

    if not history:
        raise ValueError("fit/checkpoint history cannot be empty")
    selected: Mapping[str, object] | None = None
    selected_key: tuple[float, float, float] | None = None
    for expected_pass, row in enumerate(history):
        pass_index = row.get("pass")
        if (
            isinstance(pass_index, bool)
            or not isinstance(pass_index, Integral)
            or int(pass_index) != expected_pass
        ):
            raise ValueError("fit/checkpoint pass indices must be contiguous from zero")
        metrics = row.get("checkpoint")
        if not isinstance(metrics, Mapping):
            raise ValueError("fit/checkpoint history lacks checkpoint metrics")
        key = fit_checkpoint_selection_key(metrics)
        if key is not None and (selected_key is None or key > selected_key):
            selected = row
            selected_key = key
    if selected is None:
        raise RuntimeError("no recovery-valid fit checkpoint is eligible")
    return selected


def fit_weighted_ridge(
    features: Tensor,
    targets: Tensor,
    example_weights: Tensor,
    *,
    constant_tolerance: float = 1e-12,
) -> WeightedRidgeModel:
    """Fit the frozen 21-scalar comparator with exact float64 conventions.

    Standardization uses normalized prompt weights and population variance.
    Constant features are mapped to zero.  The intercept is unpenalized; every
    standardized slope receives the binding ``1e-3`` ridge penalty.  Neither
    feature dimension nor ridge coefficient is caller-configurable.
    """

    if features.ndim != 2:
        raise ValueError("ridge features must have shape [N, D]")
    rows, dimensions = features.shape
    if targets.shape != (rows,) or example_weights.shape != (rows,):
        raise ValueError("ridge targets and weights must have shape [N]")
    if dimensions != RIDGE_FEATURE_DIMENSION:
        raise ValueError("frozen scalar comparator requires exactly 21 features")
    if not math.isfinite(constant_tolerance) or constant_tolerance < 0:
        raise ValueError("constant tolerance must be finite and nonnegative")

    x = features.detach().to(dtype=torch.float64, device="cpu")
    y = targets.detach().to(dtype=torch.float64, device="cpu")
    weights = example_weights.detach().to(dtype=torch.float64, device="cpu")
    for name, value in (("features", x), ("targets", y), ("weights", weights)):
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"ridge {name} must be finite")
    if bool((weights <= 0).any()):
        raise ValueError("ridge weights must be strictly positive")
    normalized = weights / weights.sum()
    mean = (normalized[:, None] * x).sum(dim=0)
    variance = (normalized[:, None] * (x - mean).square()).sum(dim=0)
    scale = variance.sqrt()
    constant = scale <= constant_tolerance
    safe_scale = torch.where(constant, torch.ones_like(scale), scale)
    standardized = (x - mean) / safe_scale
    standardized[:, constant] = 0.0

    design = torch.cat(
        [torch.ones(rows, 1, dtype=torch.float64), standardized], dim=1
    )
    weighted_design = normalized[:, None] * design
    gram = design.T @ weighted_design
    penalty = (
        torch.eye(dimensions + 1, dtype=torch.float64) * RIDGE_COEFFICIENT
    )
    penalty[0, 0] = 0.0
    right_hand_side = design.T @ (normalized * y)
    parameters = torch.linalg.solve(gram + penalty, right_hand_side)
    return WeightedRidgeModel(
        feature_mean=mean,
        feature_scale=safe_scale,
        constant_features=constant,
        coefficients=parameters[1:],
        intercept=parameters[0],
        ridge=RIDGE_COEFFICIENT,
    )


def deterministic_bootstrap_indices(
    prompt_count: int,
    *,
    replicates: int,
    seed: int,
    prompt_set_sha256: str,
) -> Tensor:
    """Counter-based, library-independent prompt-cluster resampling indices."""

    if prompt_count < 1 or replicates < 1:
        raise ValueError("prompt_count and replicates must be positive")
    if seed < 0:
        raise ValueError("bootstrap seed must be nonnegative")
    _validate_sha256("prompt-set hash", prompt_set_sha256)
    mask = (1 << 64) - 1

    def splitmix64(value: int) -> int:
        value = (value + 0x9E3779B97F4A7C15) & mask
        value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & mask
        value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & mask
        return value ^ (value >> 31)

    protocol_seed = int.from_bytes(
        hashlib.sha256(
            b"pros-bootstrap-v1\0"
            + str(seed).encode("ascii")
            + b"\0"
            + prompt_set_sha256.encode("ascii")
        ).digest()[:8],
        "big",
    )
    output = torch.empty(replicates, prompt_count, dtype=torch.int64)
    for replicate in range(replicates):
        state = splitmix64(protocol_seed ^ replicate)
        for draw in range(prompt_count):
            state = splitmix64(state ^ draw)
            output[replicate, draw] = state % prompt_count
    return output
