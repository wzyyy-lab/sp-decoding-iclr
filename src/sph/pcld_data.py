"""Data and sidecar contracts for PCLD-16R.

Offline target tensors are carried by the training batch, but the only helper
that constructs production head inputs returns exactly the four online fields
accepted by :class:`sph.pcld.PCLD16Head`.
"""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import random
from typing import Any, Iterable

import torch
from torch import Tensor

from sph.japd import strict_joint_two_frontier_metric
from sph.japd_data import load_rollout_records, record_key
from sph.pcld import (
    BLOCK_LENGTH,
    CANDIDATES,
    HIDDEN_SIZE,
    LATENT_SCALE_FLOOR,
    calibrate_numeric_epsilon,
    candidate_gold_ranks,
    continuous_clean_support,
    stable_teacher_rows,
)


PCLD_FORWARD_FIELDS = frozenset(
    {"hidden", "candidate_lm_rows", "candidate_logits", "base_logsumexp"}
)

# Frozen before the first PCLD capacity optimizer update.  The legacy value is
# stored in the JAPD manifest; the other three are deterministic, read-only
# decompositions of the authoritative PCLD capacity sidecar.  Keeping all four
# prevents the training-support definition from silently replacing the
# pre-registered strict-J2 evaluation population.
PCLD_CAPACITY_J2_DENOMINATORS = {
    "legacy": 411,
    "authoritative": 403,
    "authoritative_numeric": 402,
    "stable": 314,
}
PCLD_CAPACITY_EPSILON_NUM = 0.24676132202148438
PCLD_CAPACITY_MARGIN_THRESHOLD = 0.49352264404296875
PCLD_CAPACITY_STABLE_EFFECTIVE_BLOCKS = 503
PCLD_CAPACITY_STABLE_SUPPORT_ROWS = 4754


def load_manifest(path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text())
    if manifest.get("format") != "japd_manifest_v1" or not manifest.get("complete"):
        raise RuntimeError(f"unsupported or incomplete prompt manifest: {path}")
    if manifest.get("label_fields_used_for_selection") != []:
        raise RuntimeError("the frozen prompt manifest used label fields")
    splits = manifest.get("prompt_splits")
    if not isinstance(splits, dict):
        raise RuntimeError("PCLD manifest lacks prompt splits")
    expected_counts = {"fit": 1589, "select": 199, "diagnostic": 199}
    prompt_sets: dict[str, set[str]] = {}
    for name, expected_count in expected_counts.items():
        values = splits.get(name)
        if not isinstance(values, list) or len(values) != expected_count:
            raise RuntimeError(
                f"PCLD manifest {name} split must contain {expected_count} prompts"
            )
        prompt_set = {str(value) for value in values}
        if len(prompt_set) != len(values):
            raise RuntimeError(f"PCLD manifest {name} split contains duplicates")
        prompt_sets[name] = prompt_set
    for left, right in (("fit", "select"), ("fit", "diagnostic"), ("select", "diagnostic")):
        overlap = prompt_sets[left] & prompt_sets[right]
        if overlap:
            raise RuntimeError(
                f"PCLD manifest prompt splits {left}/{right} overlap"
            )
    return manifest


def validate_manifest_source(
    manifest: dict[str, Any], *, rollout: Path, split: str
) -> None:
    expected_rollout = str(rollout.resolve())
    if manifest.get("source_rollout") != expected_rollout:
        raise RuntimeError("PCLD manifest source rollout mismatch")
    if manifest.get("source_split") != split:
        raise RuntimeError("PCLD manifest source split mismatch")


def capacity_expected_j2_denominator(manifest: dict[str, Any]) -> int:
    value = manifest.get("capacity", {}).get(
        "strict_multi_repair_blocks_diagnostic_only"
    )
    if not isinstance(value, int) or value != 411:
        raise RuntimeError("PCLD capacity manifest must freeze J2 denominator at 411")
    return value


def capacity_expected_j2_denominators(
    manifest: dict[str, Any],
) -> dict[str, int]:
    legacy = capacity_expected_j2_denominator(manifest)
    expected = dict(PCLD_CAPACITY_J2_DENOMINATORS)
    if expected["legacy"] != legacy:
        raise RuntimeError("PCLD capacity legacy J2 receipt mismatch")
    return expected


def _serialized_record_key(record: dict[str, Any]) -> list[Any]:
    sample_id, anchor_offset, context_length = record_key(record)
    return [sample_id, anchor_offset, context_length]


def capacity_support_summary(
    records: list[dict[str, Any]], epsilon_num: float
) -> dict[str, Any]:
    """Recompute the four frozen capacity support populations.

    ``legacy`` is the binding strict-J2 population.  The other branches audit
    PCLD teacher geometry and numerical hygiene; only ``stable`` is used by
    trainable losses.
    """

    keys = [record_key(record) for record in records]
    if len(records) != 512 or len(set(keys)) != 512:
        raise RuntimeError("PCLD capacity support receipt requires 512 unique records")
    candidate_ids = torch.stack(
        [record["base_topk_ids"].long() for record in records]
    )
    gold_ids = torch.stack([record["gold_ids"].long() for record in records])
    gold_ranks = candidate_gold_ranks(candidate_ids, gold_ids)
    legacy_top1 = torch.stack(
        [record["target_top1_ids"].long() for record in records]
    )
    authoritative_top1 = torch.stack(
        [record["pcld_authoritative_top1_ids"].long() for record in records]
    )
    fp32_top1 = torch.stack(
        [record["pcld_fp32_top1_ids"].long() for record in records]
    )
    margins = torch.stack(
        [record["pcld_target_top1_margins"].float() for record in records]
    )
    stable_rows = stable_teacher_rows(
        authoritative_top1, fp32_top1, margins, epsilon_num
    )
    authoritative_matches_gold = authoritative_top1.eq(gold_ids)
    conditions = {
        "legacy": legacy_top1.eq(gold_ids),
        "authoritative": authoritative_matches_gold,
        "authoritative_numeric": (
            authoritative_matches_gold & authoritative_top1.eq(fp32_top1)
        ),
        "stable": authoritative_matches_gold & stable_rows,
    }
    zero_prediction = torch.zeros_like(gold_ranks)
    branches: dict[str, Any] = {}
    for name, condition in conditions.items():
        metric = strict_joint_two_frontier_metric(
            zero_prediction, gold_ranks, condition
        )
        eligible_keys = sorted(
            [list(keys[index]) for index in metric.eligible.nonzero().flatten().tolist()]
        )
        branches[name] = {
            "j2_denominator": int(metric.denominator),
            "eligible_keys": eligible_keys,
        }

    stable_support, stable_horizons = continuous_clean_support(
        gold_ranks, authoritative_top1, gold_ids, stable_rows
    )
    ordered = sorted(range(len(records)), key=lambda index: keys[index])
    summary = {
        "records": len(records),
        "semantic_keys": [_serialized_record_key(records[index]) for index in ordered],
        "epsilon_num": float(epsilon_num),
        "margin_threshold": float(2.0 * epsilon_num),
        "branches": branches,
        "stable_effective_blocks": int(stable_horizons.gt(0).sum()),
        "stable_support_rows": int(stable_support.sum()),
        "stable_horizons": [
            {
                "key": _serialized_record_key(records[index]),
                "horizon": int(stable_horizons[index]),
            }
            for index in ordered
        ],
    }
    expected_scalars = {
        "epsilon_num": PCLD_CAPACITY_EPSILON_NUM,
        "margin_threshold": PCLD_CAPACITY_MARGIN_THRESHOLD,
        "stable_effective_blocks": PCLD_CAPACITY_STABLE_EFFECTIVE_BLOCKS,
        "stable_support_rows": PCLD_CAPACITY_STABLE_SUPPORT_ROWS,
    }
    for name, expected in PCLD_CAPACITY_J2_DENOMINATORS.items():
        actual = summary["branches"][name]["j2_denominator"]
        if actual != expected:
            raise RuntimeError(
                f"PCLD capacity {name} J2 denominator {actual} != {expected}"
            )
    for name, expected in expected_scalars.items():
        if summary[name] != expected:
            raise RuntimeError(
                f"PCLD capacity support {name} {summary[name]} != {expected}"
            )
    return summary


def build_capacity_support_receipt(
    records: list[dict[str, Any]],
    epsilon_num: float,
    *,
    rollout: Path,
    manifest: Path,
    target: Path,
    sidecar: Path,
    split: str,
    group: str,
) -> dict[str, Any]:
    if split != "train" or group != "capacity":
        raise RuntimeError("PCLD capacity receipt requires train/capacity")
    return {
        "format": "pcld_capacity_support_receipt_v1",
        "complete": True,
        "source_rollout": str(rollout.resolve()),
        "source_manifest": str(manifest.resolve()),
        "target": str(target.resolve()),
        "sidecar": str(sidecar.resolve()),
        "split": split,
        "group": group,
        "sidecar_replay_verified": True,
        "support": capacity_support_summary(records, epsilon_num),
    }


def validate_capacity_support_receipt(
    path: Path,
    records: list[dict[str, Any]],
    epsilon_num: float,
    *,
    rollout: Path,
    manifest: Path,
    target: Path,
    sidecar: Path,
    split: str,
    group: str,
    replay_report: dict[str, Any],
) -> dict[str, Any]:
    receipt = json.loads(path.read_text())
    if replay_report.get("verified") is not True:
        raise RuntimeError("PCLD capacity receipt requires verified sidecar replay")
    expected = build_capacity_support_receipt(
        records,
        epsilon_num,
        rollout=rollout,
        manifest=manifest,
        target=target,
        sidecar=sidecar,
        split=split,
        group=group,
    )
    if receipt != expected:
        raise RuntimeError("PCLD capacity support receipt differs from recomputation")
    return receipt


def capacity_manifest_keys(
    manifest: dict[str, Any],
) -> set[tuple[str, int, int]]:
    items = manifest.get("capacity", {}).get("records")
    if not isinstance(items, list) or len(items) != 512:
        raise RuntimeError("PCLD capacity group must contain exactly 512 records")
    keys = [
        (
            str(item["sample_id"]),
            int(item["anchor_offset"]),
            int(item["context_length"]),
        )
        for item in items
    ]
    if len(set(keys)) != 512 or len({key[0] for key in keys}) != 512:
        raise RuntimeError("PCLD capacity group must use 512 unique prompts/records")
    return set(keys)


def select_manifest_group(
    records: list[dict[str, Any]],
    manifest: dict[str, Any],
    group: str,
) -> list[dict[str, Any]]:
    if group == "all":
        selected = list(records)
    elif group == "capacity":
        keys = capacity_manifest_keys(manifest)
        selected = [record for record in records if record_key(record) in keys]
        if {record_key(record) for record in selected} != keys:
            raise RuntimeError("capacity manifest does not align with the rollout")
    elif group in {"fit", "select", "diagnostic"}:
        prompts = manifest.get("prompt_splits", {}).get(group)
        if not isinstance(prompts, list) or not prompts:
            raise RuntimeError(f"manifest has no prompt split {group}")
        prompt_set = {str(value) for value in prompts}
        if len(prompt_set) != len(prompts):
            raise RuntimeError(f"manifest prompt split {group} contains duplicates")
        selected = [
            record for record in records if str(record["sample_id"]) in prompt_set
        ]
        if {str(record["sample_id"]) for record in selected} != prompt_set:
            raise RuntimeError(f"manifest prompt split {group} does not align")
    else:
        raise ValueError(f"unsupported PCLD manifest group {group!r}")
    keys = [record_key(record) for record in selected]
    if not selected or len(keys) != len(set(keys)):
        raise RuntimeError(f"PCLD group {group} is empty or has duplicate records")
    return selected


def select_balanced_smoke_records(
    records: list[dict[str, Any]], *, count: int = 32
) -> list[dict[str, Any]]:
    """Select a deterministic cross-domain, distinct-prompt mechanics subset."""

    if count < 3:
        raise ValueError("balanced smoke requires at least three records")
    by_domain: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen_prompts: set[str] = set()
    for record in records:
        sample_id = str(record["sample_id"])
        if sample_id in seen_prompts:
            continue
        seen_prompts.add(sample_id)
        by_domain[str(record["domain"])].append(record)
    if set(by_domain) != {"chat", "code", "math"}:
        raise RuntimeError("smoke selection requires chat/code/math domains")
    result: list[dict[str, Any]] = []
    domain_order = ("chat", "code", "math")
    cursors = {domain: 0 for domain in domain_order}
    while len(result) < count:
        progressed = False
        for domain in domain_order:
            index = cursors[domain]
            if index < len(by_domain[domain]) and len(result) < count:
                result.append(by_domain[domain][index])
                cursors[domain] += 1
                progressed = True
        if not progressed:
            raise RuntimeError(f"only {len(result)} distinct prompts available for smoke")
    return result


def validate_sidecar_item(item: dict[str, Any]) -> None:
    expected = {
        "base_logsumexp": (BLOCK_LENGTH,),
        "base_candidate_logits": (BLOCK_LENGTH, CANDIDATES),
        "target_hidden": (BLOCK_LENGTH, HIDDEN_SIZE),
        "target_candidate_logits": (BLOCK_LENGTH, CANDIDATES),
        "authoritative_top1_ids": (BLOCK_LENGTH,),
        "fp32_top1_ids": (BLOCK_LENGTH,),
        "target_top1_margins": (BLOCK_LENGTH,),
        "centered_max_errors": (BLOCK_LENGTH,),
    }
    record_key(item)
    for name, shape in expected.items():
        value = item.get(name)
        if not isinstance(value, Tensor) or tuple(value.shape) != shape:
            raise RuntimeError(
                f"PCLD sidecar field {name} must have shape {shape}"
            )
    for name in (
        "base_logsumexp",
        "base_candidate_logits",
        "target_hidden",
        "target_candidate_logits",
        "target_top1_margins",
        "centered_max_errors",
    ):
        if not bool(torch.isfinite(item[name].float()).all().item()):
            raise RuntimeError(f"PCLD sidecar field {name} is non-finite")
    cancellation_error = item.get("residual_cancellation_max_error")
    if not isinstance(cancellation_error, (float, int)) or not bool(
        torch.isfinite(torch.tensor(float(cancellation_error))).item()
    ):
        raise RuntimeError("PCLD residual-cancellation diagnostic is invalid")


def load_pcld_sidecar(
    root: Path,
) -> tuple[dict[str, Any], dict[tuple[str, int, int], dict[str, Any]]]:
    metadata = json.loads((root / "metadata.json").read_text())
    if metadata.get("format") != "pcld_sidecar_v1" or not metadata.get(
        "collection_complete", False
    ):
        raise RuntimeError(f"unsupported or incomplete PCLD sidecar: {root}")
    values: dict[tuple[str, int, int], dict[str, Any]] = {}
    for shard in sorted(root.glob("shard-*.pt")):
        items = torch.load(shard, map_location="cpu", weights_only=False)
        for item in items:
            validate_sidecar_item(item)
            key = record_key(item)
            if key in values:
                raise RuntimeError(f"duplicate PCLD sidecar key: {key}")
            values[key] = item
    if len(values) != int(metadata.get("records", -1)):
        raise RuntimeError("PCLD sidecar record count differs from metadata")
    return metadata, values


def validate_sidecar_source(
    metadata: dict[str, Any],
    *,
    rollout: Path,
    target: Path,
    split: str,
    group: str,
) -> None:
    expected = {
        "source_rollout": str(rollout.resolve()),
        "target": str(target.resolve()),
        "split": split,
        "group": group,
    }
    mismatches = {
        key: (metadata.get(key), value)
        for key, value in expected.items()
        if metadata.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"PCLD sidecar source contract mismatch: {mismatches}")


def validate_sidecar_receipt(
    root: Path,
    metadata: dict[str, Any],
    *,
    require_manual_records: int = 0,
) -> dict[str, Any]:
    path = root / "replay_report.json"
    if not path.is_file():
        raise RuntimeError(f"PCLD sidecar lacks GPU replay receipt: {path}")
    report = json.loads(path.read_text())
    if report.get("format") != "pcld_sidecar_replay_v1":
        raise RuntimeError("unsupported PCLD sidecar replay receipt")
    required = (
        "verified",
        "base_lattice_exact",
        "target_hidden_allclose",
        "target_candidate_scores_allclose",
        "numeric_authority_exact",
    )
    failed = [name for name in required if report.get(name) is not True]
    if failed:
        raise RuntimeError(f"PCLD sidecar replay failed: {failed}")
    if int(report.get("records", -1)) != int(metadata.get("records", -2)):
        raise RuntimeError("PCLD replay/metadata record count mismatch")
    if report.get("sidecar") != str(root.resolve()):
        raise RuntimeError("PCLD replay path mismatch")
    if report.get("source_rollout") != metadata.get("source_rollout"):
        raise RuntimeError("PCLD replay source mismatch")
    for name in ("target", "split", "group"):
        if report.get(name) != metadata.get(name):
            raise RuntimeError(f"PCLD replay {name} mismatch")
    if require_manual_records:
        if int(report.get("manual_parity_records", -1)) < require_manual_records:
            raise RuntimeError("PCLD replay lacks required manual-prefix coverage")
        manual_required = (
            "manual_parity_passed",
            "manual_row_alignment_exact",
            "manual_stable_top1_exact",
            "manual_row0_alignment_exact",
            "manual_row15_alignment_exact",
            "manual_row0_stable_top1_exact",
            "manual_row15_stable_top1_exact",
        )
        failed_manual = [
            name for name in manual_required if report.get(name) is not True
        ]
        if failed_manual:
            raise RuntimeError(
                f"PCLD manual-prefix replay receipt failed: {failed_manual}"
            )
    return report


def attach_pcld_sidecar(
    records: Iterable[dict[str, Any]],
    sidecar: dict[tuple[str, int, int], dict[str, Any]],
    *,
    require_exact_keys: bool,
) -> list[dict[str, Any]]:
    attached: list[dict[str, Any]] = []
    used: set[tuple[str, int, int]] = set()
    for source in records:
        key = record_key(source)
        item = sidecar.get(key)
        if item is None:
            raise RuntimeError(f"PCLD sidecar lacks rollout key {key}")
        record = dict(source)
        for name in (
            "base_logsumexp",
            "base_candidate_logits",
            "target_hidden",
            "target_candidate_logits",
            "authoritative_top1_ids",
            "fp32_top1_ids",
            "target_top1_margins",
            "centered_max_errors",
            "residual_cancellation_max_error",
        ):
            if name in item:
                record[f"pcld_{name}"] = item[name]
        attached.append(record)
        used.add(key)
    if require_exact_keys and set(sidecar) != used:
        raise RuntimeError("PCLD sidecar keys differ from the selected rollout group")
    return attached


def calibrate_epsilon_from_records(records: list[dict[str, Any]]) -> float:
    authoritative = torch.stack(
        [record["pcld_authoritative_top1_ids"].long() for record in records]
    )
    replay = torch.stack(
        [record["pcld_fp32_top1_ids"].long() for record in records]
    )
    errors = torch.stack(
        [record["pcld_centered_max_errors"].float() for record in records]
    )
    return float(calibrate_numeric_epsilon(authoritative, replay, errors).item())


def record_support(record: dict[str, Any], epsilon_num: float) -> Tensor:
    candidate_ids = record["base_topk_ids"].long().unsqueeze(0)
    gold_ids = record["gold_ids"].long().unsqueeze(0)
    ranks = candidate_gold_ranks(candidate_ids, gold_ids)
    stable = stable_teacher_rows(
        record["pcld_authoritative_top1_ids"].long().unsqueeze(0),
        record["pcld_fp32_top1_ids"].long().unsqueeze(0),
        record["pcld_target_top1_margins"].float().unsqueeze(0),
        epsilon_num,
    )
    support, _ = continuous_clean_support(
        ranks,
        record["pcld_authoritative_top1_ids"].long().unsqueeze(0),
        gold_ids,
        stable,
    )
    return support[0]


def filter_effective_records(
    records: list[dict[str, Any]], epsilon_num: float
) -> list[dict[str, Any]]:
    result = [record for record in records if bool(record_support(record, epsilon_num).any())]
    if not result:
        raise RuntimeError("PCLD training group has no effective clean-prefix blocks")
    return result


def compute_latent_scale(
    records: list[dict[str, Any]], epsilon_num: float
) -> tuple[Tensor, int]:
    """Compute train-only per-channel population std over supported rows."""

    total = torch.zeros(HIDDEN_SIZE, dtype=torch.float64)
    total_square = torch.zeros(HIDDEN_SIZE, dtype=torch.float64)
    count = 0
    for record in records:
        support = record_support(record, epsilon_num)
        residual = (
            record["pcld_target_hidden"].float()
            - record["parallel_hidden"].float()
        )[support]
        if residual.numel() == 0:
            continue
        residual64 = residual.to(torch.float64)
        total += residual64.sum(dim=0)
        total_square += residual64.square().sum(dim=0)
        count += residual64.shape[0]
    if count < 1:
        raise RuntimeError("cannot compute PCLD latent scale without supported rows")
    mean = total / count
    variance = (total_square / count - mean.square()).clamp_min(0.0)
    scale = variance.sqrt().clamp_min(LATENT_SCALE_FLOOR).float()
    if not bool(torch.isfinite(scale).all().item()):
        raise RuntimeError("PCLD latent scale is non-finite")
    return scale, count


def collate_pcld_records(
    records: list[dict[str, Any]],
    *,
    epsilon_num: float,
    require_effective: bool,
) -> dict[str, Any]:
    if not records:
        raise ValueError("cannot collate an empty PCLD batch")
    candidate_ids = torch.stack([record["base_topk_ids"].long() for record in records])
    # Use the exact FP32 Top16 values replayed from the BF16 vocabulary GEMM,
    # not the canonical rollout's compact float16 storage representation.
    candidate_logits = torch.stack(
        [record["pcld_base_candidate_logits"].float() for record in records]
    )
    gold_ids = torch.stack([record["gold_ids"].long() for record in records])
    gold_ranks = candidate_gold_ranks(candidate_ids, gold_ids)
    authoritative_top1 = torch.stack(
        [record["pcld_authoritative_top1_ids"].long() for record in records]
    )
    legacy_top1 = torch.stack(
        [record["target_top1_ids"].long() for record in records]
    )
    fp32_top1 = torch.stack(
        [record["pcld_fp32_top1_ids"].long() for record in records]
    )
    target_margins = torch.stack(
        [record["pcld_target_top1_margins"].float() for record in records]
    )
    stable = stable_teacher_rows(
        authoritative_top1, fp32_top1, target_margins, epsilon_num
    )
    authoritative_matches_gold = authoritative_top1.eq(gold_ids)
    numeric_agreement = authoritative_top1.eq(fp32_top1)
    support, horizons = continuous_clean_support(
        gold_ranks, authoritative_top1, gold_ids, stable
    )
    if require_effective and bool(horizons.eq(0).any().item()):
        bad = [
            record_key(record)
            for record, horizon in zip(records, horizons.tolist(), strict=True)
            if horizon == 0
        ]
        raise RuntimeError(f"PCLD sampled horizon-zero blocks: {bad}")
    return {
        "hidden": torch.stack([record["parallel_hidden"] for record in records]),
        "candidate_ids": candidate_ids,
        "candidate_logits": candidate_logits,
        "base_logsumexp": torch.stack(
            [record["pcld_base_logsumexp"].float() for record in records]
        ),
        "gold_ids": gold_ids,
        "policy_ids": torch.stack([record["policy_ids"].long() for record in records]),
        "gold_candidate_ranks": gold_ranks,
        "target_residual": torch.stack(
            [
                record["pcld_target_hidden"].float()
                - record["parallel_hidden"].float()
                for record in records
            ]
        ),
        "target_candidate_logits": torch.stack(
            [record["pcld_target_candidate_logits"].float() for record in records]
        ),
        "target_top1_ids": authoritative_top1,
        "stable_rows": stable,
        # Evaluation-only support definitions.  None is accepted by the
        # production forward helper or any PCLD training loss.
        "legacy_j2_target_matches_gold": legacy_top1.eq(gold_ids),
        "authoritative_j2_target_matches_gold": authoritative_matches_gold,
        "authoritative_numeric_j2_target_matches_gold": (
            authoritative_matches_gold & numeric_agreement
        ),
        "stable_j2_target_matches_gold": authoritative_matches_gold & stable,
        "support_mask": support,
        "horizons": horizons,
        "sample_ids": [str(record["sample_id"]) for record in records],
        "domains": [str(record["domain"]) for record in records],
        "anchor_offsets": torch.tensor(
            [int(record["anchor_offset"]) for record in records], dtype=torch.long
        ),
        "context_lengths": torch.tensor(
            [int(record["context_length"]) for record in records], dtype=torch.long
        ),
    }


def pcld_forward_inputs(batch: dict[str, Any], lm_head_weight: Tensor) -> dict[str, Tensor]:
    """Build the exact target-free production call from a training batch."""

    values = {
        "hidden": batch["hidden"],
        "candidate_lm_rows": lm_head_weight[batch["candidate_ids"]],
        "candidate_logits": batch["candidate_logits"],
        "base_logsumexp": batch["base_logsumexp"],
    }
    if set(values) != PCLD_FORWARD_FIELDS:
        raise AssertionError("PCLD production input whitelist drifted")
    return values


def group_record_indices_by_prompt(records: list[dict[str, Any]]) -> dict[str, list[int]]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        grouped[str(record["sample_id"])].append(index)
    if not grouped:
        raise ValueError("cannot group an empty PCLD dataset")
    return dict(grouped)


def sample_prompt_balanced_records(
    records: list[dict[str, Any]],
    grouped_indices: dict[str, list[int]],
    *,
    batch_size: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    """Sample prompt uniformly, then one effective block uniformly."""

    if batch_size < 1:
        raise ValueError("PCLD batch size must be positive")
    prompts = tuple(sorted(grouped_indices))
    if not prompts:
        raise ValueError("PCLD prompt sampler has no prompts")
    batch: list[dict[str, Any]] = []
    for _ in range(batch_size):
        sample_id = prompts[rng.randrange(len(prompts))]
        indices = grouped_indices[sample_id]
        batch.append(records[indices[rng.randrange(len(indices))]])
    return batch
