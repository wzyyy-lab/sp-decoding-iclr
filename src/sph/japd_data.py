"""Data contracts for JAPD-16 full-block training and evaluation."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
import json
import random
from typing import Any, Iterable

import torch
from torch import Tensor

from sph.japd import (
    BLOCK_LENGTH,
    CANDIDATES,
    candidate_gold_ranks,
    clean_support,
    validate_full16_lattice,
)


HEAD_BATCH_FIELDS = frozenset(
    {
        "hidden",
        "candidate_ids",
        "candidate_logits",
        "base_logsumexp",
        "anchor_ids",
        "gold_ids",
        "policy_ids",
        "gold_candidate_ranks",
        "target_candidate_logits",
        "target_matches_gold",
        "effective_blocks_per_prompt",
        "sample_ids",
        "domains",
        "anchor_offsets",
        "context_lengths",
    }
)

FORBIDDEN_ONLINE_FEATURE_FIELDS = frozenset(
    {
        "target_anchor_early_feature",
        "target_boundary_feature",
        "target_hidden",
        "target_logits",
        "selected_token_embeddings",
    }
)


def record_key(record: dict[str, Any]) -> tuple[str, int, int]:
    """Stable semantic key for a rollout block and its derived sidecar."""

    return (
        str(record["sample_id"]),
        int(record["anchor_offset"]),
        int(record["context_length"]),
    )


def validate_rollout_record(record: dict[str, Any]) -> None:
    expected = {
        "parallel_hidden": (BLOCK_LENGTH, 2560),
        "base_topk_ids": (BLOCK_LENGTH, CANDIDATES),
        "base_topk_logits": (BLOCK_LENGTH, CANDIDATES),
        "gold_ids": (BLOCK_LENGTH,),
        "target_candidate_logits": (BLOCK_LENGTH, CANDIDATES),
        "target_top1_ids": (BLOCK_LENGTH,),
        "policy_ids": (BLOCK_LENGTH,),
    }
    for name, shape in expected.items():
        value = record.get(name)
        if not isinstance(value, Tensor) or tuple(value.shape) != shape:
            raise RuntimeError(
                f"record field {name} must have shape {shape}, got "
                f"{None if value is None else tuple(value.shape)}"
            )
    validate_full16_lattice(record["base_topk_ids"].unsqueeze(0).long())
    if "sample_id" not in record or "domain" not in record:
        raise RuntimeError("rollout record lacks sample/domain identity")
    record_key(record)


def load_rollout_records(
    root: Path,
    *,
    split: str,
    max_records: int = 0,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    metadata = json.loads((root / "metadata.json").read_text())
    if metadata.get("format") != "gfpr_rollout_v1":
        raise RuntimeError(f"unsupported rollout format at {root}")
    if not metadata.get("collection_complete", False):
        raise RuntimeError(f"rollout collection is incomplete: {root}")
    records: list[dict[str, Any]] = []
    seen: set[tuple[str, int, int]] = set()
    for shard in sorted(root.glob("shard-*.pt")):
        shard_records = torch.load(
            shard, map_location="cpu", weights_only=False
        )
        for record in shard_records:
            if str(record["split"]) != split:
                continue
            validate_rollout_record(record)
            key = record_key(record)
            if key in seen:
                raise RuntimeError(f"duplicate rollout identity: {key}")
            seen.add(key)
            records.append(record)
            if max_records and len(records) >= max_records:
                return metadata, records
    if not records:
        raise ValueError(f"no records for split {split!r} under {root}")
    return metadata, records


def load_lse_sidecar(
    root: Path,
) -> tuple[dict[str, Any], dict[tuple[str, int, int], Tensor]]:
    metadata = json.loads((root / "metadata.json").read_text())
    if metadata.get("format") != "japd_base_lse_v1":
        raise RuntimeError(f"unsupported JAPD sidecar format at {root}")
    if not metadata.get("collection_complete", False):
        raise RuntimeError(f"incomplete JAPD sidecar: {root}")
    values: dict[tuple[str, int, int], Tensor] = {}
    for shard in sorted(root.glob("shard-*.pt")):
        for item in torch.load(shard, map_location="cpu", weights_only=False):
            key = (
                str(item["sample_id"]),
                int(item["anchor_offset"]),
                int(item["context_length"]),
            )
            lse = item.get("base_logsumexp")
            if not isinstance(lse, Tensor) or tuple(lse.shape) != (BLOCK_LENGTH,):
                raise RuntimeError(f"invalid base_logsumexp for {key}")
            if key in values:
                raise RuntimeError(f"duplicate sidecar identity: {key}")
            values[key] = lse.float()
    expected = int(metadata.get("records", len(values)))
    if len(values) != expected:
        raise RuntimeError(
            f"sidecar record count mismatch: metadata {expected}, loaded {len(values)}"
        )
    return metadata, values


def attach_lse_sidecar(
    records: Iterable[dict[str, Any]],
    sidecar: dict[tuple[str, int, int], Tensor],
    *,
    require_exact_keys: bool = False,
) -> list[dict[str, Any]]:
    attached: list[dict[str, Any]] = []
    used: set[tuple[str, int, int]] = set()
    for source in records:
        key = record_key(source)
        if key not in sidecar:
            raise RuntimeError(f"sidecar lacks rollout identity: {key}")
        record = dict(source)
        record["base_logsumexp"] = sidecar[key]
        attached.append(record)
        used.add(key)
    extras = set(sidecar).difference(used)
    if require_exact_keys and extras:
        raise RuntimeError(
            f"sidecar has {len(extras)} records outside the selected rollout"
        )
    return attached


def effective_record_mask(records: list[dict[str, Any]]) -> Tensor:
    effective: list[bool] = []
    for record in records:
        ids = record["base_topk_ids"].long().unsqueeze(0)
        gold = record["gold_ids"].long().unsqueeze(0)
        ranks = candidate_gold_ranks(ids, gold)
        target_matches = record["target_top1_ids"].long().eq(
            record["gold_ids"].long()
        ).unsqueeze(0)
        _, horizon = clean_support(ranks, target_matches)
        effective.append(bool(horizon.item() > 0))
    return torch.tensor(effective, dtype=torch.bool)


def stratified_prompt_split(
    records: list[dict[str, Any]],
    *,
    seed: int = 20260810,
    split_counts_by_domain: dict[str, tuple[int, int, int]] | None = None,
) -> dict[str, set[str]]:
    """Label-independent fit/select/diagnostic split by prompt and domain."""

    prompt_domains: dict[str, str] = {}
    for record in records:
        sample_id = str(record["sample_id"])
        domain = str(record["domain"])
        previous = prompt_domains.setdefault(sample_id, domain)
        if previous != domain:
            raise RuntimeError(f"prompt {sample_id} appears in multiple domains")
    by_domain: dict[str, list[str]] = defaultdict(list)
    for sample_id, domain in prompt_domains.items():
        by_domain[domain].append(sample_id)
    defaults = {
        "math": (533, 67, 67),
        "code": (532, 66, 67),
        "chat": (524, 66, 65),
    }
    requested = defaults if split_counts_by_domain is None else split_counts_by_domain
    unknown = set(by_domain).symmetric_difference(requested)
    if unknown:
        raise RuntimeError(f"domain split contract mismatch: {sorted(unknown)}")
    result = {"fit": set(), "select": set(), "diagnostic": set()}
    rng = random.Random(seed)
    for domain in sorted(by_domain):
        prompts = sorted(by_domain[domain])
        rng.shuffle(prompts)
        fit_count, select_count, diagnostic_count = requested[domain]
        if fit_count + select_count + diagnostic_count != len(prompts):
            raise RuntimeError(
                f"split counts for {domain} do not cover {len(prompts)} prompts"
            )
        fit_end = fit_count
        select_end = fit_end + select_count
        result["fit"].update(prompts[:fit_end])
        result["select"].update(prompts[fit_end:select_end])
        result["diagnostic"].update(prompts[select_end:])
    if result["fit"] & result["select"] or result["fit"] & result["diagnostic"] or result["select"] & result["diagnostic"]:
        raise AssertionError("prompt split overlap")
    if set().union(*result.values()) != set(prompt_domains):
        raise AssertionError("prompt split does not cover the source prompts")
    return result


def effective_blocks_per_prompt(
    records: list[dict[str, Any]],
) -> dict[str, int]:
    mask = effective_record_mask(records)
    counts: Counter[str] = Counter(
        str(record["sample_id"])
        for record, keep in zip(records, mask.tolist(), strict=True)
        if keep
    )
    return dict(counts)


def collate_japd_records(
    records: list[dict[str, Any]],
    *,
    prompt_effective_counts: dict[str, int],
    require_effective: bool = True,
) -> dict[str, Any]:
    """Whitelist-only collate; target online features cannot enter the head batch."""

    if not records:
        raise ValueError("cannot collate an empty JAPD batch")
    candidate_ids = torch.stack(
        [record["base_topk_ids"].long() for record in records]
    )
    candidate_logits = torch.stack(
        [record["base_topk_logits"].float() for record in records]
    )
    validate_full16_lattice(candidate_ids, candidate_logits)
    gold_ids = torch.stack([record["gold_ids"].long() for record in records])
    gold_candidate_ranks = candidate_gold_ranks(candidate_ids, gold_ids)
    target_matches_gold = torch.stack(
        [
            record["target_top1_ids"].long().eq(record["gold_ids"].long())
            for record in records
        ]
    )
    _, horizons = clean_support(gold_candidate_ranks, target_matches_gold)
    if require_effective and bool(horizons.eq(0).any().item()):
        bad = [
            record_key(record)
            for record, horizon in zip(records, horizons.tolist(), strict=True)
            if horizon == 0
        ]
        raise RuntimeError(
            f"JAPD collate received horizon-zero blocks outside B_p^+: {bad}"
        )
    sample_ids = [str(record["sample_id"]) for record in records]
    counts = []
    for sample_id in sample_ids:
        count = int(prompt_effective_counts.get(sample_id, 0))
        if require_effective and count < 1:
            raise RuntimeError(f"sampled ineffective prompt/block: {sample_id}")
        counts.append(count)
    batch = {
        "hidden": torch.stack(
            [record["parallel_hidden"] for record in records]
        ),
        "candidate_ids": candidate_ids,
        "candidate_logits": candidate_logits,
        "base_logsumexp": torch.stack(
            [record["base_logsumexp"].float() for record in records]
        ),
        "anchor_ids": torch.tensor(
            [int(record["anchor_token_id"]) for record in records],
            dtype=torch.long,
        ),
        "gold_ids": gold_ids,
        # Released Domino is an offline evaluation baseline only.  It is
        # deliberately absent from GlobalDirectCandidateSelector.forward.
        "policy_ids": torch.stack(
            [record["policy_ids"].long() for record in records]
        ),
        "gold_candidate_ranks": gold_candidate_ranks,
        "target_candidate_logits": torch.stack(
            [record["target_candidate_logits"].float() for record in records]
        ),
        "target_matches_gold": target_matches_gold,
        "effective_blocks_per_prompt": torch.tensor(counts, dtype=torch.float32),
        "sample_ids": sample_ids,
        "domains": [str(record["domain"]) for record in records],
        "anchor_offsets": torch.tensor(
            [int(record["anchor_offset"]) for record in records], dtype=torch.long
        ),
        "context_lengths": torch.tensor(
            [int(record["context_length"]) for record in records], dtype=torch.long
        ),
    }
    if set(batch) != HEAD_BATCH_FIELDS:
        raise AssertionError("JAPD collate output deviates from the whitelist")
    if set(batch).intersection(FORBIDDEN_ONLINE_FEATURE_FIELDS):
        raise AssertionError("forbidden target online feature entered the batch")
    return batch
