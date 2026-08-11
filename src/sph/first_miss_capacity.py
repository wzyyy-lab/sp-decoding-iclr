"""Deterministic manifest support for FMAS same-subset capacity probes."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any

import torch


MANIFEST_VERSION = 1
SELECTION_PROTOCOL = "global_direct_deterministic_capacity_subset_v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def describe_capacity_record(
    record: dict[str, Any], *, candidate_k: int
) -> dict[str, Any]:
    """Return an identity and canonical action descriptor for one block."""

    topk = record["base_topk_ids"][:, :candidate_k].long()
    gold = record["gold_ids"].long()
    if topk.ndim != 2 or gold.ndim != 1 or topk.shape[0] != gold.shape[0]:
        raise ValueError("capacity record has inconsistent lattice shapes")
    if candidate_k < 2 or topk.shape[1] != candidate_k:
        raise ValueError("capacity record does not contain declared K")

    base_correct = topk[:, 0].eq(gold)
    if bool(base_correct.all()):
        target_kind = "keep_full_correct"
        first_miss_position = None
        gold_rank = None
        gain = 0
        action = 0
    else:
        first_miss = int(
            (~base_correct).to(torch.int64).argmax().item()
        )
        matches = topk[first_miss].eq(gold[first_miss])
        if not bool(matches.any()):
            target_kind = "keep_out_of_k"
            first_miss_position = first_miss
            gold_rank = None
            gain = 0
            action = 0
        else:
            rank = int(matches.to(torch.int64).argmax().item())
            if rank == 0:
                raise RuntimeError(
                    "base first miss cannot have gold at candidate rank zero"
                )
            repaired = base_correct.clone()
            repaired[first_miss] = True
            oracle_prefix = int(
                repaired.to(torch.int64).cumprod(dim=0).sum().item()
            )
            target_kind = "edit"
            first_miss_position = first_miss
            gold_rank = rank
            gain = oracle_prefix - first_miss
            action = 1 + first_miss * (candidate_k - 1) + (rank - 1)

    return {
        "sample_id": str(record["sample_id"]),
        "anchor_offset": int(record["anchor_offset"]),
        "context_length": int(record["context_length"]),
        "anchor_token_id": int(record["anchor_token_id"]),
        "domain": str(record["domain"]),
        "split": str(record["split"]),
        "target_kind": target_kind,
        "first_miss_position": first_miss_position,
        "gold_rank": gold_rank,
        "gain": gain,
        "action": action,
    }


def build_capacity_manifest(
    records: list[dict[str, Any]],
    *,
    source_metadata_path: Path,
    candidate_k: int,
    seed: int,
    opportunity_fraction: float,
) -> dict[str, Any]:
    """Build a deterministic, timestamp-free manifest for selected records."""

    entries = [
        describe_capacity_record(record, candidate_k=candidate_k)
        for record in records
    ]
    identity_keys = [
        (
            entry["sample_id"],
            entry["anchor_offset"],
            entry["context_length"],
        )
        for entry in entries
    ]
    if len(set(identity_keys)) != len(identity_keys):
        raise RuntimeError("capacity subset contains duplicate block identities")

    kind_counts = Counter(entry["target_kind"] for entry in entries)
    position_counts = Counter(
        str(entry["first_miss_position"])
        for entry in entries
        if entry["target_kind"] == "edit"
    )
    rank_counts = Counter(
        str(entry["gold_rank"])
        for entry in entries
        if entry["target_kind"] == "edit"
    )
    gain_counts = Counter(
        str(entry["gain"])
        for entry in entries
        if entry["target_kind"] == "edit"
    )
    action_counts = Counter(str(entry["action"]) for entry in entries)
    domain_counts = Counter(entry["domain"] for entry in entries)
    subset_sha256 = _canonical_sha256(entries)
    return {
        "manifest_version": MANIFEST_VERSION,
        "selection_protocol": SELECTION_PROTOCOL,
        "source_metadata_path": str(source_metadata_path.resolve()),
        "source_metadata_sha256": sha256_file(source_metadata_path),
        "candidate_k": int(candidate_k),
        "seed": int(seed),
        "opportunity_fraction": float(opportunity_fraction),
        "blocks": len(entries),
        "subset_sha256": subset_sha256,
        "composition": {
            "target_kind": dict(sorted(kind_counts.items())),
            "edit_position": dict(sorted(position_counts.items(), key=lambda x: int(x[0]))),
            "edit_gold_rank": dict(sorted(rank_counts.items(), key=lambda x: int(x[0]))),
            "edit_gain": dict(sorted(gain_counts.items(), key=lambda x: int(x[0]))),
            "target_action": dict(sorted(action_counts.items(), key=lambda x: int(x[0]))),
            "domain": dict(sorted(domain_counts.items())),
        },
        "entries": entries,
    }


def verify_capacity_manifest(
    manifest: dict[str, Any],
    records: list[dict[str, Any]],
    *,
    source_metadata_path: Path,
    candidate_k: int,
    seed: int,
    opportunity_fraction: float,
) -> dict[str, Any]:
    """Rebuild and byte-semantically compare a frozen capacity manifest."""

    expected = build_capacity_manifest(
        records,
        source_metadata_path=source_metadata_path,
        candidate_k=candidate_k,
        seed=seed,
        opportunity_fraction=opportunity_fraction,
    )
    if manifest != expected:
        differing = sorted(
            key
            for key in set(manifest) | set(expected)
            if manifest.get(key) != expected.get(key)
        )
        raise RuntimeError(
            "capacity manifest does not match deterministic selection: "
            f"{differing}"
        )
    return expected

