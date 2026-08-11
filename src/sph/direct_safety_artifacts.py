"""Reviewed artifact contracts for staged PROS-Gate experiments.

Unlike the Gate-0 modules, this module intentionally owns filesystem I/O for
R079 and later stages.  Its helpers are deterministic, content-addressed, and
fail closed on split, identity, tensor, outcome, or provenance drift.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
from typing import Any, Iterable, Mapping, Sequence
import uuid

import torch
from torch import Tensor

from sph.direct_safety_gate import (
    BLOCK_LENGTH,
    CANDIDATE_COUNT,
    POSITION_FEATURE_DIMENSION,
    binary_outcomes_from_tokens,
)
from sph.direct_safety_numeric_policy import (
    assert_numeric_policy_binding,
    assert_portable_feature_relations,
    numeric_policy_receipt,
)
from sph.direct_safety_protocol import (
    BlockKey,
    CapacityRecord,
    DEFAULT_SPLIT_COUNTS,
    build_prompt_split,
    capacity_records_sha256,
    ordered_block_keys_sha256,
    select_capacity_records,
)
from sph.source_closure import SOURCE_CLOSURE_PROTOCOL


ARTIFACT_FORMAT_VERSION = 1
OUTCOME_ARTIFACT_FORMAT_VERSION = 2
SPLIT_MANIFEST_PROTOCOL = "pros-gate-phase3-split-manifest-v1"
OUTCOME_ARTIFACT_PROTOCOL = "pros-gate-direct-outcomes-v2"
CAPACITY_ARTIFACT_PROTOCOL = "pros-gate-capacity-artifact-v1"
PHASE3_PROMPTS = 1_987
PHASE3_BLOCKS = 15_886
REQUIRED_EXCLUSION_ROLES = frozenset(
    {"producer_train", "validation", "reserved"}
)
VALID_OUTCOME_SPLITS = frozenset({"fit", "checkpoint", "falsifier"})


def _require_sha256(name: str, value: Any) -> str:
    if not isinstance(value, str) or len(value) != 64 or value != value.lower():
        raise ValueError(f"{name} must be a canonical lowercase SHA256")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError(f"{name} must be hexadecimal") from error
    return value


def _validated_source_closure(value: Mapping[str, Any]) -> dict[str, Any]:
    expected_fields = {
        "protocol",
        "source_manifest_sha256",
        "source_file_count",
        "source_entries_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != expected_fields:
        raise ValueError("source closure summary fields differ")
    if value.get("protocol") != SOURCE_CLOSURE_PROTOCOL:
        raise ValueError("source closure protocol differs")
    count = value.get("source_file_count")
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError("source closure file count is invalid")
    return {
        "protocol": SOURCE_CLOSURE_PROTOCOL,
        "source_manifest_sha256": _require_sha256(
            "source manifest hash", value.get("source_manifest_sha256")
        ),
        "source_file_count": count,
        "source_entries_sha256": _require_sha256(
            "source entries hash", value.get("source_entries_sha256")
        ),
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def prompt_set_sha256(prompt_ids: Iterable[str]) -> str:
    values = sorted(set(prompt_ids))
    if not values or any(not value or "\0" in value for value in values):
        raise ValueError("prompt IDs must be nonempty and contain no NUL")
    return hashlib.sha256("\n".join(values).encode("utf-8")).hexdigest()


def atomic_write_json(path: Path, value: Any) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("wb") as handle:
            handle.write(canonical_json_bytes(value))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_torch_save(path: Path, value: Any) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        torch.save(value, temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def block_key_from_mapping(record: Mapping[str, Any]) -> BlockKey:
    return BlockKey(
        sample_id=record["sample_id"],
        anchor_offset=record["anchor_offset"],
        context_length=record["context_length"],
    )


def _identity_rows(
    records: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, str], list[dict[str, Any]]]:
    prompt_domains: dict[str, str] = {}
    keys: set[BlockKey] = set()
    rows: list[dict[str, Any]] = []
    for record in records:
        key = block_key_from_mapping(record)
        key.serialize()
        domain = record.get("domain")
        if not isinstance(domain, str) or not domain:
            raise ValueError("every block needs a nonempty domain")
        previous = prompt_domains.setdefault(key.sample_id, domain)
        if previous != domain:
            raise ValueError("one prompt appears under multiple domains")
        if key in keys:
            raise ValueError("block identities must be unique")
        keys.add(key)
        rows.append(
            {
                "sample_id": key.sample_id,
                "anchor_offset": int(key.anchor_offset),
                "context_length": int(key.context_length),
                "domain": domain,
            }
        )
    if not rows:
        raise ValueError("identity records cannot be empty")
    rows.sort(
        key=lambda row: block_key_from_mapping(row).serialize()
    )
    return prompt_domains, rows


def build_phase3_split_manifest(
    records: Sequence[Mapping[str, Any]],
    *,
    canonical_metadata_sha256: str,
    exclusion_prompt_sets: Mapping[str, Iterable[str]],
    exclusion_manifest_sha256: Mapping[str, str],
    split_counts: Mapping[str, Mapping[str, int]] = DEFAULT_SPLIT_COUNTS,
    expected_prompts: int = PHASE3_PROMPTS,
    expected_blocks: int = PHASE3_BLOCKS,
    provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Freeze prompt assignments and overlap proofs before Direct outcomes."""

    prompt_domains, identity_rows = _identity_rows(records)
    if len(prompt_domains) != expected_prompts:
        raise ValueError(
            f"Phase-3 prompt count differs: {len(prompt_domains)} != {expected_prompts}"
        )
    if len(identity_rows) != expected_blocks:
        raise ValueError(
            f"Phase-3 block count differs: {len(identity_rows)} != {expected_blocks}"
        )
    if set(exclusion_prompt_sets) != REQUIRED_EXCLUSION_ROLES:
        raise ValueError("exclusion roles must be producer_train/validation/reserved")
    if set(exclusion_manifest_sha256) != REQUIRED_EXCLUSION_ROLES:
        raise ValueError("exclusion hash roles differ from required roles")

    materialized_exclusions: dict[str, set[str]] = {}
    exclusion_provenance: dict[str, Any] = {}
    phase3_prompts = set(prompt_domains)
    for role in sorted(REQUIRED_EXCLUSION_ROLES):
        digest = _require_sha256(
            f"{role} manifest hash", exclusion_manifest_sha256[role]
        )
        values = set(exclusion_prompt_sets[role])
        if any(not isinstance(value, str) or not value for value in values):
            raise ValueError(f"{role} prompt IDs must be nonempty strings")
        overlap = phase3_prompts & values
        if overlap:
            raise ValueError(
                f"Phase-3 prompts overlap {role}: {sorted(overlap)[:3]}"
            )
        materialized_exclusions[role] = values
        exclusion_provenance[role] = {
            "manifest_sha256": digest,
            "prompts": len(values),
            "prompt_set_sha256": prompt_set_sha256(values),
            "overlap": 0,
        }
    for left_index, left in enumerate(sorted(materialized_exclusions)):
        for right in sorted(materialized_exclusions)[left_index + 1 :]:
            # Exclusion collections may overlap one another; record the fact
            # without treating it as training leakage.
            exclusion_provenance[left][f"overlap_with_{right}"] = len(
                materialized_exclusions[left] & materialized_exclusions[right]
            )

    assignments = build_prompt_split(
        prompt_domains,
        canonical_metadata_sha256,
        split_counts=split_counts,
    )
    prompts = [
        {
            "sample_id": sample_id,
            "domain": prompt_domains[sample_id],
            "split": assignments[sample_id],
        }
        for sample_id in sorted(prompt_domains)
    ]
    blocks = [
        {**row, "split": assignments[row["sample_id"]]}
        for row in identity_rows
    ]
    prompt_counts = Counter(row["split"] for row in prompts)
    block_counts = Counter(row["split"] for row in blocks)
    domain_prompt_counts = Counter(
        (row["domain"], row["split"]) for row in prompts
    )
    ordered_keys = [block_key_from_mapping(row) for row in blocks]
    return {
        "format_version": ARTIFACT_FORMAT_VERSION,
        "protocol": SPLIT_MANIFEST_PROTOCOL,
        "canonical_metadata_sha256": canonical_metadata_sha256,
        "prompt_count": len(prompts),
        "block_count": len(blocks),
        "prompt_set_sha256": prompt_set_sha256(prompt_domains),
        "ordered_block_keys_sha256": ordered_block_keys_sha256(ordered_keys),
        "prompt_counts_by_split": dict(sorted(prompt_counts.items())),
        "block_counts_by_split": dict(sorted(block_counts.items())),
        "prompt_counts_by_domain_split": {
            f"{domain}:{split}": count
            for (domain, split), count in sorted(domain_prompt_counts.items())
        },
        "exclusion_provenance": exclusion_provenance,
        "provenance": dict(provenance or {}),
        "prompts": prompts,
        "blocks": blocks,
    }


def verify_phase3_split_manifest(
    manifest: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    *,
    canonical_metadata_sha256: str,
    exclusion_prompt_sets: Mapping[str, Iterable[str]],
    exclusion_manifest_sha256: Mapping[str, str],
    split_counts: Mapping[str, Mapping[str, int]] = DEFAULT_SPLIT_COUNTS,
    expected_prompts: int = PHASE3_PROMPTS,
    expected_blocks: int = PHASE3_BLOCKS,
    provenance: Mapping[str, Any] | None = None,
) -> None:
    rebuilt = build_phase3_split_manifest(
        records,
        canonical_metadata_sha256=canonical_metadata_sha256,
        exclusion_prompt_sets=exclusion_prompt_sets,
        exclusion_manifest_sha256=exclusion_manifest_sha256,
        split_counts=split_counts,
        expected_prompts=expected_prompts,
        expected_blocks=expected_blocks,
        provenance=provenance,
    )
    if dict(manifest) != rebuilt:
        differing = sorted(
            key
            for key in set(manifest) | set(rebuilt)
            if manifest.get(key) != rebuilt.get(key)
        )
        raise RuntimeError(f"split manifest differs from reconstruction: {differing}")


def split_assignment_map(manifest: Mapping[str, Any]) -> dict[str, str]:
    if manifest.get("protocol") != SPLIT_MANIFEST_PROTOCOL:
        raise ValueError("unexpected split-manifest protocol")
    prompts = manifest.get("prompts")
    if not isinstance(prompts, list) or not prompts:
        raise ValueError("split manifest has no prompt entries")
    result: dict[str, str] = {}
    for row in prompts:
        if not isinstance(row, dict):
            raise ValueError("split prompt entries must be objects")
        sample_id = row.get("sample_id")
        split = row.get("split")
        if not isinstance(sample_id, str) or split not in VALID_OUTCOME_SPLITS:
            raise ValueError("invalid prompt assignment")
        if sample_id in result:
            raise ValueError("duplicate prompt assignment")
        result[sample_id] = split
    if len(result) != int(manifest.get("prompt_count", -1)):
        raise ValueError("split prompt cardinality mismatch")
    return result


def _require_tensor(
    record: Mapping[str, Any],
    name: str,
    *,
    shape: tuple[int, ...],
    dtype: torch.dtype,
) -> Tensor:
    value = record.get(name)
    if not isinstance(value, Tensor):
        raise ValueError(f"{name} must be a tensor")
    if tuple(value.shape) != shape or value.dtype != dtype or value.device.type != "cpu":
        raise ValueError(
            f"{name} must have shape {shape}, dtype {dtype}, on CPU"
        )
    if value.is_floating_point() and not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must be finite")
    return value


def validate_outcome_record(
    record: Mapping[str, Any],
    *,
    expected_split: str,
) -> CapacityRecord:
    """Validate one self-contained Direct-outcome/feature record."""

    if expected_split not in VALID_OUTCOME_SPLITS:
        raise ValueError("unexpected outcome split")
    key = block_key_from_mapping(record)
    key.serialize()
    if record.get("split") != expected_split:
        raise ValueError("record split differs from physical artifact split")
    assert_numeric_policy_binding(
        record.get("numeric_policy_id"),
        record.get("numeric_policy_sha256"),
    )
    domain = record.get("domain")
    if not isinstance(domain, str) or not domain:
        raise ValueError("record domain must be nonempty")

    features = _require_tensor(
        record,
        "position_features",
        shape=(BLOCK_LENGTH, POSITION_FEATURE_DIMENSION),
        dtype=torch.float32,
    )
    direct_path = _require_tensor(
        record,
        "direct_path",
        shape=(BLOCK_LENGTH,),
        dtype=torch.int64,
    )
    change_mask = _require_tensor(
        record,
        "change_mask",
        shape=(BLOCK_LENGTH,),
        dtype=torch.bool,
    )
    candidate_ids = _require_tensor(
        record,
        "candidate_ids",
        shape=(BLOCK_LENGTH, CANDIDATE_COUNT),
        dtype=torch.int64,
    )
    gold_ids = _require_tensor(
        record,
        "gold_ids",
        shape=(BLOCK_LENGTH,),
        dtype=torch.int64,
    )
    candidate_logits = _require_tensor(
        record,
        "candidate_logits",
        shape=(BLOCK_LENGTH, CANDIDATE_COUNT),
        dtype=torch.float32,
    )
    base_logsumexp = _require_tensor(
        record,
        "base_logsumexp",
        shape=(BLOCK_LENGTH,),
        dtype=torch.float32,
    )
    direct_scores = _require_tensor(
        record,
        "direct_scores",
        shape=(BLOCK_LENGTH, CANDIDATE_COUNT),
        dtype=torch.float32,
    )
    residual_scores = _require_tensor(
        record,
        "direct_residual_scores",
        shape=(BLOCK_LENGTH, CANDIDATE_COUNT),
        dtype=torch.float32,
    )
    base_log_probs = _require_tensor(
        record,
        "base_log_probs",
        shape=(BLOCK_LENGTH, CANDIDATE_COUNT),
        dtype=torch.float32,
    )
    assert_portable_feature_relations(
        features,
        direct_path,
        change_mask,
        candidate_logits,
        base_logsumexp,
        direct_scores,
        residual_scores,
        base_log_probs,
    )

    outcomes = binary_outcomes_from_tokens(
        direct_path[None], candidate_ids[None], gold_ids[None]
    )
    base_length = int(outcomes.base_lengths[0])
    direct_length = int(outcomes.direct_lengths[0])
    # Persist/hash the mathematically exact Python ratio rather than a
    # device-dependent float32 rendering of the same integer outcome.
    gain = (direct_length - base_length) / float(BLOCK_LENGTH)
    if record.get("base_length") != base_length:
        raise ValueError("saved base length differs from token reconstruction")
    if record.get("direct_length") != direct_length:
        raise ValueError("saved Direct length differs from token reconstruction")
    if record.get("base_first_token_correct") is not (base_length > 0):
        raise ValueError("saved base first-token witness is inconsistent")
    if record.get("direct_first_token_correct") is not (direct_length > 0):
        raise ValueError("saved Direct first-token witness is inconsistent")
    reported_gain = record.get("normalized_gain")
    if (
        isinstance(reported_gain, bool)
        or not isinstance(reported_gain, (int, float))
        or not math.isfinite(float(reported_gain))
        or not math.isclose(float(reported_gain), gain, rel_tol=0.0, abs_tol=1e-7)
    ):
        raise ValueError("saved normalized gain is inconsistent")
    return CapacityRecord(
        block_key=key,
        normalized_gain=gain,
        direct_changed=bool(change_mask.any()),
    )


def _outcome_summary(
    records: Sequence[Mapping[str, Any]],
    *,
    split: str,
) -> dict[str, Any]:
    validated = [
        validate_outcome_record(record, expected_split=split) for record in records
    ]
    serialized_keys = [item.block_key.serialize() for item in validated]
    if len(set(serialized_keys)) != len(serialized_keys):
        raise ValueError("outcome records must have unique block identities")
    prompt_ids = [item.block_key.sample_id for item in validated]
    gains = [float(item.normalized_gain) for item in validated]
    return {
        "blocks": len(records),
        "prompts": len(set(prompt_ids)),
        "prompt_set_sha256": prompt_set_sha256(prompt_ids),
        "ordered_block_keys_sha256": ordered_block_keys_sha256(
            [item.block_key for item in validated]
        ),
        "beneficial_blocks": sum(value > 0 for value in gains),
        "harmful_blocks": sum(value < 0 for value in gains),
        "neutral_blocks": sum(value == 0 for value in gains),
        "changed_neutral_blocks": sum(
            value == 0 and item.direct_changed
            for value, item in zip(gains, validated, strict=True)
        ),
        "base_token_mass": sum(int(record["base_length"]) for record in records),
        "direct_token_mass": sum(
            int(record["direct_length"]) for record in records
        ),
    }


def write_outcome_bundle(
    output: Path,
    records: Sequence[Mapping[str, Any]],
    *,
    split: str,
    split_manifest_sha256: str,
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    """Atomically create one physically isolated fit/checkpoint/falsifier bundle."""

    if output.exists():
        raise FileExistsError(f"refusing to overwrite bundle: {output}")
    if split not in VALID_OUTCOME_SPLITS:
        raise ValueError("unexpected outcome split")
    if not records:
        raise ValueError("outcome bundle cannot be empty")
    split_manifest_sha256 = _require_sha256(
        "split manifest hash", split_manifest_sha256
    )
    ordered = sorted(
        [dict(record) for record in records],
        key=lambda record: block_key_from_mapping(record).serialize(),
    )
    summary = _outcome_summary(ordered, split=split)
    temporary = output.with_name(f".{output.name}.{uuid.uuid4().hex}.tmp")
    if temporary.exists():
        raise FileExistsError(f"temporary bundle already exists: {temporary}")
    temporary.mkdir(parents=True)
    try:
        records_path = temporary / "records.pt"
        torch.save(ordered, records_path)
        with records_path.open("rb") as handle:
            os.fsync(handle.fileno())
        metadata = {
            "format_version": OUTCOME_ARTIFACT_FORMAT_VERSION,
            "protocol": OUTCOME_ARTIFACT_PROTOCOL,
            "numeric_policy": numeric_policy_receipt(),
            "split": split,
            "split_manifest_sha256": split_manifest_sha256,
            "records_sha256": sha256_file(records_path),
            "summary": summary,
            "provenance": dict(provenance),
        }
        metadata_path = temporary / "metadata.json"
        with metadata_path.open("wb") as handle:
            handle.write(canonical_json_bytes(metadata))
            handle.flush()
            os.fsync(handle.fileno())
        output.parent.mkdir(parents=True, exist_ok=True)
        os.replace(temporary, output)
        return metadata
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def load_outcome_bundle(
    root: Path,
    *,
    expected_split: str,
    expected_metadata_sha256: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    metadata_path = root / "metadata.json"
    records_path = root / "records.pt"
    if expected_metadata_sha256 is not None:
        if sha256_file(metadata_path) != expected_metadata_sha256:
            raise RuntimeError("outcome metadata hash mismatch")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("format_version") != OUTCOME_ARTIFACT_FORMAT_VERSION:
        raise RuntimeError("unexpected outcome artifact format version")
    if metadata.get("protocol") != OUTCOME_ARTIFACT_PROTOCOL:
        raise RuntimeError("unexpected outcome artifact protocol")
    if metadata.get("split") != expected_split:
        raise RuntimeError("outcome artifact split mismatch")
    if metadata.get("numeric_policy") != numeric_policy_receipt():
        raise RuntimeError("outcome artifact numeric policy differs")
    if sha256_file(records_path) != metadata.get("records_sha256"):
        raise RuntimeError("outcome records hash mismatch")
    records = torch.load(records_path, map_location="cpu", weights_only=False)
    if not isinstance(records, list) or not records:
        raise RuntimeError("outcome records must be a nonempty list")
    summary = _outcome_summary(records, split=expected_split)
    if summary != metadata.get("summary"):
        raise RuntimeError("outcome summary differs from records")
    return records, metadata


def select_capacity_from_fit_bundle(
    fit_records: Sequence[Mapping[str, Any]],
    *,
    producer_checkpoint_sha256: str,
    producer_metrics_sha256: str,
    canonical_metadata_sha256: str,
    split_manifest_sha256: str,
) -> list[dict[str, Any]]:
    candidates = [
        validate_outcome_record(record, expected_split="fit")
        for record in fit_records
    ]
    selected = select_capacity_records(
        candidates,
        producer_checkpoint_sha256,
        producer_metrics_sha256,
        canonical_metadata_sha256,
        split_manifest_sha256,
    )
    by_key = {
        block_key_from_mapping(record): dict(record) for record in fit_records
    }
    return [by_key[item.block_key] for item in selected]


def write_capacity_bundle(
    output: Path,
    records: Sequence[Mapping[str, Any]],
    *,
    parent_fit_metadata_sha256: str,
    producer_checkpoint_sha256: str,
    producer_metrics_sha256: str,
    canonical_metadata_sha256: str,
    split_manifest_sha256: str,
    source_closure: Mapping[str, Any],
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite bundle: {output}")
    parent_fit_metadata_sha256 = _require_sha256(
        "parent fit metadata hash", parent_fit_metadata_sha256
    )
    producer_checkpoint_sha256 = _require_sha256(
        "producer checkpoint hash", producer_checkpoint_sha256
    )
    producer_metrics_sha256 = _require_sha256(
        "producer metrics hash", producer_metrics_sha256
    )
    canonical_metadata_sha256 = _require_sha256(
        "canonical metadata hash", canonical_metadata_sha256
    )
    split_manifest_sha256 = _require_sha256(
        "split manifest hash", split_manifest_sha256
    )
    source_closure = _validated_source_closure(source_closure)
    validated = [
        validate_outcome_record(record, expected_split="fit") for record in records
    ]
    if len(validated) != 512:
        raise ValueError("capacity bundle must contain exactly 512 records")
    prompts = {item.block_key.sample_id for item in validated}
    if len(prompts) != 512:
        raise ValueError("capacity bundle must be prompt unique")
    composition = Counter(
        "beneficial"
        if item.normalized_gain > 0
        else "harmful"
        if item.normalized_gain < 0
        else "changed-neutral"
        if item.direct_changed
        else "ineligible"
        for item in validated
    )
    if composition != Counter(
        {"beneficial": 256, "harmful": 128, "changed-neutral": 128}
    ):
        raise ValueError("capacity composition differs from 256/128/128")
    semantic_sha256 = capacity_records_sha256(validated)

    temporary = output.with_name(f".{output.name}.{uuid.uuid4().hex}.tmp")
    temporary.mkdir(parents=True)
    try:
        records_path = temporary / "records.pt"
        torch.save([dict(record) for record in records], records_path)
        with records_path.open("rb") as handle:
            os.fsync(handle.fileno())
        metadata = {
            "format_version": ARTIFACT_FORMAT_VERSION,
            "protocol": CAPACITY_ARTIFACT_PROTOCOL,
            "records_sha256": sha256_file(records_path),
            "semantic_selection_sha256": semantic_sha256,
            "blocks": 512,
            "prompts": 512,
            "composition": dict(sorted(composition.items())),
            "parent_fit_metadata_sha256": parent_fit_metadata_sha256,
            "producer_checkpoint_sha256": producer_checkpoint_sha256,
            "producer_metrics_sha256": producer_metrics_sha256,
            "canonical_metadata_sha256": canonical_metadata_sha256,
            "split_manifest_sha256": split_manifest_sha256,
            "source_closure": source_closure,
        }
        metadata_path = temporary / "metadata.json"
        with metadata_path.open("wb") as handle:
            handle.write(canonical_json_bytes(metadata))
            handle.flush()
            os.fsync(handle.fileno())
        output.parent.mkdir(parents=True, exist_ok=True)
        os.replace(temporary, output)
        return metadata
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def load_capacity_bundle(
    root: Path,
    *,
    expected_metadata_sha256: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    metadata_path = root / "metadata.json"
    records_path = root / "records.pt"
    if expected_metadata_sha256 is not None:
        if sha256_file(metadata_path) != expected_metadata_sha256:
            raise RuntimeError("capacity metadata hash mismatch")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("format_version") != ARTIFACT_FORMAT_VERSION:
        raise RuntimeError("unexpected capacity artifact format version")
    if metadata.get("protocol") != CAPACITY_ARTIFACT_PROTOCOL:
        raise RuntimeError("unexpected capacity artifact protocol")
    if sha256_file(records_path) != metadata.get("records_sha256"):
        raise RuntimeError("capacity records hash mismatch")
    records = torch.load(records_path, map_location="cpu", weights_only=False)
    if not isinstance(records, list):
        raise RuntimeError("capacity records must be a list")
    validated = [
        validate_outcome_record(record, expected_split="fit") for record in records
    ]
    if len(validated) != 512 or len(
        {item.block_key.sample_id for item in validated}
    ) != 512:
        raise RuntimeError("capacity cardinality/prompt uniqueness mismatch")
    if capacity_records_sha256(validated) != metadata.get(
        "semantic_selection_sha256"
    ):
        raise RuntimeError("capacity semantic selection hash mismatch")
    try:
        _validated_source_closure(metadata.get("source_closure"))
    except (TypeError, ValueError) as error:
        raise RuntimeError("capacity source closure summary is invalid") from error
    composition = Counter(
        "beneficial"
        if item.normalized_gain > 0
        else "harmful"
        if item.normalized_gain < 0
        else "changed-neutral"
        if item.direct_changed
        else "ineligible"
        for item in validated
    )
    expected_composition = Counter(
        {"beneficial": 256, "harmful": 128, "changed-neutral": 128}
    )
    if composition != expected_composition:
        raise RuntimeError("capacity composition differs from 256/128/128")
    if dict(sorted(composition.items())) != metadata.get("composition"):
        raise RuntimeError("capacity composition differs from metadata")
    return records, metadata
