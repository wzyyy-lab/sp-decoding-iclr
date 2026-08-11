#!/usr/bin/env python3
"""Independent, read-only auditor for staged PROS-Gate artifacts.

This module intentionally does not import ``direct_safety_artifacts`` or the
materializer.  It reimplements the artifact-side checks from persisted values
and uses only the frozen identity/ranking primitives from the Gate-0 protocol.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
import uuid

import torch
from torch import Tensor

from sph.data import CanonicalBlockDataset
from sph.direct_safety_protocol import (
    BlockKey,
    CapacityRecord,
    DEFAULT_SPLIT_COUNTS,
    build_prompt_split,
    capacity_records_sha256,
    ordered_block_keys_sha256,
    select_capacity_records,
)


FORMAT_VERSION = 1
OUTCOME_FORMAT_VERSION = 2
SPLIT_PROTOCOL = "pros-gate-phase3-split-manifest-v1"
OUTCOME_PROTOCOL = "pros-gate-direct-outcomes-v2"
CAPACITY_PROTOCOL = "pros-gate-capacity-artifact-v1"
SOURCE_CLOSURE_PROTOCOL = "pros-gate-first-party-source-closure-v1"
NUMERIC_POLICY_PROTOCOL = "pros-gate-cross-device-numeric-policy-spec-v2"
NUMERIC_POLICY_ID = "pros-gate-cross-device-numeric-policy-v2"
NUMERIC_POLICY_SHA256 = (
    "cbd80345e7249707931f71b29c65722ec8910263d51b7d649c5dd5c04fc4d4f0"
)
AUDIT_EPS32 = 2.0**-23
AUDIT_ADD_SUB_ULPS = 2
AUDIT_ADD_SUB_HALF_WIDTH_CAP = 2.0**-14
AUDIT_ENTROPY_ABS_ENVELOPE = 2.0**-17
AUDIT_LSE_ULPS = 8
AUDIT_LSE_ABS_FLOOR = 2.0**-20
AUDIT_RETAINED_OUTER_ULPS = 2
AUDIT_RETAINED_OUTER_FLOOR = 2.0**-20
AUDIT_MAX_LSE_SOURCE_ULP = 2.0**-16
AUDIT_MATERIAL_MUTATION = 1.0e-4
SAME_DEVICE_FEATURE_RELATION_COUNT = 15
AUDIT_NUMERIC_POLICY_SPEC: dict[str, Any] = {
    "protocol": NUMERIC_POLICY_PROTOCOL,
    "id": NUMERIC_POLICY_ID,
    "format": "IEEE-754 binary32 persisted tensors; float64 references",
    "field_classes": {
        "exact": [
            "direct_path",
            "change_mask",
            "selected_state_copy",
            "base_state_copy",
            "change_scalar",
        ],
        "add_sub": [
            "state_difference",
            "base_log_probs",
            "direct_scores",
            "total_margin",
            "residual_margin",
            "dflash_margin",
        ],
        "normalized_neighbor": ["rank", "position"],
        "entropy": ["entropy"],
        "retained_mass": ["retained_mass"],
    },
    "producer_same_device": {
        "relation_count": SAME_DEVICE_FEATURE_RELATION_COUNT,
        "rule": "bitwise exact reconstruction before host copy",
    },
    "relations": {
        "exact": {"rule": "bitwise equality"},
        "normalized_neighbor": {
            "endpoints": "0 and 1 exact",
            "interior_float32_neighbors": 1,
            "range": "[0,1]",
        },
        "add_sub": {
            "source_scale_floor": 1.0,
            "half_width_source_ulps": AUDIT_ADD_SUB_ULPS,
            "half_width_cap_hex": AUDIT_ADD_SUB_HALF_WIDTH_CAP.hex(),
        },
        "entropy": {
            "absolute_envelope_hex": AUDIT_ENTROPY_ABS_ENVELOPE.hex(),
            "range": "[0,1]",
            "normalizer": "ln(16)",
        },
        "retained_mass": {
            "lse_envelope": "8*ulp32(scale)+2^-20",
            "lse_source_ulp_cap_hex": AUDIT_MAX_LSE_SOURCE_ULP.hex(),
            "outer_float32_neighbors": AUDIT_RETAINED_OUTER_ULPS,
            "outer_floor_hex": AUDIT_RETAINED_OUTER_FLOOR.hex(),
            "analytic_cap": "E_lse/2+2^-20+4*2^-23",
            "analytic_cap_strict_upper_hex": AUDIT_MATERIAL_MUTATION.hex(),
            "subset_invariant": "logsumexp(top16)<=base_logsumexp+E_lse",
            "range": "[-1, propagated_upper]",
        },
    },
}
SOURCE_SPLIT = "train"
AUDITABLE_OUTCOME_SPLITS = frozenset({"fit", "checkpoint"})
EXCLUSION_ROLES = frozenset({"producer_train", "validation", "reserved"})
EXPECTED_PROMPTS = 1_987
EXPECTED_BLOCKS = 15_886
EXPECTED_METADATA_SHA256 = (
    "0dbca3e9667f6578b02559e2934ee79a677e45d058c68271c94db6e8f4338320"
)
EXPECTED_DIRECT_CHECKPOINT_SHA256 = (
    "9486d976c115fc4313864bb5e2b9b84ddf0145311e58f838b0925c184d6d9a0e"
)
EXPECTED_DIRECT_METRICS_SHA256 = (
    "9ec91a1faceec05c1f798482d9ebf45a8ed194f2bddfaa18228317476aad4aef"
)
PROJECT = Path(__file__).resolve().parents[1]
EXPECTED_SOURCE_DATA = (
    PROJECT / "artifacts/canonical/qwen3_4b_phase3_tier1_10035436"
).resolve()
EXPECTED_DIRECT_RUN = (
    PROJECT
    / "artifacts/training/gcls_v4_feature_100k_10133585/"
    "compact_axial_additive_d64_full_seed0"
).resolve()
EXPECTED_TARGET = Path(
    "/hpc2hdd/home/zwang668/sp decoding/hf_assets/models/Qwen3-4B"
).resolve()
SOURCE_MANIFEST = (
    PROJECT
    / "refine-logs/direct-safety-gate/R079_SOURCE_CLOSURE_NUMERIC_V2.json"
).resolve()
EXPECTED_EXCLUSION_SOURCES: dict[str, tuple[dict[str, Any], ...]] = {
    "producer_train": (
        {
            "path": str(
                (PROJECT / "artifacts/manifests/open_perfectblend_100k_v2.jsonl").resolve()
            ),
            "bytes": 92_590_866,
            "sha256": "b05087a56e8e717605415026421f7bae23092eb7cb9509361a36932f80260e3a",
            "selected_splits": ["train"],
            "row_counts_by_split": {"train": 100_000},
        },
    ),
    "validation": (
        {
            "path": str(
                (PROJECT / "artifacts/manifests/phase3_development_v3.jsonl").resolve()
            ),
            "bytes": 1_418_202,
            "sha256": "e16374068e9c8904214fbf282b4adb6187a0b099db5c37e79660fc46a2801d01",
            "selected_splits": ["validation_gate", "validation_select"],
            "row_counts_by_split": {
                "train": 2_000,
                "validation_gate": 150,
                "validation_select": 150,
            },
        },
    ),
    "reserved": (
        {
            "path": str(
                (PROJECT / "artifacts/manifests/phase3_reserved_test_v3.jsonl").resolve()
            ),
            "bytes": 429_359,
            "sha256": "ae25467fbb52b7091c8d9a5f98776b11ccf76e87e781850b5638734548a53bb4",
            "selected_splits": ["test"],
            "row_counts_by_split": {"test": 600},
        },
    ),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _audit_numeric_policy_receipt() -> dict[str, Any]:
    observed = _canonical_json_sha256(AUDIT_NUMERIC_POLICY_SPEC)
    if observed != NUMERIC_POLICY_SHA256:
        raise RuntimeError("independent numeric-policy digest differs")
    return {
        "protocol": NUMERIC_POLICY_PROTOCOL,
        "id": NUMERIC_POLICY_ID,
        "sha256": NUMERIC_POLICY_SHA256,
        "spec": json.loads(
            json.dumps(
                AUDIT_NUMERIC_POLICY_SPEC,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        ),
    }


def _prompt_set_sha256(prompt_ids: Iterable[str]) -> str:
    values = sorted(set(prompt_ids))
    if not values:
        raise RuntimeError("cannot hash an empty prompt set")
    return hashlib.sha256("\n".join(values).encode("utf-8")).hexdigest()


def _require_sha256(name: str, value: Any) -> str:
    if not isinstance(value, str) or len(value) != 64 or value != value.lower():
        raise RuntimeError(f"{name} is not a canonical lowercase SHA256")
    try:
        int(value, 16)
    except ValueError as error:
        raise RuntimeError(f"{name} is not hexadecimal") from error
    return value


def _discover_first_party_python() -> tuple[str, ...]:
    paths = [
        path.relative_to(PROJECT).as_posix()
        for path in (PROJECT / "src/sph").rglob("*.py")
        if path.is_file()
    ]
    paths.extend(
        path.relative_to(PROJECT).as_posix()
        for path in (PROJECT / "scripts").glob("*.py")
        if path.is_file()
    )
    result = tuple(sorted(paths))
    if not result or len(set(result)) != len(result):
        raise RuntimeError("audited first-party source surface is invalid")
    return result


def _verify_source_manifest(
    expected_manifest_sha256: str | None = None,
) -> dict[str, Any]:
    """Independently verify the complete reviewed first-party source closure."""

    if SOURCE_MANIFEST.is_symlink() or not SOURCE_MANIFEST.is_file():
        raise RuntimeError("reviewed source manifest is missing or is a symlink")
    manifest_sha256 = sha256_file(SOURCE_MANIFEST)
    if expected_manifest_sha256 is not None and manifest_sha256 != _require_sha256(
        "expected source manifest hash", expected_manifest_sha256
    ):
        raise RuntimeError("source manifest SHA256 differs from reviewed input")
    value = json.loads(SOURCE_MANIFEST.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError("source manifest must be a JSON object")
    if value.get("protocol") != SOURCE_CLOSURE_PROTOCOL:
        raise RuntimeError("source manifest protocol differs")
    if value.get("roots") != ["scripts/*.py", "src/sph/**/*.py"]:
        raise RuntimeError("source manifest roots differ")
    rows = value.get("files")
    if not isinstance(rows, list) or not rows:
        raise RuntimeError("source manifest has no entries")
    normalized: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != {"path", "bytes", "sha256"}:
            raise RuntimeError("source manifest entry fields differ")
        relative = row["path"]
        byte_count = row["bytes"]
        if (
            not isinstance(relative, str)
            or not relative
            or "\0" in relative
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
        ):
            raise RuntimeError("source manifest path is invalid")
        if isinstance(byte_count, bool) or not isinstance(byte_count, int) or byte_count < 1:
            raise RuntimeError("source manifest byte count is invalid")
        digest = _require_sha256("source file hash", row["sha256"])
        normalized.append(
            {"path": relative, "bytes": byte_count, "sha256": digest}
        )
    relative_paths = tuple(row["path"] for row in normalized)
    if relative_paths != tuple(sorted(relative_paths)) or len(set(relative_paths)) != len(
        relative_paths
    ):
        raise RuntimeError("source manifest paths are not sorted and unique")
    if relative_paths != _discover_first_party_python():
        raise RuntimeError("source manifest does not close the first-party surface")
    for row in normalized:
        path = PROJECT / row["path"]
        if (
            path.is_symlink()
            or not path.is_file()
            or not path.resolve().is_relative_to(PROJECT.resolve())
        ):
            raise RuntimeError(f"audited source is not a regular project file: {row['path']}")
        if path.stat().st_size != row["bytes"] or sha256_file(path) != row["sha256"]:
            raise RuntimeError(f"audited source identity differs: {row['path']}")
    return {
        "protocol": SOURCE_CLOSURE_PROTOCOL,
        "source_manifest_sha256": manifest_sha256,
        "source_file_count": len(normalized),
        "source_entries_sha256": _canonical_json_sha256(normalized),
    }


def _atomic_report(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite audit report: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(
                dict(value),
                handle,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _parse_role_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise RuntimeError("exclusions must use ROLE=JSONL")
    role, path = value.split("=", 1)
    if role not in EXCLUSION_ROLES or not path:
        raise RuntimeError(f"invalid exclusion input: {value!r}")
    return role, Path(path).resolve()


def _read_exclusions(
    values: Sequence[str],
) -> tuple[dict[str, set[str]], dict[str, str], dict[str, list[dict[str, Any]]]]:
    paths: dict[str, list[Path]] = defaultdict(list)
    for value in values:
        role, path = _parse_role_path(value)
        if path in paths[role]:
            raise RuntimeError(f"duplicate exclusion path for {role}")
        paths[role].append(path)
    if set(paths) != EXCLUSION_ROLES:
        raise RuntimeError("auditor requires all three exclusion roles")

    prompt_sets: dict[str, set[str]] = {}
    hashes: dict[str, str] = {}
    provenance: dict[str, list[dict[str, Any]]] = {}
    for role in sorted(EXCLUSION_ROLES):
        prompts: set[str] = set()
        entries: list[dict[str, Any]] = []
        frozen_entries = [dict(item) for item in EXPECTED_EXCLUSION_SOURCES[role]]
        frozen_entries.sort(key=lambda item: str(item.get("path", "")))
        for path in sorted(paths[role]):
            matching = [item for item in frozen_entries if item.get("path") == str(path)]
            if len(matching) != 1:
                raise RuntimeError(f"{role} exclusion path differs from frozen identity")
            frozen = matching[0]
            identity = {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            if any(identity[name] != frozen.get(name) for name in identity):
                raise RuntimeError(f"{role} exclusion files differ from frozen identities")
            selected = frozen.get("selected_splits")
            if (
                not isinstance(selected, list)
                or not selected
                or selected != sorted(set(selected))
                or any(not isinstance(value, str) or not value for value in selected)
            ):
                raise RuntimeError(f"{role} frozen split filter is invalid")
            local: set[str] = set()
            row_counts: Counter[str] = Counter()
            with path.open(encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    if not line.strip():
                        continue
                    try:
                        row = json.loads(line)
                        sample_id = row["sample_id"]
                        split = row["split"]
                    except (json.JSONDecodeError, KeyError, TypeError) as error:
                        raise RuntimeError(
                            f"malformed exclusion {path}:{line_number}"
                        ) from error
                    if (
                        not isinstance(sample_id, str)
                        or not sample_id
                        or "\0" in sample_id
                    ):
                        raise RuntimeError(
                            f"invalid sample ID in {path}:{line_number}"
                        )
                    if not isinstance(split, str) or not split or "\0" in split:
                        raise RuntimeError(
                            f"invalid split ID in {path}:{line_number}"
                        )
                    row_counts[split] += 1
                    if split in selected:
                        if sample_id in local:
                            raise RuntimeError(
                                f"duplicate selected prompt in {path}:{line_number}"
                            )
                        local.add(sample_id)
            if not local:
                raise RuntimeError(f"exclusion split filter selected no prompts: {path}")
            if not set(selected).issubset(row_counts):
                raise RuntimeError(f"exclusion manifest lacks a selected split: {path}")
            entry = {
                **identity,
                "selected_splits": selected,
                "row_counts_by_split": dict(sorted(row_counts.items())),
            }
            if entry != frozen:
                raise RuntimeError(f"{role} exclusion split census differs from freeze")
            entries.append(entry)
            if prompts.intersection(local):
                raise RuntimeError(f"{role} exclusion sources repeat selected prompts")
            prompts.update(local)
        if len(entries) != len(frozen_entries):
            raise RuntimeError(f"{role} exclusion source count differs from freeze")
        semantic = [
            {
                "bytes": item["bytes"],
                "sha256": item["sha256"],
                "selected_splits": item["selected_splits"],
                "row_counts_by_split": item["row_counts_by_split"],
            }
            for item in entries
        ]
        prompt_sets[role] = prompts
        hashes[role] = _canonical_json_sha256(
            {"role": role, "files": semantic}
        )
        provenance[role] = entries
    return prompt_sets, hashes, provenance


def _block_key(row: Mapping[str, Any]) -> BlockKey:
    key = BlockKey(
        row["sample_id"], row["anchor_offset"], row["context_length"]
    )
    key.serialize()
    return key


def audit_split_manifest_values(
    manifest: Mapping[str, Any],
    source_records: Sequence[Mapping[str, Any]],
    *,
    metadata_sha256: str,
    exclusion_prompt_sets: Mapping[str, set[str]],
    exclusion_hashes: Mapping[str, str],
    source_collection: Path | None = None,
    exclusion_sources: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
    source_closure: Mapping[str, Any] | None = None,
    expected_prompts: int = EXPECTED_PROMPTS,
    expected_blocks: int = EXPECTED_BLOCKS,
    split_counts: Mapping[str, Mapping[str, int]] = DEFAULT_SPLIT_COUNTS,
) -> dict[str, Any]:
    if manifest.get("format_version") != FORMAT_VERSION:
        raise RuntimeError("split manifest format version differs")
    if manifest.get("protocol") != SPLIT_PROTOCOL:
        raise RuntimeError("split manifest protocol differs")
    _require_sha256("canonical metadata hash", metadata_sha256)
    if manifest.get("canonical_metadata_sha256") != metadata_sha256:
        raise RuntimeError("split manifest canonical metadata hash differs")
    if len(source_records) != expected_blocks:
        raise RuntimeError("source block cardinality differs")

    prompt_domains: dict[str, str] = {}
    source_rows: list[dict[str, Any]] = []
    source_keys: set[BlockKey] = set()
    for record in source_records:
        if record.get("split") != SOURCE_SPLIT:
            raise RuntimeError("split audit source is not canonical train only")
        key = _block_key(record)
        if key in source_keys:
            raise RuntimeError("source contains duplicate block identities")
        source_keys.add(key)
        domain = record.get("domain")
        if not isinstance(domain, str) or not domain:
            raise RuntimeError("source domain is invalid")
        previous = prompt_domains.setdefault(key.sample_id, domain)
        if previous != domain:
            raise RuntimeError("one prompt occurs under multiple domains")
        source_rows.append(
            {
                "sample_id": key.sample_id,
                "anchor_offset": int(key.anchor_offset),
                "context_length": int(key.context_length),
                "domain": domain,
            }
        )
    if len(prompt_domains) != expected_prompts:
        raise RuntimeError("source prompt cardinality differs")
    assignments = build_prompt_split(
        prompt_domains, metadata_sha256, split_counts=split_counts
    )
    expected_prompt_rows = [
        {
            "sample_id": sample_id,
            "domain": prompt_domains[sample_id],
            "split": assignments[sample_id],
        }
        for sample_id in sorted(prompt_domains)
    ]
    source_rows.sort(key=lambda row: _block_key(row).serialize())
    expected_block_rows = [
        {**row, "split": assignments[row["sample_id"]]}
        for row in source_rows
    ]
    if manifest.get("prompts") != expected_prompt_rows:
        raise RuntimeError("split prompt assignments differ from reconstruction")
    if manifest.get("blocks") != expected_block_rows:
        raise RuntimeError("split block assignments differ from reconstruction")
    if manifest.get("prompt_count") != expected_prompts:
        raise RuntimeError("split prompt count differs")
    if manifest.get("block_count") != expected_blocks:
        raise RuntimeError("split block count differs")
    if manifest.get("prompt_set_sha256") != _prompt_set_sha256(prompt_domains):
        raise RuntimeError("split prompt-set hash differs")
    ordered_hash = ordered_block_keys_sha256(
        [_block_key(row) for row in expected_block_rows]
    )
    if manifest.get("ordered_block_keys_sha256") != ordered_hash:
        raise RuntimeError("split ordered block-key hash differs")

    prompt_counter = Counter(row["split"] for row in expected_prompt_rows)
    block_counter = Counter(row["split"] for row in expected_block_rows)
    domain_counter = Counter(
        (row["domain"], row["split"]) for row in expected_prompt_rows
    )
    if manifest.get("prompt_counts_by_split") != dict(sorted(prompt_counter.items())):
        raise RuntimeError("split prompt counts differ")
    if manifest.get("block_counts_by_split") != dict(sorted(block_counter.items())):
        raise RuntimeError("split block counts differ")
    expected_domain_counts = {
        f"{domain}:{split}": count
        for (domain, split), count in sorted(domain_counter.items())
    }
    if manifest.get("prompt_counts_by_domain_split") != expected_domain_counts:
        raise RuntimeError("split domain/prompt counts differ")

    phase3 = set(prompt_domains)
    proof = manifest.get("exclusion_provenance")
    if not isinstance(proof, dict) or set(proof) != EXCLUSION_ROLES:
        raise RuntimeError("split exclusion proof roles differ")
    for left_index, role in enumerate(sorted(EXCLUSION_ROLES)):
        values = exclusion_prompt_sets[role]
        if phase3 & values:
            raise RuntimeError(f"Phase-3 overlaps audited exclusion role {role}")
        row = proof[role]
        expected_fields = {
            "manifest_sha256": exclusion_hashes[role],
            "prompts": len(values),
            "prompt_set_sha256": _prompt_set_sha256(values),
            "overlap": 0,
        }
        for name, expected in expected_fields.items():
            if row.get(name) != expected:
                raise RuntimeError(f"split exclusion proof differs for {role}.{name}")
        for right in sorted(EXCLUSION_ROLES)[left_index + 1 :]:
            if row.get(f"overlap_with_{right}") != len(
                values & exclusion_prompt_sets[right]
            ):
                raise RuntimeError("cross-exclusion overlap count differs")

    if source_collection is not None and exclusion_sources is not None:
        if source_closure is None:
            raise RuntimeError("split audit requires the reviewed source closure")
        expected_provenance = {
            "source_collection": str(source_collection.resolve()),
            "source_split": SOURCE_SPLIT,
            "source_metadata_sha256": metadata_sha256,
            "exclusion_sources": {
                role: [dict(item) for item in entries]
                for role, entries in sorted(exclusion_sources.items())
            },
            "source_closure": dict(source_closure),
        }
        if manifest.get("provenance") != expected_provenance:
            raise RuntimeError("split source provenance differs")
    return {
        "status": "GO",
        "prompt_count": expected_prompts,
        "block_count": expected_blocks,
        "prompt_counts_by_split": dict(sorted(prompt_counter.items())),
        "block_counts_by_split": dict(sorted(block_counter.items())),
        "ordered_block_keys_sha256": ordered_hash,
        "canonical_metadata_sha256": metadata_sha256,
        "exclusion_manifest_sha256": dict(sorted(exclusion_hashes.items())),
        "exclusion_sources": {
            role: [dict(item) for item in entries]
            for role, entries in sorted((exclusion_sources or {}).items())
        },
    }


def _tensor(
    row: Mapping[str, Any],
    name: str,
    shape: tuple[int, ...],
    dtype: torch.dtype,
) -> Tensor:
    value = row.get(name)
    if (
        not isinstance(value, Tensor)
        or tuple(value.shape) != shape
        or value.dtype != dtype
        or value.device.type != "cpu"
    ):
        raise RuntimeError(f"invalid persisted tensor contract: {name}")
    if value.is_floating_point() and not bool(torch.isfinite(value).all()):
        raise RuntimeError(f"non-finite persisted tensor: {name}")
    return value


def _audit_require_float32(name: str, value: Tensor) -> None:
    if value.dtype != torch.float32 or not bool(torch.isfinite(value).all()):
        raise RuntimeError(f"independent numeric input is invalid for {name}")


def _audit_source_scale_ulp32(
    operands: Sequence[Tensor], reference64: Tensor
) -> Tensor:
    reference = reference64.to(dtype=torch.float64)
    scale = torch.ones_like(reference)
    for operand in operands:
        expanded = torch.broadcast_to(
            operand.to(dtype=torch.float64), reference.shape
        )
        scale = torch.maximum(scale, expanded.abs())
    scale = torch.maximum(scale, reference.abs())
    scale32 = scale.to(dtype=torch.float32)
    ulp = (
        torch.nextafter(scale32, torch.full_like(scale32, torch.inf)).to(
            dtype=torch.float64
        )
        - scale32.to(dtype=torch.float64)
    )
    if not bool(torch.isfinite(ulp).all()) or not bool(ulp.gt(0.0).all()):
        raise RuntimeError("independent numeric source ULP is invalid")
    return ulp


def _audit_exact(name: str, actual: Tensor, expected: Tensor) -> None:
    if (
        actual.shape != expected.shape
        or actual.device != expected.device
        or not torch.equal(actual, expected)
    ):
        raise RuntimeError(f"independent reconstruction differs for {name}")


def _audit_add_sub(
    name: str,
    actual: Tensor,
    left: Tensor,
    right: Tensor,
    *,
    operation: str,
) -> None:
    _audit_require_float32(name, actual)
    _audit_require_float32(f"{name} left", left)
    _audit_require_float32(f"{name} right", right)
    if operation == "add":
        reference = left.to(dtype=torch.float64) + right.to(dtype=torch.float64)
    elif operation == "subtract":
        reference = left.to(dtype=torch.float64) - right.to(dtype=torch.float64)
    else:
        raise RuntimeError("independent numeric operation is invalid")
    if actual.shape != reference.shape or actual.device != reference.device:
        raise RuntimeError(f"independent reconstruction differs for {name}")
    half_width = AUDIT_ADD_SUB_ULPS * _audit_source_scale_ulp32(
        (left, right), reference
    )
    if not bool(half_width.le(AUDIT_ADD_SUB_HALF_WIDTH_CAP).all()):
        raise RuntimeError(f"independent numeric cap differs for {name}")
    if not bool(
        (actual.to(dtype=torch.float64) - reference).abs().le(half_width).all()
    ):
        raise RuntimeError(f"independent reconstruction differs for {name}")


def _audit_normalized_neighbor(
    name: str,
    actual: Tensor,
    expected: Tensor,
) -> None:
    _audit_require_float32(name, actual)
    _audit_require_float32(f"{name} reference", expected)
    if actual.shape != expected.shape or actual.device != expected.device:
        raise RuntimeError(f"independent reconstruction differs for {name}")
    endpoint = expected.eq(0.0) | expected.eq(1.0)
    lower = torch.nextafter(expected, torch.full_like(expected, -torch.inf))
    upper = torch.nextafter(expected, torch.full_like(expected, torch.inf))
    accepted = actual.eq(expected) | (
        ~endpoint & (actual.eq(lower) | actual.eq(upper))
    )
    in_range = actual.ge(0.0) & actual.le(1.0)
    if not bool((accepted & in_range).all()):
        raise RuntimeError(f"independent reconstruction differs for {name}")


def _audit_entropy(name: str, actual: Tensor, logits: Tensor) -> None:
    _audit_require_float32(name, actual)
    _audit_require_float32(f"{name} logits", logits)
    logits64 = logits.to(dtype=torch.float64)
    log_q = logits64 - torch.logsumexp(logits64, dim=-1, keepdim=True)
    reference = -(log_q.exp() * log_q).sum(dim=-1) / math.log(16.0)
    if actual.shape != reference.shape or actual.device != reference.device:
        raise RuntimeError(f"independent reconstruction differs for {name}")
    actual64 = actual.to(dtype=torch.float64)
    accepted = (actual64 - reference).abs().le(AUDIT_ENTROPY_ABS_ENVELOPE)
    in_range = actual64.ge(0.0) & actual64.le(1.0)
    if not bool((accepted & in_range).all()):
        raise RuntimeError(f"independent reconstruction differs for {name}")


def _audit_retained_mass(
    name: str,
    actual: Tensor,
    logits: Tensor,
    base_logsumexp: Tensor,
) -> None:
    _audit_require_float32(name, actual)
    _audit_require_float32(f"{name} logits", logits)
    _audit_require_float32(f"{name} base logsumexp", base_logsumexp)
    logits64 = logits.to(dtype=torch.float64)
    base64 = base_logsumexp.to(dtype=torch.float64)
    lse64 = torch.logsumexp(logits64, dim=-1)
    scale = torch.maximum(torch.ones_like(lse64), logits64.amax(dim=-1).abs())
    scale = torch.maximum(scale, lse64.abs())
    scale = torch.maximum(scale, base64.abs())
    scale32 = scale.to(dtype=torch.float32)
    lse_source_ulp = (
        torch.nextafter(scale32, torch.full_like(scale32, torch.inf)).to(
            dtype=torch.float64
        )
        - scale32.to(dtype=torch.float64)
    )
    lse_envelope = AUDIT_LSE_ULPS * lse_source_ulp + AUDIT_LSE_ABS_FLOOR
    center = torch.tanh((lse64 - base64) / 2.0)
    lower = torch.tanh(((lse64 - lse_envelope) - base64) / 2.0).to(
        dtype=torch.float32
    )
    upper = torch.tanh(((lse64 + lse_envelope) - base64) / 2.0).to(
        dtype=torch.float32
    )
    for _ in range(AUDIT_RETAINED_OUTER_ULPS):
        lower = torch.nextafter(lower, torch.full_like(lower, -torch.inf))
        upper = torch.nextafter(upper, torch.full_like(upper, torch.inf))
    lower64 = lower.to(dtype=torch.float64) - AUDIT_RETAINED_OUTER_FLOOR
    upper64 = upper.to(dtype=torch.float64) + AUDIT_RETAINED_OUTER_FLOOR
    half_width = torch.maximum(
        (center - lower64).abs(), (upper64 - center).abs()
    )
    analytic_cap = (
        lse_envelope / 2.0
        + AUDIT_RETAINED_OUTER_FLOOR
        + 4.0 * AUDIT_EPS32
    )
    cap_ok = (
        lse_source_ulp.le(AUDIT_MAX_LSE_SOURCE_ULP)
        & analytic_cap.lt(AUDIT_MATERIAL_MUTATION)
        & half_width.le(analytic_cap)
    )
    subset_ok = lse64.le(base64 + lse_envelope)
    actual64 = actual.to(dtype=torch.float64)
    finite = (
        torch.isfinite(actual64)
        & torch.isfinite(center)
        & torch.isfinite(lower64)
        & torch.isfinite(upper64)
    )
    interval = actual64.ge(lower64) & actual64.le(upper64)
    in_range = actual64.ge(-1.0) & actual64.le(upper64)
    if not bool((finite & cap_ok & subset_ok & interval & in_range).all()):
        raise RuntimeError(f"independent reconstruction differs for {name}")


def _audit_portable_feature_relations(
    features: Tensor,
    path: Tensor,
    changed: Tensor,
    logits: Tensor,
    full_lse: Tensor,
    scores: Tensor,
    residual: Tensor,
    base_log_probs: Tensor,
) -> None:
    _audit_exact("Direct path", path, scores.argmax(dim=-1))
    _audit_exact("change mask", changed, path.ne(0))
    _audit_add_sub(
        "base log probabilities",
        base_log_probs,
        logits,
        full_lse[:, None],
        operation="subtract",
    )
    _audit_add_sub(
        "Direct scores", scores, base_log_probs, residual, operation="add"
    )
    _audit_add_sub(
        "state-difference feature",
        features[:, 128:192],
        features[:, :64],
        features[:, 64:128],
        operation="subtract",
    )
    gather = path[:, None]
    chosen_scores = scores.gather(1, gather).squeeze(1)
    chosen_residual = residual.gather(1, gather).squeeze(1)
    chosen_base = base_log_probs.gather(1, gather).squeeze(1)
    for name, actual, left, right in (
        ("total-margin feature", features[:, 192], chosen_scores, scores[:, 0]),
        (
            "residual-margin feature",
            features[:, 193],
            chosen_residual,
            residual[:, 0],
        ),
        (
            "DFlash-margin feature",
            features[:, 194],
            chosen_base,
            base_log_probs[:, 0],
        ),
    ):
        _audit_add_sub(name, actual, left, right, operation="subtract")
    _audit_normalized_neighbor(
        "rank feature", features[:, 195], path.float() / 15.0
    )
    _audit_normalized_neighbor(
        "position feature",
        features[:, 196],
        torch.arange(15, dtype=torch.float32) / 14.0,
    )
    _audit_exact("change scalar", features[:, 197], changed.float())
    _audit_entropy("entropy feature", features[:, 198], logits)
    _audit_retained_mass(
        "retained-mass feature", features[:, 199], logits, full_lse
    )


def _accepted_length(tokens: Tensor, gold: Tensor) -> int:
    correct = tokens.eq(gold)
    return int(correct.to(torch.int64).cumprod(dim=0).sum())


def audit_outcome_record(
    row: Mapping[str, Any], *, expected_split: str
) -> CapacityRecord:
    key = _block_key(row)
    if row.get("split") != expected_split or row.get("source_split") != SOURCE_SPLIT:
        raise RuntimeError("persisted record split boundary differs")
    if row.get("numeric_policy_id") != NUMERIC_POLICY_ID:
        raise RuntimeError("persisted record numeric policy ID differs")
    if row.get("numeric_policy_sha256") != NUMERIC_POLICY_SHA256:
        raise RuntimeError("persisted record numeric policy digest differs")
    if not isinstance(row.get("domain"), str) or not row["domain"]:
        raise RuntimeError("persisted record domain is invalid")
    features = _tensor(row, "position_features", (15, 200), torch.float32)
    path = _tensor(row, "direct_path", (15,), torch.int64)
    changed = _tensor(row, "change_mask", (15,), torch.bool)
    candidate_ids = _tensor(row, "candidate_ids", (15, 16), torch.int64)
    gold_ids = _tensor(row, "gold_ids", (15,), torch.int64)
    logits = _tensor(row, "candidate_logits", (15, 16), torch.float32)
    full_lse = _tensor(row, "base_logsumexp", (15,), torch.float32)
    scores = _tensor(row, "direct_scores", (15, 16), torch.float32)
    residual = _tensor(
        row, "direct_residual_scores", (15, 16), torch.float32
    )
    base_log_probs = _tensor(row, "base_log_probs", (15, 16), torch.float32)
    if bool(((path < 0) | (path >= 16)).any()):
        raise RuntimeError("persisted Direct path is outside [0,15]")
    _audit_portable_feature_relations(
        features,
        path,
        changed,
        logits,
        full_lse,
        scores,
        residual,
        base_log_probs,
    )

    base_tokens = candidate_ids[:, 0]
    direct_tokens = candidate_ids.gather(1, path[:, None]).squeeze(1)
    base_length = _accepted_length(base_tokens, gold_ids)
    direct_length = _accepted_length(direct_tokens, gold_ids)
    gain = (direct_length - base_length) / 15.0
    if row.get("base_length") != base_length:
        raise RuntimeError("persisted base length differs")
    if row.get("direct_length") != direct_length:
        raise RuntimeError("persisted Direct length differs")
    if row.get("base_first_token_correct") is not (base_length > 0):
        raise RuntimeError("persisted base first-token witness differs")
    if row.get("direct_first_token_correct") is not (direct_length > 0):
        raise RuntimeError("persisted Direct first-token witness differs")
    stored_gain = row.get("normalized_gain")
    if (
        isinstance(stored_gain, bool)
        or not isinstance(stored_gain, (int, float))
        or not math.isfinite(float(stored_gain))
        or not math.isclose(float(stored_gain), gain, rel_tol=0.0, abs_tol=1e-7)
    ):
        raise RuntimeError("persisted normalized gain differs")
    return CapacityRecord(key, gain, bool(changed.any()))


def _outcome_summary(
    records: Sequence[Mapping[str, Any]], *, split: str
) -> tuple[list[CapacityRecord], dict[str, Any]]:
    validated = [
        audit_outcome_record(row, expected_split=split) for row in records
    ]
    keys = [item.block_key for item in validated]
    serialized = [key.serialize() for key in keys]
    if len(set(serialized)) != len(serialized):
        raise RuntimeError("outcome artifact contains duplicate block identities")
    if serialized != sorted(serialized):
        raise RuntimeError("outcome artifact records are not in canonical order")
    gains = [item.normalized_gain for item in validated]
    prompt_ids = [item.block_key.sample_id for item in validated]
    summary = {
        "blocks": len(records),
        "prompts": len(set(prompt_ids)),
        "prompt_set_sha256": _prompt_set_sha256(prompt_ids),
        "ordered_block_keys_sha256": ordered_block_keys_sha256(keys),
        "beneficial_blocks": sum(value > 0 for value in gains),
        "harmful_blocks": sum(value < 0 for value in gains),
        "neutral_blocks": sum(value == 0 for value in gains),
        "changed_neutral_blocks": sum(
            value == 0 and item.direct_changed
            for value, item in zip(gains, validated, strict=True)
        ),
        "base_token_mass": sum(int(row["base_length"]) for row in records),
        "direct_token_mass": sum(int(row["direct_length"]) for row in records),
    }
    return validated, summary


def audit_outcome_bundle_values(
    root: Path,
    *,
    expected_split: str,
    expected_metadata_sha256: str,
    split_manifest: Mapping[str, Any],
    split_manifest_sha256: str,
    source_closure: Mapping[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    if expected_split not in AUDITABLE_OUTCOME_SPLITS:
        raise PermissionError("R079 auditor accepts only fit/checkpoint bundles")
    metadata_path = root / "metadata.json"
    records_path = root / "records.pt"
    if sha256_file(metadata_path) != _require_sha256(
        "expected outcome metadata hash", expected_metadata_sha256
    ):
        raise RuntimeError("outcome metadata hash differs from frozen input")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("format_version") != OUTCOME_FORMAT_VERSION:
        raise RuntimeError("outcome format version differs")
    if metadata.get("protocol") != OUTCOME_PROTOCOL:
        raise RuntimeError("outcome protocol differs")
    if metadata.get("split") != expected_split:
        raise RuntimeError("outcome physical split differs")
    policy_receipt = _audit_numeric_policy_receipt()
    if metadata.get("numeric_policy") != policy_receipt:
        raise RuntimeError("outcome numeric policy differs")
    if metadata.get("split_manifest_sha256") != split_manifest_sha256:
        raise RuntimeError("outcome split-manifest hash differs")
    if sha256_file(records_path) != metadata.get("records_sha256"):
        raise RuntimeError("outcome records hash differs")
    records = torch.load(records_path, map_location="cpu", weights_only=False)
    if not isinstance(records, list) or not records:
        raise RuntimeError("outcome payload is not a nonempty record list")
    validated, summary = _outcome_summary(records, split=expected_split)
    if metadata.get("summary") != summary:
        raise RuntimeError("outcome metadata summary differs")

    expected_rows = {
        _block_key(row): str(row["domain"])
        for row in split_manifest["blocks"]
        if row["split"] == expected_split
    }
    observed_rows = {
        item.block_key: str(row["domain"])
        for item, row in zip(validated, records, strict=True)
    }
    if observed_rows != expected_rows:
        raise RuntimeError("outcome identities/domains differ from split manifest")

    provenance = metadata.get("provenance")
    if not isinstance(provenance, dict):
        raise RuntimeError("outcome provenance is missing")
    expected_provenance = {
        "source_metadata_sha256": EXPECTED_METADATA_SHA256,
        "source_split": SOURCE_SPLIT,
        "direct_checkpoint_sha256": EXPECTED_DIRECT_CHECKPOINT_SHA256,
        "direct_metrics_sha256": EXPECTED_DIRECT_METRICS_SHA256,
        "target_identity_verified": True,
    }
    for name, expected in expected_provenance.items():
        if provenance.get(name) != expected:
            raise RuntimeError(f"outcome provenance differs for {name}")
    if provenance.get("numeric_policy") != policy_receipt:
        raise RuntimeError("outcome provenance numeric policy differs")
    expected_paths = {
        "source_data": str(EXPECTED_SOURCE_DATA),
        "direct_run": str(EXPECTED_DIRECT_RUN),
        "target": str(EXPECTED_TARGET),
    }
    for name, expected in expected_paths.items():
        if provenance.get(name) != expected:
            raise RuntimeError(f"outcome provenance differs for {name}")
    witness = provenance.get("native_witness")
    if not isinstance(witness, dict):
        raise RuntimeError("native Direct witness is missing")
    for name in (
        "regular_vs_hooked_outputs_bitwise",
        "hooked_repeat_outputs_bitwise",
        "hooked_repeat_node_states_bitwise",
        "state_dict_unchanged_after_native",
        "state_dict_unchanged_after_hooked",
        "state_dict_unchanged_after_repeated_hooked",
        "same_device_numeric_invariants",
    ):
        if witness.get(name) is not True:
            raise RuntimeError(f"native Direct witness failed: {name}")
    if witness.get("records") != len(records) or int(witness.get("batches", 0)) < 1:
        raise RuntimeError("native Direct witness cardinality differs")
    before_state = _require_sha256(
        "native Direct state_dict hash", witness.get("state_dict_sha256_before")
    )
    if witness.get("state_dict_sha256_after") != before_state:
        raise RuntimeError("native Direct state_dict hash changed")
    if witness.get("numeric_policy_id") != NUMERIC_POLICY_ID:
        raise RuntimeError("native witness numeric policy ID differs")
    if witness.get("numeric_policy_sha256") != NUMERIC_POLICY_SHA256:
        raise RuntimeError("native witness numeric policy digest differs")
    key_count = witness.get("state_dict_key_count")
    checks = witness.get("state_dict_checks")
    if (
        isinstance(key_count, bool)
        or not isinstance(key_count, int)
        or key_count < 1
        or isinstance(checks, bool)
        or not isinstance(checks, int)
        or checks != 3 * int(witness["batches"])
    ):
        raise RuntimeError("native Direct state_dict witness counts differ")
    numeric_batches = witness.get("same_device_numeric_batches")
    numeric_checks = witness.get("same_device_numeric_relation_checks")
    if (
        isinstance(numeric_batches, bool)
        or not isinstance(numeric_batches, int)
        or numeric_batches != int(witness["batches"])
        or isinstance(numeric_checks, bool)
        or not isinstance(numeric_checks, int)
        or numeric_checks != SAME_DEVICE_FEATURE_RELATION_COUNT * numeric_batches
    ):
        raise RuntimeError("same-device numeric witness counts differ")
    current_source_closure = dict(source_closure or _verify_source_manifest())
    if provenance.get("source_closure_start") != current_source_closure:
        raise RuntimeError("outcome starting source closure differs")
    if provenance.get("source_closure_end") != current_source_closure:
        raise RuntimeError("outcome source closure changed during materialization")
    return records, metadata, {
        "status": "GO",
        "split": expected_split,
        "metadata_sha256": expected_metadata_sha256,
        "records_sha256": metadata["records_sha256"],
        "split_manifest_sha256": split_manifest_sha256,
        "canonical_metadata_sha256": EXPECTED_METADATA_SHA256,
        "direct_checkpoint_sha256": EXPECTED_DIRECT_CHECKPOINT_SHA256,
        "direct_metrics_sha256": EXPECTED_DIRECT_METRICS_SHA256,
        "source_manifest_sha256": current_source_closure[
            "source_manifest_sha256"
        ],
        "numeric_policy_id": NUMERIC_POLICY_ID,
        "numeric_policy_sha256": NUMERIC_POLICY_SHA256,
        "same_device_numeric_batches": numeric_batches,
        "same_device_numeric_relation_checks": numeric_checks,
        **summary,
    }


def audit_capacity_bundle_values(
    root: Path,
    *,
    expected_metadata_sha256: str,
    fit_records: Sequence[Mapping[str, Any]],
    fit_metadata_sha256: str,
    producer_checkpoint_sha256: str,
    producer_metrics_sha256: str,
    canonical_metadata_sha256: str,
    split_manifest_sha256: str,
    source_closure: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    metadata_path = root / "metadata.json"
    records_path = root / "records.pt"
    if sha256_file(metadata_path) != _require_sha256(
        "expected capacity metadata hash", expected_metadata_sha256
    ):
        raise RuntimeError("capacity metadata hash differs from frozen input")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("format_version") != FORMAT_VERSION:
        raise RuntimeError("capacity format version differs")
    if metadata.get("protocol") != CAPACITY_PROTOCOL:
        raise RuntimeError("capacity protocol differs")
    if sha256_file(records_path) != metadata.get("records_sha256"):
        raise RuntimeError("capacity records hash differs")
    records = torch.load(records_path, map_location="cpu", weights_only=False)
    if not isinstance(records, list):
        raise RuntimeError("capacity payload is not a record list")
    validated = [
        audit_outcome_record(row, expected_split="fit") for row in records
    ]
    if len(validated) != 512:
        raise RuntimeError("capacity does not contain exactly 512 records")
    prompts = {item.block_key.sample_id for item in validated}
    if len(prompts) != 512:
        raise RuntimeError("capacity records are not prompt unique")
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
    exact = Counter({"beneficial": 256, "harmful": 128, "changed-neutral": 128})
    if composition != exact:
        raise RuntimeError("capacity composition differs from exact 256/128/128")
    if metadata.get("composition") != dict(sorted(exact.items())):
        raise RuntimeError("capacity metadata composition differs")
    semantic_hash = capacity_records_sha256(validated)
    if metadata.get("semantic_selection_sha256") != semantic_hash:
        raise RuntimeError("capacity semantic-selection hash differs")

    fit_candidates = [
        audit_outcome_record(row, expected_split="fit") for row in fit_records
    ]
    independently_selected = select_capacity_records(
        fit_candidates,
        producer_checkpoint_sha256,
        producer_metrics_sha256,
        canonical_metadata_sha256,
        split_manifest_sha256,
    )
    if validated != independently_selected:
        raise RuntimeError("capacity selection differs from independent fit replay")
    expected_metadata_fields = {
        "blocks": 512,
        "prompts": 512,
        "parent_fit_metadata_sha256": fit_metadata_sha256,
        "producer_checkpoint_sha256": producer_checkpoint_sha256,
        "producer_metrics_sha256": producer_metrics_sha256,
        "canonical_metadata_sha256": canonical_metadata_sha256,
        "split_manifest_sha256": split_manifest_sha256,
    }
    for name, expected in expected_metadata_fields.items():
        if metadata.get(name) != expected:
            raise RuntimeError(f"capacity metadata differs for {name}")
    current_source_closure = dict(source_closure or _verify_source_manifest())
    if metadata.get("source_closure") != current_source_closure:
        raise RuntimeError("capacity metadata names a different source closure")
    return {
        "status": "GO",
        "blocks": 512,
        "prompts": 512,
        "composition": dict(sorted(exact.items())),
        "semantic_selection_sha256": semantic_hash,
        "capacity_metadata_sha256": expected_metadata_sha256,
        "capacity_records_sha256": metadata["records_sha256"],
        "fit_metadata_sha256": fit_metadata_sha256,
        "split_manifest_sha256": split_manifest_sha256,
        "canonical_metadata_sha256": canonical_metadata_sha256,
        "direct_checkpoint_sha256": producer_checkpoint_sha256,
        "direct_metrics_sha256": producer_metrics_sha256,
        "source_manifest_sha256": current_source_closure[
            "source_manifest_sha256"
        ],
    }


def _add_exclusions(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--exclusion", action="append", required=True)


def _add_source_manifest(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--expected-source-manifest-sha256", required=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)

    split = commands.add_parser("split")
    split.add_argument("--data", type=Path, required=True)
    split.add_argument("--split-manifest", type=Path, required=True)
    split.add_argument("--expected-split-manifest-sha256", required=True)
    split.add_argument("--output", type=Path, required=True)
    _add_source_manifest(split)
    _add_exclusions(split)

    outcomes = commands.add_parser("outcomes")
    outcomes.add_argument("--split-manifest", type=Path, required=True)
    outcomes.add_argument("--expected-split-manifest-sha256", required=True)
    outcomes.add_argument("--fit-bundle", type=Path, required=True)
    outcomes.add_argument("--expected-fit-metadata-sha256", required=True)
    outcomes.add_argument("--checkpoint-bundle", type=Path, required=True)
    outcomes.add_argument("--expected-checkpoint-metadata-sha256", required=True)
    outcomes.add_argument("--output", type=Path, required=True)
    _add_source_manifest(outcomes)

    capacity = commands.add_parser("capacity")
    capacity.add_argument("--split-manifest", type=Path, required=True)
    capacity.add_argument("--expected-split-manifest-sha256", required=True)
    capacity.add_argument("--fit-bundle", type=Path, required=True)
    capacity.add_argument("--expected-fit-metadata-sha256", required=True)
    capacity.add_argument("--capacity-bundle", type=Path, required=True)
    capacity.add_argument("--expected-capacity-metadata-sha256", required=True)
    capacity.add_argument("--output", type=Path, required=True)
    _add_source_manifest(capacity)
    return parser.parse_args()


def _load_split(path: Path, expected_hash: str) -> tuple[dict[str, Any], str]:
    observed = sha256_file(path)
    if observed != _require_sha256("expected split manifest hash", expected_hash):
        raise RuntimeError("split manifest hash differs from frozen input")
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("protocol") != SPLIT_PROTOCOL:
        raise RuntimeError("unexpected split manifest protocol")
    return value, observed


def main() -> None:
    args = parse_args()
    source_closure = _verify_source_manifest(
        args.expected_source_manifest_sha256
    )
    if args.command == "split":
        manifest, manifest_hash = _load_split(
            args.split_manifest, args.expected_split_manifest_sha256
        )
        metadata_hash = sha256_file(args.data / "metadata.json")
        if metadata_hash != EXPECTED_METADATA_SHA256:
            raise RuntimeError("audited source metadata differs from freeze")
        collection = CanonicalBlockDataset(args.data, split=SOURCE_SPLIT)
        exclusions, exclusion_hashes, exclusion_sources = _read_exclusions(
            args.exclusion
        )
        report = audit_split_manifest_values(
            manifest,
            collection.records,
            metadata_sha256=metadata_hash,
            exclusion_prompt_sets=exclusions,
            exclusion_hashes=exclusion_hashes,
            source_collection=args.data,
            exclusion_sources=exclusion_sources,
            source_closure=source_closure,
        )
        report["split_manifest_sha256"] = manifest_hash
        report["source_manifest_sha256"] = source_closure[
            "source_manifest_sha256"
        ]
    elif args.command == "outcomes":
        manifest, manifest_hash = _load_split(
            args.split_manifest, args.expected_split_manifest_sha256
        )
        fit_root = args.fit_bundle.resolve()
        checkpoint_root = args.checkpoint_bundle.resolve()
        if fit_root == checkpoint_root or fit_root in checkpoint_root.parents or checkpoint_root in fit_root.parents:
            raise RuntimeError("fit/checkpoint artifacts are not physically isolated")
        _, _, fit_report = audit_outcome_bundle_values(
            fit_root,
            expected_split="fit",
            expected_metadata_sha256=args.expected_fit_metadata_sha256,
            split_manifest=manifest,
            split_manifest_sha256=manifest_hash,
            source_closure=source_closure,
        )
        _, _, checkpoint_report = audit_outcome_bundle_values(
            checkpoint_root,
            expected_split="checkpoint",
            expected_metadata_sha256=args.expected_checkpoint_metadata_sha256,
            split_manifest=manifest,
            split_manifest_sha256=manifest_hash,
            source_closure=source_closure,
        )
        if set(row["sample_id"] for row in manifest["prompts"] if row["split"] == "fit") & set(
            row["sample_id"] for row in manifest["prompts"] if row["split"] == "checkpoint"
        ):
            raise RuntimeError("fit/checkpoint prompt sets overlap")
        report = {
            "status": "GO",
            "split_manifest_sha256": manifest_hash,
            "fit_metadata_sha256": args.expected_fit_metadata_sha256,
            "checkpoint_metadata_sha256": (
                args.expected_checkpoint_metadata_sha256
            ),
            "canonical_metadata_sha256": EXPECTED_METADATA_SHA256,
            "direct_checkpoint_sha256": EXPECTED_DIRECT_CHECKPOINT_SHA256,
            "direct_metrics_sha256": EXPECTED_DIRECT_METRICS_SHA256,
            "source_manifest_sha256": source_closure[
                "source_manifest_sha256"
            ],
            "fit": fit_report,
            "checkpoint": checkpoint_report,
        }
    elif args.command == "capacity":
        manifest, manifest_hash = _load_split(
            args.split_manifest, args.expected_split_manifest_sha256
        )
        fit_records, _, _ = audit_outcome_bundle_values(
            args.fit_bundle,
            expected_split="fit",
            expected_metadata_sha256=args.expected_fit_metadata_sha256,
            split_manifest=manifest,
            split_manifest_sha256=manifest_hash,
            source_closure=source_closure,
        )
        report = audit_capacity_bundle_values(
            args.capacity_bundle,
            expected_metadata_sha256=args.expected_capacity_metadata_sha256,
            fit_records=fit_records,
            fit_metadata_sha256=args.expected_fit_metadata_sha256,
            producer_checkpoint_sha256=EXPECTED_DIRECT_CHECKPOINT_SHA256,
            producer_metrics_sha256=EXPECTED_DIRECT_METRICS_SHA256,
            canonical_metadata_sha256=EXPECTED_METADATA_SHA256,
            split_manifest_sha256=manifest_hash,
            source_closure=source_closure,
        )
    else:  # pragma: no cover
        raise AssertionError(args.command)
    if _verify_source_manifest(args.expected_source_manifest_sha256) != source_closure:
        raise RuntimeError("source closure changed during independent audit")
    _atomic_report(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
