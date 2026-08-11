#!/usr/bin/env python3
"""Freeze producer-OOS PROS-Gate split, outcome, and capacity artifacts.

The commands are deliberately staged.  ``split`` sees identities only before
any Direct outcomes are computed.  ``outcomes`` can materialize only fit or
checkpoint records; the one-shot falsifier is intentionally absent from this
R079 implementation.  ``capacity`` consumes only the already isolated fit
bundle and never opens canonical data, target weights, or a Direct checkpoint.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import platform
import time
from typing import Any, Iterable, Mapping, Sequence

import torch
from torch import Tensor, nn

from sph.data import CanonicalBlockDataset, collate_canonical_blocks
from sph.direct_safety_artifacts import (
    PHASE3_BLOCKS,
    PHASE3_PROMPTS,
    REQUIRED_EXCLUSION_ROLES,
    atomic_write_json,
    build_phase3_split_manifest,
    canonical_json_sha256,
    load_outcome_bundle,
    select_capacity_from_fit_bundle,
    sha256_file,
    split_assignment_map,
    verify_phase3_split_manifest,
    write_capacity_bundle,
    write_outcome_bundle,
)
from sph.direct_safety_gate import (
    binary_outcomes_from_tokens,
    direct_safety_position_features,
    freeze_direct_producer,
    frozen_direct_forward_with_states,
)
from sph.direct_safety_numeric_policy import (
    NUMERIC_POLICY_ID,
    NUMERIC_POLICY_SHA256,
    assert_same_device_feature_invariants,
    numeric_policy_receipt,
)
from sph.direct_safety_protocol import BlockKey, assert_stage_splits
from sph.global_direct_selector import GlobalDirectOutput
from sph.source_closure import verify_source_manifest

try:
    import evaluate_direct_one_edit as direct_evaluator
    import train_global_direct_selector as direct_training
except ModuleNotFoundError:  # Imported as ``scripts.*`` in CPU tests.
    from scripts import evaluate_direct_one_edit as direct_evaluator
    from scripts import train_global_direct_selector as direct_training


PROJECT = Path(__file__).resolve().parents[1]
EXPECTED_PHASE3_METADATA_SHA256 = (
    "0dbca3e9667f6578b02559e2934ee79a677e45d058c68271c94db6e8f4338320"
)
EXPECTED_DIRECT_CHECKPOINT_SHA256 = (
    "9486d976c115fc4313864bb5e2b9b84ddf0145311e58f838b0925c184d6d9a0e"
)
EXPECTED_DIRECT_METRICS_SHA256 = (
    "9ec91a1faceec05c1f798482d9ebf45a8ed194f2bddfaa18228317476aad4aef"
)
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
MATERIALIZABLE_OUTCOME_SPLITS = frozenset({"fit", "checkpoint"})
SOURCE_SPLIT = "train"


def _add_exclusion_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--exclusion",
        action="append",
        required=True,
        metavar="ROLE=JSONL",
        help=(
            "repeat for producer_train, validation, and reserved; a role may "
            "have multiple manifest files"
        ),
    )


def _add_source_manifest_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--expected-source-manifest-sha256", required=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)

    split = commands.add_parser("split")
    split.add_argument("--data", type=Path, required=True)
    split.add_argument("--output", type=Path, required=True)
    _add_source_manifest_argument(split)
    _add_exclusion_arguments(split)

    outcomes = commands.add_parser("outcomes")
    outcomes.add_argument("--data", type=Path, required=True)
    outcomes.add_argument("--split-manifest", type=Path, required=True)
    outcomes.add_argument("--expected-split-manifest-sha256", required=True)
    outcomes.add_argument("--split", choices=sorted(MATERIALIZABLE_OUTCOME_SPLITS), required=True)
    outcomes.add_argument("--direct-run", type=Path, required=True)
    outcomes.add_argument("--target", type=Path, required=True)
    outcomes.add_argument("--output", type=Path, required=True)
    outcomes.add_argument("--batch-size", type=int, default=32)
    _add_source_manifest_argument(outcomes)
    _add_exclusion_arguments(outcomes)

    capacity = commands.add_parser("capacity")
    capacity.add_argument("--fit-bundle", type=Path, required=True)
    capacity.add_argument("--expected-fit-metadata-sha256", required=True)
    capacity.add_argument("--output", type=Path, required=True)
    capacity.add_argument("--producer-checkpoint-sha256", required=True)
    capacity.add_argument("--producer-metrics-sha256", required=True)
    capacity.add_argument("--canonical-metadata-sha256", required=True)
    capacity.add_argument("--split-manifest-sha256", required=True)
    _add_source_manifest_argument(capacity)
    return parser.parse_args()


def _parse_role_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError("--exclusion must use ROLE=JSONL")
    role, raw_path = value.split("=", 1)
    if role not in REQUIRED_EXCLUSION_ROLES or not raw_path:
        raise ValueError(f"unexpected exclusion role/path: {value!r}")
    return role, Path(raw_path)


def read_jsonl_prompt_ids(
    path: Path,
    *,
    selected_splits: frozenset[str],
) -> tuple[set[str], dict[str, int]]:
    if not selected_splits or any(
        not isinstance(split, str) or not split for split in selected_splits
    ):
        raise RuntimeError("selected exclusion splits are invalid")
    prompt_ids: set[str] = set()
    row_counts: Counter[str] = Counter()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
                sample_id = value["sample_id"]
                split = value["split"]
            except (json.JSONDecodeError, KeyError, TypeError) as error:
                raise RuntimeError(
                    f"malformed exclusion manifest {path}:{line_number}"
                ) from error
            if not isinstance(sample_id, str) or not sample_id or "\0" in sample_id:
                raise RuntimeError(
                    f"invalid prompt identity in {path}:{line_number}"
                )
            if not isinstance(split, str) or not split or "\0" in split:
                raise RuntimeError(
                    f"invalid split identity in {path}:{line_number}"
                )
            row_counts[split] += 1
            if split in selected_splits:
                if sample_id in prompt_ids:
                    raise RuntimeError(
                        f"duplicate selected prompt in {path}:{line_number}"
                    )
                prompt_ids.add(sample_id)
    if not prompt_ids:
        raise RuntimeError(f"exclusion split filter selected no prompts: {path}")
    if not selected_splits.issubset(row_counts):
        raise RuntimeError(f"exclusion manifest lacks a selected split: {path}")
    return prompt_ids, dict(sorted(row_counts.items()))


def load_exclusions(
    values: Sequence[str],
    *,
    expected_sources: Mapping[str, Sequence[Mapping[str, Any]]] = (
        EXPECTED_EXCLUSION_SOURCES
    ),
) -> tuple[dict[str, set[str]], dict[str, str], dict[str, list[dict[str, Any]]]]:
    paths_by_role: dict[str, list[Path]] = defaultdict(list)
    for value in values:
        role, path = _parse_role_path(value)
        resolved = path.resolve()
        if resolved in paths_by_role[role]:
            raise ValueError(f"duplicate exclusion path for {role}: {resolved}")
        paths_by_role[role].append(resolved)
    if set(paths_by_role) != REQUIRED_EXCLUSION_ROLES:
        raise ValueError("exclusions must cover producer_train/validation/reserved")
    if set(expected_sources) != REQUIRED_EXCLUSION_ROLES:
        raise ValueError("expected exclusion identities have invalid roles")

    prompt_sets: dict[str, set[str]] = {}
    aggregate_hashes: dict[str, str] = {}
    provenance: dict[str, list[dict[str, Any]]] = {}
    for role in sorted(REQUIRED_EXCLUSION_ROLES):
        prompts: set[str] = set()
        source_entries: list[dict[str, Any]] = []
        frozen_entries = [dict(item) for item in expected_sources[role]]
        frozen_entries.sort(key=lambda item: str(item.get("path", "")))
        for path in sorted(paths_by_role[role]):
            matches = [item for item in frozen_entries if item.get("path") == str(path)]
            if len(matches) != 1:
                raise RuntimeError(f"{role} exclusion path differs from frozen identity")
            frozen = matches[0]
            identity = {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            if any(identity[name] != frozen.get(name) for name in identity):
                raise RuntimeError(f"{role} exclusion files differ from frozen identities")
            raw_selected = frozen.get("selected_splits")
            if (
                not isinstance(raw_selected, list)
                or not raw_selected
                or raw_selected != sorted(set(raw_selected))
            ):
                raise RuntimeError(f"{role} frozen split filter is invalid")
            local, row_counts = read_jsonl_prompt_ids(
                path, selected_splits=frozenset(raw_selected)
            )
            entry = {
                **identity,
                "selected_splits": raw_selected,
                "row_counts_by_split": row_counts,
            }
            if entry != frozen:
                raise RuntimeError(f"{role} exclusion split census differs from freeze")
            if prompts.intersection(local):
                raise RuntimeError(f"{role} exclusion sources repeat selected prompts")
            prompts.update(local)
            source_entries.append(entry)
        if len(source_entries) != len(frozen_entries):
            raise RuntimeError(f"{role} exclusion source count differs from freeze")
        semantic_entries = [
            {
                "bytes": item["bytes"],
                "sha256": item["sha256"],
                "selected_splits": item["selected_splits"],
                "row_counts_by_split": item["row_counts_by_split"],
            }
            for item in source_entries
        ]
        prompt_sets[role] = prompts
        aggregate_hashes[role] = canonical_json_sha256(
            {"role": role, "files": semantic_entries}
        )
        provenance[role] = source_entries
    return prompt_sets, aggregate_hashes, provenance


def _identity_records(records: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for record in records:
        if record.get("split") != SOURCE_SPLIT:
            raise RuntimeError("Phase-3 identity source must be canonical train only")
        result.append(
            {
                "sample_id": record["sample_id"],
                "anchor_offset": record["anchor_offset"],
                "context_length": record["context_length"],
                "domain": record["domain"],
            }
        )
    return result


def load_phase3_source(data: Path) -> tuple[CanonicalBlockDataset, str]:
    metadata_path = data / "metadata.json"
    metadata_sha256 = sha256_file(metadata_path)
    if metadata_sha256 != EXPECTED_PHASE3_METADATA_SHA256:
        raise RuntimeError("Phase-3 metadata differs from the frozen collection")
    collection = CanonicalBlockDataset(data, split=SOURCE_SPLIT)
    return collection, metadata_sha256


def split_provenance(
    *,
    data: Path,
    metadata_sha256: str,
    exclusion_sources: Mapping[str, Sequence[Mapping[str, Any]]],
    source_closure: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "source_collection": str(data.resolve()),
        "source_split": SOURCE_SPLIT,
        "source_metadata_sha256": metadata_sha256,
        "exclusion_sources": {
            role: [dict(item) for item in entries]
            for role, entries in sorted(exclusion_sources.items())
        },
        "source_closure": dict(source_closure),
    }


def create_split_manifest(args: argparse.Namespace) -> dict[str, Any]:
    closure_start = verify_source_manifest(
        PROJECT,
        expected_manifest_sha256=args.expected_source_manifest_sha256,
    )
    collection, metadata_sha256 = load_phase3_source(args.data)
    prompt_sets, exclusion_hashes, exclusion_sources = load_exclusions(
        args.exclusion
    )
    provenance = split_provenance(
        data=args.data,
        metadata_sha256=metadata_sha256,
        exclusion_sources=exclusion_sources,
        source_closure=closure_start.summary(),
    )
    manifest = build_phase3_split_manifest(
        _identity_records(collection.records),
        canonical_metadata_sha256=metadata_sha256,
        exclusion_prompt_sets=prompt_sets,
        exclusion_manifest_sha256=exclusion_hashes,
        expected_prompts=PHASE3_PROMPTS,
        expected_blocks=PHASE3_BLOCKS,
        provenance=provenance,
    )
    closure_end = verify_source_manifest(
        PROJECT,
        expected_manifest_sha256=args.expected_source_manifest_sha256,
    )
    if closure_end != closure_start:
        raise RuntimeError("source closure changed during split materialization")
    atomic_write_json(args.output, manifest)
    return manifest


def _assert_output_equal(
    first: GlobalDirectOutput,
    second: GlobalDirectOutput,
    *,
    label: str,
) -> None:
    for field in ("scores", "log_probs", "residual_scores", "base_log_probs"):
        if not torch.equal(getattr(first, field), getattr(second, field)):
            raise RuntimeError(f"Direct {field} changed under {label}")


def module_state_snapshot(module: nn.Module) -> dict[str, Tensor]:
    state = module.state_dict()
    if not state:
        raise RuntimeError("Direct producer state_dict is empty")
    snapshot: dict[str, Tensor] = {}
    for name, value in state.items():
        if not isinstance(name, str) or not name or not isinstance(value, Tensor):
            raise RuntimeError("Direct producer state_dict has an invalid entry")
        snapshot[name] = value.detach().to(device="cpu").contiguous().clone()
    return snapshot


def state_snapshot_sha256(snapshot: Mapping[str, Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(snapshot):
        value = snapshot[name]
        encoded_name = name.encode("utf-8")
        metadata = json.dumps(
            {"dtype": str(value.dtype), "shape": list(value.shape)},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
        raw = value.reshape(-1).view(torch.uint8).numpy().tobytes()
        for payload in (encoded_name, metadata, raw):
            digest.update(len(payload).to_bytes(8, "big"))
            digest.update(payload)
    return digest.hexdigest()


def assert_module_state_unchanged(
    module: nn.Module,
    reference: Mapping[str, Tensor],
    *,
    label: str,
) -> None:
    observed = module_state_snapshot(module)
    if observed.keys() != reference.keys():
        raise RuntimeError(f"Direct state_dict keys changed after {label}")
    for name in reference:
        if not torch.equal(observed[name], reference[name]):
            raise RuntimeError(
                f"Direct state_dict tensor changed after {label}: {name}"
            )


def _cpu_clone(value: Tensor, *, dtype: torch.dtype) -> Tensor:
    return value.detach().to(device="cpu", dtype=dtype).contiguous().clone()


def _materialize_outcome_records_for_exact_split(
    source_records: Sequence[Mapping[str, Any]],
    assignments: Mapping[str, str],
    *,
    split: str,
    producer: nn.Module,
    target_embedding: Tensor,
    device: torch.device,
    batch_size: int,
    candidate_k: int = 16,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Core reviewed outcome computation for one already-authorized split."""

    if split not in {"fit", "checkpoint", "falsifier"}:
        raise PermissionError("outcome materialization requires an exact gate split")
    assert_stage_splits(split, {split})
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    selected = [
        record
        for record in source_records
        if assignments.get(str(record.get("sample_id"))) == split
    ]
    if not selected:
        raise RuntimeError(f"no source records were assigned to {split}")
    if any(record.get("split") != SOURCE_SPLIT for record in selected):
        raise RuntimeError("outcome source includes a non-train canonical record")
    expected_prompts = {
        sample_id for sample_id, assigned in assignments.items() if assigned == split
    }
    observed_prompts = {str(record["sample_id"]) for record in selected}
    if observed_prompts != expected_prompts:
        raise RuntimeError("outcome source prompt identities differ from split manifest")

    freeze_direct_producer(producer)
    reference_state = module_state_snapshot(producer)
    reference_state_sha256 = state_snapshot_sha256(reference_state)
    target_embedding = target_embedding.detach().to(device)
    target_embedding.requires_grad_(False)
    output_records: list[dict[str, Any]] = []
    batches = 0
    same_device_relation_checks = 0
    with torch.inference_mode():
        for start in range(0, len(selected), batch_size):
            source_batch = selected[start : start + batch_size]
            batch = collate_canonical_blocks(list(source_batch), candidate_k)
            tensors = {
                name: value.to(device)
                for name, value in batch.items()
                if isinstance(value, Tensor)
            }
            candidate_ids = tensors["candidate_ids"]
            candidate_logits = tensors["candidate_logits"]
            base_logsumexp = tensors["base_logsumexp"]
            inputs = (
                tensors["hidden"],
                target_embedding[candidate_ids],
                candidate_logits,
                base_logsumexp,
                target_embedding[tensors["anchor_ids"]],
            )
            native = producer(*inputs)
            assert_module_state_unchanged(
                producer, reference_state, label="native forward"
            )
            hooked, node_states = frozen_direct_forward_with_states(
                producer, *inputs
            )
            assert_module_state_unchanged(
                producer, reference_state, label="hooked forward"
            )
            repeated, repeated_states = frozen_direct_forward_with_states(
                producer, *inputs
            )
            assert_module_state_unchanged(
                producer, reference_state, label="repeated hooked forward"
            )
            _assert_output_equal(native, hooked, label="state-capture hook")
            _assert_output_equal(hooked, repeated, label="repeat forward")
            if not torch.equal(node_states, repeated_states):
                raise RuntimeError("captured Direct node states are not bitwise stable")
            features = direct_safety_position_features(
                node_states, hooked, candidate_logits, base_logsumexp
            )
            same_device_relation_checks += assert_same_device_feature_invariants(
                features.position_features,
                features.direct_path,
                features.change_mask,
                node_states,
                candidate_logits,
                base_logsumexp,
                hooked.scores,
                hooked.residual_scores,
                hooked.base_log_probs,
            )
            outcomes = binary_outcomes_from_tokens(
                features.direct_path,
                candidate_ids,
                tensors["gold_ids"],
            )
            for local_index, source in enumerate(source_batch):
                base_length = int(outcomes.base_lengths[local_index])
                direct_length = int(outcomes.direct_lengths[local_index])
                output_records.append(
                    {
                        "sample_id": str(source["sample_id"]),
                        "anchor_offset": int(source["anchor_offset"]),
                        "context_length": int(source["context_length"]),
                        "domain": str(source["domain"]),
                        "source_split": SOURCE_SPLIT,
                        "split": split,
                        "numeric_policy_id": NUMERIC_POLICY_ID,
                        "numeric_policy_sha256": NUMERIC_POLICY_SHA256,
                        "position_features": _cpu_clone(
                            features.position_features[local_index],
                            dtype=torch.float32,
                        ),
                        "direct_path": _cpu_clone(
                            features.direct_path[local_index], dtype=torch.int64
                        ),
                        "change_mask": _cpu_clone(
                            features.change_mask[local_index], dtype=torch.bool
                        ),
                        "candidate_ids": _cpu_clone(
                            candidate_ids[local_index], dtype=torch.int64
                        ),
                        "gold_ids": _cpu_clone(
                            tensors["gold_ids"][local_index], dtype=torch.int64
                        ),
                        "candidate_logits": _cpu_clone(
                            candidate_logits[local_index], dtype=torch.float32
                        ),
                        "base_logsumexp": _cpu_clone(
                            base_logsumexp[local_index], dtype=torch.float32
                        ),
                        "direct_scores": _cpu_clone(
                            hooked.scores[local_index], dtype=torch.float32
                        ),
                        "direct_residual_scores": _cpu_clone(
                            hooked.residual_scores[local_index], dtype=torch.float32
                        ),
                        "base_log_probs": _cpu_clone(
                            hooked.base_log_probs[local_index], dtype=torch.float32
                        ),
                        "base_length": base_length,
                        "direct_length": direct_length,
                        "base_first_token_correct": base_length > 0,
                        "direct_first_token_correct": direct_length > 0,
                        "normalized_gain": (
                            direct_length - base_length
                        ) / 15.0,
                    }
                )
            batches += 1
    final_state = module_state_snapshot(producer)
    final_state_sha256 = state_snapshot_sha256(final_state)
    if final_state_sha256 != reference_state_sha256:
        raise RuntimeError("Direct state_dict semantic hash changed")
    return output_records, {
        "batches": batches,
        "records": len(output_records),
        "regular_vs_hooked_outputs_bitwise": True,
        "hooked_repeat_outputs_bitwise": True,
        "hooked_repeat_node_states_bitwise": True,
        "state_dict_key_count": len(reference_state),
        "state_dict_checks": batches * 3,
        "state_dict_sha256_before": reference_state_sha256,
        "state_dict_sha256_after": final_state_sha256,
        "state_dict_unchanged_after_native": True,
        "state_dict_unchanged_after_hooked": True,
        "state_dict_unchanged_after_repeated_hooked": True,
        "numeric_policy_id": NUMERIC_POLICY_ID,
        "numeric_policy_sha256": NUMERIC_POLICY_SHA256,
        "same_device_numeric_invariants": True,
        "same_device_numeric_batches": batches,
        "same_device_numeric_relation_checks": same_device_relation_checks,
    }


def materialize_outcome_records(
    source_records: Sequence[Mapping[str, Any]],
    assignments: Mapping[str, str],
    *,
    split: str,
    producer: nn.Module,
    target_embedding: Tensor,
    device: torch.device,
    batch_size: int,
    candidate_k: int = 16,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """R079 surface: compute only physically isolated fit/checkpoint outcomes."""

    if split not in MATERIALIZABLE_OUTCOME_SPLITS:
        raise PermissionError("R079 can materialize only fit/checkpoint outcomes")
    return _materialize_outcome_records_for_exact_split(
        source_records,
        assignments,
        split=split,
        producer=producer,
        target_embedding=target_embedding,
        device=device,
        batch_size=batch_size,
        candidate_k=candidate_k,
    )


def materialize_falsifier_outcome_records(
    source_records: Sequence[Mapping[str, Any]],
    assignments: Mapping[str, str],
    *,
    split: str,
    producer: nn.Module,
    target_embedding: Tensor,
    device: torch.device,
    batch_size: int,
    candidate_k: int = 16,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """R083-only surface: compute exactly the one reviewed falsifier split."""

    if split != "falsifier":
        raise PermissionError("R083 can materialize only falsifier outcomes")
    return _materialize_outcome_records_for_exact_split(
        source_records,
        assignments,
        split=split,
        producer=producer,
        target_embedding=target_embedding,
        device=device,
        batch_size=batch_size,
        candidate_k=candidate_k,
    )


def load_frozen_direct(
    direct_run: Path,
    target: Path,
    data_metadata: Mapping[str, Any],
    *,
    device: torch.device,
) -> tuple[nn.Module, Tensor, dict[str, Any]]:
    metrics_path = direct_run / "metrics.json"
    checkpoint_path = direct_run / "best.pt"
    if sha256_file(metrics_path) != EXPECTED_DIRECT_METRICS_SHA256:
        raise RuntimeError("Direct metrics differ from frozen job10133585")
    if sha256_file(checkpoint_path) != EXPECTED_DIRECT_CHECKPOINT_SHA256:
        raise RuntimeError("Direct checkpoint differs from frozen job10133585")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = direct_evaluator.validate_direct_checkpoint_contract(
        metrics, checkpoint, direct_run=direct_run
    )
    direct_training.validate_target_embedding_identity(dict(data_metadata), target)
    target_embedding = (
        direct_training.load_target_embedding(target)
        .to(device=device, dtype=torch.bfloat16)
        .detach()
    )
    block_length = int(data_metadata.get("draft_positions", 15))
    producer = direct_evaluator.build_direct_model(
        config,
        hidden_size=int(target_embedding.shape[1]),
        block_length=block_length,
    )
    producer.load_state_dict(checkpoint["model"], strict=True)
    if sum(value.numel() for value in producer.parameters()) != int(
        checkpoint["parameter_count"]
    ):
        raise RuntimeError("reconstructed Direct parameter count differs")
    producer = freeze_direct_producer(producer.to(device))
    return producer, target_embedding, config


def materialize_outcomes(args: argparse.Namespace) -> dict[str, Any]:
    closure_start = verify_source_manifest(
        PROJECT,
        expected_manifest_sha256=args.expected_source_manifest_sha256,
    )
    if not torch.cuda.is_available():
        raise RuntimeError("real Direct outcome materialization requires CUDA")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    assert_stage_splits(args.split, {args.split})
    observed_split_hash = sha256_file(args.split_manifest)
    if observed_split_hash != args.expected_split_manifest_sha256:
        raise RuntimeError("split manifest hash differs from frozen CLI input")
    split_manifest = json.loads(args.split_manifest.read_text(encoding="utf-8"))
    collection, metadata_sha256 = load_phase3_source(args.data)
    prompt_sets, exclusion_hashes, exclusion_sources = load_exclusions(
        args.exclusion
    )
    provenance = split_provenance(
        data=args.data,
        metadata_sha256=metadata_sha256,
        exclusion_sources=exclusion_sources,
        source_closure=closure_start.summary(),
    )
    identity_records = _identity_records(collection.records)
    verify_phase3_split_manifest(
        split_manifest,
        identity_records,
        canonical_metadata_sha256=metadata_sha256,
        exclusion_prompt_sets=prompt_sets,
        exclusion_manifest_sha256=exclusion_hashes,
        expected_prompts=PHASE3_PROMPTS,
        expected_blocks=PHASE3_BLOCKS,
        provenance=provenance,
    )
    assignments = split_assignment_map(split_manifest)

    start = time.perf_counter()
    device = torch.device("cuda:0")
    producer, target_embedding, config = load_frozen_direct(
        args.direct_run,
        args.target,
        collection.metadata,
        device=device,
    )
    records, native_witness = materialize_outcome_records(
        collection.records,
        assignments,
        split=args.split,
        producer=producer,
        target_embedding=target_embedding,
        device=device,
        batch_size=args.batch_size,
        candidate_k=int(config["candidate_k"]),
    )
    expected_block_keys = {
        BlockKey(
            str(row["sample_id"]),
            int(row["anchor_offset"]),
            int(row["context_length"]),
        )
        for row in split_manifest["blocks"]
        if row["split"] == args.split
    }
    observed_block_keys = {
        BlockKey(
            str(row["sample_id"]),
            int(row["anchor_offset"]),
            int(row["context_length"]),
        )
        for row in records
    }
    if observed_block_keys != expected_block_keys:
        raise RuntimeError("outcome block identities differ from split manifest")
    closure_end = verify_source_manifest(
        PROJECT,
        expected_manifest_sha256=args.expected_source_manifest_sha256,
    )
    if closure_end != closure_start:
        raise RuntimeError("source closure changed during outcome materialization")
    return write_outcome_bundle(
        args.output,
        records,
        split=args.split,
        split_manifest_sha256=observed_split_hash,
        provenance={
            "job_id": os.environ.get("SLURM_JOB_ID"),
            "hostname": platform.node(),
            "device": torch.cuda.get_device_name(device),
            "seconds": time.perf_counter() - start,
            "source_data": str(args.data.resolve()),
            "source_metadata_sha256": metadata_sha256,
            "source_split": SOURCE_SPLIT,
            "direct_run": str(args.direct_run.resolve()),
            "direct_checkpoint_sha256": EXPECTED_DIRECT_CHECKPOINT_SHA256,
            "direct_metrics_sha256": EXPECTED_DIRECT_METRICS_SHA256,
            "target": str(args.target.resolve()),
            "target_identity_verified": True,
            "native_witness": native_witness,
            "numeric_policy": numeric_policy_receipt(),
            "source_closure_start": closure_start.summary(),
            "source_closure_end": closure_end.summary(),
        },
    )


def materialize_capacity(args: argparse.Namespace) -> dict[str, Any]:
    closure_start = verify_source_manifest(
        PROJECT,
        expected_manifest_sha256=args.expected_source_manifest_sha256,
    )
    assert_stage_splits("capacity", {"fit"})
    if args.producer_checkpoint_sha256 != EXPECTED_DIRECT_CHECKPOINT_SHA256:
        raise RuntimeError("capacity producer checkpoint hash differs from freeze")
    if args.producer_metrics_sha256 != EXPECTED_DIRECT_METRICS_SHA256:
        raise RuntimeError("capacity producer metrics hash differs from freeze")
    if args.canonical_metadata_sha256 != EXPECTED_PHASE3_METADATA_SHA256:
        raise RuntimeError("capacity canonical metadata hash differs from freeze")
    fit_records, fit_metadata = load_outcome_bundle(
        args.fit_bundle,
        expected_split="fit",
        expected_metadata_sha256=args.expected_fit_metadata_sha256,
    )
    if fit_metadata.get("split_manifest_sha256") != args.split_manifest_sha256:
        raise RuntimeError("fit artifact names a different split manifest")
    selected = select_capacity_from_fit_bundle(
        fit_records,
        producer_checkpoint_sha256=args.producer_checkpoint_sha256,
        producer_metrics_sha256=args.producer_metrics_sha256,
        canonical_metadata_sha256=args.canonical_metadata_sha256,
        split_manifest_sha256=args.split_manifest_sha256,
    )
    closure_end = verify_source_manifest(
        PROJECT,
        expected_manifest_sha256=args.expected_source_manifest_sha256,
    )
    if closure_end != closure_start:
        raise RuntimeError("source closure changed during capacity materialization")
    return write_capacity_bundle(
        args.output,
        selected,
        parent_fit_metadata_sha256=args.expected_fit_metadata_sha256,
        producer_checkpoint_sha256=args.producer_checkpoint_sha256,
        producer_metrics_sha256=args.producer_metrics_sha256,
        canonical_metadata_sha256=args.canonical_metadata_sha256,
        split_manifest_sha256=args.split_manifest_sha256,
        source_closure=closure_start.summary(),
    )


def main() -> None:
    args = parse_args()
    if args.command == "split":
        result = create_split_manifest(args)
        summary = {
            "output": str(args.output),
            "prompts": result["prompt_count"],
            "blocks": result["block_count"],
            "prompt_counts_by_split": result["prompt_counts_by_split"],
        }
    elif args.command == "outcomes":
        result = materialize_outcomes(args)
        summary = {
            "output": str(args.output),
            "split": result["split"],
            "summary": result["summary"],
        }
    elif args.command == "capacity":
        result = materialize_capacity(args)
        summary = {
            "output": str(args.output),
            "blocks": result["blocks"],
            "composition": result["composition"],
        }
    else:  # pragma: no cover - argparse prevents this branch.
        raise AssertionError(args.command)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
