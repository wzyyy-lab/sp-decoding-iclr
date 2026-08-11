from __future__ import annotations

import json

import pytest
import torch

from scripts.materialize_direct_safety_artifacts import (
    MATERIALIZABLE_OUTCOME_SPLITS,
    load_exclusions,
    materialize_falsifier_outcome_records,
    materialize_outcome_records,
)
from sph.direct_safety_artifacts import sha256_file, validate_outcome_record
from sph.direct_safety_numeric_policy import (
    NUMERIC_POLICY_ID,
    NUMERIC_POLICY_SHA256,
)
from sph.global_direct_selector import GlobalDirectCandidateSelector


def _write_jsonl(path, sample_ids: list[str], *, split: str) -> None:
    path.write_text(
        "".join(
            json.dumps({"sample_id": sample_id, "split": split}) + "\n"
            for sample_id in sample_ids
        ),
        encoding="utf-8",
    )


def test_exclusion_loader_requires_all_roles_and_hashes_every_source(tmp_path) -> None:
    producer = tmp_path / "producer.jsonl"
    validation = tmp_path / "validation.jsonl"
    reserved = tmp_path / "reserved.jsonl"
    _write_jsonl(producer, ["producer:a", "producer:b"], split="train")
    _write_jsonl(validation, ["validation:a"], split="validation_gate")
    _write_jsonl(reserved, ["reserved:a"], split="test")
    values = [
        f"producer_train={producer}",
        f"validation={validation}",
        f"reserved={reserved}",
    ]
    expected = {}
    for role, path, selected_splits, split_counts in (
        ("producer_train", producer, ["train"], {"train": 2}),
        (
            "validation",
            validation,
            ["validation_gate"],
            {"validation_gate": 1},
        ),
        ("reserved", reserved, ["test"], {"test": 1}),
    ):
        expected[role] = (
            {
                "path": str(path.resolve()),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "selected_splits": selected_splits,
                "row_counts_by_split": split_counts,
            },
        )
    prompt_sets, hashes, provenance = load_exclusions(
        values, expected_sources=expected
    )
    assert prompt_sets["producer_train"] == {"producer:a", "producer:b"}
    assert set(hashes) == {"producer_train", "validation", "reserved"}
    assert all(len(value) == 64 for value in hashes.values())
    assert provenance["reserved"][0]["path"] == str(reserved.resolve())
    assert provenance["reserved"][0]["bytes"] == reserved.stat().st_size

    with pytest.raises(ValueError, match="must cover"):
        load_exclusions(values[:-1], expected_sources=expected)
    with pytest.raises(ValueError, match="duplicate exclusion path"):
        load_exclusions(
            [*values, f"reserved={reserved}"], expected_sources=expected
        )
    with pytest.raises(RuntimeError, match="frozen identit"):
        load_exclusions(values)


def test_exclusion_loader_filters_combined_manifest_by_frozen_row_split(
    tmp_path,
) -> None:
    producer = tmp_path / "producer.jsonl"
    validation = tmp_path / "validation.jsonl"
    reserved = tmp_path / "reserved.jsonl"
    _write_jsonl(producer, ["producer:a"], split="train")
    validation.write_text(
        "".join(
            json.dumps({"sample_id": sample_id, "split": split}) + "\n"
            for sample_id, split in (
                ("phase3-train:a", "train"),
                ("heldout:gate", "validation_gate"),
                ("heldout:select", "validation_select"),
            )
        ),
        encoding="utf-8",
    )
    _write_jsonl(reserved, ["reserved:a"], split="test")
    paths = {
        "producer_train": producer,
        "validation": validation,
        "reserved": reserved,
    }
    filters = {
        "producer_train": ["train"],
        "validation": ["validation_gate", "validation_select"],
        "reserved": ["test"],
    }
    census = {
        "producer_train": {"train": 1},
        "validation": {
            "train": 1,
            "validation_gate": 1,
            "validation_select": 1,
        },
        "reserved": {"test": 1},
    }
    expected = {
        role: (
            {
                "path": str(path.resolve()),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "selected_splits": filters[role],
                "row_counts_by_split": census[role],
            },
        )
        for role, path in paths.items()
    }
    values = [f"{role}={path}" for role, path in paths.items()]
    prompt_sets, _, provenance = load_exclusions(
        values, expected_sources=expected
    )
    assert prompt_sets["validation"] == {"heldout:gate", "heldout:select"}
    assert "phase3-train:a" not in prompt_sets["validation"]
    assert provenance["validation"][0]["row_counts_by_split"] == census["validation"]

    tampered = json.loads(json.dumps(expected))
    tampered["validation"][0]["row_counts_by_split"]["train"] = 2
    with pytest.raises(RuntimeError, match="split census"):
        load_exclusions(values, expected_sources=tampered)


def _canonical_record(sample_id: str, *, anchor_offset: int) -> dict[str, object]:
    generator = torch.Generator().manual_seed(20260805 + anchor_offset)
    logits = -torch.arange(16, dtype=torch.float32)[None].expand(15, -1).clone()
    candidate_ids = (
        10
        + torch.arange(15, dtype=torch.int64)[:, None] * 16
        + torch.arange(16, dtype=torch.int64)[None]
    )
    return {
        "sample_id": sample_id,
        "domain": "synthetic",
        "split": "train",
        "anchor_offset": anchor_offset,
        "context_length": 64 + anchor_offset,
        "parallel_hidden": torch.randn(15, 32, generator=generator).to(
            torch.bfloat16
        ),
        "anchor_token_id": 3,
        "base_topk_ids": candidate_ids.to(torch.int32),
        "base_greedy_ids": candidate_ids[:, 0].to(torch.int32),
        "base_topk_logits": logits.to(torch.float16),
        "base_logsumexp": torch.logsumexp(logits, dim=-1) + 0.5,
        "gold_ids": candidate_ids[:, 0].to(torch.int32),
    }


def test_cpu_core_materializes_only_requested_split_with_native_witness() -> None:
    records = [
        _canonical_record("fit-prompt", anchor_offset=0),
        _canonical_record("checkpoint-prompt", anchor_offset=15),
    ]
    assignments = {
        "fit-prompt": "fit",
        "checkpoint-prompt": "checkpoint",
    }
    producer = GlobalDirectCandidateSelector(
        hidden_size=32,
        max_positions=15,
        max_candidates=16,
        model_dim=64,
        num_heads=4,
        num_layers=1,
        scope="global",
        mixer="axial",
        node_encoder="additive",
        dropout=0.0,
        initialization_seed=0,
    )
    target_embedding = torch.randn(
        300, 32, generator=torch.Generator().manual_seed(7)
    )
    materialized, witness = materialize_outcome_records(
        records,
        assignments,
        split="fit",
        producer=producer,
        target_embedding=target_embedding,
        device=torch.device("cpu"),
        batch_size=1,
    )
    assert len(materialized) == 1
    assert materialized[0]["sample_id"] == "fit-prompt"
    assert materialized[0]["source_split"] == "train"
    assert materialized[0]["numeric_policy_id"] == NUMERIC_POLICY_ID
    assert materialized[0]["numeric_policy_sha256"] == NUMERIC_POLICY_SHA256
    assert witness["batches"] == 1
    assert witness["records"] == 1
    for name in (
        "regular_vs_hooked_outputs_bitwise",
        "hooked_repeat_outputs_bitwise",
        "hooked_repeat_node_states_bitwise",
        "state_dict_unchanged_after_native",
        "state_dict_unchanged_after_hooked",
        "state_dict_unchanged_after_repeated_hooked",
    ):
        assert witness[name] is True
    assert witness["state_dict_checks"] == 3
    assert witness["state_dict_key_count"] > 0
    assert witness["state_dict_sha256_before"] == witness["state_dict_sha256_after"]
    assert witness["numeric_policy_id"] == NUMERIC_POLICY_ID
    assert witness["numeric_policy_sha256"] == NUMERIC_POLICY_SHA256
    assert witness["same_device_numeric_invariants"] is True
    assert witness["same_device_numeric_batches"] == 1
    assert witness["same_device_numeric_relation_checks"] == 15
    validate_outcome_record(materialized[0], expected_split="fit")
    assert not producer.training
    assert all(not value.requires_grad for value in producer.parameters())


def test_r079_has_no_falsifier_outcome_surface() -> None:
    assert MATERIALIZABLE_OUTCOME_SPLITS == {"fit", "checkpoint"}
    record = _canonical_record("falsifier-prompt", anchor_offset=0)
    producer = GlobalDirectCandidateSelector(
        hidden_size=32,
        max_positions=15,
        max_candidates=16,
        model_dim=64,
        num_heads=4,
        num_layers=1,
        scope="global",
        mixer="axial",
        node_encoder="additive",
        dropout=0.0,
        initialization_seed=0,
    )
    with pytest.raises(PermissionError, match="fit/checkpoint"):
        materialize_outcome_records(
            [record],
            {"falsifier-prompt": "falsifier"},
            split="falsifier",
            producer=producer,
            target_embedding=torch.randn(300, 32),
            device=torch.device("cpu"),
            batch_size=1,
        )


def test_r083_dedicated_surface_materializes_only_falsifier() -> None:
    record = _canonical_record("falsifier-prompt", anchor_offset=0)
    producer = GlobalDirectCandidateSelector(
        hidden_size=32,
        max_positions=15,
        max_candidates=16,
        model_dim=64,
        num_heads=4,
        num_layers=1,
        scope="global",
        mixer="axial",
        node_encoder="additive",
        dropout=0.0,
        initialization_seed=0,
    )
    materialized, witness = materialize_falsifier_outcome_records(
        [record],
        {"falsifier-prompt": "falsifier"},
        split="falsifier",
        producer=producer,
        target_embedding=torch.randn(
            300, 32, generator=torch.Generator().manual_seed(83)
        ),
        device=torch.device("cpu"),
        batch_size=1,
    )
    assert len(materialized) == 1
    assert materialized[0]["split"] == "falsifier"
    assert witness["records"] == 1
    validate_outcome_record(materialized[0], expected_split="falsifier")
    with pytest.raises(PermissionError, match="only falsifier"):
        materialize_falsifier_outcome_records(
            [record],
            {"falsifier-prompt": "checkpoint"},
            split="checkpoint",
            producer=producer,
            target_embedding=torch.randn(300, 32),
            device=torch.device("cpu"),
            batch_size=1,
        )


class _MutatingFrozenProducer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.residual_projection = torch.nn.Linear(64, 1)
        self.register_buffer("forward_count", torch.zeros((), dtype=torch.int64))

    def forward(
        self,
        hidden,
        candidate_embeddings,
        candidate_logits,
        base_logsumexp,
        anchor_embeddings,
    ):
        del candidate_embeddings, anchor_embeddings
        self.forward_count.add_(1)
        states = torch.zeros(
            hidden.shape[0], 15, 16, 64, device=hidden.device
        )
        residual = self.residual_projection(states).squeeze(-1)
        residual = residual - residual.mean(dim=-1, keepdim=True)
        base = candidate_logits.float() - base_logsumexp.float()[..., None]
        scores = base + residual.float()
        from sph.global_direct_selector import GlobalDirectOutput

        return GlobalDirectOutput(
            scores=scores,
            log_probs=torch.log_softmax(scores, dim=-1),
            residual_scores=residual.float(),
            base_log_probs=base,
        )


def test_native_witness_rejects_forward_mutated_registered_state() -> None:
    record = _canonical_record("mutating", anchor_offset=0)
    with pytest.raises(RuntimeError, match="state_dict tensor changed"):
        materialize_outcome_records(
            [record],
            {"mutating": "fit"},
            split="fit",
            producer=_MutatingFrozenProducer(),
            target_embedding=torch.randn(300, 32),
            device=torch.device("cpu"),
            batch_size=1,
        )
