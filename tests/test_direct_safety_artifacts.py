from __future__ import annotations

from collections import Counter
from copy import deepcopy
import json

import pytest
import torch

from sph.direct_safety_artifacts import (
    CAPACITY_ARTIFACT_PROTOCOL,
    OUTCOME_ARTIFACT_PROTOCOL,
    SPLIT_MANIFEST_PROTOCOL,
    atomic_write_json,
    build_phase3_split_manifest,
    load_capacity_bundle,
    load_outcome_bundle,
    select_capacity_from_fit_bundle,
    sha256_file,
    split_assignment_map,
    validate_outcome_record,
    verify_phase3_split_manifest,
    write_capacity_bundle,
    write_outcome_bundle,
)
from sph.direct_safety_gate import direct_safety_position_features
from sph.direct_safety_numeric_policy import (
    NUMERIC_POLICY_ID,
    NUMERIC_POLICY_SHA256,
    numeric_policy_receipt,
)
from sph.global_direct_selector import GlobalDirectOutput


CANONICAL_HASH = "0" * 64
SPLIT_HASH = "1" * 64
PRODUCER_HASH = "2" * 64
METRICS_HASH = "3" * 64
FIT_METADATA_HASH = "4" * 64
SOURCE_CLOSURE = {
    "protocol": "pros-gate-first-party-source-closure-v1",
    "source_manifest_sha256": "8" * 64,
    "source_file_count": 1,
    "source_entries_sha256": "9" * 64,
}
EXCLUSION_HASHES = {
    "producer_train": "5" * 64,
    "validation": "6" * 64,
    "reserved": "7" * 64,
}
EXCLUSION_PROMPTS = {
    "producer_train": {"producer:a", "producer:b"},
    "validation": {"validation:a"},
    "reserved": {"reserved:a"},
}


@pytest.fixture(scope="module", autouse=True)
def _single_threaded_small_tensor_checks():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def _identity_fixture() -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for domain in ("chat", "code"):
        for prompt_index in range(3):
            sample_id = f"{domain}:{prompt_index}"
            for block_index in range(2):
                records.append(
                    {
                        "sample_id": sample_id,
                        "anchor_offset": block_index * 15,
                        "context_length": 32 + block_index,
                        "domain": domain,
                    }
                )
    return records


SMALL_SPLIT_COUNTS = {
    "chat": {"fit": 1, "checkpoint": 1, "falsifier": 1},
    "code": {"fit": 1, "checkpoint": 1, "falsifier": 1},
}


def _small_split_manifest(
    records: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    values = _identity_fixture() if records is None else records
    return build_phase3_split_manifest(
        values,
        canonical_metadata_sha256=CANONICAL_HASH,
        exclusion_prompt_sets=EXCLUSION_PROMPTS,
        exclusion_manifest_sha256=EXCLUSION_HASHES,
        split_counts=SMALL_SPLIT_COUNTS,
        expected_prompts=6,
        expected_blocks=12,
    )


def _outcome_record(
    sample_id: str,
    outcome: str,
    *,
    split: str = "fit",
    anchor_offset: int = 0,
) -> dict[str, object]:
    if outcome not in {"beneficial", "harmful", "changed-neutral"}:
        raise ValueError(outcome)
    logits = -torch.arange(16, dtype=torch.float32)[None].expand(15, -1).clone()
    base_logsumexp = torch.logsumexp(logits, dim=-1) + 0.75
    base_log_probs = logits - base_logsumexp[:, None]
    residual = torch.zeros_like(base_log_probs)
    residual[0, 1] = 8.0
    residual[0] -= residual[0].mean()
    scores = base_log_probs + residual
    direct_output = GlobalDirectOutput(
        scores=scores[None],
        log_probs=torch.log_softmax(scores[None], dim=-1),
        residual_scores=residual[None],
        base_log_probs=base_log_probs[None],
    )
    generator = torch.Generator().manual_seed(20260805)
    node_states = torch.randn(1, 15, 16, 64, generator=generator)
    feature_output = direct_safety_position_features(
        node_states,
        direct_output,
        logits[None],
        base_logsumexp[None],
    )

    positions = torch.arange(15, dtype=torch.int64)[:, None]
    ranks = torch.arange(16, dtype=torch.int64)[None]
    candidate_ids = 10_000 + positions * 100 + ranks
    gold_ids = candidate_ids[:, 0].clone()
    if outcome == "beneficial":
        gold_ids[0] = candidate_ids[0, 1]
        base_length, direct_length, gain = 0, 15, 1.0
    elif outcome == "harmful":
        base_length, direct_length, gain = 15, 0, -1.0
    else:
        gold_ids[0] = 999_999
        base_length, direct_length, gain = 0, 0, 0.0

    return {
        "sample_id": sample_id,
        "anchor_offset": anchor_offset,
        "context_length": 64 + anchor_offset,
        "domain": "synthetic",
        "source_split": "train",
        "split": split,
        "numeric_policy_id": NUMERIC_POLICY_ID,
        "numeric_policy_sha256": NUMERIC_POLICY_SHA256,
        "position_features": feature_output.position_features[0],
        "direct_path": feature_output.direct_path[0],
        "change_mask": feature_output.change_mask[0],
        "candidate_ids": candidate_ids,
        "gold_ids": gold_ids,
        "candidate_logits": logits,
        "base_logsumexp": base_logsumexp,
        "direct_scores": scores,
        "direct_residual_scores": residual,
        "base_log_probs": base_log_probs,
        "base_length": base_length,
        "direct_length": direct_length,
        "base_first_token_correct": base_length > 0,
        "direct_first_token_correct": direct_length > 0,
        "normalized_gain": gain,
    }


def _capacity_records() -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for name, count in (
        ("harmful", 128),
        ("changed-neutral", 128),
        ("beneficial", 256),
    ):
        template = _outcome_record(f"template:{name}", name)
        for index in range(count):
            record = dict(template)
            record["sample_id"] = f"{name}:{index:04d}"
            record["anchor_offset"] = index
            record["context_length"] = 1000 + index
            records.append(record)
    return records


def test_split_manifest_is_identity_only_deterministic_and_reconstructable() -> None:
    records = _identity_fixture()
    first = _small_split_manifest(records)
    repeated = _small_split_manifest(list(reversed(records)))
    assert first == repeated
    assert first["protocol"] == SPLIT_MANIFEST_PROTOCOL
    assert first["prompt_count"] == 6
    assert first["block_count"] == 12
    assert first["prompt_counts_by_split"] == {
        "checkpoint": 2,
        "falsifier": 2,
        "fit": 2,
    }
    assert Counter(split_assignment_map(first).values()) == {
        "fit": 2,
        "checkpoint": 2,
        "falsifier": 2,
    }
    verify_phase3_split_manifest(
        first,
        records,
        canonical_metadata_sha256=CANONICAL_HASH,
        exclusion_prompt_sets=EXCLUSION_PROMPTS,
        exclusion_manifest_sha256=EXCLUSION_HASHES,
        split_counts=SMALL_SPLIT_COUNTS,
        expected_prompts=6,
        expected_blocks=12,
    )
    for proof in first["exclusion_provenance"].values():
        assert proof["overlap"] == 0


def test_split_manifest_fails_closed_on_leakage_duplicate_or_tamper() -> None:
    records = _identity_fixture()
    leaking = {key: set(value) for key, value in EXCLUSION_PROMPTS.items()}
    leaking["reserved"].add("chat:0")
    with pytest.raises(ValueError, match="overlap reserved"):
        build_phase3_split_manifest(
            records,
            canonical_metadata_sha256=CANONICAL_HASH,
            exclusion_prompt_sets=leaking,
            exclusion_manifest_sha256=EXCLUSION_HASHES,
            split_counts=SMALL_SPLIT_COUNTS,
            expected_prompts=6,
            expected_blocks=12,
        )
    with pytest.raises(ValueError, match="unique"):
        _small_split_manifest([*records, records[0]])

    tampered = deepcopy(_small_split_manifest(records))
    tampered["prompts"][0]["split"] = "fit"
    with pytest.raises(RuntimeError, match="differs from reconstruction"):
        verify_phase3_split_manifest(
            tampered,
            records,
            canonical_metadata_sha256=CANONICAL_HASH,
            exclusion_prompt_sets=EXCLUSION_PROMPTS,
            exclusion_manifest_sha256=EXCLUSION_HASHES,
            split_counts=SMALL_SPLIT_COUNTS,
            expected_prompts=6,
            expected_blocks=12,
        )


@pytest.mark.parametrize(
    ("outcome", "expected"),
    [
        ("beneficial", (0, 15, 1.0)),
        ("harmful", (15, 0, -1.0)),
        ("changed-neutral", (0, 0, 0.0)),
    ],
)
def test_outcome_record_independently_reconstructs_tokens_and_features(
    outcome: str,
    expected: tuple[int, int, float],
) -> None:
    record = _outcome_record(f"sample:{outcome}", outcome)
    capacity = validate_outcome_record(record, expected_split="fit")
    assert (record["base_length"], record["direct_length"]) == expected[:2]
    assert capacity.normalized_gain == expected[2]
    assert capacity.direct_changed


def test_outcome_record_requires_exact_numeric_policy_binding() -> None:
    record = _outcome_record("policy", "beneficial")
    missing = deepcopy(record)
    del missing["numeric_policy_sha256"]
    with pytest.raises(ValueError, match="policy digest"):
        validate_outcome_record(missing, expected_split="fit")
    changed = deepcopy(record)
    changed["numeric_policy_id"] = "different-policy"
    with pytest.raises(ValueError, match="policy ID"):
        validate_outcome_record(changed, expected_split="fit")


def test_fractional_gain_uses_exact_integer_ratio_not_float32_rendering() -> None:
    record = _outcome_record("fractional", "beneficial")
    record["gold_ids"][1] = 999_999
    record["direct_length"] = 1
    record["direct_first_token_correct"] = True
    record["normalized_gain"] = 1 / 15
    capacity = validate_outcome_record(record, expected_split="fit")
    assert capacity.normalized_gain == 1 / 15


def _move_float32(value: torch.Tensor, direction: float, steps: int) -> torch.Tensor:
    moved = value.clone()
    destination = torch.full_like(moved, direction)
    for _ in range(steps):
        moved = torch.nextafter(moved, destination)
    return moved


@pytest.mark.parametrize(("column", "index", "message"), [(195, 0, "rank"), (196, 7, "position")])
@pytest.mark.parametrize("direction", [-torch.inf, torch.inf])
def test_outcome_record_accepts_one_adjacent_float32_for_normalized_features(
    column: int,
    index: int,
    message: str,
    direction: float,
) -> None:
    del message
    record = deepcopy(_outcome_record("adjacent", "beneficial"))
    value = record["position_features"][index, column]
    assert 0.0 < float(value) < 1.0
    record["position_features"][index, column] = _move_float32(
        value, direction, 1
    )
    validate_outcome_record(record, expected_split="fit")


@pytest.mark.parametrize(("column", "index", "message"), [(195, 0, "rank"), (196, 7, "position")])
@pytest.mark.parametrize("direction", [-torch.inf, torch.inf])
def test_outcome_record_rejects_two_adjacent_float32_steps(
    column: int,
    index: int,
    message: str,
    direction: float,
) -> None:
    record = deepcopy(_outcome_record("two-ulp", "beneficial"))
    value = record["position_features"][index, column]
    record["position_features"][index, column] = _move_float32(
        value, direction, 2
    )
    with pytest.raises(ValueError, match=message):
        validate_outcome_record(record, expected_split="fit")


@pytest.mark.parametrize(("column", "index", "message"), [(195, 0, "rank"), (196, 7, "position")])
def test_outcome_record_rejects_material_normalized_feature_tampering(
    column: int,
    index: int,
    message: str,
) -> None:
    record = deepcopy(_outcome_record("material", "beneficial"))
    record["position_features"][index, column] += 1e-4
    with pytest.raises(ValueError, match=message):
        validate_outcome_record(record, expected_split="fit")


@pytest.mark.parametrize(
    ("column", "index", "direction", "message"),
    [
        (195, 1, torch.inf, "rank"),
        (196, 0, torch.inf, "position"),
        (196, 14, -torch.inf, "position"),
    ],
)
def test_normalized_feature_endpoints_remain_bitwise_exact(
    column: int,
    index: int,
    direction: float,
    message: str,
) -> None:
    record = deepcopy(_outcome_record("endpoint", "beneficial"))
    value = record["position_features"][index, column]
    assert float(value) in {0.0, 1.0}
    record["position_features"][index, column] = _move_float32(
        value, direction, 1
    )
    with pytest.raises(ValueError, match=message):
        validate_outcome_record(record, expected_split="fit")


def test_outcome_bundle_accepts_one_adjacent_normalized_feature(tmp_path) -> None:
    record = deepcopy(_outcome_record("bundle-adjacent", "beneficial"))
    value = record["position_features"][0, 195]
    record["position_features"][0, 195] = _move_float32(
        value, torch.inf, 1
    )
    output = tmp_path / "fit"
    write_outcome_bundle(
        output,
        [record],
        split="fit",
        split_manifest_sha256=SPLIT_HASH,
        provenance={"synthetic": True},
    )
    assert (output / "metadata.json").is_file()


@pytest.mark.parametrize(
    ("field", "mutator", "message"),
    [
        ("direct_path", lambda value: value.fill_(0), "path"),
        ("change_mask", lambda value: value.fill_(False), "change mask"),
        ("position_features", lambda value: value[:, 192].add_(1), "margin"),
        ("candidate_logits", lambda value: value.add_(1), "log probabilities"),
        ("gold_ids", lambda value: value.fill_(999_999), "base length"),
    ],
)
def test_outcome_record_rejects_semantic_tensor_tampering(
    field: str,
    mutator: object,
    message: str,
) -> None:
    record = deepcopy(_outcome_record("tamper", "harmful"))
    mutator(record[field])  # type: ignore[operator]
    with pytest.raises(ValueError, match=message):
        validate_outcome_record(record, expected_split="fit")


def test_outcome_bundle_is_atomic_content_addressed_and_split_isolated(
    tmp_path: pytest.TempPathFactory,
) -> None:
    output = tmp_path / "fit"
    records = [
        _outcome_record("beneficial", "beneficial"),
        _outcome_record("harmful", "harmful", anchor_offset=1),
        _outcome_record("neutral", "changed-neutral", anchor_offset=2),
    ]
    metadata = write_outcome_bundle(
        output,
        records,
        split="fit",
        split_manifest_sha256=SPLIT_HASH,
        provenance={"synthetic": True},
    )
    assert metadata["protocol"] == OUTCOME_ARTIFACT_PROTOCOL
    assert metadata["numeric_policy"] == numeric_policy_receipt()
    assert metadata["summary"]["beneficial_blocks"] == 1
    metadata_sha256 = sha256_file(output / "metadata.json")
    loaded, loaded_metadata = load_outcome_bundle(
        output,
        expected_split="fit",
        expected_metadata_sha256=metadata_sha256,
    )
    assert len(loaded) == 3
    assert loaded_metadata == metadata
    with pytest.raises(RuntimeError, match="split mismatch"):
        load_outcome_bundle(output, expected_split="checkpoint")
    with pytest.raises(FileExistsError, match="overwrite"):
        write_outcome_bundle(
            output,
            records,
            split="fit",
            split_manifest_sha256=SPLIT_HASH,
            provenance={},
        )

    stored = torch.load(output / "records.pt", map_location="cpu", weights_only=False)
    stored[0]["normalized_gain"] = 0.0
    torch.save(stored, output / "records.pt")
    with pytest.raises(RuntimeError, match="records hash mismatch"):
        load_outcome_bundle(output, expected_split="fit")


def test_outcome_bundle_rejects_duplicate_blocks_and_bad_hash(tmp_path) -> None:
    record = _outcome_record("duplicate", "beneficial")
    with pytest.raises(ValueError, match="unique block"):
        write_outcome_bundle(
            tmp_path / "duplicates",
            [record, record],
            split="fit",
            split_manifest_sha256=SPLIT_HASH,
            provenance={},
        )
    with pytest.raises(ValueError, match="canonical lowercase SHA256"):
        write_outcome_bundle(
            tmp_path / "bad-hash",
            [record],
            split="fit",
            split_manifest_sha256="bad",
            provenance={},
        )


def test_outcome_loader_rejects_numeric_policy_metadata_tamper(tmp_path) -> None:
    output = tmp_path / "policy-tamper"
    write_outcome_bundle(
        output,
        [_outcome_record("policy-tamper", "beneficial")],
        split="fit",
        split_manifest_sha256=SPLIT_HASH,
        provenance={},
    )
    metadata_path = output / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["numeric_policy"]["sha256"] = "0" * 64
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(RuntimeError, match="numeric policy"):
        load_outcome_bundle(output, expected_split="fit")


def test_capacity_selection_bundle_and_loader_enforce_exact_contract(tmp_path) -> None:
    candidates = _capacity_records()
    selected = select_capacity_from_fit_bundle(
        candidates,
        producer_checkpoint_sha256=PRODUCER_HASH,
        producer_metrics_sha256=METRICS_HASH,
        canonical_metadata_sha256=CANONICAL_HASH,
        split_manifest_sha256=SPLIT_HASH,
    )
    repeated = select_capacity_from_fit_bundle(
        list(reversed(candidates)),
        producer_checkpoint_sha256=PRODUCER_HASH,
        producer_metrics_sha256=METRICS_HASH,
        canonical_metadata_sha256=CANONICAL_HASH,
        split_manifest_sha256=SPLIT_HASH,
    )
    assert [row["sample_id"] for row in selected] == [
        row["sample_id"] for row in repeated
    ]
    output = tmp_path / "capacity"
    metadata = write_capacity_bundle(
        output,
        selected,
        parent_fit_metadata_sha256=FIT_METADATA_HASH,
        producer_checkpoint_sha256=PRODUCER_HASH,
        producer_metrics_sha256=METRICS_HASH,
        canonical_metadata_sha256=CANONICAL_HASH,
        split_manifest_sha256=SPLIT_HASH,
        source_closure=SOURCE_CLOSURE,
    )
    assert metadata["protocol"] == CAPACITY_ARTIFACT_PROTOCOL
    assert metadata["composition"] == {
        "beneficial": 256,
        "changed-neutral": 128,
        "harmful": 128,
    }
    loaded, loaded_metadata = load_capacity_bundle(
        output,
        expected_metadata_sha256=sha256_file(output / "metadata.json"),
    )
    assert len(loaded) == 512
    assert loaded_metadata == metadata

    tampered = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
    tampered["composition"] = {"beneficial": 512}
    (output / "metadata.json").write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(RuntimeError, match="composition differs from metadata"):
        load_capacity_bundle(output)


def test_capacity_writer_rejects_wrong_composition_and_atomic_json_no_overwrite(
    tmp_path,
) -> None:
    records = _capacity_records()
    with pytest.raises(ValueError, match="exactly 512"):
        write_capacity_bundle(
            tmp_path / "short",
            records[:-1],
            parent_fit_metadata_sha256=FIT_METADATA_HASH,
            producer_checkpoint_sha256=PRODUCER_HASH,
            producer_metrics_sha256=METRICS_HASH,
            canonical_metadata_sha256=CANONICAL_HASH,
            split_manifest_sha256=SPLIT_HASH,
            source_closure=SOURCE_CLOSURE,
        )
    manifest_path = tmp_path / "manifest.json"
    atomic_write_json(manifest_path, {"value": 1})
    with pytest.raises(FileExistsError, match="overwrite"):
        atomic_write_json(manifest_path, {"value": 2})
