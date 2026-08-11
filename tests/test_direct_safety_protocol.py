from __future__ import annotations

from collections import Counter
import math

import pytest
import torch

from sph.direct_safety_protocol import (
    CAPACITY_ADJUDICATION_SCHEMA,
    BlockKey,
    CapacityRecord,
    SavedGateRecord,
    assert_disjoint_prompt_sets,
    assert_stage_splits,
    build_prompt_split,
    capacity_gate_passes,
    capacity_record_rank_sha256,
    capacity_records_sha256,
    complete_pass_schedule,
    deterministic_bootstrap_indices,
    earliest_exact_minimum,
    fit_checkpoint_selection_key,
    fit_weighted_ridge,
    half_up_warmup_steps,
    ordered_block_keys,
    ordered_block_keys_sha256,
    reconstruct_saved_gate_evaluation,
    select_capacity_records,
    selected_capacity_checkpoint,
    selected_fit_checkpoint,
)


METADATA_HASH = "0" * 64
MANIFEST_HASH = "1" * 64
PRODUCER_HASH = "2" * 64
METRICS_HASH = "3" * 64


def exact_prompt_domains() -> dict[str, str]:
    result: dict[str, str] = {}
    for domain, count in (("chat", 655), ("code", 665), ("math", 667)):
        for index in range(count):
            result[f"{domain}:{index:04d}"] = domain
    return result


def test_exact_phase3_split_counts_and_determinism() -> None:
    prompts = exact_prompt_domains()
    first = build_prompt_split(prompts, METADATA_HASH)
    second = build_prompt_split(dict(reversed(list(prompts.items()))), METADATA_HASH)
    assert first == second
    by_domain_split = Counter((prompts[key], split) for key, split in first.items())
    assert by_domain_split == {
        ("chat", "fit"): 523,
        ("chat", "checkpoint"): 66,
        ("chat", "falsifier"): 66,
        ("code", "fit"): 531,
        ("code", "checkpoint"): 67,
        ("code", "falsifier"): 67,
        ("math", "fit"): 533,
        ("math", "checkpoint"): 67,
        ("math", "falsifier"): 67,
    }
    assert Counter(first.values()) == {
        "fit": 1587,
        "checkpoint": 200,
        "falsifier": 200,
    }


def test_prompt_split_is_identity_only_and_fails_wrong_counts() -> None:
    small = {"chat:a": "chat", "chat:b": "chat", "code:a": "code"}
    counts = {
        "chat": {"fit": 1, "checkpoint": 1, "falsifier": 0},
        "code": {"fit": 1, "checkpoint": 0, "falsifier": 0},
    }
    split = build_prompt_split(
        small, METADATA_HASH, split_counts=counts
    )
    assert Counter(split.values()) == {"fit": 2, "checkpoint": 1}
    with pytest.raises(ValueError, match="prompt count"):
        build_prompt_split(
            {"chat:a": "chat"}, METADATA_HASH, split_counts=counts
        )


def test_disjoint_prompt_guard_does_not_need_examples() -> None:
    assert_disjoint_prompt_sets(
        {"fit": {"a", "b"}, "checkpoint": {"c"}, "falsifier": {"d"}}
    )
    with pytest.raises(ValueError, match="overlap"):
        assert_disjoint_prompt_sets(
            {"fit": {"a", "b"}, "checkpoint": {"b", "c"}}
        )


def test_stage_split_allowlist_rejects_every_reserved_surface() -> None:
    assert_stage_splits("fit", {"fit"})
    assert_stage_splits("checkpoint", {"checkpoint"})
    assert_stage_splits("falsifier", {"falsifier"})
    assert_stage_splits(
        "manifest_identity", {"fit", "checkpoint", "falsifier", "opb_train"}
    )
    for forbidden in (
        "validation_gate",
        "validation_select",
        "formal",
        "formal_test",
        "reserved",
        "test",
    ):
        with pytest.raises(PermissionError):
            assert_stage_splits("fit", {"fit", forbidden})
    with pytest.raises(ValueError, match="unknown"):
        assert_stage_splits("development", {"fit"})


def test_block_key_serialization_and_pass_order_are_version_independent() -> None:
    keys = [
        BlockKey("sample-z", 8, 101),
        BlockKey("sample-a", 0, 17),
        BlockKey("sample-b", 24, 53),
    ]
    assert keys[1].serialize() == b"sample-a\x000\x0017"
    assert BlockKey("样本", 12, 0).serialize() == "样本\0".encode() + b"12\x000"
    first = ordered_block_keys(
        keys, pass_index=0, training_manifest_sha256=MANIFEST_HASH
    )
    repeated = ordered_block_keys(
        list(reversed(keys)),
        pass_index=0,
        training_manifest_sha256=MANIFEST_HASH,
    )
    second_pass = ordered_block_keys(
        keys, pass_index=1, training_manifest_sha256=MANIFEST_HASH
    )
    assert first == repeated
    assert first != second_pass
    assert ordered_block_keys_sha256(first) == ordered_block_keys_sha256(repeated)
    with pytest.raises(ValueError, match="unique"):
        ordered_block_keys(
            [keys[0], keys[0]],
            pass_index=0,
            training_manifest_sha256=MANIFEST_HASH,
        )
    for invalid in (1.5, True, -1, "1", None):
        with pytest.raises(ValueError):
            BlockKey("sample", invalid, 2).serialize()  # type: ignore[arg-type]
        with pytest.raises(ValueError):
            BlockKey("sample", 2, invalid).serialize()  # type: ignore[arg-type]
        with pytest.raises(ValueError):
            ordered_block_keys(
                keys,
                pass_index=invalid,  # type: ignore[arg-type]
                training_manifest_sha256=MANIFEST_HASH,
            )
    with pytest.raises(ValueError, match="lowercase"):
        ordered_block_keys(
            keys,
            pass_index=0,
            training_manifest_sha256="A" * 64,
        )


def capacity_candidates() -> list[CapacityRecord]:
    records: list[CapacityRecord] = []
    for outcome, count, gain, changed in (
        ("harm", 128, -1 / 15, True),
        ("neutral", 128, 0.0, True),
        ("benefit", 256, 2 / 15, True),
    ):
        for index in range(count):
            records.append(
                CapacityRecord(
                    BlockKey(f"{outcome}:{index:04d}", index, 100 + index),
                    gain,
                    changed,
                )
            )
    return records


def test_capacity_selection_exact_unique_deterministic_and_fail_closed() -> None:
    records = capacity_candidates()
    selected = select_capacity_records(
        records,
        PRODUCER_HASH,
        METRICS_HASH,
        METADATA_HASH,
        MANIFEST_HASH,
    )
    repeated = select_capacity_records(
        list(reversed(records)),
        PRODUCER_HASH,
        METRICS_HASH,
        METADATA_HASH,
        MANIFEST_HASH,
    )
    assert selected == repeated
    assert len(selected) == 512
    assert len({record.block_key.sample_id for record in selected}) == 512
    assert sum(record.normalized_gain < 0 for record in selected) == 128
    assert sum(
        record.normalized_gain == 0 and record.direct_changed
        for record in selected
    ) == 128
    assert sum(record.normalized_gain > 0 for record in selected) == 256
    assert capacity_records_sha256(selected) == capacity_records_sha256(repeated)
    assert capacity_record_rank_sha256(
        BlockKey("golden", 7, 29),
        PRODUCER_HASH,
        METRICS_HASH,
        METADATA_HASH,
        MANIFEST_HASH,
    ) == "b36a12025c351f15de47db197c8c3371cbf155c879164b75a4ea01054feb2946"

    with pytest.raises(RuntimeError, match="beneficial"):
        select_capacity_records(
            records[:-1],
            PRODUCER_HASH,
            METRICS_HASH,
            METADATA_HASH,
            MANIFEST_HASH,
        )


def test_capacity_scarcity_order_excludes_prompts_globally() -> None:
    records = capacity_candidates()
    records.append(
        CapacityRecord(BlockKey("harm:0000", 999, 999), 1 / 15, True)
    )
    selected = select_capacity_records(
        records,
        PRODUCER_HASH,
        METRICS_HASH,
        METADATA_HASH,
        MANIFEST_HASH,
    )
    selected_for_shared = [
        record for record in selected if record.block_key.sample_id == "harm:0000"
    ]
    assert len(selected_for_shared) == 1
    assert selected_for_shared[0].normalized_gain < 0


def test_capacity_rejects_nonzero_gain_without_direct_path_change() -> None:
    records = capacity_candidates()
    for gain in (1 / 15, -1 / 15):
        impossible = CapacityRecord(
            BlockKey(f"impossible:{gain}", 0, 1), gain, False
        )
        with pytest.raises(ValueError, match="requires a changed Direct path"):
            select_capacity_records(
                [impossible, *records],
                PRODUCER_HASH,
                METRICS_HASH,
                METADATA_HASH,
                MANIFEST_HASH,
            )
        with pytest.raises(ValueError, match="requires a changed Direct path"):
            capacity_records_sha256([impossible])


def test_complete_pass_and_half_up_warmup_arithmetic() -> None:
    capacity = complete_pass_schedule(512, batch_size=32)
    assert capacity.steps_per_pass == 16
    assert capacity.passes == 320
    assert capacity.total_steps == 5120
    assert capacity.warmup_steps == 205
    fit = complete_pass_schedule(12_680, batch_size=64)
    assert fit.steps_per_pass == 199
    assert fit.passes == 25
    assert fit.total_steps == 4975
    assert fit.warmup_steps == 199
    assert half_up_warmup_steps(5120) == 205


def passing_capacity_metrics(loss: float) -> dict[str, float | int | bool]:
    return {
        "prompt_weighted_loss": loss,
        "regret_bound_violation_count": 0,
        "record_count": 512,
        "prompt_count": 512,
        "beneficial_count": 256,
        "beneficial_apply_count": 254,
        "harmful_count": 128,
        "harmful_keep_count": 127,
        "harm_avoidance_numerator": 127,
        "harm_avoidance_denominator": 128,
        "neutral_count": 128,
        "utility_optimal_count": 509,
        "base_eal": 1.0,
        "method_eal": 2.9,
        "oracle_eal": 3.0,
        "recovery_numerator": 1.9,
        "recovery_denominator": 2.0,
        "oracle_recovery": 0.95,
        "harmful_apply_count": 1,
        "values_finite": True,
        "gradients_finite": True,
    }


def test_capacity_gate_checks_every_binding_boundary() -> None:
    assert CAPACITY_ADJUDICATION_SCHEMA == "pros-capacity-adjudication-v2"
    passing = passing_capacity_metrics(0.5)
    assert capacity_gate_passes(passing, 10.0)
    failures: dict[str, float | int | bool] = {
        "prompt_weighted_loss": 0.5000001,
        "regret_bound_violation_count": 1,
        "record_count": 511,
        "prompt_count": 511,
        "beneficial_count": 255,
        "beneficial_apply_count": 253,
        "harmful_count": 127,
        "harmful_keep_count": 126,
        "neutral_count": 127,
        "utility_optimal_count": 508,
        "oracle_recovery": 0.949999,
        "harmful_apply_count": 2,
        "values_finite": False,
        "gradients_finite": False,
    }
    for name, value in failures.items():
        altered = dict(passing)
        altered[name] = value
        assert not capacity_gate_passes(altered, 10.0), name
    assert not capacity_gate_passes(passing, 0.0)
    assert not capacity_gate_passes(passing, float("nan"))
    negative = dict(passing)
    negative["prompt_weighted_loss"] = -1e-6
    assert not capacity_gate_passes(negative, 10.0)
    fractional = dict(passing)
    fractional["utility_optimal_count"] = 509.5
    assert not capacity_gate_passes(fractional, 10.0)
    upper = dict(passing)
    upper["oracle_recovery"] = 1.0 + 1.1e-6
    assert not capacity_gate_passes(upper, 10.0)
    forged = dict(passing)
    forged["method_eal"] = 2.0
    assert not capacity_gate_passes(forged, 10.0)
    inconsistent_denominator = dict(passing)
    inconsistent_denominator["recovery_denominator"] = -1.0
    assert not capacity_gate_passes(inconsistent_denominator, 10.0)
    zero_denominator = dict(passing)
    zero_denominator["oracle_eal"] = zero_denominator["base_eal"]
    assert not capacity_gate_passes(zero_denominator, 10.0)
    negative_denominator = dict(passing)
    negative_denominator["oracle_eal"] = 0.5
    assert not capacity_gate_passes(negative_denominator, 10.0)
    nan_denominator = dict(passing)
    nan_denominator["oracle_eal"] = float("nan")
    assert not capacity_gate_passes(nan_denominator, 10.0)
    inconsistent_harm_alias = dict(passing)
    inconsistent_harm_alias["harm_avoidance_numerator"] = 126
    assert not capacity_gate_passes(inconsistent_harm_alias, 10.0)
    inconsistent_harm_count = dict(passing)
    inconsistent_harm_count["harm_avoidance_denominator"] = 127
    assert not capacity_gate_passes(inconsistent_harm_count, 10.0)
    missing_harm_alias = dict(passing)
    del missing_harm_alias["harmful_keep_count"]
    assert not capacity_gate_passes(missing_harm_alias, 10.0)


def test_earliest_minimum_and_selected_capacity_pass_are_binding() -> None:
    assert earliest_exact_minimum([2.0, 1.0, 1.0, 3.0]) == 1
    history = []
    for index in range(321):
        row = passing_capacity_metrics(float(321 - index))
        row["pass"] = index
        history.append(row)
    assert selected_capacity_checkpoint(history)["pass"] == 320

    tied = [dict(row) for row in history]
    tied[319]["prompt_weighted_loss"] = 1.0
    assert selected_capacity_checkpoint(tied)["pass"] == 319

    nonselected_rescue = [dict(row) for row in history]
    nonselected_rescue[10]["prompt_weighted_loss"] = 0.5
    nonselected_rescue[320]["prompt_weighted_loss"] = 0.25
    nonselected_rescue[320]["beneficial_apply_count"] = 253
    with pytest.raises(RuntimeError, match="selected"):
        selected_capacity_checkpoint(nonselected_rescue)


def test_saved_record_reconstruction_hand_golden_and_denominators() -> None:
    records = [
        SavedGateRecord(BlockKey("a", 0, 20), 2, 5, 1.0, True, True),
        SavedGateRecord(BlockKey("a", 1, 21), 6, 3, 0.0, True, True),
        SavedGateRecord(BlockKey("b", 0, 20), 1, 4, -0.5, True, True),
        SavedGateRecord(BlockKey("b", 1, 21), 4, 4, 2.0, True, True),
    ]
    report = reconstruct_saved_gate_evaluation(records)
    assert report.apply_direct == (True, False, False, True)
    assert report.method_lengths == (5, 6, 1, 4)
    assert report.oracle_lengths == (5, 6, 4, 4)
    assert report.normalized_gains == pytest.approx((0.2, -0.2, 0.2, 0.0))
    assert report.per_block_loss == pytest.approx((0.0, 0.2, 0.3, 0.0))
    assert report.decoded_regret == pytest.approx((0.0, 0.0, 0.2, 0.0))
    assert report.bound_slack == pytest.approx((0.0, 0.2, 0.1, 0.0))
    metrics = report.metrics
    assert metrics["record_count"] == 4
    assert metrics["prompt_count"] == 2
    assert metrics["base_eal"] == pytest.approx(3.25)
    assert metrics["direct_eal"] == pytest.approx(4.0)
    assert metrics["method_eal"] == pytest.approx(4.0)
    assert metrics["oracle_eal"] == pytest.approx(4.75)
    assert metrics["oracle_recovery"] == pytest.approx(0.5)
    assert metrics["prompt_weighted_gain_hinge"] == pytest.approx(0.125)
    assert metrics["prompt_weighted_decoded_regret"] == pytest.approx(0.05)
    assert metrics["regret_bound_violation_count"] == 0
    assert metrics["benefit_recall_numerator"] == 1
    assert metrics["benefit_recall_denominator"] == 2
    assert metrics["benefit_recall"] == pytest.approx(0.5)
    assert metrics["harm_avoidance_numerator"] == 1
    assert metrics["harmful_keep_count"] == 1
    assert metrics["harm_avoidance_denominator"] == 1
    assert metrics["harm_avoidance"] == pytest.approx(1.0)
    assert metrics["tie_agreement_numerator"] == 0
    assert metrics["tie_agreement_denominator"] == 1
    assert metrics["utility_optimal_numerator"] == 3
    assert metrics["utility_optimal_denominator"] == 4
    assert metrics["false_apply_numerator"] == 1
    assert metrics["false_apply_denominator"] == 2
    assert metrics["base_first_token_count"] == 4
    assert metrics["direct_first_token_count"] == 4
    assert metrics["method_first_token_count"] == 4
    assert metrics["oracle_first_token_count"] == 4


def test_saved_record_evaluator_is_capacity_gate_schema_compatible() -> None:
    records = [
        SavedGateRecord(
            BlockKey(f"beneficial:{index}", 0, 1),
            0,
            1,
            1.0,
            False,
            True,
        )
        for index in range(256)
    ]
    records.extend(
        SavedGateRecord(
            BlockKey(f"harmful:{index}", 0, 1),
            1,
            0,
            -1.0,
            True,
            False,
        )
        for index in range(128)
    )
    records.extend(
        SavedGateRecord(
            BlockKey(f"neutral:{index}", 0, 1),
            1,
            1,
            0.0,
            True,
            True,
        )
        for index in range(128)
    )
    replay = reconstruct_saved_gate_evaluation(records)
    metrics = {
        **{name: value for name, value in replay.metrics.items() if value is not None},
        "prompt_weighted_loss": replay.metrics["prompt_weighted_gain_hinge"],
        "utility_optimal_count": replay.metrics["utility_optimal_numerator"],
        "values_finite": True,
        "gradients_finite": True,
    }
    assert metrics["harmful_keep_count"] == 128
    assert metrics["harmful_keep_count"] == metrics["harm_avoidance_numerator"]
    assert capacity_gate_passes(metrics, 1.0)

    for name, value in (
        ("harmful_keep_count", 127),
        ("harm_avoidance_numerator", 127),
        ("harm_avoidance_denominator", 127),
    ):
        tampered = dict(metrics)
        tampered[name] = value
        assert not capacity_gate_passes(tampered, 1.0), name


def test_false_apply_uses_all_nonbeneficial_blocks_as_denominator() -> None:
    records = [
        SavedGateRecord(
            BlockKey(f"benefit:{index}", 0, 1),
            0,
            1,
            1.0,
            False,
            True,
        )
        for index in range(4)
    ]
    records.extend(
        [
            SavedGateRecord(
                BlockKey("harm", 0, 1), 1, 0, 1.0, True, False
            ),
            SavedGateRecord(
                BlockKey("neutral", 0, 1), 0, 0, 1.0, False, False
            ),
        ]
    )
    metrics = reconstruct_saved_gate_evaluation(records).metrics
    assert metrics["apply_count"] == 6
    assert metrics["false_apply_numerator"] == 2
    assert metrics["false_apply_denominator"] == 2
    assert metrics["false_apply_fraction"] == pytest.approx(1.0)


def test_saved_record_zero_rate_denominators_and_corruption_fail_closed() -> None:
    only_benefit = [
        SavedGateRecord(BlockKey("p", 0, 1), 1, 2, -1.0, True, True)
    ]
    metrics = reconstruct_saved_gate_evaluation(only_benefit).metrics
    assert metrics["harm_avoidance_denominator"] == 0
    assert metrics["harm_avoidance"] is None
    assert metrics["tie_agreement_denominator"] == 0
    assert metrics["tie_agreement"] is None
    assert metrics["false_apply_denominator"] == 0
    assert metrics["false_apply_fraction"] is None

    corrupt = [
        SavedGateRecord(BlockKey("p", 0, 1), 16, 2, 0.0, True, True)
    ]
    with pytest.raises(ValueError, match=r"\[0, 15\]"):
        reconstruct_saved_gate_evaluation(corrupt)
    nonfinite = [
        SavedGateRecord(BlockKey("p", 0, 1), 1, 2, float("nan"), True, True)
    ]
    with pytest.raises(ValueError, match="finite"):
        reconstruct_saved_gate_evaluation(nonfinite)
    bad_first_witness = [
        SavedGateRecord(BlockKey("p", 0, 1), 1, 2, 0.0, False, True)
    ]
    with pytest.raises(ValueError, match="first-token witness"):
        reconstruct_saved_gate_evaluation(bad_first_witness)
    bad_zero_witness = [
        SavedGateRecord(BlockKey("p", 0, 1), 0, 2, 0.0, True, True)
    ]
    with pytest.raises(ValueError, match="first-token witness"):
        reconstruct_saved_gate_evaluation(bad_zero_witness)
    duplicate = only_benefit + only_benefit
    with pytest.raises(ValueError, match="unique"):
        reconstruct_saved_gate_evaluation(duplicate)
    no_oracle_gap = [
        SavedGateRecord(BlockKey("p", 0, 1), 2, 2, 0.0, True, True)
    ]
    with pytest.raises(ValueError, match="positive"):
        reconstruct_saved_gate_evaluation(no_oracle_gap)
    nonstrict = reconstruct_saved_gate_evaluation(
        no_oracle_gap, require_valid_recovery=False
    )
    assert nonstrict.metrics["recovery_denominator"] == 0.0
    assert nonstrict.metrics["recovery_numerator"] == 0.0
    assert nonstrict.metrics["oracle_recovery"] is None
    ineligible = {
        **nonstrict.metrics,
        "values_finite": True,
        "gradients_finite": True,
    }
    assert fit_checkpoint_selection_key(ineligible) is None
    with pytest.raises(RuntimeError, match="no recovery-valid"):
        selected_fit_checkpoint([{"pass": 0, "checkpoint": ineligible}])


def test_saved_replay_can_report_negative_recovery_for_checkpoint_ineligibility() -> None:
    records = [
        SavedGateRecord(BlockKey("benefit", 0, 1), 0, 1, 0.0, False, True),
        SavedGateRecord(BlockKey("harm", 0, 1), 1, 0, 1.0, True, False),
    ]
    with pytest.raises(ValueError, match="outside"):
        reconstruct_saved_gate_evaluation(records)
    report = reconstruct_saved_gate_evaluation(
        records, require_valid_recovery=False
    )
    assert report.metrics["oracle_recovery"] == pytest.approx(-1.0)


def _fit_checkpoint_metrics(
    *, eal: float, harmed: int, hinge: float, recovery: float = 0.5
) -> dict[str, float | int | bool]:
    return {
        "base_eal": 1.0,
        "method_eal": eal,
        "oracle_eal": 3.0,
        "recovery_numerator": eal - 1.0,
        "recovery_denominator": 2.0,
        "oracle_recovery": recovery,
        "harmed_numerator": harmed,
        "prompt_weighted_gain_hinge": hinge,
        "values_finite": True,
        "gradients_finite": True,
    }


def test_fit_checkpoint_selection_is_strict_lexicographic_and_earliest() -> None:
    identity = _fit_checkpoint_metrics(eal=1.0, harmed=0, hinge=0.4, recovery=0.0)
    first = _fit_checkpoint_metrics(eal=2.0, harmed=3, hinge=0.2)
    same_eal_safer = _fit_checkpoint_metrics(eal=2.0, harmed=2, hinge=0.3)
    same_eal_harm_lower_hinge = _fit_checkpoint_metrics(
        eal=2.0, harmed=2, hinge=0.1
    )
    tied = dict(same_eal_harm_lower_hinge)
    history = [
        {"pass": 0, "checkpoint": identity},
        {"pass": 1, "checkpoint": first},
        {"pass": 2, "checkpoint": same_eal_safer},
        {"pass": 3, "checkpoint": same_eal_harm_lower_hinge},
        {"pass": 4, "checkpoint": tied},
    ]
    assert fit_checkpoint_selection_key(first) == (2.0, -3.0, -0.2)
    assert selected_fit_checkpoint(history)["pass"] == 3

    invalid = dict(first)
    invalid["method_eal"] = 0.5
    invalid["recovery_numerator"] = -0.5
    invalid["oracle_recovery"] = -0.25
    assert fit_checkpoint_selection_key(invalid) is None
    with pytest.raises(RuntimeError, match="no recovery-valid"):
        selected_fit_checkpoint([{"pass": 0, "checkpoint": invalid}])


def test_weighted_ridge_binding_1e3_numeric_reference() -> None:
    x = torch.full((5, 21), 7.0)
    x[:, 0] = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    y = 2.0 + 3.0 * x[:, 0]
    weights = torch.tensor([1.0, 2.0, 1.0, 2.0, 1.0])
    model = fit_weighted_ridge(x, y, weights)
    assert model.feature_mean.dtype == torch.float64
    assert model.constant_features.tolist() == [False] + [True] * 20
    prediction = model.predict(x)
    expected_slope = 3.0 / 1.001
    expected = 2.0 + expected_slope * x[:, 0].double()
    torch.testing.assert_close(prediction, expected, atol=1e-12, rtol=0)
    assert model.ridge == 1e-3
    assert model.intercept.item() == pytest.approx(2.0, abs=1e-12)
    assert model.coefficients[0].item() == pytest.approx(
        3.0 * math.sqrt(12.0 / 7.0) / 1.001,
        abs=1e-12,
    )
    assert torch.equal(model.coefficients[1:], torch.zeros(20, dtype=torch.float64))
    with pytest.raises(ValueError, match="exactly 21"):
        fit_weighted_ridge(x[:, :2], y, weights)
    with pytest.raises(TypeError):
        fit_weighted_ridge(x, y, weights, ridge=0.5)  # type: ignore[call-arg]


def test_deterministic_bootstrap_has_frozen_counter_semantics() -> None:
    first = deterministic_bootstrap_indices(
        7, replicates=5, seed=20260805, prompt_set_sha256="a" * 64
    )
    second = deterministic_bootstrap_indices(
        7, replicates=5, seed=20260805, prompt_set_sha256="a" * 64
    )
    other = deterministic_bootstrap_indices(
        7, replicates=5, seed=20260806, prompt_set_sha256="a" * 64
    )
    assert torch.equal(first, second)
    assert not torch.equal(first, other)
    assert first.shape == (5, 7)
    assert int(first.min()) >= 0
    assert int(first.max()) < 7
