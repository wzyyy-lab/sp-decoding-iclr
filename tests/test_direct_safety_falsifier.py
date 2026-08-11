from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import signal
import tempfile
from types import SimpleNamespace
from typing import Callable

import pytest
import torch

import scripts.evaluate_direct_safety_falsifier as falsifier
import sph.direct_safety_publication as publication
from sph.direct_safety_artifacts import sha256_file
from sph.direct_safety_gate import direct_safety_position_features
from sph.direct_safety_numeric_policy import (
    NUMERIC_POLICY_ID,
    NUMERIC_POLICY_SHA256,
)
from sph.direct_safety_publication import (
    FALSIFIER_PURPOSE,
    PUBLICATION_READY_NAME,
    commit_publication,
    publication_identity,
    ready_pending_path,
    reserve_publication_directory,
    verify_published_directory,
)
from sph.global_direct_selector import GlobalDirectOutput


ENTRYPOINT = "scripts/evaluate_direct_safety_falsifier.py"


def _publication_fixture(
    tmp_path: Path,
    *,
    scientific_status: str = "FAIL",
    include_metrics: bool = True,
) -> tuple[Path, dict[str, object]]:
    source = tmp_path / "source"
    (source / "scripts").mkdir(parents=True)
    entrypoint = source / ENTRYPOINT
    entrypoint.write_text("print('frozen')\n", encoding="utf-8")
    source_manifest = source / "SOURCE_MANIFEST.json"
    source_manifest.write_text(
        json.dumps(
            {
                "protocol": "pros-gate-first-party-source-closure-v1",
                "roots": ["scripts/*.py", "src/sph/**/*.py"],
                "files": [
                    {
                        "path": ENTRYPOINT,
                        "bytes": entrypoint.stat().st_size,
                        "sha256": sha256_file(entrypoint),
                    }
                ],
            },
            sort_keys=True,
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )
    output = tmp_path / "pros_gate_falsifier_123" / "seed0"
    identity = publication_identity(
        output,
        job_id="123",
        purpose=FALSIFIER_PURPOSE,
        entrypoint_path=ENTRYPOINT,
        entrypoint_sha256=sha256_file(entrypoint),
        wrapper_sha256="b" * 64,
        source_manifest_sha256=sha256_file(source_manifest),
    )
    reserve_publication_directory(output, identity)
    snapshot = output / "source_snapshot"
    (snapshot / "scripts").mkdir(parents=True)
    (snapshot / "SOURCE_MANIFEST.json").write_bytes(source_manifest.read_bytes())
    (snapshot / ENTRYPOINT).write_bytes(entrypoint.read_bytes())
    (output / "payload").mkdir()
    (output / "payload/value.txt").write_text("immutable\n", encoding="utf-8")
    binding: dict[str, object] = {
        "identity": identity,
        "scientific_status": scientific_status,
        "input_identities_end": {"input": {"sha256": "c" * 64}},
        "source_closure_end": {
            "protocol": "pros-gate-first-party-source-closure-v1",
            "source_manifest_sha256": sha256_file(source_manifest),
            "source_file_count": 1,
            "source_entries_sha256": "d" * 64,
        },
    }
    if include_metrics:
        (output / "metrics.json").write_text(
            json.dumps(
                {
                    "scientific_status": binding["scientific_status"],
                    "input_identities_end": binding["input_identities_end"],
                    "source_closure_end": binding["source_closure_end"],
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
            encoding="utf-8",
        )
    return output, binding


def test_r083_publication_ready_is_atomic_and_science_separate(tmp_path: Path) -> None:
    output, binding = _publication_fixture(tmp_path, scientific_status="FAIL")
    assert not (output / PUBLICATION_READY_NAME).exists()
    summary = commit_publication(output, binding)
    assert summary == verify_published_directory(output, expected_binding=binding)
    assert summary["status"] == "READY"
    pending = ready_pending_path(output)
    ready = output / PUBLICATION_READY_NAME
    assert pending.stat().st_dev == ready.stat().st_dev
    assert pending.stat().st_ino == ready.stat().st_ino
    assert pending.stat().st_nlink == ready.stat().st_nlink == 2
    assert pending.stat().st_mode & 0o777 == ready.stat().st_mode & 0o777 == 0o400


def test_r083_publication_refuses_collision_and_mutation(tmp_path: Path) -> None:
    output, binding = _publication_fixture(tmp_path)
    with pytest.raises(FileExistsError, match="overwrite R083 run"):
        reserve_publication_directory(output, binding["identity"])
    commit_publication(output, binding)
    (output / "payload/value.txt").write_text("mutated\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="payload tree differs"):
        verify_published_directory(output, expected_binding=binding)


@pytest.mark.parametrize(
    "mutation", ["missing_pending", "copied_ready", "third_link", "wrong_mode"]
)
def test_r083_consumer_rejects_broken_ready_commit_link(
    tmp_path: Path, mutation: str
) -> None:
    output, binding = _publication_fixture(tmp_path)
    commit_publication(output, binding)
    ready = output / PUBLICATION_READY_NAME
    pending = ready_pending_path(output)
    if mutation == "missing_pending":
        pending.unlink()
    elif mutation == "copied_ready":
        payload = ready.read_bytes()
        ready.unlink()
        ready.write_bytes(payload)
        ready.chmod(0o400)
    elif mutation == "third_link":
        os.link(pending, output.parent / "third-ready-link")
    else:
        ready.chmod(0o600)
    with pytest.raises(RuntimeError, match="pending|hard-link|mode"):
        verify_published_directory(output, expected_binding=binding)


@pytest.mark.parametrize("kind", ["file", "directory", "symlink"])
def test_r083_directory_reservation_is_exclusive_for_existing_targets(
    tmp_path: Path, kind: str
) -> None:
    output = tmp_path / "existing"
    if kind == "file":
        output.write_text("winner\n", encoding="utf-8")
    elif kind == "directory":
        output.mkdir()
        (output / "winner").write_text("preserve\n", encoding="utf-8")
    else:
        target = tmp_path / "winner"
        target.mkdir()
        output.symlink_to(target, target_is_directory=True)
    identity = publication_identity(
        output,
        job_id="unit-test",
        purpose=publication.SMOKE_PURPOSE,
        entrypoint_path=ENTRYPOINT,
        entrypoint_sha256="a" * 64,
        wrapper_sha256="b" * 64,
        source_manifest_sha256="c" * 64,
    )
    with pytest.raises(FileExistsError, match="overwrite R083 run"):
        reserve_publication_directory(output, identity)
    assert output.exists() or output.is_symlink()


def test_r083_ready_race_preserves_competitor_and_complete_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output, binding = _publication_fixture(tmp_path)
    real_link = publication._link_ready_no_replace

    def raced_link(pending: Path, destination: Path) -> None:
        (destination / PUBLICATION_READY_NAME).write_text(
            "concurrent-winner\n", encoding="utf-8"
        )
        real_link(pending, destination)

    monkeypatch.setattr(publication, "_link_ready_no_replace", raced_link)
    with pytest.raises(FileExistsError, match="overwrite R083 READY"):
        commit_publication(output, binding)
    assert (output / PUBLICATION_READY_NAME).read_text(encoding="utf-8") == (
        "concurrent-winner\n"
    )
    assert (output / "payload/value.txt").read_text(encoding="utf-8") == (
        "immutable\n"
    )
    assert (output / publication.PUBLICATION_MANIFEST_NAME).is_file()
    assert ready_pending_path(output).is_file()


@pytest.mark.parametrize(
    "mutation",
    ["missing_ready", "truncated_ready", "missing_payload", "extra", "tamper", "symlink"],
)
def test_r083_consumer_rejects_incomplete_or_mutated_tree(
    tmp_path: Path, mutation: str
) -> None:
    output, binding = _publication_fixture(tmp_path)
    commit_publication(output, binding)
    if mutation == "missing_ready":
        (output / PUBLICATION_READY_NAME).unlink()
    elif mutation == "truncated_ready":
        ready = output / PUBLICATION_READY_NAME
        ready.chmod(0o600)
        ready.write_text("{", encoding="utf-8")
    elif mutation == "missing_payload":
        (output / "payload/value.txt").unlink()
    elif mutation == "extra":
        (output / "extra").write_text("unexpected\n", encoding="utf-8")
    elif mutation == "tamper":
        (output / "payload/value.txt").write_text("changed\n", encoding="utf-8")
    else:
        (output / "unexpected-link").symlink_to(output / "metrics.json")
    with pytest.raises((RuntimeError, json.JSONDecodeError)):
        verify_published_directory(output, expected_binding=binding)


@pytest.mark.parametrize("failure_point", ["tree_fsync", "pending", "link", "after_link"])
def test_r083_publication_failure_injection_never_deletes_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_point: str,
) -> None:
    output, binding = _publication_fixture(tmp_path)
    if failure_point == "tree_fsync":
        monkeypatch.setattr(
            publication,
            "_fsync_publication_tree",
            lambda *_: (_ for _ in ()).throw(OSError("injected tree fsync")),
        )
    elif failure_point == "pending":
        monkeypatch.setattr(
            publication,
            "_write_ready_pending",
            lambda *_: (_ for _ in ()).throw(OSError("injected pending")),
        )
    elif failure_point == "link":
        monkeypatch.setattr(
            publication,
            "_link_ready_no_replace",
            lambda *_: (_ for _ in ()).throw(OSError("injected link")),
        )
    else:

        def link_then_fail(pending: Path, destination: Path) -> None:
            os.link(
                pending,
                destination / PUBLICATION_READY_NAME,
                follow_symlinks=False,
            )
            raise OSError("injected final directory fsync")

        monkeypatch.setattr(publication, "_link_ready_no_replace", link_then_fail)
    with pytest.raises(OSError, match="injected"):
        commit_publication(output, binding)
    assert output.is_dir()
    assert (output / "metrics.json").is_file()
    assert (output / "payload/value.txt").is_file()


def _valid_outcome_record() -> dict[str, object]:
    logits = -torch.arange(16, dtype=torch.float32)[None].expand(15, -1).clone()
    full_lse = torch.logsumexp(logits, dim=-1) + 0.75
    base = logits - full_lse[:, None]
    residual = torch.zeros_like(base)
    residual[0, 1] = 8.0
    residual[0] -= residual[0].mean()
    scores = base + residual
    direct = GlobalDirectOutput(
        scores=scores[None],
        log_probs=torch.log_softmax(scores[None], dim=-1),
        residual_scores=residual[None],
        base_log_probs=base[None],
    )
    states = torch.randn(
        1, 15, 16, 64, generator=torch.Generator().manual_seed(20260806)
    )
    features = direct_safety_position_features(
        states, direct, logits[None], full_lse[None]
    )
    ids = (
        10_000
        + torch.arange(15, dtype=torch.int64)[:, None] * 100
        + torch.arange(16, dtype=torch.int64)[None]
    )
    gold = ids[:, 0].clone()
    gold[0] = ids[0, 1]
    return {
        "sample_id": "synthetic:falsifier",
        "anchor_offset": 0,
        "context_length": 64,
        "domain": "synthetic",
        "source_split": "train",
        "split": "falsifier",
        "numeric_policy_id": NUMERIC_POLICY_ID,
        "numeric_policy_sha256": NUMERIC_POLICY_SHA256,
        "position_features": features.position_features[0],
        "direct_path": features.direct_path[0],
        "change_mask": features.change_mask[0],
        "candidate_ids": ids,
        "gold_ids": gold,
        "candidate_logits": logits,
        "base_logsumexp": full_lse,
        "direct_scores": scores,
        "direct_residual_scores": residual,
        "base_log_probs": base,
        "base_length": 0,
        "direct_length": 15,
        "base_first_token_correct": False,
        "direct_first_token_correct": True,
        "normalized_gain": 1.0,
    }


def test_r083_outcome_roundtrip_uses_tensor_aware_exact_equality(
    tmp_path: Path,
) -> None:
    record = _valid_outcome_record()
    metadata = falsifier._write_outcome_bundle(
        tmp_path / "bundle",
        [record],
        split_manifest_sha256="a" * 64,
        provenance={"test": "tensor-bearing-roundtrip"},
    )
    reloaded, reloaded_metadata = falsifier.load_outcome_bundle(
        tmp_path / "bundle", expected_split="falsifier"
    )
    assert reloaded_metadata == metadata
    assert falsifier._nested_exact_equal(reloaded, [record])
    changed = [{**reloaded[0], "candidate_ids": reloaded[0]["candidate_ids"].clone()}]
    changed[0]["candidate_ids"][0, 0] += 1
    assert not falsifier._nested_exact_equal(reloaded, changed)


@pytest.mark.parametrize(
    "case", ["nonfinite_pros", "nonfinite_ridge", "regret_violation"]
)
def test_r083_scientific_evaluation_failures_commit_ready_exit2_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    case: str,
) -> None:
    output, fixture_binding = _publication_fixture(
        tmp_path, include_metrics=False
    )
    outcomes = output / "falsifier_outcomes"
    outcomes.mkdir()
    (outcomes / "metadata.json").write_text("{}", encoding="utf-8")
    torch.save([{"synthetic": True}], outcomes / "records.pt")
    source_summary = dict(fixture_binding["source_closure_end"])
    closure = SimpleNamespace(summary=lambda: source_summary)
    input_identity = {"synthetic": {"sha256": "e" * 64}}
    monkeypatch.setattr(falsifier, "verify_source_manifest", lambda *a, **k: closure)
    monkeypatch.setattr(
        falsifier, "_capture_input_identities", lambda _: input_identity
    )
    monkeypatch.setattr(
        falsifier.torch.cuda, "get_device_name", lambda _: "synthetic-device"
    )
    passing_pros = {
        "values_finite": True,
        "regret_bound_violation_count": 0,
        "recovery_denominator": 1.0,
        "oracle_recovery": 0.90,
        "method_eal": 5.2,
        "base_eal": 5.0,
        "direct_eal": 5.1,
        "harmed_fraction": 0.05,
        "method_first_token_count": 99,
        "direct_first_token_count": 100,
    }
    if case == "nonfinite_pros":
        reason = "nonfinite_pros_values"
        system = "pros"
        error: BaseException | None = ValueError("sidecar scores must be finite")
        scores = None
        metrics = None
    elif case == "nonfinite_ridge":
        reason = "nonfinite_ridge_values"
        system = "ridge"
        error = None
        scores = {
            "pros": torch.tensor([1.0]),
            "ridge": torch.tensor([float("inf")]),
        }
        metrics = {"pros": passing_pros}
    else:
        reason = "pros_regret_bound_violation"
        system = "pros"
        error = RuntimeError("saved-record evaluation violated the regret bound")
        scores = {"pros": torch.tensor([1.0])}
        metrics = None

    report, scientific_pass = falsifier._publish_scientific_evaluation_failure(
        args=SimpleNamespace(
            source_manifest=tmp_path / "SOURCE_MANIFEST.json",
            expected_source_manifest_sha256="f" * 64,
        ),
        output=output,
        identity=fixture_binding["identity"],
        closure_start=closure,
        input_identities_start=input_identity,
        job_id="123",
        start=0.0,
        device=torch.device("cpu"),
        reason=reason,
        system=system,
        error=error,
        frozen_bindings={
            "r082_publication": {"status": "READY"},
            "split_audit": {"status": "BOUND"},
        },
        native_witness={
            "regular_vs_hooked_outputs_bitwise": True,
            "hooked_repeat_outputs_bitwise": True,
            "hooked_repeat_node_states_bitwise": True,
            "state_dict_sha256_before": "a" * 64,
            "state_dict_sha256_after": "a" * 64,
        },
        outcome_metadata={"summary": {"blocks": 1}},
        selected_payload={
            "protocol": "pros-gate-fit-checkpoint-training-v1",
            "pass": 5,
            "completed_updates": 995,
        },
        ridge_payload={
            "protocol": "pros-gate-fit-ridge-freeze-v1",
            "feature_dimension": 21,
            "ridge": 0.001,
        },
        selected_gradients_finite=True,
        scores_by_system=scores,
        metrics_by_system=metrics,
    )
    assert scientific_pass is False
    assert report["scientific_status"] == "FAIL"
    assert report["scientific_failure"]["reason"] == reason
    assert report["gate_checks"]["exact_identity_and_data_boundary"] is True
    assert verify_published_directory(output)["status"] == "READY"


def test_only_preregistered_scientific_evaluation_errors_are_intercepted() -> None:
    assert falsifier._scientific_evaluation_failure(
        ValueError("sidecar scores must be finite"), system="pros"
    ) == "nonfinite_pros_values"
    assert falsifier._scientific_evaluation_failure(
        FloatingPointError("saved gate scores are non-finite"), system="ridge"
    ) == "nonfinite_ridge_values"
    assert falsifier._scientific_evaluation_failure(
        RuntimeError("saved-record evaluation violated the regret bound"),
        system="pros",
    ) == "pros_regret_bound_violation"
    assert (
        falsifier._scientific_evaluation_failure(
            RuntimeError("unrelated operational failure"), system="pros"
        )
        is None
    )


def _fork_barrier_results(actions: list[Callable[[], None]]) -> list[int]:
    read_descriptor, write_descriptor = os.pipe()
    children: list[int] = []
    for action in actions:
        process_id = os.fork()
        if process_id == 0:
            os.close(write_descriptor)
            try:
                os.read(read_descriptor, 1)
                action()
            except FileExistsError:
                os._exit(17)
            except BaseException:
                os._exit(99)
            os._exit(0)
        children.append(process_id)
    os.close(read_descriptor)
    os.write(write_descriptor, b"x" * len(children))
    os.close(write_descriptor)
    result: list[int] = []
    for process_id in children:
        _, status = os.waitpid(process_id, 0)
        result.append(os.waitstatus_to_exitcode(status))
    return sorted(result)


@pytest.mark.skipif(
    "PROS_R083_GPFS_PUBLICATION_ROOT" not in os.environ,
    reason="explicit real-GPFS R083 publication integration only",
)
def test_real_gpfs_r083_publication_races_and_crash_boundaries() -> None:
    integration_parent = Path(
        os.environ["PROS_R083_GPFS_PUBLICATION_ROOT"]
    ).resolve()
    integration_parent.mkdir(parents=True, exist_ok=True)
    root = Path(
        tempfile.mkdtemp(prefix="pros-r083-publication-integration.", dir=integration_parent)
    )
    try:
        clean_root = root / "clean"
        clean_root.mkdir()
        clean, clean_binding = _publication_fixture(clean_root)
        clean_summary = commit_publication(clean, clean_binding)
        assert verify_published_directory(
            clean, expected_binding=clean_binding
        ) == clean_summary

        mkdir_root = root / "mkdir-race"
        mkdir_root.mkdir()
        mkdir_output = mkdir_root / "winner"
        mkdir_identity = publication_identity(
            mkdir_output,
            job_id="gpfs-mkdir-race",
            purpose=publication.SMOKE_PURPOSE,
            entrypoint_path=ENTRYPOINT,
            entrypoint_sha256="a" * 64,
            wrapper_sha256="b" * 64,
            source_manifest_sha256="c" * 64,
        )
        assert _fork_barrier_results(
            [
                lambda: reserve_publication_directory(mkdir_output, mkdir_identity),
                lambda: reserve_publication_directory(mkdir_output, mkdir_identity),
            ]
        ) == [0, 17]

        link_root = root / "ready-race"
        link_output = link_root / "output"
        link_identity = publication_identity(
            link_output,
            job_id="gpfs-ready-race",
            purpose=publication.SMOKE_PURPOSE,
            entrypoint_path=ENTRYPOINT,
            entrypoint_sha256="a" * 64,
            wrapper_sha256="b" * 64,
            source_manifest_sha256="c" * 64,
        )
        reserve_publication_directory(link_output, link_identity)
        pending_paths = [link_root / "pending-a", link_root / "pending-b"]
        for index, pending in enumerate(pending_paths):
            with pending.open("xb") as handle:
                handle.write(f"candidate-{index}\n".encode("utf-8"))
                handle.flush()
                os.fsync(handle.fileno())
        publication._fsync_directory(link_root)
        assert _fork_barrier_results(
            [
                lambda: publication._link_ready_no_replace(
                    pending_paths[0], link_output
                ),
                lambda: publication._link_ready_no_replace(
                    pending_paths[1], link_output
                ),
            ]
        ) == [0, 17]
        assert (link_output / PUBLICATION_READY_NAME).read_text(
            encoding="utf-8"
        ) in {"candidate-0\n", "candidate-1\n"}

        before_root = root / "crash-before-link"
        before_root.mkdir()
        before_output, before_binding = _publication_fixture(before_root)
        before_read, before_write = os.pipe()
        before_child = os.fork()
        if before_child == 0:
            os.close(before_read)

            def stop_before_link(_: Path, __: Path) -> None:
                os.write(before_write, b"1")
                signal.pause()

            publication._link_ready_no_replace = stop_before_link
            commit_publication(before_output, before_binding)
            os._exit(99)
        os.close(before_write)
        assert os.read(before_read, 1) == b"1"
        os.close(before_read)
        os.kill(before_child, signal.SIGKILL)
        os.waitpid(before_child, 0)
        assert before_output.is_dir()
        assert not (before_output / PUBLICATION_READY_NAME).exists()
        assert (before_output / publication.PUBLICATION_MANIFEST_NAME).is_file()
        assert ready_pending_path(before_output).is_file()

        after_root = root / "crash-after-link"
        after_root.mkdir()
        after_output, after_binding = _publication_fixture(after_root)
        after_read, after_write = os.pipe()
        after_child = os.fork()
        if after_child == 0:
            os.close(after_read)

            def stop_before_final_fsync(pending: Path, destination: Path) -> None:
                os.link(
                    pending,
                    destination / PUBLICATION_READY_NAME,
                    follow_symlinks=False,
                )
                os.write(after_write, b"1")
                signal.pause()

            publication._link_ready_no_replace = stop_before_final_fsync
            commit_publication(after_output, after_binding)
            os._exit(99)
        os.close(after_write)
        assert os.read(after_read, 1) == b"1"
        os.close(after_read)
        os.kill(after_child, signal.SIGKILL)
        os.waitpid(after_child, 0)
        assert after_output.is_dir()
        assert (after_output / PUBLICATION_READY_NAME).is_file()
        assert ready_pending_path(after_output).stat().st_ino == (
            after_output / PUBLICATION_READY_NAME
        ).stat().st_ino
    finally:
        shutil.rmtree(root)


def test_frozen_gate_checks_are_exactly_conjunctive() -> None:
    pros = {
        "values_finite": True,
        "regret_bound_violation_count": 0,
        "recovery_denominator": 1.0,
        "oracle_recovery": 0.90,
        "method_eal": 5.2,
        "base_eal": 5.0,
        "direct_eal": 5.1,
        "harmed_fraction": 0.05,
        "method_first_token_count": 99,
        "direct_first_token_count": 100,
    }
    comparators = {
        "ridge": {"oracle_recovery": 0.85, "recovery_denominator": 1.0},
        "always_keep": {"oracle_recovery": 0.0, "recovery_denominator": 1.0},
        "always_direct": {"oracle_recovery": 0.50, "recovery_denominator": 1.0},
    }
    checks = falsifier._gate_checks(
        pros,
        comparators,
        identity_checks_passed=True,
        gradients_finite=True,
    )
    assert all(checks.values())
    failed = dict(pros)
    failed["harmed_fraction"] = 0.0500000001
    assert falsifier._gate_checks(
        failed,
        comparators,
        identity_checks_passed=True,
        gradients_finite=True,
    )["harmed_fraction_at_most_0p05"] is False


def test_exact_linear_quantiles_have_frozen_interpolation() -> None:
    values = [0.0, 10.0, 20.0, 30.0]
    assert falsifier._linear_quantile(values, 0.0) == 0.0
    assert falsifier._linear_quantile(values, 0.25) == 7.5
    assert falsifier._linear_quantile(values, 0.5) == 15.0
    assert falsifier._linear_quantile(values, 1.0) == 30.0
    with pytest.raises(ValueError):
        falsifier._linear_quantile([], 0.5)


def _bootstrap_records() -> list[dict[str, object]]:
    return [
        {
            "sample_id": "p0",
            "anchor_offset": 0,
            "context_length": 1,
            "base_length": 1,
            "direct_length": 2,
            "base_first_token_correct": True,
            "direct_first_token_correct": True,
        },
        {
            "sample_id": "p1",
            "anchor_offset": 0,
            "context_length": 1,
            "base_length": 2,
            "direct_length": 1,
            "base_first_token_correct": True,
            "direct_first_token_correct": True,
        },
    ]


def test_prompt_cluster_bootstrap_is_deterministic(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(falsifier, "FALSIFIER_PROMPTS", 2)
    monkeypatch.setattr(falsifier, "BOOTSTRAP_REPLICATES", 20)
    scores = {
        "pros": torch.tensor([1.0, -1.0]),
        "always_direct": torch.ones(2),
    }
    first = falsifier._bootstrap_report(
        _bootstrap_records(), scores, prompt_set_sha256="a" * 64
    )
    second = falsifier._bootstrap_report(
        _bootstrap_records(), scores, prompt_set_sha256="a" * 64
    )
    assert first == second
    assert first["replicates"] == 20
    assert first["systems"]["pros"]["harmed_fraction_ci95"] == [0.0, 0.0]


def test_zero_denominator_bootstrap_is_reported_without_operational_crash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(falsifier, "FALSIFIER_PROMPTS", 2)
    monkeypatch.setattr(falsifier, "BOOTSTRAP_REPLICATES", 20)
    records = _bootstrap_records()
    for row in records:
        row["direct_length"] = row["base_length"]
    report = falsifier._bootstrap_report(
        records,
        {"pros": torch.ones(2)},
        prompt_set_sha256="b" * 64,
    )
    pros = report["systems"]["pros"]
    assert pros["oracle_recovery_ci95"] is None
    assert pros["valid_recovery_replicates"] == 0
    assert pros["invalid_recovery_replicates"] == 20

    point = {
        "values_finite": True,
        "regret_bound_violation_count": 0,
        "recovery_denominator": 0.0,
        "oracle_recovery": None,
        "method_eal": 2.0,
        "base_eal": 2.0,
        "direct_eal": 2.0,
        "harmed_fraction": 0.0,
        "method_first_token_count": 2,
        "direct_first_token_count": 2,
    }
    checks = falsifier._gate_checks(
        point,
        {},
        identity_checks_passed=True,
        gradients_finite=True,
    )
    assert checks["finite_positive_unclipped_recovery"] is False
    assert checks["recovery_at_least_0p90"] is False


def _small_frozen_inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    data = tmp_path / "data"
    target = tmp_path / "target"
    data.mkdir()
    target.mkdir()
    shard_paths = [data / "shard-00000.pt", data / "shard-00001.pt"]
    for index, path in enumerate(shard_paths):
        path.write_bytes(f"shard-{index}\n".encode("utf-8"))
    (target / "config.json").write_text("{}\n", encoding="utf-8")
    (target / "embedding.safetensors").write_bytes(b"frozen-embedding\n")
    (target / "model.safetensors.index.json").write_text(
        json.dumps(
            {"weight_map": {"model.embed_tokens.weight": "embedding.safetensors"}}
        ),
        encoding="utf-8",
    )

    def identity(path: Path) -> dict[str, object]:
        return {
            "path": path.name,
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }

    metadata_path = data / "metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "shards": [
                    {**identity(path), "blocks": 1} for path in shard_paths
                ],
                "provenance": {
                    "target_files": [
                        identity(target / name)
                        for name in (
                            "config.json",
                            "embedding.safetensors",
                            "model.safetensors.index.json",
                        )
                    ]
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return data, target, metadata_path


def test_canonical_and_target_input_identities_fail_closed_on_mutation(
    tmp_path: Path,
) -> None:
    data, target, _ = _small_frozen_inputs(tmp_path)
    assert len(falsifier._canonical_shard_identities(data)) == 2
    target_identity = falsifier._target_embedding_identities(data, target)
    assert target_identity["embedding_shard"] == "embedding.safetensors"
    assert len(target_identity["files"]) == 3

    (data / "shard-00000.pt").write_bytes(b"tampered\n")
    with pytest.raises(RuntimeError, match="canonical shard identity"):
        falsifier._canonical_shard_identities(data)
    (target / "config.json").write_text('{"tampered":true}\n', encoding="utf-8")
    with pytest.raises(RuntimeError, match="target embedding input differs"):
        falsifier._target_embedding_identities(data, target)


def test_coordinated_input_mutation_changes_start_end_identity_chain(
    tmp_path: Path,
) -> None:
    data, target, metadata_path = _small_frozen_inputs(tmp_path)
    before = {
        "metadata": falsifier._file_identity(metadata_path, "metadata"),
        "shards": falsifier._canonical_shard_identities(data),
        "target": falsifier._target_embedding_identities(data, target),
    }
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    shard = data / "shard-00000.pt"
    shard.write_bytes(b"coordinated-new-shard\n")
    metadata["shards"][0]["bytes"] = shard.stat().st_size
    metadata["shards"][0]["sha256"] = sha256_file(shard)
    metadata_path.write_text(json.dumps(metadata, sort_keys=True), encoding="utf-8")
    after = {
        "metadata": falsifier._file_identity(metadata_path, "metadata"),
        "shards": falsifier._canonical_shard_identities(data),
        "target": falsifier._target_embedding_identities(data, target),
    }
    assert after != before


def test_r083_entrypoint_has_no_tuning_or_later_data_cli() -> None:
    source = (Path(__file__).parents[1] / ENTRYPOINT).read_text(encoding="utf-8")
    forbidden = (
        "--threshold",
        "--seed",
        "--checkpoint-pass",
        "--validation",
        "--reserved",
        "--formal",
        "--fit-bundle",
    )
    assert all(token not in source for token in forbidden)
    assert "--r082-output" in source
    assert "materialize_falsifier_outcome_records" in source
