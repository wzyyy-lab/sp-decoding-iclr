from __future__ import annotations

from collections.abc import Callable
import json
import math
import os
from pathlib import Path
import shutil
import signal
import tempfile

import pytest
import torch

import scripts.train_direct_safety_fit as fit_module
from scripts.train_direct_safety_fit import (
    BATCH_SIZE,
    FINAL_BATCH_SIZE,
    FIT_PROMPTS,
    FIT_RECORDS,
    PASSES,
    PEAK_LEARNING_RATE,
    STEPS_PER_PASS,
    TOTAL_UPDATES,
    WARMUP_UPDATES,
    _assert_input_identities_unchanged,
    _commit_publication,
    _link_ready_no_replace,
    _ready_pending_path,
    _reserve_publication_directory,
    _training_tensors,
    _write_json,
    build_fit_order_manifest,
    capture_input_identities,
    domain_slice_metrics,
    evaluate_model,
    evaluate_scores,
    fit_batch_sizes,
    fit_learning_rate,
    fit_minibatch_objective,
    freeze_ridge_from_fit,
    load_fit_order_manifest,
    publication_identity,
    verify_published_directory,
)
from sph.direct_safety_artifacts import sha256_file
from sph.direct_safety_gate import DirectSafetySidecar
from sph.direct_safety_protocol import (
    BlockKey,
    complete_pass_schedule,
    fit_checkpoint_selection_key,
)


@pytest.fixture(scope="module", autouse=True)
def _single_threaded_small_tensor_checks():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def _features(path: torch.Tensor, offset: float) -> torch.Tensor:
    values = torch.zeros(15, 200, dtype=torch.float32)
    values[:, 192] = offset
    values[:, 193] = offset / 2.0
    values[:, 194] = -offset / 3.0
    values[:, 196] = torch.arange(15, dtype=torch.float32) / 14.0
    values[:, 198] = 0.5
    values[:, 199] = -0.25
    values[:, 195] = path.float() / 15.0
    values[:, 197] = path.ne(0).float()
    return values


def _records() -> list[dict[str, object]]:
    outcomes = [
        ("p0", "chat", 0, 1),
        ("p0", "chat", 1, 0),
        ("p1", "code", 0, 1),
        ("p1", "code", 0, 0),
    ]
    result: list[dict[str, object]] = []
    for index, (sample_id, domain, base, direct) in enumerate(outcomes):
        path = torch.zeros(15, dtype=torch.int64)
        if index != 3:
            path[index] = 1
        result.append(
            {
                "sample_id": sample_id,
                "domain": domain,
                "anchor_offset": index,
                "context_length": 100 + index,
                "position_features": _features(path, 0.1 * (index + 1)),
                "direct_path": path,
                "change_mask": path.ne(0),
                "normalized_gain": (direct - base) / 15.0,
                "base_length": base,
                "direct_length": direct,
                "base_first_token_correct": base > 0,
                "direct_first_token_correct": direct > 0,
            }
        )
    return result


def test_frozen_r082_schedule_and_learning_rate_endpoints() -> None:
    schedule = complete_pass_schedule(
        FIT_RECORDS, batch_size=BATCH_SIZE, max_updates=5_120
    )
    assert schedule.records == FIT_RECORDS
    assert schedule.steps_per_pass == STEPS_PER_PASS == 199
    assert schedule.passes == PASSES == 25
    assert schedule.total_steps == TOTAL_UPDATES == 4_975
    assert schedule.warmup_steps == WARMUP_UPDATES == 199
    sizes = fit_batch_sizes()
    assert len(sizes) == STEPS_PER_PASS
    assert sizes == (BATCH_SIZE,) * 198 + (FINAL_BATCH_SIZE,)
    assert FINAL_BATCH_SIZE == 14
    assert sum(sizes) == FIT_RECORDS
    assert FIT_PROMPTS == 1_587
    assert fit_learning_rate(0) == pytest.approx(
        PEAK_LEARNING_RATE / WARMUP_UPDATES
    )
    assert fit_learning_rate(WARMUP_UPDATES - 1) == PEAK_LEARNING_RATE
    assert fit_learning_rate(WARMUP_UPDATES) < PEAK_LEARNING_RATE
    assert fit_learning_rate(TOTAL_UPDATES - 1) == pytest.approx(0.0)
    with pytest.raises(ValueError):
        fit_learning_rate(TOTAL_UPDATES)
    with pytest.raises(ValueError):
        fit_learning_rate(True)  # type: ignore[arg-type]


def test_prompt_mass_objective_handles_short_final_batch_exactly() -> None:
    scores = torch.tensor([0.0, 0.0, 0.0], requires_grad=True)
    gains = torch.tensor([1.0, -1.0, 0.0])
    weights = torch.tensor([0.75, 0.75, 1.5])
    loss, contribution = fit_minibatch_objective(
        scores,
        gains,
        weights,
        total_records=3,
        steps_per_pass=2,
    )
    assert contribution.item() == pytest.approx(0.5)
    assert loss.item() == pytest.approx(1.0)
    loss.backward()
    torch.testing.assert_close(scores.grad, torch.tensor([-0.5, 0.5, 0.0]))


def test_epoch_zero_replay_and_negative_recovery_are_recorded() -> None:
    records = _records()
    tensors = _training_tensors(
        records,
        torch.device("cpu"),
        expected_records=4,
        expected_prompts=2,
    )
    model = DirectSafetySidecar(initialization_seed=0)
    metrics, rows, scores = evaluate_model(
        model, records, tensors, gradients_finite=True
    )
    assert scores.tolist() == [0.0] * 4
    assert len(rows) == 4
    assert metrics["method_eal"] == metrics["base_eal"]
    assert metrics["oracle_recovery"] == 0.0
    assert fit_checkpoint_selection_key(metrics) is not None

    harmful_only = torch.tensor([-1.0, 1.0, -1.0, 0.0])
    negative, _ = evaluate_scores(
        records,
        tensors,
        harmful_only,
        values_finite=True,
        gradients_finite=True,
        verify_tensor_loss=False,
    )
    assert negative["oracle_recovery"] < 0.0
    assert fit_checkpoint_selection_key(negative) is None
    slices = domain_slice_metrics(
        records,
        scores,
        values_finite=True,
        gradients_finite=True,
    )
    assert set(slices) == {"chat", "code"}


def test_fit_orders_are_frozen_and_semantically_reloaded(tmp_path: Path) -> None:
    keys = [BlockKey(f"prompt:{index}", index, 100 + index) for index in range(4)]
    manifest = build_fit_order_manifest(keys, "a" * 64, passes=3)
    assert manifest["records"] == 4
    assert manifest["passes"] == 3
    assert len(manifest["orders"]) == 3
    path = tmp_path / "orders.json"
    _write_json(path, manifest)
    loaded = load_fit_order_manifest(
        path,
        expected_sha256=sha256_file(path),
        expected_training_manifest_sha256="a" * 64,
        expected_keys=keys,
        expected_passes=3,
    )
    assert all(set(order) == set(keys) for order in loaded)
    manifest["orders"][0]["block_keys"][0], manifest["orders"][0][
        "block_keys"
    ][1] = (
        manifest["orders"][0]["block_keys"][1],
        manifest["orders"][0]["block_keys"][0],
    )
    path.write_text(
        json.dumps(
            manifest,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="semantic hash"):
        load_fit_order_manifest(
            path,
            expected_sha256=sha256_file(path),
            expected_training_manifest_sha256="a" * 64,
            expected_keys=keys,
            expected_passes=3,
        )


def test_float64_ridge_is_frozen_reloaded_and_hashed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    records = _records()
    tensors = _training_tensors(
        records,
        torch.device("cpu"),
        expected_records=4,
        expected_prompts=2,
    )
    captured: dict[str, torch.Tensor] = {}
    original_fit = fit_module.fit_weighted_ridge

    def capture_fit(
        features: torch.Tensor,
        targets: torch.Tensor,
        weights: torch.Tensor,
    ):
        captured["targets"] = targets.clone()
        captured["weights"] = weights.clone()
        return original_fit(features, targets, weights)

    monkeypatch.setattr(fit_module, "fit_weighted_ridge", capture_fit)
    model, receipt = freeze_ridge_from_fit(
        records,
        tensors,
        output=tmp_path,
        source_manifest_sha256="b" * 64,
    )
    assert model.coefficients.dtype == torch.float64
    assert model.feature_mean.dtype == torch.float64
    assert model.ridge == 1e-3
    assert captured["targets"].dtype == torch.float64
    assert captured["weights"].dtype == torch.float64
    assert captured["targets"].tolist() == [
        float(row["normalized_gain"]) for row in records
    ]
    assert math.isfinite(float(model.intercept))
    assert receipt["ridge_model_sha256"] == sha256_file(
        tmp_path / "ridge_model.pt"
    )
    frozen = json.loads(
        (tmp_path / "ridge_freeze_receipt.json").read_text(encoding="utf-8")
    )
    assert frozen["status"] == "FROZEN_BEFORE_CHECKPOINT_LOAD"
    assert frozen["ridge_model_sha256"] == receipt["ridge_model_sha256"]


def test_fit_cli_and_source_order_have_no_later_data_surface() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "scripts/train_direct_safety_fit.py"
    ).read_text(encoding="utf-8")
    assert '"--fit-bundle"' in source
    assert '"--checkpoint-bundle"' in source
    for forbidden in (
        '"--falsifier',
        '"--validation',
        '"--reserved',
        '"--formal',
        '"--data"',
        '"--target"',
        '"--direct-run"',
    ):
        assert forbidden not in source
    freeze_call = source.index(
        "ridge_model, ridge_freeze = freeze_ridge_from_fit("
    )
    checkpoint_open = source.index(
        "checkpoint_records, checkpoint_metadata = load_outcome_bundle("
    )
    assert freeze_call < checkpoint_open


def _identity_fixture(tmp_path: Path) -> dict[str, Path]:
    paths = {
        "source_manifest": tmp_path / "source.json",
        "outcomes_audit_receipt": tmp_path / "outcomes.json",
        "capacity_adjudication_receipt": tmp_path / "capacity.json",
        "fit_bundle": tmp_path / "fit",
        "checkpoint_bundle": tmp_path / "checkpoint",
    }
    for name in (
        "source_manifest",
        "outcomes_audit_receipt",
        "capacity_adjudication_receipt",
    ):
        paths[name].write_text(f'{{"name":"{name}"}}\n', encoding="utf-8")
    for name in ("fit_bundle", "checkpoint_bundle"):
        paths[name].mkdir()
        (paths[name] / "metadata.json").write_text("{}\n", encoding="utf-8")
        (paths[name] / "records.pt").write_bytes(b"records\n")
    return paths


def test_end_identity_capture_rehashes_receipts(tmp_path: Path) -> None:
    paths = _identity_fixture(tmp_path)
    before = capture_input_identities(**paths)
    paths["outcomes_audit_receipt"].write_text(
        '{"name":"mutated"}\n', encoding="utf-8"
    )
    after = capture_input_identities(**paths)
    assert before["outcomes_audit_receipt"] != after["outcomes_audit_receipt"]
    with pytest.raises(RuntimeError, match="outcomes_audit_receipt"):
        _assert_input_identities_unchanged(before, after)


def _publication_fixture(
    tmp_path: Path, *, scientific_status: str = "FAIL"
) -> tuple[Path, dict[str, object]]:
    trainer = Path(fit_module.__file__).resolve()
    trainer_sha256 = sha256_file(trainer)
    source_manifest = tmp_path / "source-manifest.json"
    _write_json(
        source_manifest,
        {
            "files": [
                {
                    "path": "scripts/train_direct_safety_fit.py",
                    "bytes": trainer.stat().st_size,
                    "sha256": trainer_sha256,
                }
            ]
        },
    )
    output = tmp_path / "published"
    identity = publication_identity(
        output,
        job_id="unit-test",
        purpose="FILESYSTEM_SMOKE",
        wrapper_sha256="a" * 64,
        source_manifest_sha256=sha256_file(source_manifest),
    )
    _reserve_publication_directory(output, identity)
    snapshot = output / "source_snapshot"
    (snapshot / "scripts").mkdir(parents=True)
    shutil.copy2(source_manifest, snapshot / "SOURCE_MANIFEST.json")
    shutil.copy2(trainer, snapshot / "scripts/train_direct_safety_fit.py")
    binding: dict[str, object] = {
        "identity": identity,
        "scientific_status": scientific_status,
        "input_identities_end": {"fixture": {"sha256": "b" * 64}},
        "source_closure_end": {
            "source_manifest_sha256": sha256_file(source_manifest)
        },
    }
    _write_json(output / "metrics.json", binding)
    (output / "nested").mkdir()
    (output / "nested/evidence.txt").write_text("complete\n", encoding="utf-8")
    return output, binding


def test_directory_publication_is_ready_committed_and_science_separate(
    tmp_path: Path,
) -> None:
    output, binding = _publication_fixture(tmp_path, scientific_status="FAIL")
    summary = _commit_publication(output, binding)
    assert summary == verify_published_directory(
        output, expected_binding=binding
    )
    assert summary["status"] == "READY"
    assert json.loads((output / "metrics.json").read_text(encoding="utf-8"))[
        "scientific_status"
    ] == "FAIL"
    pending = _ready_pending_path(output)
    ready = output / fit_module.PUBLICATION_READY_NAME
    assert pending.stat().st_dev == ready.stat().st_dev
    assert pending.stat().st_ino == ready.stat().st_ino


@pytest.mark.parametrize("kind", ["file", "directory", "symlink"])
def test_directory_reservation_is_exclusive_for_every_existing_target(
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
        purpose="FILESYSTEM_SMOKE",
        wrapper_sha256="a" * 64,
        source_manifest_sha256="b" * 64,
    )
    with pytest.raises(FileExistsError, match="overwrite"):
        _reserve_publication_directory(output, identity)
    assert output.exists() or output.is_symlink()


def test_ready_race_preserves_competitor_and_complete_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output, binding = _publication_fixture(tmp_path)
    real_link = _link_ready_no_replace

    def raced_link(pending: Path, destination: Path) -> None:
        (destination / fit_module.PUBLICATION_READY_NAME).write_text(
            "concurrent-winner\n", encoding="utf-8"
        )
        real_link(pending, destination)

    monkeypatch.setattr(fit_module, "_link_ready_no_replace", raced_link)
    with pytest.raises(FileExistsError, match="overwrite"):
        _commit_publication(output, binding)
    assert (output / fit_module.PUBLICATION_READY_NAME).read_text(
        encoding="utf-8"
    ) == "concurrent-winner\n"
    assert (output / "nested/evidence.txt").read_text(encoding="utf-8") == (
        "complete\n"
    )
    assert (output / fit_module.PUBLICATION_MANIFEST_NAME).is_file()
    assert _ready_pending_path(output).is_file()


@pytest.mark.parametrize(
    "mutation",
    ["missing_ready", "truncated_ready", "missing_payload", "extra", "tamper", "symlink"],
)
def test_publication_consumer_rejects_every_incomplete_or_mutated_tree(
    tmp_path: Path, mutation: str
) -> None:
    output, binding = _publication_fixture(tmp_path)
    _commit_publication(output, binding)
    if mutation == "missing_ready":
        (output / fit_module.PUBLICATION_READY_NAME).unlink()
    elif mutation == "truncated_ready":
        ready = output / fit_module.PUBLICATION_READY_NAME
        ready.chmod(0o600)
        ready.write_text(
            "{", encoding="utf-8"
        )
    elif mutation == "missing_payload":
        (output / "nested/evidence.txt").unlink()
    elif mutation == "extra":
        (output / "extra").write_text("unexpected\n", encoding="utf-8")
    elif mutation == "tamper":
        (output / "nested/evidence.txt").write_text("changed\n", encoding="utf-8")
    else:
        (output / "unexpected-link").symlink_to(output / "metrics.json")
    with pytest.raises((RuntimeError, json.JSONDecodeError)):
        verify_published_directory(output, expected_binding=binding)


@pytest.mark.parametrize("failure_point", ["tree_fsync", "pending", "link", "after_link"])
def test_publication_failure_injection_never_deletes_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_point: str,
) -> None:
    output, binding = _publication_fixture(tmp_path)
    if failure_point == "tree_fsync":
        monkeypatch.setattr(
            fit_module,
            "_fsync_publication_tree",
            lambda *_: (_ for _ in ()).throw(OSError("injected tree fsync")),
        )
    elif failure_point == "pending":
        monkeypatch.setattr(
            fit_module,
            "_write_ready_pending",
            lambda *_: (_ for _ in ()).throw(OSError("injected pending")),
        )
    elif failure_point == "link":
        monkeypatch.setattr(
            fit_module,
            "_link_ready_no_replace",
            lambda *_: (_ for _ in ()).throw(OSError("injected link")),
        )
    else:
        def link_then_fail(pending: Path, destination: Path) -> None:
            os.link(pending, destination / fit_module.PUBLICATION_READY_NAME)
            raise OSError("injected final directory fsync")

        monkeypatch.setattr(fit_module, "_link_ready_no_replace", link_then_fail)
    with pytest.raises(OSError, match="injected"):
        _commit_publication(output, binding)
    assert output.is_dir()
    assert (output / "metrics.json").is_file()
    assert (output / "nested/evidence.txt").is_file()


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
    "PROS_GPFS_PUBLICATION_ROOT" not in os.environ,
    reason="explicit real-GPFS publication integration only",
)
def test_real_gpfs_publication_commit_races_and_crash_boundaries() -> None:
    integration_parent = Path(os.environ["PROS_GPFS_PUBLICATION_ROOT"]).resolve()
    integration_parent.mkdir(parents=True, exist_ok=True)
    root = Path(
        tempfile.mkdtemp(prefix="pros-r082-publication-integration.", dir=integration_parent)
    )
    try:
        clean_root = root / "clean"
        clean_root.mkdir()
        clean, clean_binding = _publication_fixture(clean_root)
        clean_summary = _commit_publication(clean, clean_binding)
        assert verify_published_directory(
            clean, expected_binding=clean_binding
        ) == clean_summary
        assert (clean / "nested/evidence.txt").stat().st_size > 0

        mkdir_root = root / "mkdir-race"
        mkdir_root.mkdir()
        mkdir_output = mkdir_root / "winner"
        mkdir_identity = publication_identity(
            mkdir_output,
            job_id="gpfs-mkdir-race",
            purpose="FILESYSTEM_SMOKE",
            wrapper_sha256="a" * 64,
            source_manifest_sha256="b" * 64,
        )
        assert _fork_barrier_results(
            [
                lambda: _reserve_publication_directory(
                    mkdir_output, mkdir_identity
                ),
                lambda: _reserve_publication_directory(
                    mkdir_output, mkdir_identity
                ),
            ]
        ) == [0, 17]
        assert (mkdir_output / fit_module.PUBLICATION_RESERVATION_NAME).is_file()

        link_root = root / "ready-race"
        link_output = link_root / "output"
        link_identity = publication_identity(
            link_output,
            job_id="gpfs-ready-race",
            purpose="FILESYSTEM_SMOKE",
            wrapper_sha256="a" * 64,
            source_manifest_sha256="b" * 64,
        )
        _reserve_publication_directory(link_output, link_identity)
        pending_paths = [link_root / "pending-a", link_root / "pending-b"]
        for index, pending in enumerate(pending_paths):
            with pending.open("xb") as handle:
                handle.write(f"candidate-{index}\n".encode("utf-8"))
                handle.flush()
                os.fsync(handle.fileno())
        fit_module._fsync_directory(link_root)
        assert _fork_barrier_results(
            [
                lambda: _link_ready_no_replace(pending_paths[0], link_output),
                lambda: _link_ready_no_replace(pending_paths[1], link_output),
            ]
        ) == [0, 17]
        assert (link_output / fit_module.PUBLICATION_READY_NAME).read_text(
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

            fit_module._link_ready_no_replace = stop_before_link
            _commit_publication(before_output, before_binding)
            os._exit(99)
        os.close(before_write)
        assert os.read(before_read, 1) == b"1"
        os.close(before_read)
        os.kill(before_child, signal.SIGKILL)
        os.waitpid(before_child, 0)
        assert before_output.is_dir()
        assert not (before_output / fit_module.PUBLICATION_READY_NAME).exists()
        assert (before_output / fit_module.PUBLICATION_MANIFEST_NAME).is_file()
        assert _ready_pending_path(before_output).is_file()

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
                    destination / fit_module.PUBLICATION_READY_NAME,
                    follow_symlinks=False,
                )
                os.write(after_write, b"1")
                signal.pause()

            fit_module._link_ready_no_replace = stop_before_final_fsync
            _commit_publication(after_output, after_binding)
            os._exit(99)
        os.close(after_write)
        assert os.read(after_read, 1) == b"1"
        os.close(after_read)
        os.kill(after_child, signal.SIGKILL)
        os.waitpid(after_child, 0)
        assert after_output.is_dir()
        assert (after_output / fit_module.PUBLICATION_READY_NAME).is_file()
        assert _ready_pending_path(after_output).stat().st_ino == (
            after_output / fit_module.PUBLICATION_READY_NAME
        ).stat().st_ino
        # A caller must still require successful job completion; visible READY
        # alone cannot prove that the final directory fsync returned.
    finally:
        shutil.rmtree(root)
