from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import scripts.audit_direct_safety_artifacts as audit_module
from scripts.audit_direct_safety_artifacts import _verify_source_manifest
from sph.source_closure import (
    SOURCE_CLOSURE_PROTOCOL,
    discover_first_party_python,
    snapshot_source_closure,
    verify_source_manifest,
)


PROJECT = Path(__file__).resolve().parents[1]
REVIEWED_MANIFEST_SHA256 = (
    "204c025305a9665803e714708dc0eab29394644d5905ad76f1715c7309020878"
)
REVIEWED_MANIFEST = (
    PROJECT
    / "refine-logs/direct-safety-gate/R083_SOURCE_CLOSURE_RESCUE_V2.json"
)
HISTORICAL_R083_V1_MANIFEST_SHA256 = (
    "b78da0f9e6203e7b481302a1dd48e2683bf2b8ae8431cf5bf8aaf36ee6e275ba"
)
HISTORICAL_R083_V1_MANIFEST = (
    PROJECT / "refine-logs/direct-safety-gate/R083_SOURCE_CLOSURE_V1.json"
)
HISTORICAL_R082_MANIFEST_SHA256 = (
    "f36291a961ea793dbaa888950bc4312d8b53954fcc5ecdb01a5caad4af97e184"
)
HISTORICAL_R082_MANIFEST = (
    PROJECT
    / "refine-logs/direct-safety-gate/R082_SOURCE_CLOSURE_PUBLICATION_RESCUE_V1.json"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_small_manifest(project: Path, manifest: Path) -> None:
    paths = discover_first_party_python(project)
    rows = [
        {
            "path": relative,
            "bytes": (project / relative).stat().st_size,
            "sha256": _sha256(project / relative),
        }
        for relative in paths
    ]
    manifest.write_text(
        json.dumps(
            {
                "protocol": SOURCE_CLOSURE_PROTOCOL,
                "roots": ["scripts/*.py", "src/sph/**/*.py"],
                "files": rows,
            }
        ),
        encoding="utf-8",
    )


def _small_project(tmp_path: Path) -> tuple[Path, Path]:
    project = tmp_path / "project"
    (project / "src/sph/nested").mkdir(parents=True)
    (project / "scripts").mkdir()
    (project / "src/sph/__init__.py").write_text("VALUE = 1\n", encoding="utf-8")
    (project / "src/sph/nested/module.py").write_text(
        "VALUE = 2\n", encoding="utf-8"
    )
    (project / "scripts/entry.py").write_text("VALUE = 3\n", encoding="utf-8")
    manifest = project / "manifest.json"
    _write_small_manifest(project, manifest)
    return project, manifest


def test_reviewed_manifest_exactly_closes_current_first_party_surface(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    closure = verify_source_manifest(
        PROJECT,
        REVIEWED_MANIFEST,
        expected_manifest_sha256=REVIEWED_MANIFEST_SHA256,
    )
    monkeypatch.setattr(audit_module, "SOURCE_MANIFEST", REVIEWED_MANIFEST)
    independent = _verify_source_manifest()
    assert closure.manifest_sha256 == REVIEWED_MANIFEST_SHA256
    assert closure.summary() == independent
    assert tuple(entry.path for entry in closure.entries) == (
        discover_first_party_python(PROJECT)
    )
    assert len(closure.entries) == 63
    with pytest.raises(RuntimeError, match="reviewed input"):
        verify_source_manifest(
            PROJECT, REVIEWED_MANIFEST, expected_manifest_sha256="0" * 64
        )


def test_historical_r082_manifest_remains_byte_immutable() -> None:
    assert _sha256(HISTORICAL_R082_MANIFEST) == HISTORICAL_R082_MANIFEST_SHA256
    value = json.loads(HISTORICAL_R082_MANIFEST.read_text(encoding="utf-8"))
    assert len(value["files"]) == 61


def test_rejected_r083_v1_manifest_remains_byte_immutable() -> None:
    assert _sha256(HISTORICAL_R083_V1_MANIFEST) == (
        HISTORICAL_R083_V1_MANIFEST_SHA256
    )
    value = json.loads(HISTORICAL_R083_V1_MANIFEST.read_text(encoding="utf-8"))
    assert len(value["files"]) == 63


def test_source_manifest_rejects_unlisted_and_modified_sources(tmp_path: Path) -> None:
    project, manifest = _small_project(tmp_path)
    verify_source_manifest(project, manifest)

    extra = project / "scripts/extra.py"
    extra.write_text("VALUE = 4\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="surface differs"):
        verify_source_manifest(project, manifest)
    extra.unlink()

    (project / "scripts/entry.py").write_text("VALUE = 99\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="byte count differs|SHA256 differs"):
        verify_source_manifest(project, manifest)


def test_complete_source_snapshot_preserves_manifest_and_relative_paths(
    tmp_path: Path,
) -> None:
    project, manifest = _small_project(tmp_path)
    closure = verify_source_manifest(project, manifest)
    snapshot = tmp_path / "snapshot"
    snapshot_source_closure(project, closure, snapshot)
    assert _sha256(snapshot / "SOURCE_MANIFEST.json") == closure.manifest_sha256
    for entry in closure.entries:
        copied = snapshot / entry.path
        assert copied.stat().st_size == entry.bytes
        assert _sha256(copied) == entry.sha256
