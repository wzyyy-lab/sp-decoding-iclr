from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from scripts.replay_direct_safety_capacity_adjudication import (
    _atomic_write_json,
    repair_legacy_capacity_row,
    require_unchanged_source_closure,
)
from sph.source_closure import SOURCE_CLOSURE_PROTOCOL, verify_source_manifest


def legacy_row() -> dict[str, int]:
    return {
        "harmful_count": 128,
        "harmful_apply_count": 1,
        "harm_avoidance_numerator": 127,
        "harm_avoidance_denominator": 128,
    }


def test_legacy_capacity_repair_adds_only_proven_alias() -> None:
    original = legacy_row()
    repaired = repair_legacy_capacity_row(original)
    assert original == legacy_row()
    assert repaired == {**original, "harmful_keep_count": 127}


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("harmful_apply_count", 129),
        ("harm_avoidance_numerator", 126),
        ("harm_avoidance_denominator", 127),
        ("harmful_count", 128.0),
        ("harmful_apply_count", True),
    ],
)
def test_legacy_capacity_repair_rejects_inconsistent_counts(
    name: str, value: object
) -> None:
    row: dict[str, object] = legacy_row()
    row[name] = value
    with pytest.raises(RuntimeError):
        repair_legacy_capacity_row(row)


def test_legacy_capacity_repair_rejects_preexisting_alias() -> None:
    row = {**legacy_row(), "harmful_keep_count": 127}
    with pytest.raises(RuntimeError, match="already"):
        repair_legacy_capacity_row(row)


def test_atomic_receipt_publication_is_no_clobber(tmp_path: Path) -> None:
    receipt = tmp_path / "receipt.json"
    _atomic_write_json(receipt, {"writer": "first"})
    first = receipt.read_bytes()
    with pytest.raises(FileExistsError, match="overwrite"):
        _atomic_write_json(receipt, {"writer": "second"})
    assert receipt.read_bytes() == first


def test_atomic_receipt_preserves_target_that_appears_at_publish(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipt = tmp_path / "receipt.json"
    original_link = os.link

    def raced_link(source: str | Path, destination: str | Path) -> None:
        Path(destination).write_bytes(b"concurrent-winner\n")
        original_link(source, destination)

    monkeypatch.setattr(os, "link", raced_link)
    with pytest.raises(FileExistsError, match="overwrite"):
        _atomic_write_json(receipt, {"writer": "loser"})
    assert receipt.read_bytes() == b"concurrent-winner\n"
    assert not list(tmp_path.glob(".*.tmp"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_source_change_blocks_receipt_publication(tmp_path: Path) -> None:
    project = tmp_path / "project"
    (project / "src/sph").mkdir(parents=True)
    (project / "scripts").mkdir()
    source = project / "src/sph/module.py"
    entry = project / "scripts/entry.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    entry.write_text("VALUE = 1\n", encoding="utf-8")
    files = []
    for path in (entry, source):
        files.append(
            {
                "path": path.relative_to(project).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    manifest = project / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "protocol": SOURCE_CLOSURE_PROTOCOL,
                "roots": ["scripts/*.py", "src/sph/**/*.py"],
                "files": files,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    manifest_sha256 = _sha256(manifest)
    initial = verify_source_manifest(
        project, manifest, expected_manifest_sha256=manifest_sha256
    )
    source.write_text("VALUE = 2\n", encoding="utf-8")
    receipt = tmp_path / "receipt.json"
    with pytest.raises(RuntimeError, match="SHA256 differs"):
        require_unchanged_source_closure(
            initial,
            project=project,
            source_manifest=manifest,
            expected_source_manifest_sha256=manifest_sha256,
        )
        _atomic_write_json(receipt, {"should_not_publish": True})
    assert not receipt.exists()
