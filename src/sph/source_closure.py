"""Verify and snapshot the reviewed first-party Python source closure.

The source manifest is deliberately external to the Python surface that it
describes, so Slurm can pin the manifest SHA256 without creating a circular
file hash.  A valid manifest must enumerate the complete conservative source
surface: every Python file below ``src/sph`` and every top-level Python entry
point below ``scripts``.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any, Mapping


SOURCE_CLOSURE_PROTOCOL = "pros-gate-first-party-source-closure-v1"
SOURCE_MANIFEST_RELATIVE_PATH = Path(
    "refine-logs/direct-safety-gate/R079_SOURCE_CLOSURE_NUMERIC_V2.json"
)


@dataclass(frozen=True)
class SourceEntry:
    path: str
    bytes: int
    sha256: str


@dataclass(frozen=True)
class VerifiedSourceClosure:
    manifest_path: Path
    manifest_sha256: str
    entries: tuple[SourceEntry, ...]

    def summary(self) -> dict[str, Any]:
        return {
            "protocol": SOURCE_CLOSURE_PROTOCOL,
            "source_manifest_sha256": self.manifest_sha256,
            "source_file_count": len(self.entries),
            "source_entries_sha256": source_entries_sha256(self.entries),
        }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(name: str, value: Any) -> str:
    if not isinstance(value, str) or len(value) != 64 or value != value.lower():
        raise RuntimeError(f"{name} is not a canonical lowercase SHA256")
    try:
        int(value, 16)
    except ValueError as error:
        raise RuntimeError(f"{name} is not hexadecimal") from error
    return value


def discover_first_party_python(project: Path) -> tuple[str, ...]:
    """Return the conservative, exact first-party Python execution surface."""

    root = project.resolve()
    sph_root = root / "src/sph"
    scripts_root = root / "scripts"
    if not sph_root.is_dir() or not scripts_root.is_dir():
        raise RuntimeError("project lacks src/sph or scripts")
    paths = [
        path.relative_to(root).as_posix()
        for path in sph_root.rglob("*.py")
        if path.is_file()
    ]
    paths.extend(
        path.relative_to(root).as_posix()
        for path in scripts_root.glob("*.py")
        if path.is_file()
    )
    ordered = tuple(sorted(paths))
    if not ordered or len(set(ordered)) != len(ordered):
        raise RuntimeError("first-party Python discovery is empty or duplicated")
    return ordered


def source_entries_sha256(entries: tuple[SourceEntry, ...]) -> str:
    payload = [
        {"path": entry.path, "bytes": entry.bytes, "sha256": entry.sha256}
        for entry in entries
    ]
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _safe_source_path(project: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative or "\0" in relative:
        raise RuntimeError("source manifest contains an invalid path")
    candidate = Path(relative)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise RuntimeError("source manifest path escapes the project")
    path = project / candidate
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"source manifest entry is not a regular file: {relative}")
    if not path.resolve().is_relative_to(project.resolve()):
        raise RuntimeError(f"source manifest entry escapes the project: {relative}")
    return path


def verify_source_manifest(
    project: Path,
    manifest_path: Path | None = None,
    *,
    expected_manifest_sha256: str | None = None,
) -> VerifiedSourceClosure:
    """Fail closed unless the reviewed manifest exactly matches the source tree."""

    root = project.resolve()
    manifest = (
        manifest_path.resolve()
        if manifest_path is not None
        else (root / SOURCE_MANIFEST_RELATIVE_PATH).resolve()
    )
    if manifest.is_symlink() or not manifest.is_file():
        raise RuntimeError("reviewed source manifest is missing or is a symlink")
    manifest_sha256 = sha256_file(manifest)
    if expected_manifest_sha256 is not None and manifest_sha256 != _require_sha256(
        "expected source manifest hash", expected_manifest_sha256
    ):
        raise RuntimeError("source manifest SHA256 differs from the reviewed input")
    value = json.loads(manifest.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError("source manifest must be a JSON object")
    if value.get("protocol") != SOURCE_CLOSURE_PROTOCOL:
        raise RuntimeError("source manifest protocol differs")
    if value.get("roots") != ["scripts/*.py", "src/sph/**/*.py"]:
        raise RuntimeError("source manifest roots differ from the closed surface")
    rows = value.get("files")
    if not isinstance(rows, list) or not rows:
        raise RuntimeError("source manifest has no file entries")

    entries: list[SourceEntry] = []
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != {"path", "bytes", "sha256"}:
            raise RuntimeError("source manifest file entry has unexpected fields")
        relative = row["path"]
        byte_count = row["bytes"]
        if isinstance(byte_count, bool) or not isinstance(byte_count, int) or byte_count < 1:
            raise RuntimeError("source manifest contains an invalid byte count")
        entries.append(
            SourceEntry(
                path=relative,
                bytes=byte_count,
                sha256=_require_sha256("source file hash", row["sha256"]),
            )
        )
    relative_paths = tuple(entry.path for entry in entries)
    if relative_paths != tuple(sorted(relative_paths)) or len(set(relative_paths)) != len(
        relative_paths
    ):
        raise RuntimeError("source manifest entries are not sorted and unique")
    discovered = discover_first_party_python(root)
    if relative_paths != discovered:
        missing = sorted(set(discovered) - set(relative_paths))
        extra = sorted(set(relative_paths) - set(discovered))
        raise RuntimeError(
            f"source manifest surface differs; missing={missing[:3]} extra={extra[:3]}"
        )
    for entry in entries:
        source = _safe_source_path(root, entry.path)
        if source.stat().st_size != entry.bytes:
            raise RuntimeError(f"source byte count differs: {entry.path}")
        if sha256_file(source) != entry.sha256:
            raise RuntimeError(f"source SHA256 differs: {entry.path}")
    return VerifiedSourceClosure(
        manifest_path=manifest,
        manifest_sha256=manifest_sha256,
        entries=tuple(entries),
    )


def snapshot_source_closure(
    project: Path,
    verified: VerifiedSourceClosure,
    output: Path,
) -> None:
    """Copy the complete verified closure, preserving project-relative paths."""

    if output.exists():
        raise FileExistsError(f"refusing to overwrite source snapshot: {output}")
    current = verify_source_manifest(
        project,
        verified.manifest_path,
        expected_manifest_sha256=verified.manifest_sha256,
    )
    if current != verified:
        raise RuntimeError("source closure changed before snapshot")
    output.mkdir(parents=True)
    try:
        manifest_copy = output / "SOURCE_MANIFEST.json"
        shutil.copy2(verified.manifest_path, manifest_copy)
        if sha256_file(manifest_copy) != verified.manifest_sha256:
            raise RuntimeError("snapshotted source manifest hash differs")
        for entry in verified.entries:
            source = _safe_source_path(project.resolve(), entry.path)
            destination = output / entry.path
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            if destination.stat().st_size != entry.bytes:
                raise RuntimeError(f"snapshotted source byte count differs: {entry.path}")
            if sha256_file(destination) != entry.sha256:
                raise RuntimeError(f"snapshotted source SHA256 differs: {entry.path}")
        for path in [manifest_copy, *(output / entry.path for entry in verified.entries)]:
            with path.open("rb") as handle:
                os.fsync(handle.fileno())
    except Exception:
        shutil.rmtree(output)
        raise


def _main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--expected-manifest-sha256", required=True)
    args = parser.parse_args()
    closure = verify_source_manifest(
        args.project,
        args.manifest,
        expected_manifest_sha256=args.expected_manifest_sha256,
    )
    print(
        json.dumps(
            closure.summary(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ),
        flush=True,
    )


if __name__ == "__main__":
    _main()
