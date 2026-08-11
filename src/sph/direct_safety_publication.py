"""Durable no-clobber publication for the one-shot R083 falsifier.

The directory is visible while incomplete, but becomes consumable only when a
parent-side, fsynced receipt is hard-linked as ``READY.json``.  Missing READY
is the sole incomplete state.  The output tree is never mutated after READY.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import stat
from typing import Any, Mapping
import uuid

from sph.direct_safety_artifacts import sha256_file
from sph.source_closure import verify_source_manifest


PUBLICATION_PROTOCOL = "pros-gate-r083-directory-commit-v1"
PUBLICATION_MANIFEST_NAME = "PUBLICATION_MANIFEST.json"
PUBLICATION_READY_NAME = "READY.json"
PUBLICATION_RESERVATION_NAME = "RESERVATION.json"
FALSIFIER_PURPOSE = "R083_FALSIFIER"
SMOKE_PURPOSE = "FILESYSTEM_SMOKE"


def _require_regular_file(path: Path, name: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"{name} is missing or is a symlink")


def _require_canonical_sha256(name: str, value: Any) -> str:
    if not isinstance(value, str) or len(value) != 64 or value != value.lower():
        raise RuntimeError(f"{name} is not a canonical lowercase SHA256")
    try:
        int(value, 16)
    except ValueError as error:
        raise RuntimeError(f"{name} is not hexadecimal") from error
    return value


def _require_relative_entrypoint(value: Any) -> str:
    if not isinstance(value, str) or not value or "\0" in value:
        raise RuntimeError("R083 entrypoint path is invalid")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or path.as_posix() != value:
        raise RuntimeError("R083 entrypoint path escapes the project")
    return value


def _write_json(path: Path, value: Any) -> None:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o600)
    try:
        view = memoryview(encoded)
        while view:
            written = os.write(descriptor, view)
            if written < 1:
                raise OSError(f"failed to write publication file: {path}")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_path(path: Path) -> None:
    _require_regular_file(path, "publication payload")
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _open_directory(path: Path) -> int:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    return os.open(path, flags)


def _fsync_directory(path: Path) -> None:
    descriptor = _open_directory(path)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def publication_identity(
    output: Path,
    *,
    job_id: str,
    purpose: str,
    entrypoint_path: str,
    entrypoint_sha256: str,
    wrapper_sha256: str,
    source_manifest_sha256: str,
) -> dict[str, Any]:
    if not isinstance(job_id, str) or not job_id or any(
        character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-"
        for character in job_id
    ):
        raise RuntimeError("R083 publication job identity is invalid")
    if purpose not in {FALSIFIER_PURPOSE, SMOKE_PURPOSE}:
        raise RuntimeError("R083 publication purpose differs")
    identity = {
        "job_id": job_id,
        "seed": 0,
        "purpose": purpose,
        "output": str(output.resolve()),
        "entrypoint_path": _require_relative_entrypoint(entrypoint_path),
        "entrypoint_sha256": _require_canonical_sha256(
            "R083 entrypoint hash", entrypoint_sha256
        ),
        "wrapper_sha256": _require_canonical_sha256(
            "R083 wrapper hash", wrapper_sha256
        ),
        "source_manifest_sha256": _require_canonical_sha256(
            "R083 source manifest hash", source_manifest_sha256
        ),
    }
    if purpose == FALSIFIER_PURPOSE and (
        not job_id.isdecimal()
        or output.name != "seed0"
        or output.parent.name != f"pros_gate_falsifier_{job_id}"
    ):
        raise RuntimeError("R083 output path does not match its job and seed")
    return identity


def _reservation_payload(identity: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "protocol": PUBLICATION_PROTOCOL,
        "state": "UNCOMMITTED_UNTIL_READY_JSON",
        "identity": dict(identity),
    }


def reserve_publication_directory(
    output: Path, identity: Mapping[str, Any]
) -> Path:
    """Exclusively reserve a visible output directory without deleting failure evidence."""

    output.parent.mkdir(parents=True, exist_ok=True)
    if output.parent.is_symlink() or not output.parent.is_dir():
        raise RuntimeError("R083 output parent is missing or is a symlink")
    parent_descriptor = _open_directory(output.parent)
    try:
        try:
            os.mkdir(output.name, mode=0o700, dir_fd=parent_descriptor)
        except FileExistsError as error:
            raise FileExistsError(f"refusing to overwrite R083 run: {output}") from error
        os.fsync(parent_descriptor)
    finally:
        os.close(parent_descriptor)
    _write_json(output / PUBLICATION_RESERVATION_NAME, _reservation_payload(identity))
    (output / PUBLICATION_RESERVATION_NAME).chmod(0o400)
    _fsync_path(output / PUBLICATION_RESERVATION_NAME)
    _fsync_directory(output)
    return output


def _publication_tree(
    output: Path,
) -> tuple[list[str], list[dict[str, int | str]]]:
    if output.is_symlink() or not output.is_dir():
        raise RuntimeError("R083 publication root is missing or is a symlink")
    directories: list[str] = []
    files: list[dict[str, int | str]] = []
    excluded = {PUBLICATION_MANIFEST_NAME, PUBLICATION_READY_NAME}
    for current, directory_names, file_names in os.walk(
        output, topdown=True, followlinks=False
    ):
        current_path = Path(current)
        directory_names.sort()
        file_names.sort()
        for name in directory_names:
            path = current_path / name
            if path.is_symlink() or not path.is_dir():
                raise RuntimeError("R083 publication contains a non-directory entry")
            directories.append(path.relative_to(output).as_posix())
        for name in file_names:
            path = current_path / name
            relative = path.relative_to(output).as_posix()
            if path.is_symlink() or not path.is_file():
                raise RuntimeError("R083 publication contains a non-regular file")
            if relative in excluded:
                continue
            files.append(
                {
                    "path": relative,
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    directories.sort()
    files.sort(key=lambda row: str(row["path"]))
    return directories, files


def _load_json_object(path: Path, name: str) -> dict[str, Any]:
    _require_regular_file(path, name)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"{name} must be a JSON object")
    return value


def _publication_manifest(
    output: Path, binding: Mapping[str, Any]
) -> dict[str, Any]:
    directories, files = _publication_tree(output)
    reservation = _load_json_object(
        output / PUBLICATION_RESERVATION_NAME, "R083 reservation"
    )
    if reservation != _reservation_payload(binding.get("identity", {})):
        raise RuntimeError("R083 reservation schema differs")
    return {
        "protocol": PUBLICATION_PROTOCOL,
        "binding": dict(binding),
        "directories": directories,
        "files": files,
        "payload_directory_count": len(directories),
        "payload_file_count": len(files),
    }


def _fsync_publication_tree(output: Path, manifest: Mapping[str, Any]) -> None:
    files = manifest.get("files")
    directories = manifest.get("directories")
    if not isinstance(files, list) or not isinstance(directories, list):
        raise RuntimeError("R083 publication manifest tree schema differs")
    for row in files:
        if not isinstance(row, Mapping) or not isinstance(row.get("path"), str):
            raise RuntimeError("R083 publication manifest file schema differs")
        _fsync_path(output / str(row["path"]))
    _fsync_path(output / PUBLICATION_MANIFEST_NAME)
    for relative in sorted(
        directories,
        key=lambda value: (str(value).count("/"), str(value)),
        reverse=True,
    ):
        _fsync_directory(output / str(relative))
    _fsync_directory(output)


def _ready_payload(manifest_sha256: str, manifest: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "protocol": PUBLICATION_PROTOCOL,
        "status": "READY",
        "publication_manifest_sha256": manifest_sha256,
        "payload_directory_count": manifest["payload_directory_count"],
        "payload_file_count": manifest["payload_file_count"],
    }


def ready_pending_path(output: Path) -> Path:
    return output.parent / f".{output.name}.READY.pending"


def _verify_ready_commit_link(output: Path, ready: Path) -> Path:
    """Require the retained pending receipt to be the exact READY inode."""

    pending = ready_pending_path(output)
    _require_regular_file(ready, "R083 READY receipt")
    _require_regular_file(pending, "R083 READY pending receipt")
    ready_stat = os.stat(ready, follow_symlinks=False)
    pending_stat = os.stat(pending, follow_symlinks=False)
    if (ready_stat.st_dev, ready_stat.st_ino) != (
        pending_stat.st_dev,
        pending_stat.st_ino,
    ):
        raise RuntimeError("R083 READY and pending receipts are not one hard link")
    if ready_stat.st_nlink != 2 or pending_stat.st_nlink != 2:
        raise RuntimeError("R083 READY receipt hard-link count differs from two")
    if stat.S_IMODE(ready_stat.st_mode) != 0o400 or stat.S_IMODE(
        pending_stat.st_mode
    ) != 0o400:
        raise RuntimeError("R083 READY receipt mode differs from 0400")
    return pending


def _write_ready_pending(output: Path, payload: Mapping[str, Any]) -> Path:
    pending = ready_pending_path(output)
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(pending, flags, 0o400)
    try:
        view = memoryview(encoded)
        while view:
            written = os.write(descriptor, view)
            if written < 1:
                raise OSError("failed to write R083 READY pending receipt")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    _fsync_directory(output.parent)
    return pending


def _link_ready_no_replace(pending: Path, output: Path) -> None:
    parent_descriptor = _open_directory(output.parent)
    output_descriptor = _open_directory(output)
    try:
        try:
            os.link(
                pending.name,
                PUBLICATION_READY_NAME,
                src_dir_fd=parent_descriptor,
                dst_dir_fd=output_descriptor,
                follow_symlinks=False,
            )
        except FileExistsError as error:
            raise FileExistsError(
                f"refusing to overwrite R083 READY receipt: {output}"
            ) from error
        os.fsync(output_descriptor)
    finally:
        os.close(output_descriptor)
        os.close(parent_descriptor)


def _validate_binding(output: Path, binding: Any) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    if not isinstance(binding, Mapping) or set(binding) != {
        "identity",
        "scientific_status",
        "input_identities_end",
        "source_closure_end",
    }:
        raise RuntimeError("R083 publication binding schema differs")
    identity = binding.get("identity")
    if not isinstance(identity, Mapping) or set(identity) != {
        "job_id",
        "seed",
        "purpose",
        "output",
        "entrypoint_path",
        "entrypoint_sha256",
        "wrapper_sha256",
        "source_manifest_sha256",
    }:
        raise RuntimeError("R083 publication identity schema differs")
    if identity.get("output") != str(output.resolve()) or identity.get("seed") != 0:
        raise RuntimeError("R083 publication output or seed identity differs")
    for name in ("entrypoint_sha256", "wrapper_sha256", "source_manifest_sha256"):
        _require_canonical_sha256(f"R083 publication {name}", identity.get(name))
    _require_relative_entrypoint(identity.get("entrypoint_path"))
    purpose = identity.get("purpose")
    job_id = identity.get("job_id")
    if purpose == FALSIFIER_PURPOSE:
        if (
            not isinstance(job_id, str)
            or not job_id.isdecimal()
            or output.name != "seed0"
            or output.parent.name != f"pros_gate_falsifier_{job_id}"
        ):
            raise RuntimeError("R083 committed output does not match job and seed")
        if binding.get("scientific_status") not in {"PASS", "FAIL"}:
            raise RuntimeError("R083 falsifier scientific status differs")
    elif purpose == SMOKE_PURPOSE:
        if binding.get("scientific_status") != "NOT_APPLICABLE_FILESYSTEM_SMOKE":
            raise RuntimeError("R083 smoke scientific status differs")
    else:
        raise RuntimeError("R083 committed publication purpose differs")
    if not isinstance(binding.get("input_identities_end"), Mapping) or not isinstance(
        binding.get("source_closure_end"), Mapping
    ):
        raise RuntimeError("R083 binding identities are not mappings")
    return binding, identity


def verify_published_directory(
    output: Path, *, expected_binding: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    ready_path = output / PUBLICATION_READY_NAME
    manifest_path = output / PUBLICATION_MANIFEST_NAME
    ready = _load_json_object(ready_path, "R083 READY receipt")
    _verify_ready_commit_link(output, ready_path)
    manifest = _load_json_object(manifest_path, "R083 publication manifest")
    if set(ready) != {
        "protocol",
        "status",
        "publication_manifest_sha256",
        "payload_directory_count",
        "payload_file_count",
    }:
        raise RuntimeError("R083 READY receipt fields differ")
    if set(manifest) != {
        "protocol",
        "binding",
        "directories",
        "files",
        "payload_directory_count",
        "payload_file_count",
    }:
        raise RuntimeError("R083 publication manifest fields differ")
    if ready.get("protocol") != PUBLICATION_PROTOCOL or ready.get("status") != "READY":
        raise RuntimeError("R083 READY receipt protocol or status differs")
    if manifest.get("protocol") != PUBLICATION_PROTOCOL:
        raise RuntimeError("R083 publication manifest protocol differs")
    binding, identity = _validate_binding(output, manifest.get("binding"))
    if expected_binding is not None and dict(binding) != dict(expected_binding):
        raise RuntimeError("R083 publication binding differs from the expected input")
    manifest_sha256 = sha256_file(manifest_path)
    if ready.get("publication_manifest_sha256") != manifest_sha256:
        raise RuntimeError("R083 READY receipt names a different manifest")
    observed_directories, observed_files = _publication_tree(output)
    if manifest.get("directories") != observed_directories or manifest.get(
        "files"
    ) != observed_files:
        raise RuntimeError("R083 payload tree differs from its committed manifest")
    if manifest.get("payload_directory_count") != len(observed_directories) or manifest.get(
        "payload_file_count"
    ) != len(observed_files):
        raise RuntimeError("R083 publication manifest cardinality differs")
    if ready.get("payload_directory_count") != len(observed_directories) or ready.get(
        "payload_file_count"
    ) != len(observed_files):
        raise RuntimeError("R083 READY receipt cardinality differs")
    reservation = _load_json_object(
        output / PUBLICATION_RESERVATION_NAME, "R083 reservation"
    )
    if reservation != _reservation_payload(identity):
        raise RuntimeError("R083 reservation changed after commit")
    metrics = _load_json_object(output / "metrics.json", "R083 metrics")
    for name in ("scientific_status", "input_identities_end", "source_closure_end"):
        if metrics.get(name) != binding.get(name):
            raise RuntimeError(f"R083 publication binding differs from metrics: {name}")
    file_rows = {
        str(row["path"]): row
        for row in observed_files
        if isinstance(row, Mapping) and isinstance(row.get("path"), str)
    }
    manifest_row = file_rows.get("source_snapshot/SOURCE_MANIFEST.json")
    entrypoint_relative = f"source_snapshot/{identity['entrypoint_path']}"
    entrypoint_row = file_rows.get(entrypoint_relative)
    if (
        not isinstance(manifest_row, Mapping)
        or manifest_row.get("sha256") != identity.get("source_manifest_sha256")
        or not isinstance(entrypoint_row, Mapping)
        or entrypoint_row.get("sha256") != identity.get("entrypoint_sha256")
    ):
        raise RuntimeError("R083 publication source snapshot identity differs")
    source_manifest = _load_json_object(
        output / "source_snapshot/SOURCE_MANIFEST.json",
        "R083 snapshotted source manifest",
    )
    source_rows = source_manifest.get("files")
    if not isinstance(source_rows, list):
        raise RuntimeError("R083 snapshotted source manifest lacks file entries")
    entrypoint_rows = [
        row
        for row in source_rows
        if isinstance(row, Mapping) and row.get("path") == identity["entrypoint_path"]
    ]
    if len(entrypoint_rows) != 1 or entrypoint_rows[0].get("sha256") != identity.get(
        "entrypoint_sha256"
    ):
        raise RuntimeError("R083 source closure names a different entrypoint")
    return {
        "protocol": PUBLICATION_PROTOCOL,
        "status": "READY",
        "publication_manifest_sha256": manifest_sha256,
        "ready_sha256": sha256_file(ready_path),
        "payload_directory_count": len(observed_directories),
        "payload_file_count": len(observed_files),
    }


def commit_publication(
    output: Path, binding: Mapping[str, Any]
) -> dict[str, Any]:
    _validate_binding(output, binding)
    for control in (PUBLICATION_MANIFEST_NAME, PUBLICATION_READY_NAME):
        path = output / control
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"refusing to replace R083 control: {path}")
    manifest = _publication_manifest(output, binding)
    manifest_path = output / PUBLICATION_MANIFEST_NAME
    _write_json(manifest_path, manifest)
    if _publication_manifest(output, binding) != manifest:
        raise RuntimeError("R083 payload changed while constructing its manifest")
    _fsync_publication_tree(output, manifest)
    if _publication_manifest(output, binding) != manifest:
        raise RuntimeError("R083 payload changed while fsyncing its manifest")
    manifest_sha256 = sha256_file(manifest_path)
    pending = _write_ready_pending(output, _ready_payload(manifest_sha256, manifest))
    if _publication_manifest(output, binding) != manifest:
        raise RuntimeError("R083 payload changed before READY commit")
    _link_ready_no_replace(pending, output)
    return verify_published_directory(output, expected_binding=binding)


def publication_filesystem_smoke(
    parent: Path,
    *,
    project: Path,
    job_id: str,
    wrapper_sha256: str,
    source_manifest: Path,
    source_manifest_sha256: str,
    entrypoint_path: str,
) -> dict[str, Any]:
    """Exercise the exact R083 directory-commit protocol before any data open."""

    parent.mkdir(parents=True, exist_ok=True)
    closure = verify_source_manifest(
        project,
        source_manifest,
        expected_manifest_sha256=source_manifest_sha256,
    )
    entrypoint_relative = _require_relative_entrypoint(entrypoint_path)
    entrypoint = project.resolve() / entrypoint_relative
    _require_regular_file(entrypoint, "R083 smoke entrypoint")
    probe = parent / f".pros-gate-r083-publication-smoke.{uuid.uuid4().hex}"
    identity = publication_identity(
        probe,
        job_id=job_id,
        purpose=SMOKE_PURPOSE,
        entrypoint_path=entrypoint_relative,
        entrypoint_sha256=sha256_file(entrypoint),
        wrapper_sha256=wrapper_sha256,
        source_manifest_sha256=source_manifest_sha256,
    )
    source_closure = closure.summary()
    binding = {
        "identity": identity,
        "scientific_status": "NOT_APPLICABLE_FILESYSTEM_SMOKE",
        "input_identities_end": {},
        "source_closure_end": source_closure,
    }
    reserve_publication_directory(probe, identity)
    snapshot = probe / "source_snapshot"
    (snapshot / Path(entrypoint_relative).parent).mkdir(parents=True)
    shutil.copy2(source_manifest, snapshot / "SOURCE_MANIFEST.json")
    shutil.copy2(entrypoint, snapshot / entrypoint_relative)
    _fsync_path(snapshot / "SOURCE_MANIFEST.json")
    _fsync_path(snapshot / entrypoint_relative)
    _write_json(
        probe / "metrics.json",
        {
            "scientific_status": binding["scientific_status"],
            "input_identities_end": binding["input_identities_end"],
            "source_closure_end": source_closure,
        },
    )
    summary = commit_publication(probe, binding)
    if verify_published_directory(probe, expected_binding=binding) != summary:
        raise RuntimeError("R083 publication smoke replay differs")
    return {
        **summary,
        "smoke": "PASS",
        "output": str(probe),
        "pending": str(ready_pending_path(probe)),
    }
