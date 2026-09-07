"""Pure file helpers for proofreading recovery and save history.

This module has no Qt or napari dependency.  Recovery snapshots and history
files are deliberately separate from the formal ``proofread.json`` schema.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import re
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

RECOVERY_SCHEMA_VERSION = 1
HISTORY_LIMIT = 10
_HISTORY_NAME = re.compile(r"\d{8}T\d{6}\.\d{6}Z_[0-9a-f]{16}(?:_\d+)?\.json")


class RecoveryFileError(ValueError):
    """Raised when a recovery file is corrupt or uses an unknown schema."""


@dataclass(frozen=True)
class FileIdentity:
    """Cheap identity used to decide whether a cached full hash is reusable."""

    device: int
    inode: int
    size: int
    mtime_ns: int


@dataclass(frozen=True)
class StableFileHash:
    sha256: str
    identity: FileIdentity


@dataclass(frozen=True)
class RecoveryCandidate:
    path: Path
    session_uuid: str | None
    utc_time: str | None
    revision: int | None
    formal_path: str | None
    error: str | None = None


@dataclass(frozen=True)
class HistoryVersion:
    path: Path
    utc_time: str
    size: int
    sha256: str


def utc_now_text() -> str:
    """Return a stable, sortable UTC timestamp."""
    return datetime.now(UTC).isoformat(timespec="microseconds").replace(
        "+00:00", "Z"
    )


def file_identity(path: str | Path) -> FileIdentity:
    stat = Path(path).stat()
    return FileIdentity(
        device=int(stat.st_dev),
        inode=int(stat.st_ino),
        size=int(stat.st_size),
        mtime_ns=int(stat.st_mtime_ns),
    )


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def hash_file_stable(
    path: str | Path, *, cancelled: Callable[[], bool] | None = None
) -> StableFileHash:
    """Hash a complete file and reject a file that changes during the scan."""
    source = Path(path)
    before = file_identity(source)
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            if cancelled is not None and cancelled():
                raise InterruptedError("file hashing cancelled")
            digest.update(chunk)
    after = file_identity(source)
    if before != after:
        raise OSError("source file changed while it was being hashed")
    return StableFileHash(digest.hexdigest(), after)


def fingerprint_file(path: str | Path) -> str:
    """Return the SHA256 of the exact bytes currently stored at *path*."""
    return hash_file_stable(path).sha256


def canonical_json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n"
    ).encode("utf-8")


def recovery_directory(roi_path: str | Path) -> Path:
    roi = Path(roi_path)
    return roi.parent / f"{roi.name}.proofread-recovery"


def recovery_path(roi_path: str | Path, session_uuid: str) -> Path:
    return recovery_directory(roi_path) / f"{session_uuid}.recovery.json"


def history_directory(formal_path: str | Path) -> Path:
    target = Path(formal_path)
    return target.parent / f"{target.name}.history"


def is_history_version(path: str | Path) -> bool:
    candidate = Path(path)
    return candidate.parent.name.endswith(".history") and bool(_HISTORY_NAME.fullmatch(candidate.name))


def write_temp_bytes(target: str | Path, data: bytes) -> Path:
    """Write and fsync a sibling temporary file without publishing it."""
    destination = Path(target)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
    except Exception:
        with contextlib.suppress(OSError):
            os.close(fd)
        with contextlib.suppress(OSError):
            os.unlink(temporary)
        raise
    return Path(temporary)


def discard_temp(path: str | Path | None) -> None:
    if path is not None:
        with contextlib.suppress(OSError):
            Path(path).unlink()


def _reject_constant(value: str) -> Any:
    raise RecoveryFileError(f"non-finite JSON constant {value!r} is not allowed")


def _reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise RecoveryFileError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def read_recovery(path: str | Path) -> dict[str, Any]:
    """Read and validate the recovery envelope shape, but not store state."""
    source = Path(path)
    try:
        payload = json.loads(
            source.read_text(encoding="utf-8"),
            parse_constant=_reject_constant,
            object_pairs_hook=_reject_duplicates,
        )
    except RecoveryFileError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RecoveryFileError(f"cannot load recovery: {exc}") from exc
    validate_recovery_envelope(payload)
    return payload


def validate_recovery_envelope(payload: Any) -> None:
    """Validate metadata identically for file and in-memory recovery input."""
    if not isinstance(payload, dict):
        raise RecoveryFileError("recovery root must be an object")
    required = {
        "recovery_schema_version",
        "session_uuid",
        "utc_time",
        "revision",
        "raw",
        "image_signature",
        "formal",
        "working_state",
        "saved_state",
    }
    if set(payload) != required:
        raise RecoveryFileError("recovery envelope fields are invalid")
    version = payload.get("recovery_schema_version")
    if type(version) is not int or version != RECOVERY_SCHEMA_VERSION:
        raise RecoveryFileError("unsupported recovery_schema_version")
    session_uuid = payload.get("session_uuid")
    if not isinstance(session_uuid, str) or not session_uuid:
        raise RecoveryFileError("invalid recovery session_uuid")
    utc_time = payload.get("utc_time")
    if not isinstance(utc_time, str) or not utc_time:
        raise RecoveryFileError("invalid recovery utc_time")
    revision = payload.get("revision")
    if isinstance(revision, bool) or not isinstance(revision, int) or revision < 1:
        raise RecoveryFileError("invalid recovery revision")
    if not isinstance(payload.get("raw"), dict):
        raise RecoveryFileError("invalid recovery raw metadata")
    formal = payload.get("formal")
    if not isinstance(formal, dict) or set(formal) != {"path", "sha256"}:
        raise RecoveryFileError("invalid recovery formal baseline")
    path_value = formal.get("path")
    digest_value = formal.get("sha256")
    if path_value is not None and not isinstance(path_value, str):
        raise RecoveryFileError("invalid recovery formal path")
    if digest_value is not None and (
        not isinstance(digest_value, str) or len(digest_value) != 64
    ):
        raise RecoveryFileError("invalid recovery formal fingerprint")
    if not isinstance(payload.get("working_state"), dict) or not isinstance(
        payload.get("saved_state"), dict
    ):
        raise RecoveryFileError("invalid recovery state")


def list_recovery_candidates(roi_path: str | Path) -> list[RecoveryCandidate]:
    directory = recovery_directory(roi_path)
    try:
        paths = list(directory.glob("*.recovery.json"))
    except OSError:
        return []
    candidates: list[RecoveryCandidate] = []
    for path in paths:
        try:
            payload = read_recovery(path)
            formal = payload["formal"]
            candidate = RecoveryCandidate(
                path=path,
                session_uuid=payload["session_uuid"],
                utc_time=payload["utc_time"],
                revision=payload["revision"],
                formal_path=formal["path"],
            )
        except (OSError, RecoveryFileError) as exc:
            candidate = RecoveryCandidate(
                path=path,
                session_uuid=None,
                utc_time=None,
                revision=None,
                formal_path=None,
                error=str(exc),
            )
        candidates.append(candidate)
    return sorted(
        candidates,
        key=lambda item: (item.utc_time or "", item.path.name),
        reverse=True,
    )


def _history_timestamp_from_name(path: Path) -> str:
    stamp = path.name.split("_", 1)[0]
    try:
        parsed = datetime.strptime(stamp, "%Y%m%dT%H%M%S.%fZ").replace(
            tzinfo=UTC
        )
    except ValueError:
        return datetime.fromtimestamp(
            path.stat().st_mtime, tz=UTC
        ).isoformat().replace("+00:00", "Z")
    return parsed.isoformat(timespec="seconds").replace("+00:00", "Z")


def list_history_versions(formal_path: str | Path) -> list[HistoryVersion]:
    directory = history_directory(formal_path)
    try:
        paths = [
            path for path in directory.glob("*.json")
            if is_history_version(path)
            and path.is_file()
        ]
    except OSError:
        return []
    versions: list[HistoryVersion] = []
    for path in paths:
        try:
            data = path.read_bytes()
            versions.append(
                HistoryVersion(
                    path=path,
                    utc_time=_history_timestamp_from_name(path),
                    size=len(data),
                    sha256=sha256_bytes(data),
                )
            )
        except OSError:
            continue
    return sorted(versions, key=lambda item: item.path.name, reverse=True)


def backup_formal_bytes(formal_path: str | Path, data: bytes) -> Path:
    """Preserve exact prior bytes, deduplicating identical historical data."""
    target = Path(formal_path)
    digest = sha256_bytes(data)
    for version in list_history_versions(target):
        if version.sha256 == digest and version.path.read_bytes() == data:
            return version.path
    directory = history_directory(target)
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    destination = directory / f"{stamp}_{digest[:16]}.json"
    suffix = 1
    while destination.exists():
        destination = directory / f"{stamp}_{digest[:16]}_{suffix}.json"
        suffix += 1
    temporary = write_temp_bytes(destination, data)
    try:
        os.replace(temporary, destination)
    except Exception:
        discard_temp(temporary)
        raise
    return destination


def trim_history(
    formal_path: str | Path,
    *,
    limit: int = HISTORY_LIMIT,
) -> str | None:
    """Retain newest distinct versions; return a warning on cleanup failure."""
    if limit < 0:
        raise ValueError("history limit must be non-negative")
    versions = list_history_versions(formal_path)
    seen: set[str] = set()
    keep: set[Path] = set()
    for version in versions:
        if version.sha256 in seen:
            continue
        seen.add(version.sha256)
        if len(keep) < limit:
            keep.add(version.path)
    failures: list[str] = []
    for version in versions:
        if version.path in keep:
            continue
        try:
            version.path.unlink()
        except OSError as exc:
            failures.append(f"{version.path.name}: {exc}")
    if failures:
        return "History cleanup failed: " + "; ".join(failures)
    return None


__all__ = (
    "FileIdentity",
    "HISTORY_LIMIT",
    "HistoryVersion",
    "RECOVERY_SCHEMA_VERSION",
    "RecoveryCandidate",
    "RecoveryFileError",
    "StableFileHash",
    "backup_formal_bytes",
    "canonical_json_bytes",
    "discard_temp",
    "file_identity",
    "fingerprint_file",
    "hash_file_stable",
    "history_directory",
    "list_history_versions",
    "list_recovery_candidates",
    "read_recovery",
    "recovery_directory",
    "recovery_path",
    "sha256_bytes",
    "trim_history",
    "utc_now_text",
    "write_temp_bytes",
)
