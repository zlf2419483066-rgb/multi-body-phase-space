"""Small, immutable sidecar storage for independence evidence.

The candidate catalog and checkpoint formats are read-only inputs.  This
module writes a separate JSON sidecar whose payload and source/profile/request
bindings are checked on every read.  It intentionally has no directory scan,
resume scheduler, or batch writer.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

from ..data_paths import default_output
from .contracts import fraction_text
from .sources import SourceBatch, SourceSnapshot, verify_source_unchanged


STORAGE_SCHEMA_VERSION = "F5-I-sidecar-v1"


class StorageError(ValueError):
    """Malformed, incomplete, or conflicting sidecar storage."""


class StorageConflictError(StorageError):
    """A sidecar binding does not match the requested recovery context."""


def default_sidecar_path(run_id: str, name: str) -> Path:
    """Return one explicit D-drive independence sidecar path.

    Both arguments are restricted to a single path component.  This helper
    only computes a path; it does not create directories or enumerate runs.
    """

    for value, label in ((run_id, "run_id"), (name, "name")):
        if not isinstance(value, str) or not value or value in {".", ".."}:
            raise ValueError(f"{label} must be one non-empty path component")
        if Path(value).name != value or Path(value).parent != Path(".") or any(char in value for char in "\\/:"):
            raise ValueError(f"{label} must be one non-empty path component")
    return default_output("independence", run_id, name)


def _jsonable(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if hasattr(value, "value") and isinstance(value.value, str):
        return value.value
    if isinstance(value, Fraction):
        return fraction_text(value)
    return value


def _json_bytes(value: Any) -> bytes:
    return json.dumps(
        _jsonable(value), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _sha256(value: Any) -> str:
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def _payload(value: Any, name: str) -> dict[str, Any]:
    if hasattr(value, "to_dict"):
        value = value.to_dict()
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping or expose to_dict()")
    return {str(key): _jsonable(item) for key, item in value.items()}


def _binding(value: Any, name: str) -> tuple[dict[str, Any], str]:
    normalized = _payload(value, name)
    return normalized, _sha256(normalized)


def _snapshot_fingerprint(snapshot: Mapping[str, Any]) -> str:
    required = {"schema_version", "source_kind", "source_path", "files", "content_sha256"}
    if not required.issubset(snapshot):
        raise StorageError("source binding is missing immutable snapshot fields")
    base = {key: snapshot[key] for key in required}
    return _sha256(base)


def _source_binding(source: SourceBatch | SourceSnapshot | Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    if isinstance(source, SourceBatch):
        verify_source_unchanged(source)
        snapshot = source.snapshot.to_dict()
    elif isinstance(source, SourceSnapshot):
        snapshot = source.to_dict()
    else:
        snapshot = _payload(source, "source")
    fingerprint = snapshot.get("fingerprint")
    if not isinstance(fingerprint, str) or not fingerprint:
        raise ValueError("source binding must contain a snapshot fingerprint")
    expected = _snapshot_fingerprint(snapshot)
    if fingerprint != expected:
        raise StorageConflictError("source snapshot fingerprint mismatch")
    return snapshot, fingerprint


@dataclass(frozen=True, slots=True)
class StoredSidecar:
    """A decoded sidecar with verified content and context bindings."""

    payload: Mapping[str, Any]
    source: Mapping[str, Any]
    profile: Mapping[str, Any]
    request: Mapping[str, Any]
    producer_version: str
    payload_sha256: str
    source_fingerprint: str
    profile_sha256: str
    request_sha256: str
    schema_version: str = STORAGE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != STORAGE_SCHEMA_VERSION:
            raise StorageConflictError("unsupported sidecar storage version")
        for value, name in (
            (self.payload, "payload"),
            (self.source, "source"),
            (self.profile, "profile"),
            (self.request, "request"),
        ):
            if not isinstance(value, Mapping):
                raise TypeError(f"{name} must be a mapping")
        if not isinstance(self.producer_version, str) or not self.producer_version:
            raise ValueError("producer_version must be non-empty")
        if _sha256(self.payload) != self.payload_sha256:
            raise StorageError("sidecar payload checksum mismatch")
        if _sha256(self.profile) != self.profile_sha256:
            raise StorageError("sidecar profile checksum mismatch")
        if _sha256(self.request) != self.request_sha256:
            raise StorageError("sidecar request checksum mismatch")
        if not isinstance(self.source_fingerprint, str) or not self.source_fingerprint:
            raise StorageError("sidecar source fingerprint is missing")
        declared_source = self.source.get("fingerprint")
        if declared_source != self.source_fingerprint:
            raise StorageError("sidecar source declaration disagrees with source fingerprint")
        if _snapshot_fingerprint(self.source) != self.source_fingerprint:
            raise StorageError("sidecar source snapshot checksum mismatch")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "producer_version": self.producer_version,
            "payload": _jsonable(self.payload),
            "payload_sha256": self.payload_sha256,
            "source": _jsonable(self.source),
            "source_fingerprint": self.source_fingerprint,
            "profile": _jsonable(self.profile),
            "profile_sha256": self.profile_sha256,
            "request": _jsonable(self.request),
            "request_sha256": self.request_sha256,
        }


def _build_sidecar(payload: Any, source: SourceBatch | SourceSnapshot | Mapping[str, Any],
                   profile: Any, request: Any, producer_version: str) -> StoredSidecar:
    if not isinstance(producer_version, str) or not producer_version:
        raise ValueError("producer_version must be non-empty")
    payload_map = _payload(payload, "payload")
    source_map, source_fingerprint = _source_binding(source)
    profile_map, profile_sha = _binding(profile, "profile")
    request_map, request_sha = _binding(request, "request")
    return StoredSidecar(
        payload=payload_map,
        source=source_map,
        profile=profile_map,
        request=request_map,
        producer_version=producer_version,
        payload_sha256=_sha256(payload_map),
        source_fingerprint=source_fingerprint,
        profile_sha256=profile_sha,
        request_sha256=request_sha,
    )


def _source_paths(source: SourceBatch | SourceSnapshot | Mapping[str, Any]) -> set[Path]:
    if isinstance(source, SourceBatch):
        snapshot = source.snapshot
    elif isinstance(source, SourceSnapshot):
        snapshot = source
    else:
        return set()
    if snapshot.source_path is None:
        return set()
    root = Path(snapshot.source_path).resolve()
    return {(root / item.name).resolve() for item in snapshot.files}


def _atomic_create(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as handle:
            temporary = handle.name
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        # A hard-link publication is atomic and fails if a sidecar already
        # exists, so recovery never silently overwrites an older result.
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise FileExistsError(path) from exc
        finally:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
            temporary = None
    finally:
        if temporary is not None:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass


def write_sidecar(path: str | Path, payload: Any, *,
                  source: SourceBatch | SourceSnapshot | Mapping[str, Any],
                  profile: Any, request: Any,
                  producer_version: str = "F5-I-producer-v1") -> StoredSidecar:
    """Atomically create one new sidecar without touching any source file."""

    target = Path(path).expanduser().resolve()
    source_paths = _source_paths(source)
    if target in source_paths:
        raise StorageConflictError("sidecar target is an existing source file")
    if target.exists():
        raise FileExistsError(target)
    sidecar = _build_sidecar(payload, source, profile, request, producer_version)
    _atomic_create(target, _json_bytes(sidecar.to_dict()) + b"\n")
    return sidecar


def _parse_sidecar(raw: Mapping[str, Any]) -> StoredSidecar:
    required = {
        "schema_version", "producer_version", "payload", "payload_sha256",
        "source", "source_fingerprint", "profile", "profile_sha256",
        "request", "request_sha256",
    }
    missing = required - set(raw)
    if missing:
        raise StorageError(f"sidecar is missing {sorted(missing)!r}")
    return StoredSidecar(
        payload=raw["payload"], source=raw["source"], profile=raw["profile"],
        request=raw["request"], producer_version=raw["producer_version"],
        payload_sha256=raw["payload_sha256"], source_fingerprint=raw["source_fingerprint"],
        profile_sha256=raw["profile_sha256"], request_sha256=raw["request_sha256"],
        schema_version=raw["schema_version"],
    )


def read_sidecar(path: str | Path, *, source: SourceBatch | SourceSnapshot | Mapping[str, Any] | None = None,
                 profile: Any | None = None, request: Any | None = None,
                 producer_version: str | None = None) -> StoredSidecar:
    """Read one exact sidecar and reject all version/profile/source conflicts."""

    target = Path(path).expanduser().resolve()
    try:
        raw = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise StorageError(f"cannot read sidecar: {target}") from exc
    if not isinstance(raw, Mapping):
        raise StorageError("sidecar root must be a mapping")
    sidecar = _parse_sidecar(raw)
    if producer_version is not None and sidecar.producer_version != producer_version:
        raise StorageConflictError("sidecar producer version conflicts with recovery request")
    if source is not None:
        source_map, fingerprint = _source_binding(source)
        if fingerprint != sidecar.source_fingerprint or source_map != dict(sidecar.source):
            raise StorageConflictError("sidecar source binding conflicts with recovery source")
    if profile is not None:
        profile_map, digest = _binding(profile, "profile")
        if digest != sidecar.profile_sha256 or profile_map != dict(sidecar.profile):
            raise StorageConflictError("sidecar profile binding conflicts with recovery profile")
    if request is not None:
        request_map, digest = _binding(request, "request")
        if digest != sidecar.request_sha256 or request_map != dict(sidecar.request):
            raise StorageConflictError("sidecar request binding conflicts with recovery request")
    if source is not None and isinstance(source, SourceBatch):
        verify_source_unchanged(source)
    return sidecar


__all__ = [
    "STORAGE_SCHEMA_VERSION", "StorageError", "StorageConflictError",
    "StoredSidecar", "default_sidecar_path", "write_sidecar", "read_sidecar",
]
