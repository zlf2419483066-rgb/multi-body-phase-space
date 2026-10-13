"""Immutable evidence payloads for five-body invariant reviews.

This module owns only the evidence envelope.  Physics profiles, coordinate
construction, and exact algebra remain owned by their respective modules.  A
certificate is deliberately a sidecar: it contains a copy of the source
identity and candidate record, but never mutates ``CandidateMomentumRecord``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from types import MappingProxyType
from typing import Any, Mapping

from .contracts import fraction_text


EVIDENCE_SCHEMA_VERSION = "F5-I-evidence-v1"
OUTCOMES = ("pass", "reject", "unknown")
INPUT_STATUSES = ("valid", "unsupported", "invalid")
EXECUTION_STATUSES = ("complete", "unsupported", "failed", "not_run")
CERTIFICATE_KINDS = (
    "affine_dependence",
    "dimension_bound",
    "polynomial_partition",
    "polynomial_dependence",
    "regular_point",
    "physical_exclusion",
    "analytic_family",
    "unknown_record",
)


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


def _deep_freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _deep_freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_deep_freeze(item) for item in value)
    return value


def _freeze_mapping(value: Mapping[str, Any] | None) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError("certificate metadata must be a mapping")
    return _deep_freeze(value)


@dataclass(frozen=True, slots=True)
class EvidenceCertificate:
    """Serializable producer evidence for one complete fixed-label identity.

    ``payload`` carries the exact algebra objects in the backend's stable
    serialization form (H, h, N columns, directional derivatives, D, C and
    optional R/F).  The verifier must rebuild them from ``source_record`` and
    ``identity``; ``producer`` is provenance, never a truth flag.
    """

    outcome: str
    kind: str
    source_record: Mapping[str, Any]
    identity: Mapping[str, Any]
    request: Mapping[str, Any]
    profile: Mapping[str, Any]
    variables: tuple[Any, ...]
    B: tuple[tuple[Any, ...], ...]
    b: tuple[Any, ...]
    payload: Mapping[str, Any] = field(default_factory=dict)
    input_status: str = "valid"
    execution_status: str = "complete"
    reason: str = ""
    producer: Mapping[str, Any] = field(default_factory=dict)
    verifier: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = EVIDENCE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != EVIDENCE_SCHEMA_VERSION:
            raise ValueError(f"unsupported evidence schema: {self.schema_version!r}")
        if self.outcome not in OUTCOMES:
            raise ValueError(f"unsupported evidence outcome: {self.outcome!r}")
        if self.kind not in CERTIFICATE_KINDS:
            raise ValueError(f"unsupported evidence kind: {self.kind!r}")
        if self.input_status not in INPUT_STATUSES:
            raise ValueError(f"unsupported input status: {self.input_status!r}")
        if self.execution_status not in EXECUTION_STATUSES:
            raise ValueError(f"unsupported execution status: {self.execution_status!r}")
        if not isinstance(self.reason, str):
            raise TypeError("certificate reason must be a string")
        if not isinstance(self.source_record, Mapping):
            raise TypeError("certificate source_record must be a mapping")
        if not isinstance(self.identity, Mapping):
            raise TypeError("certificate identity must be a mapping")
        if not isinstance(self.request, Mapping) or not isinstance(self.profile, Mapping):
            raise TypeError("certificate request and profile must be mappings")
        if not isinstance(self.payload, Mapping):
            raise TypeError("certificate payload must be a mapping")
        if not isinstance(self.variables, tuple):
            object.__setattr__(self, "variables", tuple(self.variables))
        if not isinstance(self.B, tuple):
            object.__setattr__(self, "B", tuple(tuple(row) for row in self.B))
        if not isinstance(self.b, tuple):
            object.__setattr__(self, "b", tuple(self.b))
        object.__setattr__(self, "source_record", _freeze_mapping(self.source_record))
        object.__setattr__(self, "identity", _freeze_mapping(self.identity))
        object.__setattr__(self, "request", _freeze_mapping(self.request))
        object.__setattr__(self, "profile", _freeze_mapping(self.profile))
        object.__setattr__(self, "payload", _freeze_mapping(self.payload))
        object.__setattr__(self, "producer", _freeze_mapping(self.producer))
        object.__setattr__(self, "verifier", _freeze_mapping(self.verifier))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "outcome": self.outcome,
            "kind": self.kind,
            "input_status": self.input_status,
            "execution_status": self.execution_status,
            "reason": self.reason,
            "source_record": _jsonable(self.source_record),
            "identity": _jsonable(self.identity),
            "request": _jsonable(self.request),
            "profile": _jsonable(self.profile),
            "variables": _jsonable(self.variables),
            "B": _jsonable(self.B),
            "b": _jsonable(self.b),
            "payload": _jsonable(self.payload),
            "producer": _jsonable(self.producer),
            "verifier": _jsonable(self.verifier),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "EvidenceCertificate":
        if not isinstance(data, Mapping):
            raise TypeError("evidence certificate must be a mapping")
        required = {
            "schema_version", "outcome", "kind", "source_record", "identity",
            "request", "profile", "variables", "B", "b", "payload",
        }
        missing = required - set(data)
        if missing:
            raise ValueError(f"evidence certificate is missing {sorted(missing)!r}")
        return cls(
            outcome=data["outcome"], kind=data["kind"],
            source_record=data["source_record"], identity=data["identity"],
            request=data["request"], profile=data["profile"],
            variables=tuple(data["variables"]),
            B=tuple(tuple(row) for row in data["B"]), b=tuple(data["b"]),
            payload=data.get("payload", {}), input_status=data.get("input_status", "valid"),
            execution_status=data.get("execution_status", "complete"),
            reason=data.get("reason", ""), producer=data.get("producer", {}),
            verifier=data.get("verifier", {}), schema_version=data["schema_version"],
        )


@dataclass(frozen=True, slots=True)
class VerificationReport:
    """Independent readback result; ``accepted`` is verifier-owned."""

    accepted: bool
    status: str
    checks: tuple[Mapping[str, Any], ...] = ()
    errors: tuple[str, ...] = ()
    shared_cas: bool = False
    rebuilt: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.status not in OUTCOMES:
            raise ValueError(f"unsupported verification status: {self.status!r}")
        if type(self.accepted) is not bool:
            raise TypeError("verification accepted must be a boolean")
        object.__setattr__(self, "checks", tuple(dict(item) for item in self.checks))
        object.__setattr__(self, "errors", tuple(str(item) for item in self.errors))
        object.__setattr__(self, "rebuilt", _freeze_mapping(self.rebuilt))

    def to_dict(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "status": self.status,
            "checks": _jsonable(self.checks),
            "errors": list(self.errors),
            "shared_cas": self.shared_cas,
            "rebuilt": _jsonable(self.rebuilt),
        }


def make_certificate(**kwargs: Any) -> EvidenceCertificate:
    """Construct a certificate while keeping the schema in one place."""

    return EvidenceCertificate(**kwargs)


def parse_certificate(data: Mapping[str, Any] | EvidenceCertificate) -> EvidenceCertificate:
    if isinstance(data, EvidenceCertificate):
        return data
    return EvidenceCertificate.from_dict(data)


__all__ = [
    "EVIDENCE_SCHEMA_VERSION", "OUTCOMES", "INPUT_STATUSES", "EXECUTION_STATUSES", "CERTIFICATE_KINDS",
    "EvidenceCertificate", "VerificationReport", "make_certificate", "parse_certificate",
]
