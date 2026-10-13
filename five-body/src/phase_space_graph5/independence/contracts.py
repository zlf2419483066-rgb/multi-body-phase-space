"""Shared contracts for the five-body independence review.

This module contains data contracts only.  It deliberately has no dependency
on the existing topology, canonicalization, or candidate-generation modules.
All numeric values crossing a JSON boundary are canonical rational strings;
binary floating point is never accepted as an exact coefficient.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import re
from typing import Any, Mapping, Sequence


PROFILE_VERSION = "F5-I-profile-v1"
EVIDENCE_SCHEMA_VERSION = "F5-I-evidence-v1"
COORDINATE_VERSION = "F5-I-affine-coordinate-v1"

_RATIONAL_RE = re.compile(r"^[+-]?[0-9]+(?:/[0-9]+)?$")


def as_fraction(value: int | str | Fraction) -> Fraction:
    """Return an exact rational, rejecting booleans, floats, and decimals."""

    if isinstance(value, bool):
        raise TypeError("boolean values are not valid exact rationals")
    if isinstance(value, Fraction):
        return value
    if isinstance(value, int):
        return Fraction(value)
    if isinstance(value, str):
        if not _RATIONAL_RE.fullmatch(value):
            raise ValueError("rational text must be an integer or numerator/denominator")
        numerator, separator, denominator = value.partition("/")
        if not separator:
            return Fraction(int(numerator), 1)
        if int(denominator) == 0:
            raise ValueError("rational denominator cannot be zero")
        return Fraction(int(numerator), int(denominator))
    raise TypeError("exact rationals must be int, Fraction, or canonical rational text")


def fraction_text(value: int | str | Fraction) -> str:
    """Serialize an exact rational in the one JSON representation we accept."""

    rational = as_fraction(value)
    if rational.denominator == 1:
        return str(rational.numerator)
    return f"{rational.numerator}/{rational.denominator}"


def _require_string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _require_tuple_strings(values: Sequence[object], name: str) -> tuple[str, ...]:
    result = tuple(_require_string(value, name) for value in values)
    return result


def _optional_rank(value: object, name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer or null")
    return value


REGULAR_DOMAIN_CONDITIONS = (
    "three_plus_one_minkowski",
    "metric_signature_+---",
    "fixed_future_timelike_total_momentum",
    "five_nonnegative_mass_squares",
    "parent_mass_above_sum_of_particle_masses",
    "on_shell_positive_energy_momenta",
    "momentum_conservation",
    "span_rank_four",
    "gram_rank_four",
)


@dataclass(frozen=True, slots=True)
class RegularDomainContract:
    """The explicit 3+1 regular physical domain used by the theorem.

    The contract is intentionally closed: callers cannot replace it with an
    arbitrary list of conditions and still claim the full-domain theorem.
    ``threshold_status`` records whether the square-root threshold has been
    checked by an exact backend; no floating approximation is accepted.
    """

    spacetime: str = "3+1"
    metric: str = "(+---)"
    kind: str = "full_regular_physical"
    dimension: int = 8
    conditions: tuple[str, ...] = REGULAR_DOMAIN_CONDITIONS
    nonempty_evidence: str = "analytic_t_family"
    threshold_status: str = "unsupported"

    def __post_init__(self) -> None:
        object.__setattr__(self, "conditions", tuple(self.conditions))
        if self.spacetime != "3+1":
            raise ValueError("only the explicit 3+1 domain is supported")
        if self.metric != "(+---)":
            raise ValueError("the 3+1 profile requires metric (+---)")
        if self.kind != "full_regular_physical":
            raise ValueError("only the full regular physical domain is supported")
        if self.dimension != 8:
            raise ValueError("the regular five-body domain has dimension 8")
        if tuple(self.conditions) != REGULAR_DOMAIN_CONDITIONS:
            raise ValueError("regular domain conditions are fixed by F5-I-spec-v1")
        if self.nonempty_evidence != "analytic_t_family":
            raise ValueError("the domain needs the declared analytic non-empty evidence")
        if self.threshold_status not in {"unsupported", "verified"}:
            raise ValueError("threshold_status must be unsupported or verified")

    def to_dict(self) -> dict[str, object]:
        return {
            "spacetime": self.spacetime,
            "metric": self.metric,
            "kind": self.kind,
            "dimension": self.dimension,
            "conditions": list(self.conditions),
            "nonempty_evidence": self.nonempty_evidence,
            "threshold_status": self.threshold_status,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "RegularDomainContract":
        return cls(
            spacetime=_require_string(payload.get("spacetime"), "spacetime"),
            metric=_require_string(payload.get("metric"), "metric"),
            kind=_require_string(payload.get("kind"), "kind"),
            dimension=payload.get("dimension", 0) if isinstance(payload.get("dimension"), int) else 0,
            conditions=tuple(payload.get("conditions", ())),
            nonempty_evidence=_require_string(payload.get("nonempty_evidence"), "nonempty_evidence"),
            threshold_status=_require_string(payload.get("threshold_status"), "threshold_status"),
        )


@dataclass(frozen=True, slots=True)
class PhysicsProfile:
    """A fixed exact mass profile and its explicit theorem domain."""

    M2: Fraction
    m2: tuple[Fraction, ...]
    domain: RegularDomainContract
    quantifier: str = "fixed_profile"
    coefficient_field: str = "Q"
    P_convention: str = "fixed_future_timelike_total"
    version: str = PROFILE_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "M2", as_fraction(self.M2))
        object.__setattr__(self, "m2", tuple(as_fraction(value) for value in self.m2))
        if len(self.m2) != 5:
            raise ValueError("five-body PhysicsProfile requires exactly five m² values")
        if self.M2 <= 0:
            raise ValueError("M² must be positive")
        if any(value < 0 for value in self.m2):
            raise ValueError("each m² must be non-negative")
        if not isinstance(self.domain, RegularDomainContract):
            raise TypeError("domain must be a RegularDomainContract")
        if self.quantifier != "fixed_profile":
            raise ValueError("only the fixed_profile quantifier is implemented")
        if self.coefficient_field != "Q":
            raise ValueError("the first exact backend supports coefficient field Q only")
        if self.P_convention != "fixed_future_timelike_total":
            raise ValueError("unsupported total-momentum convention")
        if self.version != PROFILE_VERSION:
            raise ValueError("unsupported PhysicsProfile version")

    @classmethod
    def regular_3_plus_1(
        cls,
        M2: int | str | Fraction,
        m2: Sequence[int | str | Fraction],
        *,
        threshold_status: str = "unsupported",
    ) -> "PhysicsProfile":
        """Build the only supported profile with all domain fields explicit."""

        return cls(
            M2=as_fraction(M2),
            m2=tuple(as_fraction(value) for value in m2),
            domain=RegularDomainContract(threshold_status=threshold_status),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "M2": fraction_text(self.M2),
            "m2": [fraction_text(value) for value in self.m2],
            "domain": self.domain.to_dict(),
            "quantifier": self.quantifier,
            "coefficient_field": self.coefficient_field,
            "P_convention": self.P_convention,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "PhysicsProfile":
        raw_m2 = payload.get("m2")
        if not isinstance(raw_m2, Sequence) or isinstance(raw_m2, (str, bytes)):
            raise TypeError("m2 must be a sequence of five rational strings")
        domain = payload.get("domain")
        if not isinstance(domain, Mapping):
            raise TypeError("domain must be an object")
        version = _require_string(payload.get("version"), "version")
        return cls(
            M2=as_fraction(payload.get("M2")),
            m2=tuple(as_fraction(value) for value in raw_m2),
            domain=RegularDomainContract.from_dict(domain),
            quantifier=_require_string(payload.get("quantifier"), "quantifier"),
            coefficient_field=_require_string(payload.get("coefficient_field"), "coefficient_field"),
            P_convention=_require_string(payload.get("P_convention"), "P_convention"),
            version=version,
        )


@dataclass(frozen=True, slots=True)
class InvariantOccurrence:
    """One appearance of an invariant in the reviewed candidate record."""

    subset: tuple[int, ...]
    ordinal: int
    occurrence_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "subset", tuple(self.subset))
        subset = self.subset
        if len(subset) not in {2, 3, 4}:
            raise ValueError("five-body invariant occurrences have cardinality 2, 3, or 4")
        if any(isinstance(item, bool) or not isinstance(item, int) for item in subset):
            raise TypeError("occurrence particle indices must be integers")
        if tuple(sorted(subset)) != subset or len(set(subset)) != len(subset):
            raise ValueError("occurrence subsets must be sorted and contain no duplicate particle")
        if subset[0] < 1 or subset[-1] > 5:
            raise ValueError("occurrence particle indices must lie in 1..5")
        if isinstance(self.ordinal, bool) or not isinstance(self.ordinal, int) or self.ordinal < 0:
            raise ValueError("occurrence ordinal must be a non-negative integer")
        if self.occurrence_id is not None:
            _require_string(self.occurrence_id, "occurrence_id")

    @property
    def label(self) -> str:
        return "s" + "".join(str(item) for item in self.subset)

    def to_dict(self) -> dict[str, object]:
        return {
            "subset": list(self.subset),
            "ordinal": self.ordinal,
            "occurrence_id": self.occurrence_id,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "InvariantOccurrence":
        subset = payload.get("subset")
        if not isinstance(subset, Sequence) or isinstance(subset, (str, bytes)):
            raise TypeError("subset must be an integer sequence")
        ordinal = payload.get("ordinal")
        if not isinstance(ordinal, int) or isinstance(ordinal, bool):
            raise TypeError("ordinal must be an integer")
        occurrence_id = payload.get("occurrence_id")
        if occurrence_id is not None and not isinstance(occurrence_id, str):
            raise TypeError("occurrence_id must be text or null")
        return cls(tuple(subset), ordinal, occurrence_id)


@dataclass(frozen=True, slots=True)
class ReviewedCandidate:
    """Read-only binding to a candidate source record.

    ``review_status='reviewed'`` is a source assertion used to select the
    input path; it is not a trusted proof.  A source adapter/verifier must
    still revalidate the original record before accepting a certificate.
    """

    topology_key: str
    identity_key: str
    mode: str
    source_digest: str
    occurrences: tuple[InvariantOccurrence, ...]
    review_status: str = "reviewed"

    def __post_init__(self) -> None:
        object.__setattr__(self, "occurrences", tuple(self.occurrences))
        for value, name in (
            (self.topology_key, "topology_key"),
            (self.identity_key, "identity_key"),
            (self.mode, "mode"),
            (self.source_digest, "source_digest"),
        ):
            _require_string(value, name)
        if self.review_status != "reviewed":
            raise ValueError("independence coordinates require a reviewed candidate record")
        ordered = self.occurrences
        if any(not isinstance(item, InvariantOccurrence) for item in ordered):
            raise TypeError("occurrences must contain InvariantOccurrence values")
        ordinals = tuple(item.ordinal for item in ordered)
        if ordinals != tuple(range(len(ordered))):
            raise ValueError("occurrences must have unique source ordinals 0..d-1 in order")

    def to_dict(self) -> dict[str, object]:
        return {
            "topology_key": self.topology_key,
            "identity_key": self.identity_key,
            "mode": self.mode,
            "source_digest": self.source_digest,
            "occurrences": [item.to_dict() for item in self.occurrences],
            "review_status": self.review_status,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "ReviewedCandidate":
        raw = payload.get("occurrences")
        if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
            raise TypeError("occurrences must be a sequence")
        return cls(
            topology_key=_require_string(payload.get("topology_key"), "topology_key"),
            identity_key=_require_string(payload.get("identity_key"), "identity_key"),
            mode=_require_string(payload.get("mode"), "mode"),
            source_digest=_require_string(payload.get("source_digest"), "source_digest"),
            occurrences=tuple(
                InvariantOccurrence.from_dict(item) if isinstance(item, Mapping) else (_ for _ in ()).throw(TypeError("invalid occurrence"))
                for item in raw
            ),
            review_status=_require_string(payload.get("review_status"), "review_status"),
        )


@dataclass(frozen=True, slots=True)
class EvaluationRequest:
    """A fixed-profile request for one reviewed label record."""

    profile: PhysicsProfile
    candidate: ReviewedCandidate
    requested_scope: str = "full_regular_physical"
    requested_quantifier: str = "fixed_profile_general"

    def __post_init__(self) -> None:
        if not isinstance(self.profile, PhysicsProfile):
            raise TypeError("profile must be PhysicsProfile")
        if not isinstance(self.candidate, ReviewedCandidate):
            raise TypeError("candidate must be ReviewedCandidate")
        if self.requested_scope != "full_regular_physical":
            raise ValueError("only the full regular physical scope is supported")
        if self.requested_quantifier != "fixed_profile_general":
            raise ValueError("only fixed-profile general independence is supported")

    def to_dict(self) -> dict[str, object]:
        return {
            "profile": self.profile.to_dict(),
            "candidate": self.candidate.to_dict(),
            "requested_scope": self.requested_scope,
            "requested_quantifier": self.requested_quantifier,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "EvaluationRequest":
        profile = payload.get("profile")
        candidate = payload.get("candidate")
        if not isinstance(profile, Mapping) or not isinstance(candidate, Mapping):
            raise TypeError("profile and candidate must be objects")
        return cls(
            profile=PhysicsProfile.from_dict(profile),
            candidate=ReviewedCandidate.from_dict(candidate),
            requested_scope=_require_string(payload.get("requested_scope"), "requested_scope"),
            requested_quantifier=_require_string(payload.get("requested_quantifier"), "requested_quantifier"),
        )


@dataclass(frozen=True, slots=True)
class LabelEvaluation:
    """One independent evaluation or an explicit unknown/input failure."""

    topology_key: str
    identity_key: str
    mode: str
    profile: PhysicsProfile
    occurrences: tuple[InvariantOccurrence, ...]
    status: str
    input_status: str
    execution_status: str
    affine_rank: int | None = None
    point_rank: int | None = None
    generic_rank: int | None = None
    certificate: Mapping[str, object] | None = None
    checks: tuple[str, ...] = ()
    provenance: Mapping[str, object] | None = None
    schema_version: str = EVIDENCE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "occurrences", tuple(self.occurrences))
        object.__setattr__(self, "checks", tuple(self.checks))
        for value, name in ((self.topology_key, "topology_key"), (self.identity_key, "identity_key"), (self.mode, "mode")):
            _require_string(value, name)
        if not isinstance(self.profile, PhysicsProfile):
            raise TypeError("profile must be PhysicsProfile")
        if any(not isinstance(item, InvariantOccurrence) for item in self.occurrences):
            raise TypeError("occurrences must contain InvariantOccurrence values")
        if self.status not in {"pass", "reject", "unknown"}:
            raise ValueError("status must be pass, reject, or unknown")
        if self.input_status not in {"valid", "invalid", "unsupported"}:
            raise ValueError("input_status must be valid, invalid, or unsupported")
        if self.execution_status not in {"complete", "unsupported", "failed", "not_run"}:
            raise ValueError("unsupported execution_status")
        for value, name in ((self.affine_rank, "affine_rank"), (self.point_rank, "point_rank"), (self.generic_rank, "generic_rank")):
            if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 0):
                raise ValueError(f"{name} must be a non-negative integer or null")
        if self.schema_version != EVIDENCE_SCHEMA_VERSION:
            raise ValueError("unsupported evidence schema version")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "topology_key": self.topology_key,
            "identity_key": self.identity_key,
            "mode": self.mode,
            "profile": self.profile.to_dict(),
            "occurrences": [item.to_dict() for item in self.occurrences],
            "status": self.status,
            "input_status": self.input_status,
            "execution_status": self.execution_status,
            "affine_rank": self.affine_rank,
            "point_rank": self.point_rank,
            "generic_rank": self.generic_rank,
            "certificate": dict(self.certificate) if self.certificate is not None else None,
            "checks": list(self.checks),
            "provenance": dict(self.provenance) if self.provenance is not None else None,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "LabelEvaluation":
        profile = payload.get("profile")
        raw_occurrences = payload.get("occurrences")
        if not isinstance(profile, Mapping) or not isinstance(raw_occurrences, Sequence) or isinstance(raw_occurrences, (str, bytes)):
            raise TypeError("profile and occurrences must be objects/sequences")
        certificate = payload.get("certificate")
        provenance = payload.get("provenance")
        if certificate is not None and not isinstance(certificate, Mapping):
            raise TypeError("certificate must be an object or null")
        if provenance is not None and not isinstance(provenance, Mapping):
            raise TypeError("provenance must be an object or null")
        checks = payload.get("checks", ())
        if not isinstance(checks, Sequence) or isinstance(checks, (str, bytes)):
            raise TypeError("checks must be a sequence")
        # ``verified`` is intentionally not interpreted here.  It remains an
        # untrusted producer assertion until a verifier recomputes the checks.
        return cls(
            topology_key=_require_string(payload.get("topology_key"), "topology_key"),
            identity_key=_require_string(payload.get("identity_key"), "identity_key"),
            mode=_require_string(payload.get("mode"), "mode"),
            profile=PhysicsProfile.from_dict(profile),
            occurrences=tuple(
                InvariantOccurrence.from_dict(item) if isinstance(item, Mapping) else (_ for _ in ()).throw(TypeError("invalid occurrence"))
                for item in raw_occurrences
            ),
            status=_require_string(payload.get("status"), "status"),
            input_status=_require_string(payload.get("input_status"), "input_status"),
            execution_status=_require_string(payload.get("execution_status"), "execution_status"),
            affine_rank=_optional_rank(payload.get("affine_rank"), "affine_rank"),
            point_rank=_optional_rank(payload.get("point_rank"), "point_rank"),
            generic_rank=_optional_rank(payload.get("generic_rank"), "generic_rank"),
            certificate=dict(certificate) if certificate is not None else None,
            checks=tuple(checks),
            provenance=dict(provenance) if provenance is not None else None,
            schema_version=_require_string(payload.get("schema_version"), "schema_version"),
        )


@dataclass(frozen=True, slots=True)
class TopologyEvaluation:
    """Topology aggregation with source and independence coverage separated."""

    topology_key: str
    mode: str
    profile: PhysicsProfile
    status: str
    source_coverage: Mapping[str, object]
    independence_coverage: Mapping[str, object]
    label_evaluations: tuple[LabelEvaluation, ...]
    witness_identity_key: str | None = None
    unknown_reasons: tuple[str, ...] = ()
    schema_version: str = EVIDENCE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "label_evaluations", tuple(self.label_evaluations))
        object.__setattr__(self, "unknown_reasons", tuple(self.unknown_reasons))
        _require_string(self.topology_key, "topology_key")
        if self.witness_identity_key is not None:
            _require_string(self.witness_identity_key, "witness_identity_key")
        _require_string(self.mode, "mode")
        if not isinstance(self.profile, PhysicsProfile):
            raise TypeError("profile must be PhysicsProfile")
        if self.status not in {"pass", "reject", "unknown"}:
            raise ValueError("status must be pass, reject, or unknown")
        if any(not isinstance(item, LabelEvaluation) for item in self.label_evaluations):
            raise TypeError("label_evaluations must contain LabelEvaluation values")
        if self.schema_version != EVIDENCE_SCHEMA_VERSION:
            raise ValueError("unsupported evidence schema version")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "topology_key": self.topology_key,
            "witness_identity_key": self.witness_identity_key,
            "mode": self.mode,
            "profile": self.profile.to_dict(),
            "status": self.status,
            "source_coverage": dict(self.source_coverage),
            "independence_coverage": dict(self.independence_coverage),
            "label_evaluations": [item.to_dict() for item in self.label_evaluations],
            "unknown_reasons": list(self.unknown_reasons),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "TopologyEvaluation":
        profile = payload.get("profile")
        source_coverage = payload.get("source_coverage")
        independence_coverage = payload.get("independence_coverage")
        raw_labels = payload.get("label_evaluations")
        if not isinstance(profile, Mapping) or not isinstance(source_coverage, Mapping) or not isinstance(independence_coverage, Mapping):
            raise TypeError("profile and coverage fields must be objects")
        if not isinstance(raw_labels, Sequence) or isinstance(raw_labels, (str, bytes)):
            raise TypeError("label_evaluations must be a sequence")
        reasons = payload.get("unknown_reasons", ())
        if not isinstance(reasons, Sequence) or isinstance(reasons, (str, bytes)):
            raise TypeError("unknown_reasons must be a sequence")
        return cls(
            topology_key=_require_string(payload.get("topology_key"), "topology_key"),
            witness_identity_key=(
                _require_string(payload.get("witness_identity_key"), "witness_identity_key")
                if payload.get("witness_identity_key") is not None
                else None
            ),
            mode=_require_string(payload.get("mode"), "mode"),
            profile=PhysicsProfile.from_dict(profile),
            status=_require_string(payload.get("status"), "status"),
            source_coverage=dict(source_coverage),
            independence_coverage=dict(independence_coverage),
            label_evaluations=tuple(
                LabelEvaluation.from_dict(item) if isinstance(item, Mapping) else (_ for _ in ()).throw(TypeError("invalid label evaluation"))
                for item in raw_labels
            ),
            unknown_reasons=tuple(reasons),
            schema_version=_require_string(payload.get("schema_version"), "schema_version"),
        )


def unsupported_threshold_reason() -> str:
    """Stable reason used when Q² data cannot prove ``M > sum(m_i)``."""

    return "exact comparison of M=sqrt(M²) with sum_i sqrt(m_i²) is unavailable"


__all__ = [
    "PROFILE_VERSION",
    "EVIDENCE_SCHEMA_VERSION",
    "COORDINATE_VERSION",
    "REGULAR_DOMAIN_CONDITIONS",
    "as_fraction",
    "fraction_text",
    "RegularDomainContract",
    "PhysicsProfile",
    "InvariantOccurrence",
    "ReviewedCandidate",
    "EvaluationRequest",
    "LabelEvaluation",
    "TopologyEvaluation",
    "unsupported_threshold_reason",
]
