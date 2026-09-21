"""Stable records shared by the two formula paths.

These records are deliberately plain and serialisable.  A status of
``unsupported`` or ``pending`` is part of the result and is never converted
to ``verified`` by a successful reference calculation.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
import json
from typing import Any, Mapping

SCHEMA_VERSION = "FM-formula-v2"


def _json(value: Any) -> Any:
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if isinstance(value, tuple):
        return [_json(v) for v in value]
    if isinstance(value, list):
        return [_json(v) for v in value]
    if isinstance(value, Mapping):
        return {str(k): _json(v) for k, v in value.items()}
    return value


@dataclass(frozen=True, slots=True)
class FormulaOptions:
    normalization: str = "jing_psf"
    density_mode: str = "per_branch"
    direct_profile_policy: str = "finite_analytic_profiles"
    reference_profile: str = "sequential_12_34"
    schema_version: str = SCHEMA_VERSION
    rule_versions: tuple[tuple[str, str], ...] = (("FM", "v2"), ("LM-12", "IM-4-formula-v1"))

    def __post_init__(self) -> None:
        if self.normalization not in {"jing_psf", "standard_2pi4"}:
            raise ValueError(f"unknown normalization: {self.normalization!r}")
        if self.density_mode not in {"per_branch", "summed"}:
            raise ValueError(f"unknown density_mode: {self.density_mode!r}")
        if self.direct_profile_policy != "finite_analytic_profiles":
            raise ValueError("unsupported direct_profile_policy")
        if self.reference_profile != "sequential_12_34":
            raise ValueError("unsupported reference_profile")
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError("unsupported formula schema_version")

    def to_dict(self) -> dict[str, Any]:
        return _json(asdict(self))


@dataclass(frozen=True, slots=True)
class RuleStep:
    rule_id: str
    before: str
    after: str
    eliminated: tuple[str, ...] = ()
    jacobian: str = "1"
    conditions: tuple[str, ...] = ()
    evidence: str = "implementation"

    def to_dict(self) -> dict[str, Any]:
        return _json(asdict(self))


@dataclass(frozen=True, slots=True)
class BranchRecord:
    key: str
    equations: tuple[str, ...]
    root_condition: str
    back_substitution: tuple[str, ...]
    domain: tuple[str, ...]
    multiplicity: int = 1
    coefficient: str = "1"
    summed: bool = False

    def to_dict(self) -> dict[str, Any]:
        return _json(asdict(self))


@dataclass(frozen=True, slots=True)
class CoordinateMap:
    source: tuple[str, ...]
    target: tuple[str, ...]
    forward: tuple[str, ...]
    inverse: tuple[str, ...]
    jacobian: str
    status: str = "built"

    def to_dict(self) -> dict[str, Any]:
        return _json(asdict(self))


@dataclass(frozen=True, slots=True)
class RawExpression:
    expression: str
    factors: tuple[str, ...]
    double_measures: tuple[str, ...]
    cancellations: tuple[str, ...]
    status: str = "built"
    steps: tuple[RuleStep, ...] = ()
    factor_sources: tuple[dict[str, Any], ...] = ()
    measure_sources: tuple[dict[str, Any], ...] = ()
    cancellation_sources: tuple[dict[str, Any], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return _json(asdict(self))


@dataclass(frozen=True, slots=True)
class CovariantResult:
    expression: str
    remaining_constraints: tuple[str, ...]
    status: str
    steps: tuple[RuleStep, ...] = ()
    diagnostics: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return _json(asdict(self))


@dataclass(frozen=True, slots=True)
class DirectFormula:
    profile: str
    target_variables: tuple[str, ...]
    angle_variables: tuple[str, ...]
    density_expression: str
    momentum_expressions: tuple[str, ...]
    jacobian: str
    branches: tuple[BranchRecord, ...]
    domain: tuple[str, ...]
    status: str
    coordinate_map: CoordinateMap | None = None
    diagnostics: tuple[str, ...] = ()
    numeric_density: Any = field(default=None, repr=False, compare=False)
    numeric_momenta: Any = field(default=None, repr=False, compare=False)
    normalization: str = ""
    differential_measure: str = ""
    angle_definitions: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data.pop("numeric_density", None)
        data.pop("numeric_momenta", None)
        return _json(data)


@dataclass(frozen=True, slots=True)
class BridgeFormula:
    profile: str
    variables: tuple[str, ...]
    density_expression: str
    momentum_expressions: tuple[str, ...]
    constraints: tuple[str, ...]
    status: str
    coordinate_map: CoordinateMap | None = None
    numeric_density: Any = field(default=None, repr=False, compare=False)
    numeric_momenta: Any = field(default=None, repr=False, compare=False)
    normalization: str = ""
    differential_measure: str = ""
    angle_definitions: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data.pop("numeric_density", None)
        data.pop("numeric_momenta", None)
        return _json(data)


@dataclass(frozen=True, slots=True)
class BenchmarkReport:
    status: str
    profile: str
    points: int
    compared: int
    max_absolute_residual: float | None
    max_relative_residual: float | None
    coordinate_residual: float | None
    branch_status: str
    failures: tuple[dict[str, Any], ...] = ()
    diagnostics: tuple[str, ...] = ()
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _json(asdict(self))


@dataclass(frozen=True, slots=True)
class FormulaArtifact:
    identity: dict[str, Any] | None
    options: FormulaOptions
    raw_expression: RawExpression | None
    covariant_result: CovariantResult | None
    direct_formula: DirectFormula | None
    bridge_formula: BridgeFormula | None
    internal_benchmark: BenchmarkReport | None
    statuses: dict[str, str]
    diagnostics: tuple[str, ...] = ()
    input_card: dict[str, Any] | None = None
    input_digest: str | None = None

    @property
    def direct_verified(self) -> bool:
        return bool(
            self.statuses.get("direct") == "verified"
            and self.direct_formula is not None
            and self.direct_formula.status == "verified"
            and self.statuses.get("covariant") == "verified"
            and self.covariant_result is not None
            and self.covariant_result.status == "verified"
            and self.statuses.get("benchmark") == "verified"
            and self.internal_benchmark is not None
            and self.internal_benchmark.status == "verified"
            and self.internal_benchmark.compared > 0
            and not self.internal_benchmark.failures
        )

    def to_dict(self) -> dict[str, Any]:
        return _json({
            "artifact_type": "four_body_formula",
            "schema_version": SCHEMA_VERSION,
            "identity": self.identity,
            "options": self.options,
            "raw_expression": self.raw_expression,
            "covariant_result": self.covariant_result,
            "direct_formula": self.direct_formula,
            "bridge_formula": self.bridge_formula,
            "internal_benchmark": self.internal_benchmark,
            "statuses": self.statuses,
            "diagnostics": self.diagnostics,
            "input_card": self.input_card,
            "input_digest": self.input_digest,
        })

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
