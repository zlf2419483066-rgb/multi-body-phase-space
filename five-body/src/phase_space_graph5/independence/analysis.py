"""Fixed-profile five-body independence analysis.

The producer is intentionally conservative: unsupported exact algebra,
unproved thresholds, and mixed polynomial partitions become ``unknown``.  A
``LabelEvaluation`` always carries an ``EvidenceCertificate`` sidecar whose
source fields are copied from the read-only source adapter.
"""

from __future__ import annotations

from fractions import Fraction
import sys
from typing import Any, Mapping, Sequence

from . import exact_algebra as algebra
from .contracts import EvaluationRequest, LabelEvaluation
from .coordinates import build_label_affine_map, pair_coordinate_map
from .certificates import EvidenceCertificate
from .gram import GramData, GramUnsupported, build_gram_data_from_affine, validate_profile
from .sources import VerifiedSource, make_evaluation_request


Z_ORDER = ("s12", "s23", "s34", "s45", "s1234", "s2345", "s123", "s234", "s345")


class AnalysisUnknown(ValueError):
    """An exact capability or physical-domain condition is unresolved."""


def _serialize_matrix(matrix: Sequence[Sequence[object]]) -> list[list[str]]:
    from .contracts import fraction_text
    return [[fraction_text(value) for value in row] for row in matrix]


def _serialize_vector(vector: Sequence[object]) -> list[str]:
    from .contracts import fraction_text
    return [fraction_text(value) for value in vector]


def _cas_value(result: algebra.CASResult, operation: str) -> Any:
    if result.status != "available":
        raise AnalysisUnknown(result.reason or f"exact {operation} is unavailable")
    return result.value


def _is_one(polynomial: algebra.SparsePolynomial) -> bool:
    zero = (0,) * len(polynomial.variables)
    return polynomial.terms == ((zero, Fraction(1)),)


def _is_constant(polynomial: algebra.SparsePolynomial) -> bool:
    return len(polynomial.terms) <= 1 and all(not any(monomial) for monomial, _ in polynomial.terms)


def _is_zero(polynomial: algebra.SparsePolynomial) -> bool:
    return not polynomial.terms


def _poly_payload(value: algebra.SparsePolynomial) -> dict[str, object]:
    return algebra.serialize_polynomial(value)


def _loaded_cas_version() -> str | None:
    """Read an already-loaded optional CAS version without importing it."""

    module = sys.modules.get("sympy")
    version = getattr(module, "__version__", None) if module is not None else None
    return version if isinstance(version, str) else None


def _common_payload(
    source: VerifiedSource,
    request: EvaluationRequest,
    affine: Any,
    *,
    profile_check: Mapping[str, Any],
    payload: Mapping[str, Any],
    outcome: str,
    kind: str,
    reason: str,
    input_status: str = "valid",
    execution_status: str = "complete",
    checks: Sequence[str] = (),
) -> LabelEvaluation:
    evidence = EvidenceCertificate(
        outcome=outcome,
        kind=kind,
        source_record=source.source_record.to_dict(),
        identity=source.identity.to_dict(),
        request=request.to_dict(),
        profile=request.profile.to_dict(),
        variables=tuple(item.to_dict() for item in request.candidate.occurrences),
        B=tuple(tuple(row) for row in affine.B),
        b=tuple(affine.b),
        payload={"profile_check": dict(profile_check), **dict(payload)},
        input_status=input_status,
        execution_status=execution_status,
        reason=reason,
        producer={
            "module": "phase_space_graph5.independence.analysis",
            "spec_version": "F5-I-spec-v1",
            "producer_version": "F5-I-analysis-v1",
            "algebra_backend": "exact_algebra",
            "cas_backend": "optional_sympy_qq",
            "cas_version": _loaded_cas_version(),
            "shared_cas_with_verifier": True,
            "implementation_tested": False,
        },
    )
    return LabelEvaluation(
        topology_key=request.candidate.topology_key,
        identity_key=request.candidate.identity_key,
        mode=source.mode,
        profile=request.profile,
        occurrences=request.candidate.occurrences,
        status=outcome,
        input_status=input_status,
        execution_status=execution_status,
        affine_rank=payload.get("affine_rank"),
        point_rank=payload.get("point_rank"),
        generic_rank=payload.get("generic_rank"),
        certificate=evidence.to_dict(),
        checks=tuple(checks),
        provenance={"source_digest": request.candidate.source_digest,
                    "identity_digest": source.digest,
                    "profile_check": dict(profile_check)},
    )


def _unknown(
    source: VerifiedSource,
    request: EvaluationRequest,
    affine: Any,
    profile_check: Mapping[str, Any],
    reason: str,
    *,
    payload: Mapping[str, Any] | None = None,
    input_status: str = "unsupported",
    execution_status: str = "unsupported",
    checks: Sequence[str] = (),
) -> LabelEvaluation:
    return _common_payload(
        source, request, affine, profile_check=profile_check, payload=dict(payload or {}),
        outcome="unknown", kind="unknown_record", reason=reason,
        input_status=input_status, execution_status=execution_status, checks=checks,
    )


def _build_gram(profile: Any) -> GramData:
    pair_map = pair_coordinate_map(profile)
    return build_gram_data_from_affine(profile, pair_map, algebra=algebra)


def analyze_label(source: VerifiedSource, request: EvaluationRequest) -> LabelEvaluation:
    """Analyze one fixed-label source identity under one fixed QQ profile."""

    if not isinstance(source, VerifiedSource):
        raise TypeError("source must be a VerifiedSource")
    if not isinstance(request, EvaluationRequest):
        raise TypeError("request must be an EvaluationRequest")
    canonical_request = make_evaluation_request(source, request.profile)
    if canonical_request.to_dict() != request.to_dict():
        raise ValueError("request does not match the verified source binding")
    profile_check = validate_profile(request.profile)
    affine = build_label_affine_map(request.candidate, request.profile)
    B = tuple(tuple(value for value in row) for row in affine.B)
    b = tuple(affine.b)
    d = len(B)
    if profile_check.get("status") != "valid":
        return _unknown(source, request, affine, profile_check,
                        "fixed profile threshold/domain is not proven",
                        payload={"B": _serialize_matrix(B), "b": _serialize_vector(b), "affine_rank": None})
    if d == 0:
        return _unknown(source, request, affine, profile_check,
                        "empty invariant occurrence set is unsupported",
                        payload={"B": [], "b": [], "affine_rank": 0}, input_status="invalid")
    affine_rank = algebra.rank(B, ncols=9)
    base_payload: dict[str, Any] = {
        "B": _serialize_matrix(B), "b": _serialize_vector(b),
        "affine_rank": affine_rank,
    }
    if affine_rank < d:
        multipliers = algebra.left_nullspace(B, ncols=9)
        relation = {"multipliers": algebra.columns_payload(multipliers),
                    "rhs": _serialize_vector(tuple(sum((left * right for left, right in zip(vector, b)), Fraction(0)) for vector in multipliers))}
        return _common_payload(
            source, request, affine, profile_check=profile_check,
            payload={**base_payload, "left_kernel_columns": relation["multipliers"], "relations": relation},
            outcome="reject", kind="affine_dependence",
            reason="the affine coefficient rows are linearly dependent",
            checks=("exact_rank_B", "complete_left_kernel"),
        )
    if d > 8:
        return _common_payload(
            source, request, affine, profile_check=profile_check, payload=base_payload,
            outcome="reject", kind="dimension_bound",
            reason="more than eight invariant functions cannot be generally independent on the regular domain",
            checks=("exact_rank_B", "regular_domain_dimension_8"),
        )
    try:
        gram = _build_gram(request.profile)
        N = algebra.nullspace(B, ncols=9)
        if len(N) != 9 - d or any(len(column) != 9 for column in N):
            raise AnalysisUnknown("nullspace did not return a complete 9 x (9-d) column basis")
        h = _cas_value(algebra.square_free_part(gram.H), "square-free part")
        if _is_zero(gram.H) or _is_constant(gram.H) or _is_zero(h) or _is_constant(h):
            raise AnalysisUnknown("Gram polynomial is zero or constant on the declared coordinate chart")
        gradient_h = _cas_value(algebra.polynomial_gradient(h), "square-free Gram gradient")
        q = tuple(_cas_value(algebra.directional_derivative(h, column), "kernel directional derivative") for column in N)
        D = _cas_value(algebra.polynomial_gcd((h,) + q), "polynomial gcd")
        C = _cas_value(algebra.polynomial_exact_quotient(h, D), "exact h/D quotient")
        base_payload.update({
            "H": _poly_payload(gram.H), "gradient_H": [_poly_payload(item) for item in gram.gradient],
            "gradient_h": [_poly_payload(item) for item in gradient_h],
            "h": _poly_payload(h), "N": {"columns": algebra.columns_payload(N)},
            "q": [_poly_payload(item) for item in q], "D": _poly_payload(D), "C": _poly_payload(C),
        })
        if _is_one(D):
            return _common_payload(
                source, request, affine, profile_check=profile_check, payload={**base_payload, "generic_rank": d},
                outcome="pass", kind="polynomial_partition",
                reason="square-free Gram polynomial has no kernel-direction factor",
                checks=("exact_rank_B", "complete_kernel_basis", "square_free_part", "gcd_maximality"),
            )
        if _is_one(C):
            R = algebra.right_inverse(B, ncols=9)
            F = _cas_value(algebra.compose_linear(D, R, target_variables=tuple(f"w{index + 1}" for index in range(d))), "F=D(Rw)")
            restored = _cas_value(algebra.compose_linear(F, B, target_variables=Z_ORDER), "D=F(Bz)")
            if restored != D or _is_constant(F):
                raise AnalysisUnknown("polynomial dependence relation F(Bz)=D(z) failed exact reconstruction")
            base_payload.update({"R": _serialize_matrix(R), "F": _poly_payload(F), "F_variables": list(F.variables)})
            return _common_payload(
                source, request, affine, profile_check=profile_check,
                payload={**base_payload, "generic_rank": d - 1},
                outcome="reject", kind="polynomial_dependence",
                reason="all square-free Gram factors are constant along the affine kernel",
                checks=("exact_rank_B", "complete_kernel_basis", "square_free_part", "gcd_maximality", "right_inverse", "D_equals_F_of_B"),
            )
        return _common_payload(
            source, request, affine, profile_check=profile_check, payload=base_payload,
            outcome="unknown", kind="polynomial_partition",
            reason="mixed D/C partition has no supported physical exclusion or regular witness",
            checks=("exact_rank_B", "complete_kernel_basis", "square_free_part", "gcd_maximality"),
            execution_status="complete",
        )
    except AnalysisUnknown as exc:
        return _unknown(source, request, affine, profile_check, str(exc), payload=base_payload,
                        execution_status="unsupported")
    except (GramUnsupported, TypeError, ValueError) as exc:
        return _unknown(source, request, affine, profile_check, str(exc), payload=base_payload,
                        execution_status="failed")


__all__ = ["Z_ORDER", "AnalysisUnknown", "analyze_label"]
