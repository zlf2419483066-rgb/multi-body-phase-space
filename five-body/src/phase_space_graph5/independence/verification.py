"""Independent readback of five-body independence evidence."""

from __future__ import annotations

from fractions import Fraction
from typing import Any, Mapping

from . import exact_algebra as algebra
from .analysis import AnalysisUnknown, Z_ORDER, _cas_value, _is_constant, _is_one, _is_zero
from .certificates import EvidenceCertificate, VerificationReport
from .contracts import EvaluationRequest, LabelEvaluation, fraction_text
from .coordinates import build_label_affine_map, pair_coordinate_map
from .gram import GramUnsupported, build_gram_data_from_affine, validate_profile
from .sources import VerifiedSource, make_evaluation_request


def _check(checks: list[dict[str, Any]], name: str, ok: bool, detail: str = "") -> None:
    checks.append({"name": name, "ok": bool(ok), **({"detail": detail} if detail else {})})


def _matrix_payload(matrix: Any) -> list[list[str]]:
    return [[fraction_text(value) for value in row] for row in matrix]


def _vector_payload(vector: Any) -> list[str]:
    return [fraction_text(value) for value in vector]


def _poly_payload(value: algebra.SparsePolynomial) -> dict[str, object]:
    return algebra.serialize_polynomial(value)


def _parse_matrix(value: Any) -> tuple[tuple[Fraction, ...], ...]:
    if not isinstance(value, list):
        raise ValueError("matrix payload must be an array")
    return tuple(tuple(algebra.as_fraction(item) for item in row) for row in value)


def _parse_vector(value: Any) -> tuple[Fraction, ...]:
    if not isinstance(value, list):
        raise ValueError("vector payload must be an array")
    return tuple(algebra.as_fraction(item) for item in value)


def _parse_poly(value: Any) -> algebra.SparsePolynomial:
    if not isinstance(value, Mapping):
        raise ValueError("polynomial payload must be an object")
    return algebra.deserialize_polynomial(value)


def _is_zero_vector(vector: Any) -> bool:
    return all(value == 0 for value in vector)


def _columns_rows(columns: Any, dimension: int) -> tuple[tuple[Fraction, ...], ...]:
    """Turn column vectors into the row matrix used by exact algebra."""

    return algebra.transpose(columns, ncols=dimension)


def _matrix_is_zero(matrix: Any) -> bool:
    return all(value == 0 for row in matrix for value in row)


def _identity(size: int) -> tuple[tuple[Fraction, ...], ...]:
    return tuple(
        tuple(Fraction(1) if row == column else Fraction(0) for column in range(size))
        for row in range(size)
    )


def _check_physical_status(
    checks: list[dict[str, Any]], evaluation: LabelEvaluation, certificate: EvidenceCertificate,
) -> None:
    """Require complete valid input before accepting a physical result."""

    _check(checks, "physical_input_valid", evaluation.input_status == "valid" and certificate.input_status == "valid")
    _check(checks, "physical_execution_complete", evaluation.execution_status == "complete" and certificate.execution_status == "complete")


def _report(accepted: bool, status: str, checks: list[dict[str, Any]], errors: list[str], rebuilt: Mapping[str, Any] | None = None) -> VerificationReport:
    return VerificationReport(accepted=accepted, status=status, checks=tuple(checks),
                              errors=tuple(errors), shared_cas=True, rebuilt=rebuilt or {})


def verify_label_evidence(
    source: VerifiedSource,
    request: EvaluationRequest,
    evaluation: LabelEvaluation,
) -> VerificationReport:
    """Rebuild and independently verify a producer ``LabelEvaluation``.

    The producer status and ``verified``-like metadata are never used as input
    to the mathematical decision.  This verifier reconstructs the source
    request, affine map, Gram polynomial, full kernel basis, directional gcd,
    and (for C=1) the R/F relation from the current exact backend.
    """

    checks: list[dict[str, Any]] = []
    errors: list[str] = []
    rebuilt_payload: dict[str, Any] = {}
    try:
        if not isinstance(source, VerifiedSource):
            raise TypeError("source must be a VerifiedSource")
        if not isinstance(request, EvaluationRequest):
            raise TypeError("request must be an EvaluationRequest")
        if not isinstance(evaluation, LabelEvaluation):
            raise TypeError("evaluation must be a LabelEvaluation")
        canonical_request = make_evaluation_request(source, request.profile)
        source_request_ok = canonical_request.to_dict() == request.to_dict()
        _check(checks, "source_request_binding", source_request_ok)
        cert = EvidenceCertificate.from_dict(evaluation.certificate or {})
        cert_dict = cert.to_dict()
        _check(checks, "certificate_source_record", cert_dict.get("source_record") == source.source_record.to_dict())
        _check(checks, "certificate_identity", cert_dict.get("identity") == source.identity.to_dict())
        _check(checks, "certificate_request", cert_dict.get("request") == request.to_dict())
        _check(checks, "certificate_profile", cert_dict.get("profile") == request.profile.to_dict())
        expected_occurrences = [item.to_dict() for item in request.candidate.occurrences]
        _check(checks, "evaluation_topology_binding", evaluation.topology_key == request.candidate.topology_key)
        _check(checks, "evaluation_identity_binding", evaluation.identity_key == request.candidate.identity_key)
        _check(checks, "evaluation_mode_binding", evaluation.mode == request.candidate.mode == source.mode)
        _check(checks, "evaluation_profile_binding", evaluation.profile.to_dict() == request.profile.to_dict())
        _check(checks, "evaluation_occurrences_binding",
               [item.to_dict() for item in evaluation.occurrences] == expected_occurrences)
        _check(checks, "certificate_outcome_binding", cert.outcome == evaluation.status)
        _check(checks, "certificate_input_status_binding", cert.input_status == evaluation.input_status)
        _check(checks, "certificate_execution_status_binding", cert.execution_status == evaluation.execution_status)
        _check(checks, "certificate_variables_binding", cert_dict.get("variables") == expected_occurrences)
        _check(checks, "producer_status_not_trusted", True)
        binding_ok = source_request_ok and all(item["ok"] for item in checks)
        if not binding_ok:
            return _report(False, "unknown", checks, errors, rebuilt_payload)
        if cert.kind in {"regular_point", "physical_exclusion", "analytic_family"}:
            _check(checks, "unsupported_certificate_kind_is_unknown", evaluation.status == "unknown")
            return _report(False, "unknown", checks, errors, rebuilt_payload)
        if cert.kind == "unknown_record":
            _check(checks, "unknown_record_status", evaluation.status == "unknown")
            return _report(all(item["ok"] for item in checks), "unknown", checks, errors, rebuilt_payload)
        affine = build_label_affine_map(request.candidate, request.profile)
        expected_B = tuple(tuple(value for value in row) for row in affine.B)
        expected_b = tuple(affine.b)
        payload = cert_dict.get("payload")
        if not isinstance(payload, Mapping):
            raise ValueError("certificate payload is missing")
        _check(checks, "certificate_point_rank_binding", payload.get("point_rank") == evaluation.point_rank)
        _check(checks, "certificate_generic_rank_binding", payload.get("generic_rank") == evaluation.generic_rank)
        _check(checks, "B_rebuild", _parse_matrix(cert_dict.get("B")) == expected_B)
        _check(checks, "b_rebuild", _parse_vector(cert_dict.get("b")) == expected_b)
        profile_check = validate_profile(request.profile)
        rebuilt_payload.update({"B": _matrix_payload(expected_B), "b": _vector_payload(expected_b),
                                "profile_check": profile_check})
        if profile_check.get("status") != "valid":
            _check(checks, "unsupported_profile_is_unknown", evaluation.status == "unknown")
            _check(checks, "unsupported_profile_kind", cert.kind == "unknown_record")
            return _report(all(item["ok"] for item in checks), "unknown", checks, errors, rebuilt_payload)
        d = len(expected_B)
        affine_rank = algebra.rank(expected_B, ncols=9)
        _check(checks, "affine_rank", payload.get("affine_rank") == affine_rank)
        _check(checks, "evaluation_affine_rank", evaluation.affine_rank == affine_rank)
        if affine_rank < d:
            raw_multipliers = payload.get("left_kernel_columns")
            relations = payload.get("relations")
            if not isinstance(relations, Mapping):
                raise ValueError("affine dependence relations are missing")
            multipliers = algebra.columns_from_payload(raw_multipliers, dimension=d)
            relation_multipliers = algebra.columns_from_payload(relations.get("multipliers"), dimension=d)
            rhs = _parse_vector(relations.get("rhs"))
            expected_rhs = tuple(
                sum((left * right for left, right in zip(column, expected_b)), Fraction(0))
                for column in multipliers
            )
            left_kernel_ok = bool(multipliers) and any(not _is_zero_vector(column) for column in multipliers)
            annihilates_B = all(
                _matrix_is_zero(algebra.matmul((column,), expected_B, left_ncols=d, right_ncols=9))
                for column in multipliers
            )
            rhs_ok = len(rhs) == len(multipliers) and rhs == expected_rhs
            _check(checks, "left_kernel_columns_payload", payload.get("left_kernel_columns") == payload.get("relations", {}).get("multipliers"))
            _check(checks, "left_kernel_nonzero", left_kernel_ok)
            _check(checks, "left_kernel_annihilates_B", annihilates_B)
            _check(checks, "left_kernel_rhs", rhs_ok)
            _check(checks, "evaluation_generic_rank_unproved", evaluation.generic_rank is None)
            _check(checks, "evaluation_point_rank_unproved", evaluation.point_rank is None)
            _check(checks, "affine_dependency_status", evaluation.status == "reject")
            _check(checks, "affine_dependency_kind", cert_dict.get("kind") == "affine_dependence")
            _check_physical_status(checks, evaluation, cert)
            return _report(all(item["ok"] for item in checks), "reject", checks, errors, rebuilt_payload)
        if d > 8:
            _check(checks, "evaluation_generic_rank_unproved", evaluation.generic_rank is None)
            _check(checks, "evaluation_point_rank_unproved", evaluation.point_rank is None)
            _check(checks, "dimension_bound_status", evaluation.status == "reject")
            _check(checks, "dimension_bound_kind", cert_dict.get("kind") == "dimension_bound")
            _check_physical_status(checks, evaluation, cert)
            return _report(all(item["ok"] for item in checks), "reject", checks, errors, rebuilt_payload)
        pair_map = pair_coordinate_map(request.profile)
        gram = build_gram_data_from_affine(request.profile, pair_map, algebra=algebra)
        h = _cas_value(algebra.square_free_part(gram.H), "square-free part")
        if _is_zero(gram.H) or _is_constant(gram.H) or _is_zero(h) or _is_constant(h):
            _check(checks, "nonconstant_gram", False, "Gram polynomial is zero or constant")
            return _report(False, "unknown", checks, errors, rebuilt_payload)
        N = algebra.nullspace(expected_B, ncols=9)
        _check(checks, "complete_kernel_basis", len(N) == 9 - d and all(len(column) == 9 for column in N))
        _check(checks, "kernel_columns_payload", payload.get("N") == {"columns": algebra.columns_payload(N)})
        N_rows = _columns_rows(N, 9)
        _check(checks, "kernel_annihilates_B", _matrix_is_zero(algebra.matmul(expected_B, N_rows, left_ncols=9, right_ncols=9 - d)))
        _check(checks, "kernel_basis_rank", algebra.rank(N_rows, ncols=9 - d) == 9 - d)
        q = tuple(_cas_value(algebra.directional_derivative(h, column), "kernel directional derivative") for column in N)
        D = _cas_value(algebra.polynomial_gcd((h,) + q), "polynomial gcd")
        C = _cas_value(algebra.polynomial_exact_quotient(h, D), "exact h/D quotient")
        gradient_H = _cas_value(algebra.polynomial_gradient(gram.H), "Gram gradient")
        gradient_h = _cas_value(algebra.polynomial_gradient(h), "square-free Gram gradient")
        h_divisor = _cas_value(algebra.polynomial_exact_quotient(gram.H, h), "h divides H")
        degree_H = max((sum(monomial) for monomial, _ in gram.H.terms), default=0)
        remainder = gram.H
        for _ in range(degree_H):
            if _is_constant(remainder):
                break
            common = _cas_value(algebra.polynomial_gcd((remainder, h)), "H/h factor gcd")
            if _is_one(common):
                break
            remainder = _cas_value(algebra.polynomial_exact_quotient(remainder, common), "remove H/h factor")
        H_divides_h_power = not _is_zero(remainder) and _is_constant(remainder)
        square_free_gcd = _cas_value(algebra.polynomial_gcd((h,) + tuple(gradient_h)), "square-free derivative gcd")
        rebuilt_payload.update({"H": _poly_payload(gram.H), "h": _poly_payload(h),
                                "N": {"columns": algebra.columns_payload(N)},
                                "q": [_poly_payload(item) for item in q],
                                "D": _poly_payload(D), "C": _poly_payload(C)})
        _check(checks, "H_rebuild", payload.get("H") == _poly_payload(gram.H))
        _check(checks, "gradient_H_rebuild", payload.get("gradient_H") == [_poly_payload(item) for item in gradient_H] and tuple(gradient_H) == tuple(gram.gradient))
        _check(checks, "h_rebuild", payload.get("h") == _poly_payload(h))
        _check(checks, "gradient_h_rebuild", payload.get("gradient_h") == [_poly_payload(item) for item in gradient_h])
        _check(checks, "h_divides_H", h_divisor is not None)
        _check(checks, "H_divides_h_power", H_divides_h_power)
        _check(checks, "h_square_free", _is_one(square_free_gcd))
        _check(checks, "q_rebuild", payload.get("q") == [_poly_payload(item) for item in q])
        _check(checks, "gcd_maximality", payload.get("D") == _poly_payload(D))
        _check(checks, "C_rebuild", payload.get("C") == _poly_payload(C))
        D_divides_h = _cas_value(algebra.polynomial_exact_quotient(h, D), "D divides h")
        D_divides_q = tuple(_cas_value(algebra.polynomial_exact_quotient(item, D), "D divides directional derivative") for item in q)
        _check(checks, "D_divides_h", D_divides_h is not None)
        _check(checks, "D_divides_all_q", all(item is not None for item in D_divides_q))
        if _is_one(D):
            _check(checks, "evaluation_generic_rank", evaluation.generic_rank == d)
            _check(checks, "evaluation_point_rank_unproved", evaluation.point_rank is None)
            _check(checks, "pass_status", evaluation.status == "pass")
            _check(checks, "pass_kind", cert_dict.get("kind") == "polynomial_partition")
            _check_physical_status(checks, evaluation, cert)
            return _report(all(item["ok"] for item in checks), "pass", checks, errors, rebuilt_payload)
        if _is_one(C):
            R = algebra.right_inverse(expected_B, ncols=9)
            target = tuple(f"w{index + 1}" for index in range(d))
            F = _cas_value(algebra.compose_linear(D, R, target_variables=target), "F=D(Rw)")
            restored = _cas_value(algebra.compose_linear(F, expected_B, target_variables=Z_ORDER), "D=F(Bz)")
            rebuilt_payload.update({"R": _matrix_payload(R), "F": _poly_payload(F),
                                    "F_variables": list(F.variables)})
            _check(checks, "right_inverse_identity", algebra.matmul(expected_B, R, left_ncols=9, right_ncols=d) == _identity(d))
            _check(checks, "R_rebuild", payload.get("R") == _matrix_payload(R))
            _check(checks, "F_rebuild", payload.get("F") == _poly_payload(F))
            _check(checks, "F_variables_rebuild", payload.get("F_variables") == list(F.variables))
            _check(checks, "D_equals_F_of_B", restored == D)
            _check(checks, "F_nonconstant", not _is_constant(F))
            _check(checks, "evaluation_generic_rank", evaluation.generic_rank == d - 1)
            _check(checks, "evaluation_point_rank_unproved", evaluation.point_rank is None)
            _check(checks, "reject_status", evaluation.status == "reject")
            _check(checks, "reject_kind", cert_dict.get("kind") == "polynomial_dependence")
            _check_physical_status(checks, evaluation, cert)
            return _report(all(item["ok"] for item in checks), "reject", checks, errors, rebuilt_payload)
        _check(checks, "evaluation_generic_rank_unproved", evaluation.generic_rank is None)
        _check(checks, "evaluation_point_rank_unproved", evaluation.point_rank is None)
        _check(checks, "mixed_status", evaluation.status == "unknown")
        _check(checks, "mixed_kind", cert_dict.get("kind") == "polynomial_partition")
        return _report(all(item["ok"] for item in checks), "unknown", checks, errors, rebuilt_payload)
    except (AnalysisUnknown, GramUnsupported, TypeError, ValueError, KeyError, AttributeError) as exc:
        errors.append(str(exc))
        return _report(False, "unknown", checks, errors, rebuilt_payload)


__all__ = ["verify_label_evidence"]
