from __future__ import annotations

from copy import deepcopy
from fractions import Fraction
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from phase_space_graph5.fixtures import fixture_a, fixture_b  # noqa: E402
from phase_space_graph5.label_identity import canonicalize_candidate_labeling  # noqa: E402
from phase_space_graph5.physical_labels import normalize_candidate_labeling  # noqa: E402
from phase_space_graph5.independence.exact_algebra import rank  # noqa: E402
from phase_space_graph5.independence.generic_family import (  # noqa: E402
    GENERIC_EVIDENCE_SCHEMA_VERSION,
    GENERIC_FAMILY,
    PARAMETER_ORDER,
    Z_PARAMETER_ORDER,
    _certificate,
    evaluate_generic_label,
)
from phase_space_graph5.independence.generic_verification import (  # noqa: E402
    _expected_certificate,
    _independent_parameter_matrix,
    _pair_rows,
    _subset_row,
    verify_generic_label,
)
from phase_space_graph5.independence.hajer_constraints import (  # noqa: E402
    _linear_coefficients,
    _subset_expression,
    build_hajer_constraint,
)
from phase_space_graph5.independence.invariant_sets import extract_invariant_set  # noqa: E402
from phase_space_graph5.independence.sources import VerifiedSource  # noqa: E402
from phase_space_graph5.independence.coordinates import SUBSET_ORDER  # noqa: E402


def _verified_fixture(name: str) -> VerifiedSource:
    if name == "A":
        topology, labeling = fixture_a()
        source_kind = "fixture-A"
    else:
        topology, labeling = fixture_b(int(name[1:]))
        source_kind = f"fixture-{name}"
    checked = normalize_candidate_labeling(topology, labeling)
    if checked.record is None:
        raise AssertionError(f"fixture {name} did not produce a record")
    identity = canonicalize_candidate_labeling(checked.record, "fixed_labels")
    return VerifiedSource(identity, checked.record, source_kind)


def _metric_gradient(momentum: tuple[int, int, int, int]) -> tuple[Fraction, ...]:
    energy, px, py, pz = momentum
    return (Fraction(2 * energy), Fraction(-2 * px), Fraction(-2 * py), Fraction(-2 * pz))


def _direct_constraint_jacobian(
    momenta: tuple[tuple[int, int, int, int], ...], subsets: tuple[tuple[int, ...], ...],
) -> tuple[tuple[Fraction, ...], ...]:
    rows: list[tuple[Fraction, ...]] = []
    for particle, momentum in enumerate(momenta):
        row = [Fraction(0)] * 20
        row[4 * particle:4 * particle + 4] = _metric_gradient(momentum)
        rows.append(tuple(row))
    for component in range(4):
        row = [Fraction(0)] * 20
        for particle in range(5):
            row[4 * particle + component] = Fraction(1)
        rows.append(tuple(row))
    for subset in subsets:
        total = tuple(
            sum(momenta[particle - 1][component] for particle in subset)
            for component in range(4)
        )
        gradient = _metric_gradient(total)
        row = [Fraction(0)] * 20
        for particle in subset:
            row[4 * (particle - 1):4 * particle] = gradient
        rows.append(tuple(row))
    return tuple(rows)


class GenericFamilyTests(unittest.TestCase):
    def test_family_descriptor_is_closed_and_explicit(self) -> None:
        self.assertEqual(GENERIC_FAMILY.to_dict()["spacetime"], "3+1")
        self.assertEqual(GENERIC_FAMILY.to_dict()["metric"], "+---")
        self.assertEqual(GENERIC_FAMILY.to_dict()["mass_condition"], "mi>0")
        self.assertTrue(GENERIC_FAMILY.to_dict()["pairwise_distinct"])
        self.assertEqual(GENERIC_FAMILY.to_dict()["threshold_condition"], "M>sum(mi)")
        self.assertEqual(GENERIC_FAMILY.to_dict()["quantifier"], "all_declared_parameters_same_label")
        self.assertEqual(len(Z_PARAMETER_ORDER), 9)
        self.assertEqual(len(PARAMETER_ORDER), 15)

    def test_A_pass_B2_reject_B3_pass_and_verifier_rechecks_each(self) -> None:
        expected = {"A": "pass", "B2": "reject", "B3": "pass"}
        for name, status in expected.items():
            source = _verified_fixture(name)
            evidence = evaluate_generic_label(source)
            self.assertEqual(evidence["schema_version"], GENERIC_EVIDENCE_SCHEMA_VERSION)
            self.assertEqual(evidence["status"], status, name)
            self.assertEqual(len(evidence["B"]), len(evidence["occurrences"]), name)
            self.assertTrue(all(len(row) == 9 for row in evidence["B"]), name)
            self.assertTrue(all(len(row) == 15 for row in evidence["affine_parameter_matrix"]), name)
            report = verify_generic_label(source, evidence)
            self.assertTrue(report["valid"], (name, report))
            self.assertEqual(report["status"], status, name)
            self.assertEqual(report["errors"], [], name)

    def test_direct_20_component_jacobian_matches_z_rank_for_A_B2_B3(self) -> None:
        momenta = (
            (2, 1, 0, 0),
            (3, -1, 0, 0),
            (4, 0, 1, 0),
            (5, 0, 0, 1),
            (6, 0, -1, -1),
        )
        expected = {"A": 3, "B2": 3, "B3": 4}
        for name, expected_rank in expected.items():
            source = _verified_fixture(name)
            invariant_set = extract_invariant_set(source)
            subsets = tuple(item.particle_subset for item in invariant_set.occurrences)
            jacobian = _direct_constraint_jacobian(momenta, subsets)
            base_rank = rank(jacobian[:9], ncols=20)
            relative_rank = rank(jacobian, ncols=20) - base_rank
            evidence = evaluate_generic_label(source)
            self.assertEqual(base_rank, 9, name)
            self.assertEqual(relative_rank, expected_rank, name)
            self.assertEqual(evidence["affine_rank"], expected_rank, name)

    def test_all_25_symbolic_rows_match_hajer_cqct_coefficients(self) -> None:
        descriptor = build_hajer_constraint()
        pairs = _pair_rows()
        for subset in SUBSET_ORDER:
            independent = _subset_row(subset, pairs)
            hajer = _linear_coefficients(_subset_expression(descriptor, subset))
            self.assertEqual(independent, hajer, subset)

    def test_mass_columns_are_constants_and_do_not_raise_kinematic_rank(self) -> None:
        subsets = ((1, 2, 3, 4), (1, 2, 3, 5), (1, 2, 4, 5), (1, 3, 4, 5), (2, 3, 4, 5))
        full = tuple(_subset_row(subset, _pair_rows()) for subset in subsets)
        z_matrix = tuple(row[:9] for row in full)
        self.assertEqual(rank(z_matrix, ncols=9), 4)
        self.assertEqual(rank(full, ncols=15), 5)

    def test_verifier_rejects_forged_bool_int_schema_fields(self) -> None:
        source = _verified_fixture("A")
        evidence = evaluate_generic_label(source)

        family_forgery = deepcopy(evidence)
        family_forgery["family"]["pairwise_distinct"] = 1  # type: ignore[index]
        self.assertFalse(verify_generic_label(source, family_forgery)["valid"])

        occurrence_forgery = deepcopy(evidence)
        occurrence_forgery["occurrences"][0]["edge_index"] = True  # type: ignore[index]
        self.assertFalse(verify_generic_label(source, occurrence_forgery)["valid"])

        certificate_forgery = deepcopy(evidence)
        certificate_forgery["certificate"]["minor"]["row_indices"][0] = False  # type: ignore[index]
        self.assertFalse(verify_generic_label(source, certificate_forgery)["valid"])

    def test_verifier_rejects_mass_column_or_source_binding_tampering(self) -> None:
        source = _verified_fixture("A")
        evidence = evaluate_generic_label(source)

        mass_forgery = deepcopy(evidence)
        mass_forgery["affine_parameter_matrix"][0][10] = "999"  # type: ignore[index]
        self.assertFalse(verify_generic_label(source, mass_forgery)["valid"])

        key_forgery = deepcopy(evidence)
        key_forgery["identity_key"] = "forged"  # type: ignore[index]
        self.assertFalse(verify_generic_label(source, key_forgery)["valid"])

    def test_dimension_bound_and_mass_relation_certificates_are_math_only(self) -> None:
        identity9 = tuple(
            tuple(Fraction(1) if row == column else Fraction(0) for column in range(9))
            for row in range(9)
        )
        full9 = tuple(row + (Fraction(0),) * 6 for row in identity9)
        dimension_certificate = _certificate(identity9, full9, 9, 9)
        self.assertEqual(_expected_certificate(identity9, full9, 9, 9), dimension_certificate)
        self.assertEqual(dimension_certificate["kind"], "dimension_bound")
        self.assertEqual(dimension_certificate["maximum_regular_dimension"], 8)
        self.assertNotEqual(dimension_certificate.get("kind"), "nonzero_minor")

        identity8 = tuple(
            tuple(Fraction(1) if row == column else Fraction(0) for column in range(8))
            for row in range(8)
        )
        full8 = tuple(row + (Fraction(0),) * 6 for row in identity8)
        self.assertEqual(_certificate(identity8, full8, 8, 8)["kind"], "nonzero_minor")

        subsets = ((1, 2, 3, 4), (1, 2, 3, 5), (1, 2, 4, 5), (1, 3, 4, 5), (2, 3, 4, 5))
        full = tuple(_subset_row(subset, _pair_rows()) for subset in subsets)
        z_matrix = tuple(row[:9] for row in full)
        certificate = _certificate(z_matrix, full, 5, 4)
        self.assertEqual(certificate["kind"], "left_kernel")
        mass_relations = certificate["mass_constant_relations"]
        self.assertTrue(mass_relations)
        expected_mass_relation = tuple(Fraction(value) for value in (3, 1, 1, 1, 1, 1))
        self.assertTrue(
            any(
                tuple(Fraction(value) for value in relation["mass_coefficients"]) == expected_mass_relation
                for relation in mass_relations
            )
        )
        self.assertEqual(
            _expected_certificate(z_matrix, full, 5, 4),
            certificate,
        )


if __name__ == "__main__":
    unittest.main()
