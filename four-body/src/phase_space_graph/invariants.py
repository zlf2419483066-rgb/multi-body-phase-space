"""Exact invariant generation and independence analysis for small final states.

The graph project treats invariant masses as symbolic affine expressions in a
fixed set of pair invariants.  This module keeps that representation separate
from graph topology: later modules can ask whether a proposed combination of
invariants is independent before assigning it to a drawing.

Only the three- and four-body systems are intentionally covered in G02.  All
linear algebra is performed with :class:`fractions.Fraction`; no numerical
rank test or random kinematics is used.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations
from typing import Iterable, Sequence


Invariant = tuple[int, ...]
"""Canonical particle subset, for example ``(1, 2, 3)`` for ``s123``."""


def _as_fraction(value: int | Fraction) -> Fraction:
    """Convert an exact scalar and reject floats that could hide round-off."""

    if isinstance(value, bool):
        raise TypeError("boolean values are not valid exact coefficients")
    if isinstance(value, Fraction):
        return value
    if isinstance(value, int):
        return Fraction(value)
    raise TypeError("coefficients must be int or Fraction values")


def _canonical_subset(subset: Iterable[int], n: int) -> Invariant:
    """Validate and canonicalize a particle subset."""

    if n not in (3, 4):
        raise ValueError("G02 currently supports n=3 and n=4 only")
    try:
        items = tuple(subset)
    except TypeError as exc:
        raise TypeError("an invariant subset must be an iterable of particle indices") from exc
    if not items:
        raise ValueError("an invariant subset cannot be empty")
    if any(isinstance(item, bool) or not isinstance(item, int) for item in items):
        raise TypeError("particle indices must be integers")
    if len(set(items)) != len(items):
        raise ValueError("a particle may occur only once inside one invariant subset")
    canonical = tuple(sorted(items))
    if canonical[0] < 1 or canonical[-1] > n:
        raise ValueError(f"particle indices must lie in 1..{n}")
    if len(canonical) < 2 or len(canonical) >= n:
        raise ValueError("G02 accepts proper invariant subsets of size 2..n-1")
    return canonical


def invariant_label(subset: Iterable[int]) -> str:
    """Return the conventional label ``s12``, ``s123`` and so on."""

    canonical = tuple(sorted(subset))
    if (
        not canonical
        or any(
            isinstance(item, bool) or not isinstance(item, int) or item < 1
            for item in canonical
        )
        or len(set(canonical)) != len(canonical)
    ):
        raise ValueError("an invariant label needs integer particle indices")
    return "s" + "".join(str(item) for item in canonical)


def base_pair_invariants(n: int) -> tuple[Invariant, ...]:
    """Return the fixed pair-coordinate order used by all affine vectors."""

    if n not in (3, 4):
        raise ValueError("G02 currently supports n=3 and n=4 only")
    return tuple(combinations(range(1, n + 1), 2))


def candidate_invariants(n: int) -> tuple[Invariant, ...]:
    """Generate all pair and proper composite candidates in deterministic order.

    For ``n=4`` this is the six pair variables followed by the four triple
    variables, for a total of ten candidates.  For ``n=3`` only the three pair
    variables are candidates.
    """

    pairs = base_pair_invariants(n)
    if n == 3:
        return pairs
    triples = tuple(combinations(range(1, n + 1), 3))
    return pairs + triples


@dataclass(frozen=True, slots=True)
class MassExpression:
    """Exact symbolic linear combination of ``M^2`` and particle ``m_i^2``.

    ``parent_coefficient`` multiplies the total invariant mass squared
    ``M^2``.  ``particle_coefficients[i-1]`` multiplies ``m_i^2``.  Keeping
    these terms as a vector lets dependence certificates be checked exactly
    without a symbolic algebra dependency.
    """

    parent_coefficient: Fraction = Fraction(0)
    particle_coefficients: tuple[Fraction, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "parent_coefficient", _as_fraction(self.parent_coefficient))
        object.__setattr__(
            self,
            "particle_coefficients",
            tuple(_as_fraction(value) for value in self.particle_coefficients),
        )

    @classmethod
    def zero(cls, n: int) -> "MassExpression":
        return cls(Fraction(0), (Fraction(0),) * n)

    def _check_size(self, other: "MassExpression") -> None:
        if len(self.particle_coefficients) != len(other.particle_coefficients):
            raise ValueError("mass expressions must refer to the same number of particles")

    def __add__(self, other: "MassExpression") -> "MassExpression":
        self._check_size(other)
        return MassExpression(
            self.parent_coefficient + other.parent_coefficient,
            tuple(a + b for a, b in zip(self.particle_coefficients, other.particle_coefficients)),
        )

    def __sub__(self, other: "MassExpression") -> "MassExpression":
        self._check_size(other)
        return MassExpression(
            self.parent_coefficient - other.parent_coefficient,
            tuple(a - b for a, b in zip(self.particle_coefficients, other.particle_coefficients)),
        )

    def scale(self, factor: int | Fraction) -> "MassExpression":
        scalar = _as_fraction(factor)
        return MassExpression(
            scalar * self.parent_coefficient,
            tuple(scalar * value for value in self.particle_coefficients),
        )

    @property
    def is_zero(self) -> bool:
        return self.parent_coefficient == 0 and all(
            value == 0 for value in self.particle_coefficients
        )

    def evaluate(
        self,
        parent_mass_squared: int | Fraction,
        particle_mass_squares: Sequence[int | Fraction],
    ) -> Fraction:
        """Evaluate the symbolic mass terms using exact numeric substitutions."""

        if len(particle_mass_squares) != len(self.particle_coefficients):
            raise ValueError("particle-mass vector has the wrong dimension")
        return self.parent_coefficient * _as_fraction(parent_mass_squared) + sum(
            coefficient * _as_fraction(value)
            for coefficient, value in zip(self.particle_coefficients, particle_mass_squares)
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "parent_coefficient": _fraction_text(self.parent_coefficient),
            "particle_coefficients": [
                _fraction_text(value) for value in self.particle_coefficients
            ],
        }

    def format(self) -> str:
        terms: list[tuple[Fraction, str]] = []
        if self.parent_coefficient:
            terms.append((self.parent_coefficient, "M^2"))
        terms.extend(
            (coefficient, f"m{index + 1}^2")
            for index, coefficient in enumerate(self.particle_coefficients)
            if coefficient
        )
        if not terms:
            return "0"
        output = ""
        for position, (coefficient, symbol) in enumerate(terms):
            sign = "-" if coefficient < 0 else "+"
            magnitude = abs(coefficient)
            if magnitude == 1:
                factor = symbol
            else:
                factor = f"{_fraction_text(magnitude)}*{symbol}"
            if position == 0:
                output = factor if coefficient > 0 else f"-{factor}"
            else:
                output += f" {sign} {factor}"
        return output


@dataclass(frozen=True, slots=True)
class AffineExpression:
    """An invariant as ``coefficients · pair_coordinates + mass_constant``."""

    variables: tuple[Invariant, ...]
    coefficients: tuple[Fraction, ...]
    mass_constant: MassExpression

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "variables",
            tuple(tuple(variable) for variable in self.variables),
        )
        object.__setattr__(
            self,
            "coefficients",
            tuple(_as_fraction(value) for value in self.coefficients),
        )
        if len(self.variables) != len(self.coefficients):
            raise ValueError("affine variable and coefficient vectors must have equal length")

    @property
    def basis(self) -> tuple[Invariant, ...]:
        """Alias for the fixed pair-coordinate basis."""

        return self.variables

    @property
    def constant(self) -> MassExpression:
        """Alias for the symbolic mass constant."""

        return self.mass_constant

    def evaluate(
        self,
        pair_values: Sequence[int | Fraction],
        *,
        parent_mass_squared: int | Fraction | None = None,
        particle_mass_squares: Sequence[int | Fraction] | None = None,
    ) -> Fraction:
        if len(pair_values) != len(self.coefficients):
            raise ValueError("pair-value vector has the wrong dimension")
        result = sum(
            coefficient * _as_fraction(value)
            for coefficient, value in zip(self.coefficients, pair_values)
        )
        if not self.mass_constant.is_zero:
            if parent_mass_squared is None or particle_mass_squares is None:
                raise ValueError("symbolic mass terms need exact numeric substitutions")
            result += self.mass_constant.evaluate(parent_mass_squared, particle_mass_squares)
        return result

    def format(self) -> str:
        terms: list[tuple[Fraction, str]] = [
            (coefficient, invariant_label(variable))
            for variable, coefficient in zip(self.variables, self.coefficients)
            if coefficient
        ]
        output = ""
        for position, (coefficient, symbol) in enumerate(terms):
            sign = "-" if coefficient < 0 else "+"
            magnitude = abs(coefficient)
            factor = symbol if magnitude == 1 else f"{_fraction_text(magnitude)}*{symbol}"
            if position == 0:
                output = factor if coefficient > 0 else f"-{factor}"
            else:
                output += f" {sign} {factor}"
        if self.mass_constant.is_zero:
            return output or "0"
        mass = self.mass_constant.format()
        if not output:
            return mass
        if mass.startswith("-"):
            return f"{output} - {mass[1:]}"
        return f"{output} + {mass}"

    def to_dict(self) -> dict[str, object]:
        return {
            "variables": [list(variable) for variable in self.variables],
            "coefficients": [_fraction_text(value) for value in self.coefficients],
            "mass_constant": self.mass_constant.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class Constraint:
    """The one total-mass affine constraint in pair coordinates."""

    coefficients: tuple[Fraction, ...]
    rhs: MassExpression
    equation_name: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "coefficients",
            tuple(_as_fraction(value) for value in self.coefficients),
        )
        expected_size = len(base_pair_invariants(len(self.rhs.particle_coefficients)))
        if len(self.coefficients) != expected_size:
            raise ValueError("constraint coefficient vector has the wrong dimension")

    def format(self) -> str:
        lhs = " + ".join(
            invariant_label(pair) if coefficient == 1 else f"{_fraction_text(coefficient)}*{invariant_label(pair)}"
            for pair, coefficient in zip(
                base_pair_invariants(len(self.rhs.particle_coefficients)), self.coefficients
            )
            if coefficient
        ) or "0"
        return f"{lhs} = {self.rhs.format()} ({self.equation_name})"

    def to_dict(self) -> dict[str, object]:
        return {
            "coefficients": [_fraction_text(value) for value in self.coefficients],
            "rhs": self.rhs.to_dict(),
            "equation_name": self.equation_name,
        }


@dataclass(frozen=True, slots=True)
class DependenceCertificate:
    """A checked relation among selected invariants and the global constraint."""

    variables: tuple[Invariant, ...]
    multipliers: tuple[Fraction, ...]
    constraint_multiplier: Fraction
    rhs_mass: MassExpression
    verified: bool

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "multipliers",
            tuple(_as_fraction(value) for value in self.multipliers),
        )
        object.__setattr__(
            self,
            "constraint_multiplier",
            _as_fraction(self.constraint_multiplier),
        )
        if len(self.variables) != len(self.multipliers):
            raise ValueError("certificate variable and multiplier vectors must have equal length")

    def format(self) -> str:
        terms: list[str] = []
        for variable, coefficient in zip(self.variables, self.multipliers):
            if coefficient == 0:
                continue
            label = invariant_label(variable)
            magnitude = abs(coefficient)
            term = label if magnitude == 1 else f"{_fraction_text(magnitude)}*{label}"
            if not terms:
                terms.append(term if coefficient > 0 else f"-{term}")
            else:
                terms.append(("+ " if coefficient > 0 else "- ") + term)
        lhs = " ".join(terms) or "0"
        return f"{lhs} = {self.rhs_mass.format()}"

    def to_dict(self) -> dict[str, object]:
        return {
            "variables": [list(variable) for variable in self.variables],
            "multipliers": [_fraction_text(value) for value in self.multipliers],
            "constraint_multiplier": _fraction_text(self.constraint_multiplier),
            "rhs_mass": self.rhs_mass.to_dict(),
            "verified": self.verified,
            "formatted": self.format(),
        }


@dataclass(frozen=True, slots=True)
class IndependenceResult:
    """Rank and certificate report for one selected invariant list."""

    n: int
    variables: tuple[Invariant, ...]
    expressions: tuple[AffineExpression, ...]
    constraint: Constraint
    rank: int
    independent: bool
    certificate: DependenceCertificate | None

    @property
    def variable_count(self) -> int:
        return len(self.variables)

    @property
    def independent_count(self) -> int:
        return self.rank

    def to_dict(self) -> dict[str, object]:
        return {
            "n": self.n,
            "variables": [list(variable) for variable in self.variables],
            "labels": [invariant_label(variable) for variable in self.variables],
            "expressions": [expression.to_dict() for expression in self.expressions],
            "constraint": self.constraint.to_dict(),
            "rank": self.rank,
            "independent": self.independent,
            "certificate": self.certificate.to_dict() if self.certificate else None,
        }


def affine_expression(subset: Iterable[int], n: int) -> AffineExpression:
    """Build the exact pair-coordinate expression for one invariant."""

    canonical = _canonical_subset(subset, n)
    pairs = base_pair_invariants(n)
    coefficients = tuple(
        Fraction(1) if pair[0] in canonical and pair[1] in canonical else Fraction(0)
        for pair in pairs
    )
    mass_coefficients = tuple(Fraction(-1) if particle in canonical else Fraction(0) for particle in range(1, n + 1))
    if len(canonical) == 2:
        mass_coefficients = (Fraction(0),) * n
    return AffineExpression(
        variables=pairs,
        coefficients=coefficients,
        mass_constant=MassExpression(Fraction(0), mass_coefficients),
    )


def total_mass_constraint(n: int) -> Constraint:
    """Return the symbolic total-mass relation for pair coordinates."""

    pairs = base_pair_invariants(n)
    particle_factor = 2 if n == 4 else 1
    rhs = MassExpression(
        Fraction(1),
        (Fraction(particle_factor),) * n,
    )
    equation_name = "four_body_total_mass" if n == 4 else "three_body_total_mass"
    return Constraint(tuple(Fraction(1) for _ in pairs), rhs, equation_name)


def _matrix_rank(rows: Sequence[Sequence[Fraction]]) -> int:
    """Compute exact row rank by fraction Gaussian elimination."""

    if not rows:
        return 0
    matrix = [list(row) for row in rows]
    width = len(matrix[0])
    if any(len(row) != width for row in matrix):
        raise ValueError("matrix rows must have equal lengths")
    rank = 0
    for column in range(width):
        pivot = next(
            (row for row in range(rank, len(matrix)) if matrix[row][column] != 0),
            None,
        )
        if pivot is None:
            continue
        matrix[rank], matrix[pivot] = matrix[pivot], matrix[rank]
        pivot_value = matrix[rank][column]
        matrix[rank] = [value / pivot_value for value in matrix[rank]]
        for row in range(len(matrix)):
            if row == rank:
                continue
            factor = matrix[row][column]
            if factor:
                matrix[row] = [
                    left - factor * right
                    for left, right in zip(matrix[row], matrix[rank])
                ]
        rank += 1
        if rank == len(matrix):
            break
    return rank


def _nullspace(rows: Sequence[Sequence[Fraction]]) -> tuple[tuple[Fraction, ...], ...]:
    """Return a deterministic exact basis for a matrix nullspace."""

    if not rows:
        return ()
    matrix = [list(row) for row in rows]
    column_count = len(matrix[0])
    if any(len(row) != column_count for row in matrix):
        raise ValueError("matrix rows must have equal lengths")
    pivot_columns: list[int] = []
    pivot_row = 0
    for column in range(column_count):
        pivot = next(
            (row for row in range(pivot_row, len(matrix)) if matrix[row][column] != 0),
            None,
        )
        if pivot is None:
            continue
        matrix[pivot_row], matrix[pivot] = matrix[pivot], matrix[pivot_row]
        pivot_value = matrix[pivot_row][column]
        matrix[pivot_row] = [value / pivot_value for value in matrix[pivot_row]]
        for row in range(len(matrix)):
            if row == pivot_row:
                continue
            factor = matrix[row][column]
            if factor:
                matrix[row] = [
                    left - factor * right
                    for left, right in zip(matrix[row], matrix[pivot_row])
                ]
        pivot_columns.append(column)
        pivot_row += 1
        if pivot_row == len(matrix):
            break
    free_columns = [column for column in range(column_count) if column not in pivot_columns]
    vectors: list[tuple[Fraction, ...]] = []
    for free in free_columns:
        vector = [Fraction(0)] * column_count
        vector[free] = Fraction(1)
        for row, pivot in reversed(list(enumerate(pivot_columns))):
            vector[pivot] = -sum(
                matrix[row][column] * vector[column]
                for column in free_columns
            )
        vectors.append(tuple(vector))
    return tuple(vectors)


def _normalize_vector(values: Sequence[Fraction]) -> tuple[Fraction, ...]:
    first = next((value for value in values if value), None)
    if first is None:
        return tuple(Fraction(0) for _ in values)
    return tuple(value / first for value in values)


def _fraction_text(value: Fraction) -> str:
    return str(value.numerator) if value.denominator == 1 else f"{value.numerator}/{value.denominator}"


def analyze_invariants(variables: Iterable[Iterable[int]], n: int) -> IndependenceResult:
    """Analyze exact independence under the total-mass constraint."""

    if n not in (3, 4):
        raise ValueError("G02 currently supports n=3 and n=4 only")
    selected = tuple(_canonical_subset(variable, n) for variable in variables)
    expressions = tuple(affine_expression(variable, n) for variable in selected)
    constraint = total_mass_constraint(n)
    matrix_rows = [constraint.coefficients] + [expression.coefficients for expression in expressions]
    constrained_rank = _matrix_rank(matrix_rows) - 1
    if constrained_rank < 0:  # defensive guard for a future zero constraint
        constrained_rank = 0
    independent = constrained_rank == len(selected)
    certificate: DependenceCertificate | None = None
    if not independent and selected:
        # Solve lambda^T A = mu c by finding the nullspace of [A^T | -c].
        relation_rows = [
            [expression.coefficients[index] for expression in expressions]
            + [-constraint.coefficients[index]]
            for index in range(len(constraint.coefficients))
        ]
        for vector in _nullspace(relation_rows):
            first_original = next((value for value in vector[:-1] if value), None)
            if first_original is None:
                continue
            lambdas = _normalize_vector(vector[:-1])
            # Normalization above sets the first nonzero lambda to one.  Scale
            # mu by the same factor used for lambda.
            mu = vector[-1] / first_original
            lhs_mass = MassExpression.zero(n)
            for coefficient, expression in zip(lambdas, expressions):
                lhs_mass = lhs_mass + expression.mass_constant.scale(coefficient)
            rhs_mass = constraint.rhs.scale(mu) + lhs_mass
            lhs_coefficients = tuple(
                sum(
                    coefficient * expression.coefficients[index]
                    for coefficient, expression in zip(lambdas, expressions)
                )
                for index in range(len(constraint.coefficients))
            )
            verified = (
                lhs_coefficients == tuple(mu * value for value in constraint.coefficients)
            )
            certificate = DependenceCertificate(
                variables=selected,
                multipliers=lambdas,
                constraint_multiplier=mu,
                rhs_mass=rhs_mass,
                verified=verified,
            )
            break
    return IndependenceResult(
        n=n,
        variables=selected,
        expressions=expressions,
        constraint=constraint,
        rank=constrained_rank,
        independent=independent,
        certificate=certificate,
    )
