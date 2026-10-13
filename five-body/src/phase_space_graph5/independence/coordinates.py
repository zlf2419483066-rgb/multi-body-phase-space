"""Exact five-body affine invariant coordinates.

The source record remains authoritative for the order and multiplicity of
selected invariant occurrences.  This module only constructs the fixed
``x``/``z`` affine map and the 25-subset expression table; it does not infer
topology or canonical identities.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations
from typing import Iterable, Mapping, Sequence

from .contracts import (
    COORDINATE_VERSION,
    InvariantOccurrence,
    PhysicsProfile,
    ReviewedCandidate,
    as_fraction,
    fraction_text,
)


X_ORDER = (
    "s12", "s13", "s14", "s15", "s23", "s24", "s25", "s34", "s35", "s45",
)
Z_ORDER = (
    "s12", "s23", "s34", "s45", "s1234", "s2345", "s123", "s234", "s345",
)


def _subset_order() -> tuple[tuple[int, ...], ...]:
    return tuple(
        subset
        for cardinality in (2, 3, 4)
        for subset in combinations(range(1, 6), cardinality)
    )


SUBSET_ORDER = _subset_order()


def subset_label(subset: Iterable[int]) -> str:
    values = tuple(subset)
    if len(values) not in {2, 3, 4} or tuple(sorted(values)) != values:
        raise ValueError("subset must be a sorted two-, three-, or four-particle tuple")
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 1 or value > 5 for value in values):
        raise ValueError("subset particle indices must lie in 1..5")
    if len(set(values)) != len(values):
        raise ValueError("subset cannot contain duplicate particles")
    return "s" + "".join(str(value) for value in values)


@dataclass(frozen=True, slots=True)
class AffineExpression:
    """One exact expression ``constant + coefficients · z``."""

    constant: Fraction
    coefficients: tuple[Fraction, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "constant", as_fraction(self.constant))
        object.__setattr__(self, "coefficients", tuple(as_fraction(value) for value in self.coefficients))

    def add(self, other: "AffineExpression", scale: int | Fraction = 1) -> "AffineExpression":
        if len(self.coefficients) != len(other.coefficients):
            raise ValueError("affine expressions must have equal dimensions")
        factor = as_fraction(scale)
        return AffineExpression(
            self.constant + factor * other.constant,
            tuple(left + factor * right for left, right in zip(self.coefficients, other.coefficients)),
        )

    def to_dict(self, *, variable_order: Sequence[str] = Z_ORDER) -> dict[str, object]:
        if len(variable_order) != len(self.coefficients):
            raise ValueError("variable_order has the wrong dimension")
        return {
            "constant": fraction_text(self.constant),
            "coefficients": {
                name: fraction_text(value)
                for name, value in zip(variable_order, self.coefficients)
                if value
            },
        }

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, object],
        *,
        variable_order: Sequence[str] = Z_ORDER,
    ) -> "AffineExpression":
        raw = payload.get("coefficients")
        if not isinstance(raw, Mapping):
            raise TypeError("affine coefficients must be an object")
        unknown = set(raw) - set(variable_order)
        if unknown:
            raise ValueError(f"unknown affine variables: {sorted(unknown)}")
        values = tuple(as_fraction(raw.get(name, "0")) for name in variable_order)
        return cls(as_fraction(payload.get("constant")), values)


@dataclass(frozen=True, slots=True)
class AffineMap:
    """A row-major exact map from the fixed nine-dimensional ``z`` chart."""

    output_order: tuple[str, ...]
    rows: tuple[AffineExpression, ...]
    z_order: tuple[str, ...] = Z_ORDER
    source_occurrences: tuple[InvariantOccurrence, ...] = ()
    version: str = COORDINATE_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "output_order", tuple(self.output_order))
        object.__setattr__(self, "z_order", tuple(self.z_order))
        object.__setattr__(self, "rows", tuple(self.rows))
        if self.version != COORDINATE_VERSION:
            raise ValueError("unsupported affine coordinate version")
        if len(self.z_order) != 9 or tuple(self.z_order) != Z_ORDER:
            raise ValueError("five-body z coordinates have a fixed nine-variable order")
        if len(self.output_order) != len(self.rows):
            raise ValueError("output_order and rows must have equal lengths")
        if any(len(row.coefficients) != len(self.z_order) for row in self.rows):
            raise ValueError("all affine rows must have nine z coefficients")
        if any(not isinstance(item, InvariantOccurrence) for item in self.source_occurrences):
            raise TypeError("source_occurrences must contain InvariantOccurrence values")

    @property
    def B(self) -> tuple[tuple[Fraction, ...], ...]:
        """Return the row-major exact coefficient matrix."""

        return tuple(row.coefficients for row in self.rows)

    @property
    def b(self) -> tuple[Fraction, ...]:
        """Return the exact affine constant vector."""

        return tuple(row.constant for row in self.rows)

    @property
    def shape(self) -> tuple[int, int]:
        return len(self.rows), len(self.z_order)

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "output_order": list(self.output_order),
            "x_order": list(X_ORDER) if tuple(self.output_order) == X_ORDER else None,
            "z_order": list(self.z_order),
            "rows": [row.to_dict(variable_order=self.z_order) for row in self.rows],
            "B": [[fraction_text(value) for value in row] for row in self.B],
            "b": [fraction_text(value) for value in self.b],
            "source_occurrences": [item.to_dict() for item in self.source_occurrences],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "AffineMap":
        output = payload.get("output_order")
        raw_x_order = payload.get("x_order")
        z_order = payload.get("z_order")
        raw_rows = payload.get("rows")
        raw_occurrences = payload.get("source_occurrences", ())
        if not isinstance(output, Sequence) or isinstance(output, (str, bytes)):
            raise TypeError("output_order must be a sequence")
        if not isinstance(z_order, Sequence) or isinstance(z_order, (str, bytes)):
            raise TypeError("z_order must be a sequence")
        if raw_x_order is not None:
            if not isinstance(raw_x_order, Sequence) or isinstance(raw_x_order, (str, bytes)):
                raise TypeError("x_order must be a sequence or null")
            if tuple(raw_x_order) != X_ORDER:
                raise ValueError("serialized x_order disagrees with the fixed pair order")
        if not isinstance(raw_rows, Sequence) or isinstance(raw_rows, (str, bytes)):
            raise TypeError("rows must be a sequence")
        if not isinstance(raw_occurrences, Sequence) or isinstance(raw_occurrences, (str, bytes)):
            raise TypeError("source_occurrences must be a sequence")
        result = cls(
            output_order=tuple(output),
            rows=tuple(
                AffineExpression.from_dict(row, variable_order=tuple(z_order))
                if isinstance(row, Mapping)
                else (_ for _ in ()).throw(TypeError("invalid affine row"))
                for row in raw_rows
            ),
            z_order=tuple(z_order),
            source_occurrences=tuple(
                InvariantOccurrence.from_dict(item)
                if isinstance(item, Mapping)
                else (_ for _ in ()).throw(TypeError("invalid source occurrence"))
                for item in raw_occurrences
            ),
            version=payload.get("version", COORDINATE_VERSION),
        )
        raw_B = payload.get("B")
        raw_b = payload.get("b")
        if raw_B is not None:
            if not isinstance(raw_B, Sequence) or isinstance(raw_B, (str, bytes)):
                raise TypeError("B must be a row sequence")
            B = tuple(
                tuple(as_fraction(value) for value in row)
                if isinstance(row, Sequence) and not isinstance(row, (str, bytes))
                else (_ for _ in ()).throw(TypeError("B rows must be sequences"))
                for row in raw_B
            )
            if B != result.B:
                raise ValueError("serialized B disagrees with affine rows")
        if raw_b is not None:
            if not isinstance(raw_b, Sequence) or isinstance(raw_b, (str, bytes)):
                raise TypeError("b must be a sequence")
            b = tuple(as_fraction(value) for value in raw_b)
            if b != result.b:
                raise ValueError("serialized b disagrees with affine rows")
        return result


def _zero() -> AffineExpression:
    return AffineExpression(Fraction(0), (Fraction(0),) * 9)


def _z(index: int) -> AffineExpression:
    values = [Fraction(0)] * 9
    values[index] = Fraction(1)
    return AffineExpression(Fraction(0), tuple(values))


def _mass_sum(profile: PhysicsProfile, subset: Iterable[int]) -> Fraction:
    return sum((profile.m2[item - 1] for item in subset), Fraction(0))


def _pair_index(pair: tuple[int, int]) -> int:
    try:
        return X_ORDER.index("s" + "".join(str(value) for value in pair))
    except ValueError as exc:
        raise ValueError(f"unknown pair coordinate {pair}") from exc


def pair_coordinate_map(profile: PhysicsProfile) -> AffineMap:
    """Construct the ten fixed pair invariants ``x=Bz+b``."""

    rows: list[AffineExpression] = [
        _z(0),
        AffineExpression(_mass_sum(profile, (1, 2, 3)), _z(6).add(_z(0), -1).add(_z(1), -1).coefficients),
        _zero(),
        _zero(),
        _z(1),
        AffineExpression(_mass_sum(profile, (2, 3, 4)), _z(7).add(_z(1), -1).add(_z(2), -1).coefficients),
        _zero(),
        _z(2),
        AffineExpression(_mass_sum(profile, (3, 4, 5)), _z(8).add(_z(2), -1).add(_z(3), -1).coefficients),
        _z(3),
    ]
    # The three-particle rows carry mass constants μ123, μ234, μ345.
    rows[1] = AffineExpression(_mass_sum(profile, (1, 2, 3)), rows[1].coefficients)
    rows[5] = AffineExpression(_mass_sum(profile, (2, 3, 4)), rows[5].coefficients)
    rows[8] = AffineExpression(_mass_sum(profile, (3, 4, 5)), rows[8].coefficients)

    # s14 = z5 + 2μ1234 - (s12+s13+s23+s24+s34)
    row = AffineExpression(2 * _mass_sum(profile, (1, 2, 3, 4)), _z(4).coefficients)
    row = row.add(rows[0], -1).add(rows[1], -1).add(rows[4], -1).add(rows[5], -1).add(rows[7], -1)
    rows[2] = row

    # s25 = z6 + 2μ2345 - (s23+s24+s34+s35+s45)
    row = AffineExpression(2 * _mass_sum(profile, (2, 3, 4, 5)), _z(5).coefficients)
    row = row.add(rows[4], -1).add(rows[5], -1).add(rows[7], -1).add(rows[8], -1).add(rows[9], -1)
    rows[6] = row

    # s15 = K0 - all other pair invariants, K0=M²+3Σm².
    k0 = profile.M2 + 3 * sum(profile.m2, Fraction(0))
    row = AffineExpression(k0, (Fraction(0),) * 9)
    for index, expression in enumerate(rows):
        if index != 3:
            row = row.add(expression, -1)
    rows[3] = row
    return AffineMap(output_order=X_ORDER, rows=tuple(rows))


def subset_expression(profile: PhysicsProfile, subset: Iterable[int], *, pair_map: AffineMap | None = None) -> AffineExpression:
    """Return the exact affine expression for any proper five-body subset."""

    values = tuple(subset)
    subset_label(values)  # validates the occurrence order and domain
    pair_map = pair_map or pair_coordinate_map(profile)
    expression = _zero()
    for pair in combinations(values, 2):
        expression = expression.add(pair_map.rows[_pair_index(pair)])
    expression = AffineExpression(
        expression.constant - (len(values) - 2) * _mass_sum(profile, values),
        expression.coefficients,
    )
    return expression


def full_subset_affine_map(profile: PhysicsProfile) -> AffineMap:
    """Return all 25 subset masses in cardinality/lexicographic order."""

    pair_map = pair_coordinate_map(profile)
    return AffineMap(
        output_order=tuple(subset_label(subset) for subset in SUBSET_ORDER),
        rows=tuple(subset_expression(profile, subset, pair_map=pair_map) for subset in SUBSET_ORDER),
    )


def build_label_affine_map(candidate: ReviewedCandidate, profile: PhysicsProfile) -> AffineMap:
    """Build ``y=Bz+b`` from a reviewed candidate's ordered occurrences.

    The candidate record is deliberately required rather than accepting a
    detached list of labels.  This keeps the original occurrence order and
    prevents a summary or a newly inferred label from becoming the source of
    a physical certificate.
    """

    if not isinstance(candidate, ReviewedCandidate):
        raise TypeError("candidate must be a ReviewedCandidate")
    if not isinstance(profile, PhysicsProfile):
        raise TypeError("profile must be a PhysicsProfile")
    pair_map = pair_coordinate_map(profile)
    rows = tuple(subset_expression(profile, item.subset, pair_map=pair_map) for item in candidate.occurrences)
    return AffineMap(
        output_order=tuple(item.label for item in candidate.occurrences),
        rows=rows,
        source_occurrences=candidate.occurrences,
    )


__all__ = [
    "X_ORDER",
    "Z_ORDER",
    "SUBSET_ORDER",
    "AffineExpression",
    "AffineMap",
    "subset_label",
    "pair_coordinate_map",
    "subset_expression",
    "full_subset_affine_map",
    "build_label_affine_map",
]
