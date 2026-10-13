"""Exact linear algebra and sparse polynomial boundary for F5-I.

Matrices use row-major tuples and :class:`fractions.Fraction`.  Nullspace
results are tuples of column vectors, so a ``9 x (9-d)`` kernel basis is
serialized as ``columns`` with nine entries per column.  Optional SymPy code
is imported only inside the adapter function; lack of that dependency is an
explicit ``unsupported`` result and never causes a float fallback.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Iterable, Mapping, Sequence

from .contracts import as_fraction, fraction_text


Matrix = tuple[tuple[Fraction, ...], ...]
Vector = tuple[Fraction, ...]
Monomial = tuple[int, ...]


def _matrix(rows: Sequence[Sequence[object]], ncols: int | None = None) -> tuple[Matrix, int]:
    if isinstance(rows, (str, bytes)):
        raise TypeError("matrix must be a sequence of rows")
    converted = tuple(tuple(as_fraction(value) for value in row) for row in rows)
    if converted:
        width = len(converted[0])
        if any(len(row) != width for row in converted):
            raise ValueError("matrix rows must have equal lengths")
        if ncols is not None and ncols != width:
            raise ValueError("ncols disagrees with matrix rows")
        return converted, width
    if ncols is None:
        ncols = 0
    if isinstance(ncols, bool) or not isinstance(ncols, int) or ncols < 0:
        raise ValueError("ncols must be a non-negative integer")
    return converted, ncols


def matrix_shape(rows: Sequence[Sequence[object]], ncols: int | None = None) -> tuple[int, int]:
    matrix, width = _matrix(rows, ncols)
    return len(matrix), width


def identity(size: int) -> Matrix:
    if isinstance(size, bool) or not isinstance(size, int) or size < 0:
        raise ValueError("identity size must be a non-negative integer")
    return tuple(
        tuple(Fraction(1) if row == column else Fraction(0) for column in range(size))
        for row in range(size)
    )


def transpose(rows: Sequence[Sequence[object]], ncols: int | None = None) -> Matrix:
    matrix, width = _matrix(rows, ncols)
    return tuple(
        tuple(matrix[row][column] for row in range(len(matrix)))
        for column in range(width)
    )


def matmul(
    left: Sequence[Sequence[object]],
    right: Sequence[Sequence[object]],
    *,
    left_ncols: int | None = None,
    right_ncols: int | None = None,
) -> Matrix:
    """Multiply exact row-major matrices, retaining zero-dimensional cases."""

    a, a_width = _matrix(left, left_ncols)
    b, b_width = _matrix(right, right_ncols)
    if a_width != len(b):
        raise ValueError(f"matrix dimensions do not multiply: {(len(a), a_width)} and {(len(b), b_width)}")
    return tuple(
        tuple(
            sum((a[row][inner] * b[inner][column] for inner in range(a_width)), Fraction(0))
            for column in range(b_width)
        )
        for row in range(len(a))
    )


def _rref(rows: Sequence[Sequence[object]], ncols: int | None = None) -> tuple[Matrix, tuple[int, ...], int]:
    matrix, width = _matrix(rows, ncols)
    work = [list(row) for row in matrix]
    pivot_columns: list[int] = []
    pivot_row = 0
    for column in range(width):
        pivot = next((index for index in range(pivot_row, len(work)) if work[index][column]), None)
        if pivot is None:
            continue
        work[pivot_row], work[pivot] = work[pivot], work[pivot_row]
        pivot_value = work[pivot_row][column]
        work[pivot_row] = [value / pivot_value for value in work[pivot_row]]
        for row in range(len(work)):
            if row == pivot_row:
                continue
            factor = work[row][column]
            if factor:
                work[row] = [left - factor * right for left, right in zip(work[row], work[pivot_row])]
        pivot_columns.append(column)
        pivot_row += 1
        if pivot_row == len(work):
            break
    return tuple(tuple(row) for row in work), tuple(pivot_columns), width


def rank(rows: Sequence[Sequence[object]], ncols: int | None = None) -> int:
    """Return exact row rank, including the rank of an explicit empty matrix."""

    _, pivots, _ = _rref(rows, ncols)
    return len(pivots)


def nullspace(rows: Sequence[Sequence[object]], ncols: int | None = None) -> tuple[Vector, ...]:
    """Return the complete right-kernel basis as column vectors."""

    reduced, pivots, width = _rref(rows, ncols)
    pivot_set = set(pivots)
    free_columns = tuple(column for column in range(width) if column not in pivot_set)
    vectors: list[Vector] = []
    for free in free_columns:
        vector = [Fraction(0)] * width
        vector[free] = Fraction(1)
        for reduced_row, pivot in reversed(tuple(enumerate(pivots))):
            vector[pivot] = -sum(
                (reduced[reduced_row][column] * vector[column] for column in free_columns),
            )
        vectors.append(tuple(vector))
    return tuple(vectors)


def left_nullspace(rows: Sequence[Sequence[object]], ncols: int | None = None) -> tuple[Vector, ...]:
    """Return the complete left-kernel basis as columns ``λ`` with ``Aᵀλ=0``."""

    matrix, width = _matrix(rows, ncols)
    return nullspace(transpose(matrix, width), ncols=len(matrix))


def right_inverse(rows: Sequence[Sequence[object]], ncols: int | None = None) -> Matrix:
    """Return an exact ``R`` satisfying ``A R = I`` for full-row-rank A."""

    matrix, width = _matrix(rows, ncols)
    row_count = len(matrix)
    if rank(matrix, width) != row_count:
        raise ValueError("right_inverse requires full row rank")
    if row_count == 0:
        return tuple(tuple() for _ in range(width))
    augmented = tuple(
        tuple(matrix[row]) + tuple(Fraction(1) if row == col else Fraction(0) for col in range(row_count))
        for row in range(row_count)
    )
    reduced, pivots, _ = _rref(augmented, width + row_count)
    if len(pivots) != row_count or any(pivot >= width for pivot in pivots):
        raise ValueError("right_inverse pivot structure is inconsistent")
    result_rows = [tuple(Fraction(0) for _ in range(row_count)) for _ in range(width)]
    for pivot_row, pivot_column in enumerate(pivots):
        result_rows[pivot_column] = tuple(
            reduced[pivot_row][width + target] for target in range(row_count)
        )
    return tuple(result_rows)


def columns_payload(columns: Sequence[Sequence[object]]) -> list[list[str]]:
    """Serialize a column basis without changing its orientation."""

    return [[fraction_text(value) for value in column] for column in columns]


def columns_from_payload(payload: object, *, dimension: int | None = None) -> tuple[Vector, ...]:
    if not isinstance(payload, Sequence) or isinstance(payload, (str, bytes)):
        raise TypeError("columns must be a sequence")
    columns: list[Vector] = []
    for column in payload:
        if not isinstance(column, Sequence) or isinstance(column, (str, bytes)):
            raise TypeError("each column must be a sequence")
        if dimension is not None and len(column) != dimension:
            raise ValueError("column has the wrong dimension")
        columns.append(tuple(as_fraction(value) for value in column))
    return tuple(columns)


@dataclass(frozen=True, slots=True)
class CASResult:
    """Result of an optional exact backend capability."""

    status: str
    value: object | None = None
    reason: str | None = None

    def __post_init__(self) -> None:
        if self.status not in {"available", "unsupported", "error"}:
            raise ValueError("CAS status must be available, unsupported, or error")
        if self.status == "available" and self.value is None:
            raise ValueError("available CAS result needs a value")
        if self.status != "available" and not self.reason:
            raise ValueError("unsupported/error CAS result needs a reason")

    def to_dict(self) -> dict[str, object]:
        return {"status": self.status, "reason": self.reason}


@dataclass(frozen=True, slots=True)
class SparsePolynomial:
    """A canonical sparse polynomial over Q.

    ``terms`` is a sorted tuple of ``(exponent_tuple, coefficient)`` pairs;
    JSON uses an object keyed by comma-separated exponent tuples.
    """

    variables: tuple[str, ...]
    terms: tuple[tuple[Monomial, Fraction], ...] = ()

    def __post_init__(self) -> None:
        variables = tuple(self.variables)
        if any(not isinstance(name, str) or not name for name in variables):
            raise ValueError("polynomial variable names must be non-empty strings")
        if len(set(variables)) != len(variables):
            raise ValueError("polynomial variables must be unique")
        canonical: dict[Monomial, Fraction] = {}
        for monomial, coefficient in self.terms:
            exponent = tuple(monomial)
            if len(exponent) != len(variables):
                raise ValueError("monomial dimension does not match variables")
            if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in exponent):
                raise ValueError("monomial exponents must be non-negative integers")
            value = as_fraction(coefficient)
            canonical[exponent] = canonical.get(exponent, Fraction(0)) + value
        canonical = {monomial: value for monomial, value in canonical.items() if value}
        object.__setattr__(self, "variables", variables)
        object.__setattr__(self, "terms", tuple(sorted(canonical.items())))

    @classmethod
    def zero(cls, variables: Sequence[str]) -> "SparsePolynomial":
        return cls(tuple(variables))

    def to_dict(self) -> dict[str, object]:
        width = len(self.variables)
        return {
            "variables": list(self.variables),
            "terms": {
                ",".join(str(value) for value in monomial) if width else "": fraction_text(coefficient)
                for monomial, coefficient in self.terms
            },
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "SparsePolynomial":
        variables = payload.get("variables")
        raw_terms = payload.get("terms")
        if not isinstance(variables, Sequence) or isinstance(variables, (str, bytes)):
            raise TypeError("polynomial variables must be a sequence")
        if not isinstance(raw_terms, Mapping):
            raise TypeError("polynomial terms must be an object")
        names = tuple(variables)
        parsed: list[tuple[Monomial, Fraction]] = []
        for key, raw_coefficient in raw_terms.items():
            if not isinstance(key, str):
                raise TypeError("monomial keys must be text")
            if not isinstance(raw_coefficient, str):
                raise TypeError("polynomial coefficients must be rational strings")
            if len(names) == 0:
                if key != "":
                    raise ValueError("zero-variable polynomial constant key must be empty")
                monomial = ()
            else:
                parts = key.split(",")
                if len(parts) != len(names) or any(not part.isdigit() for part in parts):
                    raise ValueError("monomial keys must be comma-separated non-negative exponents")
                monomial = tuple(int(part) for part in parts)
            parsed.append((monomial, as_fraction(raw_coefficient)))
        return cls(names, tuple(parsed))

    def derivative(self, variable: str) -> "SparsePolynomial":
        if variable not in self.variables:
            raise ValueError(f"unknown polynomial variable {variable}")
        index = self.variables.index(variable)
        terms: list[tuple[Monomial, Fraction]] = []
        for monomial, coefficient in self.terms:
            if monomial[index] == 0:
                continue
            exponent = list(monomial)
            exponent[index] -= 1
            terms.append((tuple(exponent), coefficient * monomial[index]))
        return SparsePolynomial(self.variables, tuple(terms))

    def add(self, other: "SparsePolynomial") -> "SparsePolynomial":
        if self.variables != other.variables:
            raise ValueError("polynomials must use the same variable order")
        return SparsePolynomial(self.variables, self.terms + other.terms)

    def scale(self, scalar: object) -> "SparsePolynomial":
        factor = as_fraction(scalar)
        return SparsePolynomial(
            self.variables,
            tuple((monomial, factor * coefficient) for monomial, coefficient in self.terms),
        )

    def subtract(self, other: "SparsePolynomial") -> "SparsePolynomial":
        return self.add(other.scale(-1))

    def multiply(self, other: "SparsePolynomial") -> "SparsePolynomial":
        if self.variables != other.variables:
            raise ValueError("polynomials must use the same variable order")
        terms = [
            (
                tuple(left + right for left, right in zip(left_monomial, right_monomial)),
                left_coefficient * right_coefficient,
            )
            for left_monomial, left_coefficient in self.terms
            for right_monomial, right_coefficient in other.terms
        ]
        return SparsePolynomial(self.variables, tuple(terms))


def polynomial_constant(variables: Sequence[str], value: object) -> SparsePolynomial:
    variables = tuple(variables)
    coefficient = as_fraction(value)
    if not coefficient:
        return SparsePolynomial.zero(variables)
    return SparsePolynomial(tuple(variables), ((tuple(0 for _ in variables), coefficient),))


def polynomial_add(left: SparsePolynomial, right: SparsePolynomial) -> SparsePolynomial:
    return left.add(right)


def polynomial_subtract(left: SparsePolynomial, right: SparsePolynomial) -> SparsePolynomial:
    return left.subtract(right)


def polynomial_scale(polynomial: SparsePolynomial, scalar: object) -> SparsePolynomial:
    return polynomial.scale(scalar)


def polynomial_multiply(left: SparsePolynomial, right: SparsePolynomial) -> SparsePolynomial:
    return left.multiply(right)


def determinant(matrix: Sequence[Sequence[SparsePolynomial]]) -> SparsePolynomial:
    """Compute a small exact polynomial determinant by the Leibniz formula."""

    rows = tuple(tuple(row) for row in matrix)
    size = len(rows)
    if any(len(row) != size for row in rows):
        raise ValueError("determinant requires a square polynomial matrix")
    if size == 0:
        raise ValueError("determinant of an empty matrix has no variable context")
    variables = rows[0][0].variables
    if any(item.variables != variables for row in rows for item in row):
        raise ValueError("all determinant entries must use the same variables")
    result = polynomial_constant(variables, 0)
    # A local permutation generator keeps this module stdlib-only.
    def permutations(values: tuple[int, ...]) -> Iterable[tuple[int, ...]]:
        if not values:
            yield ()
            return
        for index, value in enumerate(values):
            for tail in permutations(values[:index] + values[index + 1:]):
                yield (value,) + tail

    for permutation in permutations(tuple(range(size))):
        inversions = sum(
            1 for left in range(size) for right in range(left + 1, size)
            if permutation[left] > permutation[right]
        )
        term = polynomial_constant(variables, 1 if inversions % 2 == 0 else -1)
        for row, column in enumerate(permutation):
            term = term.multiply(rows[row][column])
        result = result.add(term)
    return result


def serialize_polynomial(polynomial: SparsePolynomial) -> dict[str, object]:
    return polynomial.to_dict()


def deserialize_polynomial(payload: Mapping[str, object]) -> SparsePolynomial:
    return SparsePolynomial.from_dict(payload)


def _sympy_poly(polynomial: SparsePolynomial, sympy: Any) -> tuple[Any, tuple[Any, ...]]:
    symbols = tuple(sympy.Symbol(name) for name in polynomial.variables)
    expression = sympy.Integer(0)
    for monomial, coefficient in polynomial.terms:
        term = sympy.Rational(coefficient.numerator, coefficient.denominator)
        for symbol, exponent in zip(symbols, monomial):
            term *= symbol ** exponent
        expression += term
    return sympy.Poly(expression, *symbols, domain=sympy.QQ), symbols


def _from_sympy_poly(poly: Any, variables: tuple[str, ...]) -> SparsePolynomial:
    terms = tuple(
        (
            tuple(int(value) for value in monomial),
            Fraction(int(coefficient.p), int(coefficient.q)),
        )
        for monomial, coefficient in poly.terms()
    )
    return SparsePolynomial(variables, terms)


def _with_sympy(operation: str, callback: Any) -> CASResult:
    try:
        import sympy  # type: ignore[import-not-found]
    except ImportError:
        return CASResult("unsupported", reason="optional SymPy QQ backend is unavailable")
    try:
        return CASResult("available", callback(sympy))
    except Exception as exc:  # exact backend failure remains visible to verifiers
        return CASResult("error", reason=f"SymPy QQ {operation} failed: {type(exc).__name__}: {exc}")


def polynomial_gradient(polynomial: SparsePolynomial) -> CASResult:
    """Return all exact formal derivatives in the polynomial variable order."""

    return _with_sympy(
        "gradient",
        lambda sympy: tuple(
            _from_sympy_poly(poly.diff(symbol), polynomial.variables)
            for symbol in _sympy_poly(polynomial, sympy)[1]
            for poly in (_sympy_poly(polynomial, sympy)[0],)
        ),
    )


def gradient(polynomial: SparsePolynomial) -> CASResult:
    """Short alias used by the Gram/analysis layer."""

    return polynomial_gradient(polynomial)


def directional_derivative(polynomial: SparsePolynomial, direction: Sequence[object]) -> CASResult:
    """Return ``sum(direction[i] * d_i polynomial)`` over Q."""

    values = tuple(as_fraction(value) for value in direction)
    if len(values) != len(polynomial.variables):
        return CASResult("error", reason="direction has the wrong dimension")

    def compute(sympy: Any) -> SparsePolynomial:
        poly, symbols = _sympy_poly(polynomial, sympy)
        expression = sum(
            (sympy.Rational(value.numerator, value.denominator) * poly.diff(symbol) for value, symbol in zip(values, symbols)),
            sympy.Integer(0),
        )
        return _from_sympy_poly(sympy.Poly(expression, *symbols, domain=sympy.QQ), polynomial.variables)

    return _with_sympy("directional derivative", compute)


def square_free_part(polynomial: SparsePolynomial) -> CASResult:
    """Return SymPy QQ's square-free part, with exact sparse serialization."""

    return _with_sympy(
        "square-free part",
        lambda sympy: _from_sympy_poly(
            _sympy_poly(polynomial, sympy)[0].sqf_part().monic(), polynomial.variables
        ),
    )


def polynomial_exact_quotient(dividend: SparsePolynomial, divisor: SparsePolynomial) -> CASResult:
    """Return an exact quotient when ``divisor`` divides ``dividend`` in Q[x]."""

    if dividend.variables != divisor.variables:
        return CASResult("error", reason="dividend and divisor variables differ")

    def compute(sympy: Any) -> SparsePolynomial:
        numerator, _ = _sympy_poly(dividend, sympy)
        denominator, _ = _sympy_poly(divisor, sympy)
        return _from_sympy_poly(numerator.exquo(denominator), dividend.variables)

    return _with_sympy("exact quotient", compute)


def compose_linear(
    polynomial: SparsePolynomial,
    coefficients: Sequence[Sequence[object]],
    *,
    constants: Sequence[object] = (),
    target_variables: Sequence[str] | None = None,
) -> CASResult:
    """Compose a polynomial with exact affine linear forms.

    ``coefficients[i][j]`` and ``constants[i]`` define the replacement for
    source variable ``variables[i]``.  This is the payload operation used for
    ``F(w)=D(Rw)`` in the polynomial-dependence certificate.
    """

    matrix = tuple(tuple(as_fraction(value) for value in row) for row in coefficients)
    if len(matrix) != len(polynomial.variables):
        return CASResult("error", reason="linear composition matrix has the wrong row count")
    target_count = len(matrix[0]) if matrix else len(tuple(constants))
    if any(len(row) != target_count for row in matrix):
        return CASResult("error", reason="linear composition rows have unequal lengths")
    constant_values = tuple(as_fraction(value) for value in constants)
    if constant_values and len(constant_values) != len(polynomial.variables):
        return CASResult("error", reason="linear composition constants have the wrong dimension")
    if not constant_values:
        constant_values = (Fraction(0),) * len(polynomial.variables)
    names = tuple(target_variables) if target_variables is not None else tuple(f"w{index + 1}" for index in range(target_count))
    if len(names) != target_count or len(set(names)) != len(names):
        return CASResult("error", reason="target variable names have the wrong dimension")

    def compute(sympy: Any) -> SparsePolynomial:
        source, _ = _sympy_poly(polynomial, sympy)
        target_symbols = tuple(sympy.Symbol(name) for name in names)
        replacements = tuple(
            sympy.Rational(constant.numerator, constant.denominator)
            + sum(
                (sympy.Rational(value.numerator, value.denominator) * symbol for value, symbol in zip(row, target_symbols)),
                sympy.Integer(0),
            )
            for constant, row in zip(constant_values, matrix)
        )
        expression = source.as_expr().subs(dict(zip(source.gens, replacements)), simultaneous=True)
        return _from_sympy_poly(sympy.Poly(expression, *target_symbols, domain=sympy.QQ), names)

    return _with_sympy("linear composition", compute)


def polynomial_gcd(polynomials: Sequence[SparsePolynomial]) -> CASResult:
    """Compute a Q polynomial gcd with a lazily loaded optional SymPy backend."""

    if not polynomials:
        return CASResult("unsupported", reason="polynomial_gcd needs at least one polynomial")
    try:
        import sympy  # type: ignore[import-not-found]
    except ImportError:
        return CASResult("unsupported", reason="optional SymPy QQ backend is unavailable")
    try:
        variables = polynomials[0].variables
        if any(poly.variables != variables for poly in polynomials):
            return CASResult("error", reason="all polynomials must use the same variables")
        symbols = tuple(sympy.Symbol(name) for name in variables)
        expressions = []
        for polynomial in polynomials:
            expression = sympy.Integer(0)
            for monomial, coefficient in polynomial.terms:
                term = sympy.Rational(coefficient.numerator, coefficient.denominator)
                for symbol, exponent in zip(symbols, monomial):
                    term *= symbol ** exponent
                expression += term
            expressions.append(sympy.Poly(expression, *symbols, domain=sympy.QQ))
        gcd_poly = expressions[0]
        for expression in expressions[1:]:
            gcd_poly = gcd_poly.gcd(expression)
        gcd_poly = gcd_poly.monic()
        return CASResult("available", _from_sympy_poly(gcd_poly, variables))
    except Exception as exc:  # backend errors become explicit evidence, not floats
        return CASResult("error", reason=f"SymPy QQ gcd failed: {type(exc).__name__}: {exc}")


def inertia(rows: Sequence[Sequence[object]], ncols: int | None = None) -> CASResult:
    """Report the deliberate first-backend boundary for exact real inertia."""

    del rows, ncols
    return CASResult(
        "unsupported",
        reason="exact rational symmetric-matrix inertia is not implemented in v1",
    )


__all__ = [
    "Matrix",
    "Vector",
    "Monomial",
    "matrix_shape",
    "identity",
    "transpose",
    "matmul",
    "rank",
    "nullspace",
    "left_nullspace",
    "right_inverse",
    "columns_payload",
    "columns_from_payload",
    "CASResult",
    "SparsePolynomial",
    "polynomial_constant",
    "polynomial_add",
    "polynomial_subtract",
    "polynomial_scale",
    "polynomial_multiply",
    "determinant",
    "serialize_polynomial",
    "deserialize_polynomial",
    "polynomial_gradient",
    "gradient",
    "directional_derivative",
    "square_free_part",
    "polynomial_exact_quotient",
    "compose_linear",
    "polynomial_gcd",
    "inertia",
]
