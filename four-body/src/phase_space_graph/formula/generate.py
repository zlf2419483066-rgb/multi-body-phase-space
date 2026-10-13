"""High-level, explicit formula generation entry points."""

from __future__ import annotations

from typing import Any, Iterable
import hashlib

from ..formula_inputs import FormulaInputCard
from ..physical_labels import NormalizationResult, PhysicalLabelRecord
from .benchmark import compare_paths
from .direct_charts import build_direct_formula, select_direct_chart
from .raw import build_f0_expression, reduce_covariant
from .records import FormulaArtifact, FormulaOptions
from .reference import build_bridge_formula, build_reference_chart
from .inputs import prepare_formula_input


def generate_formula(
    record: PhysicalLabelRecord | NormalizationResult | FormulaInputCard,
    *,
    measure_convention: str | None = None,
    options: FormulaOptions | None = None,
    identity: Any = None,
    run_benchmark: bool = True,
    masses: tuple[float, ...] | None = None,
    total_mass: float | None = None,
) -> FormulaArtifact:
    convention = measure_convention or (options.normalization if options else (
        record.measure_convention if isinstance(record, FormulaInputCard) else "jing_psf"))
    if options is not None and options.normalization != convention:
        raise ValueError("measure_convention conflicts with FormulaOptions.normalization")
    card = prepare_formula_input(record, measure_convention=convention, identity=identity)
    options = options or FormulaOptions(normalization=card.measure_convention)
    source = card.to_dict()
    input_digest = hashlib.sha256(card.to_json().encode("utf-8")).hexdigest()
    raw = cov = None
    direct = bridge = benchmark = None
    diagnostics: list[str] = []
    statuses = {"f0": "pending", "covariant": "pending", "direct": "pending", "bridge": "pending", "benchmark": "pending", "source": "pending"}
    if card.availability != "ready":
        diagnostics.append(card.unavailable_reason or "formula input is unavailable")
        statuses.update({"f0": "blocked", "covariant": "blocked", "direct": "blocked", "bridge": "blocked", "benchmark": "blocked"})
        return FormulaArtifact(card.identity, options, None, None, None, None, None, statuses, tuple(diagnostics), source, input_digest)
    raw = build_f0_expression(card); statuses["f0"] = raw.status
    cov = reduce_covariant(card, raw); statuses["covariant"] = cov.status
    diagnostics.extend(cov.diagnostics)
    chart = select_direct_chart(card, options=options)
    direct = build_direct_formula(card, chart, options=options); statuses["direct"] = direct.status
    ref_chart = build_reference_chart(card, options=options)
    bridge = build_bridge_formula(card, ref_chart, options=options); statuses["bridge"] = bridge.status
    if run_benchmark and masses is not None and total_mass is not None:
        benchmark = compare_paths(direct, bridge, masses=masses, total_mass=total_mass)
        statuses["benchmark"] = benchmark.status
    elif run_benchmark:
        statuses["benchmark"] = "pending"
        diagnostics.append("benchmark requires explicit masses and total_mass")
    return FormulaArtifact(card.identity, options, raw, cov, direct, bridge, benchmark, statuses, tuple(diagnostics), source, input_digest)


def generate_formulas(records: Iterable[PhysicalLabelRecord | NormalizationResult | FormulaInputCard], **kwargs: Any) -> tuple[FormulaArtifact, ...]:
    return tuple(generate_formula(record, **kwargs) for record in records)


def generate_complete_formula(
    record: PhysicalLabelRecord | NormalizationResult | FormulaInputCard,
    *,
    measure_convention: str | None = None,
    options: FormulaOptions | None = None,
    identity: Any = None,
    run_benchmark: bool = True,
    masses: tuple[float, ...] | None = None,
    total_mass: float | None = None,
) -> FormulaArtifact:
    """Build the complete supported four-body d=2..5 formula artifact.

    The legacy :func:`generate_formula` keeps its existing direct-chart
    selection. This entry point extends the complete route with the low-loop
    mixed chart for d=2,3,4 while retaining the established D5 builder.
    """

    convention = measure_convention or (options.normalization if options else (
        record.measure_convention if isinstance(record, FormulaInputCard) else "jing_psf"))
    if options is not None and options.normalization != convention:
        raise ValueError("measure_convention conflicts with FormulaOptions.normalization")
    card = prepare_formula_input(record, measure_convention=convention, identity=identity)
    options = options or FormulaOptions(normalization=card.measure_convention)
    source = card.to_dict()
    input_digest = hashlib.sha256(card.to_json().encode("utf-8")).hexdigest()
    raw = cov = None
    direct = bridge = benchmark = None
    diagnostics: list[str] = []
    statuses = {
        "f0": "pending", "covariant": "pending", "direct": "pending",
        "bridge": "pending", "benchmark": "pending", "source": "pending",
    }
    if card.availability != "ready":
        diagnostics.append(card.unavailable_reason or "formula input is unavailable")
        statuses.update({
            "f0": "blocked", "covariant": "blocked", "direct": "blocked",
            "bridge": "blocked", "benchmark": "blocked",
        })
        return FormulaArtifact(
            card.identity, options, None, None, None, None, None,
            statuses, tuple(diagnostics), source, input_digest,
        )

    raw = build_f0_expression(card)
    statuses["f0"] = raw.status
    cov = reduce_covariant(card, raw)
    statuses["covariant"] = cov.status
    diagnostics.extend(cov.diagnostics)

    dimension = len(card.independent_variables)
    if card.n == 4 and dimension in (2, 3, 4):
        from .low_loop_direct import build_low_loop_formula

        direct = build_low_loop_formula(card, options=options)
    elif card.n == 4 and dimension == 5:
        from .d5_direct import build_d5_formula

        direct = build_d5_formula(card, options=options)
    else:
        chart = select_direct_chart(card, options=options)
        direct = build_direct_formula(card, chart, options=options)
    statuses["direct"] = direct.status

    reference_chart = build_reference_chart(card, options=options)
    bridge = build_bridge_formula(card, reference_chart, options=options)
    statuses["bridge"] = bridge.status
    if run_benchmark and masses is not None and total_mass is not None:
        benchmark = compare_paths(direct, bridge, masses=masses, total_mass=total_mass)
        statuses["benchmark"] = benchmark.status
    elif run_benchmark:
        statuses["benchmark"] = "pending"
        diagnostics.append("benchmark requires explicit masses and total_mass")
    return FormulaArtifact(
        card.identity, options, raw, cov, direct, bridge, benchmark,
        statuses, tuple(diagnostics), source, input_digest,
    )


def generate_complete_formulas(
    records: Iterable[PhysicalLabelRecord | NormalizationResult | FormulaInputCard],
    **kwargs: Any,
) -> tuple[FormulaArtifact, ...]:
    """Build complete supported artifacts for a sequence of formula inputs."""

    return tuple(generate_complete_formula(record, **kwargs) for record in records)


__all__ = [
    "generate_formula", "generate_formulas",
    "generate_complete_formula", "generate_complete_formulas",
]
