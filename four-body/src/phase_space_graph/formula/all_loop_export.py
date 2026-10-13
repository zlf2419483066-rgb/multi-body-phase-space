"""Six-step Markdown and machine-evidence export for all four-body loop classes."""

from __future__ import annotations

import json
from fractions import Fraction
import math
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence

from .all_loop_catalog import (
    AllLoopCatalog,
    AllLoopMarkedClass,
    build_all_loop_catalog,
    select_all_loop_classes,
)
from .d5_export import (
    _display_latex,
    _field,
    _formula_blocks,
    _inline_math,
    _items,
    _render_step_five,
    _render_step_four,
    _render_step_six,
    _render_step_three,
    build_d5_formula_evidence,
)
from .inputs import prepare_formula_input
from .tree_covariant import build_tree_covariant_trace


ALL_LOOP_EXPORT_RELATIVE_PATH = Path(
    "文档/验证与测试/误差分析/四体全圈143类公式_2026-10-09/run01"
)
ALL_LOOP_EXPORT_SCHEMA_VERSION = "four-body-all-loop-six-step-v1"


def _jsonable(value: Any) -> Any:
    if hasattr(value, "to_dict"):
        return _jsonable(value.to_dict())
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def _canonical_json(value: Any, *, indent: int | None = None) -> str:
    return json.dumps(
        _jsonable(value), ensure_ascii=False, sort_keys=True,
        separators=None if indent is not None else (",", ":"),
        indent=indent, allow_nan=False,
    )


def _display_data(label: str, value: Any) -> list[str]:
    """Render actual low-loop chart payloads without guessing at absent fields."""

    if value is None:
        return [f"- `{label}`：未提供。"]
    if isinstance(value, Mapping):
        rows: list[str] = []
        for key, item in value.items():
            rows.extend(_display_data(f"{label}.{key}", item))
        return rows or [f"- `{label}`：{{}}。"]
    if isinstance(value, (tuple, list)):
        if not value:
            return [f"- `{label}`：空列表。"]
        if all(not isinstance(item, (Mapping, tuple, list)) for item in value):
            rendered = "; ".join(str(item) for item in value)
            return [f"- `{label}`：{_inline_math(rendered)}"]
        rows = []
        for index, item in enumerate(value):
            rows.extend(_display_data(f"{label}[{index}]", item))
        return rows
    if isinstance(value, str) and value:
        if "\\" in value or "=" in value or value.startswith(("s_", "S_", "J_", "\\")):
            return [f"- `{label}`：{_display_latex(value)}"]
        return [f"- `{label}`：`{value}`"]
    return [f"- `{label}`：`{value}`"]


def _fraction_text(value: Fraction) -> str:
    return str(value.numerator) if value.denominator == 1 else f"{value.numerator}/{value.denominator}"


def _fraction_latex(value: Fraction) -> str:
    if value.denominator == 1:
        return str(value.numerator)
    return rf"\frac{{{value.numerator}}}{{{value.denominator}}}"


def _math_identifier(value: Any) -> str:
    name = str(value)
    subset = re.fullmatch(r"s(\d+)", name)
    if subset:
        return rf"s_{{{subset.group(1)}}}"
    return {
        "Jx": r"J_x",
        "Jy": r"J_y",
        "internal_phi": r"\varphi",
    }.get(name, name)


def _det_c_e_latex(chart: Mapping[str, Any]) -> str:
    scale = chart.get("det_C_E_scale", {})
    base = abs(Fraction(str(chart.get("det_C_E_base", "1"))))
    pieces = [_fraction_latex(base)]
    for key, symbol in (("jx1_power", r"|j_{x1}|"), ("jy1_power", r"|j_{y1}|")):
        power = int(scale.get(key, 0))
        if power == 1:
            pieces.append(symbol)
        elif power > 1:
            pieces.append(rf"{symbol}^{{{power}}}")
    return r"|\det C_E|=" + r"\,".join(pieces)


def _constraint_entry_latex(chart: Mapping[str, Any], row: int, column: int) -> str:
    base = Fraction(str(chart["constraint_matrix_base"][row][column]))
    if not base:
        return "0"
    scale = {0: r"j_{x1}", 1: r"j_{y1}", 2: ""}[column]
    magnitude = abs(base)
    coefficient = "" if magnitude == 1 and scale else _fraction_latex(magnitude)
    token = coefficient + scale
    return ("-" if base < 0 else "") + token


def _constraint_matrix_from_columns(chart: Mapping[str, Any], columns: Sequence[int]) -> str:
    rows = chart["constraint_matrix_base"]
    if not rows:
        return r"\varnothing"
    body = r" \\ ".join(
        " & ".join(_constraint_entry_latex(chart, row_index, column) for column in columns)
        for row_index in range(len(rows))
    )
    return r"\begin{pmatrix}" + body + r"\end{pmatrix}"


def _affine_equation(
    label: str,
    coefficients: Iterable[Any],
    variables: Iterable[str],
    offsets: Mapping[str, Any],
) -> str:
    terms: list[tuple[Fraction, str]] = []
    for coefficient, variable in zip(coefficients, variables):
        value = Fraction(str(coefficient))
        if value:
            terms.append((value, _math_identifier(variable)))
    for variable, coefficient in offsets.items():
        value = Fraction(str(coefficient))
        if value:
            terms.append((value, str(variable)))
    if not terms:
        rhs = "0"
    else:
        pieces: list[str] = []
        for index, (coefficient, variable) in enumerate(terms):
            magnitude = abs(coefficient)
            factor = "" if magnitude == 1 else _fraction_latex(magnitude) + r"\,"
            token = factor + variable
            if index == 0:
                pieces.append(("-" if coefficient < 0 else "") + token)
            else:
                pieces.append((" - " if coefficient < 0 else " + ") + token)
        rhs = "".join(pieces)
    return f"{_math_identifier(label)} = {rhs}"


def _formula_lines(value: Any, *, label: str | None = None) -> list[str]:
    """Render only explicit LaTeX strings, never machine-schema prose as math."""

    if isinstance(value, Mapping):
        lines: list[str] = []
        for key, expression in value.items():
            lines.extend(_formula_lines(expression, label=str(key)))
        return lines
    if isinstance(value, (tuple, list)):
        lines = []
        for expression in value:
            lines.extend(_formula_lines(expression, label=label))
        return lines
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"required LaTeX field {label or 'formula'} is missing")
    block = _display_latex(value)
    return [f"- `{label}`：{block}" if label else block]


def _variable_list_latex(values: Iterable[Any]) -> str:
    names = {
        "internal_phi": r"\varphi",
        "theta": r"\theta",
        "phi": r"\phi",
        "psi": r"\psi",
    }
    return r"(" + r",\;".join(names.get(str(value), _math_identifier(value)) for value in values) + r")"


def _render_marker_graph(card: Any) -> list[str]:
    if isinstance(card, Mapping):
        raw_record = card.get("record")
        variable_labels = card.get("variable_labels", [])
        provenance = card.get("provenance")
        status = card.get("status")
        measure_convention = (
            provenance.get("measure_convention")
            if isinstance(provenance, Mapping) else None
        )
        if measure_convention is None and isinstance(status, Mapping):
            measure_convention = status.get("measure_convention")
        if measure_convention is None:
            measure_convention = card.get("measure_convention")
    else:
        raw_record = card.record
        variable_labels = card.variable_labels
        measure_convention = card.measure_convention
    if measure_convention not in {"jing_psf", "standard_2pi4"}:
        raise ValueError("formula input card must provide a recognized measure convention")
    record = raw_record if isinstance(raw_record, dict) else {}
    topology = record.get("topology", {})
    labeling = record.get("complete_labeling", {})
    singles = {
        (row["edge_index"], row["occurrence"]): row["particle_id"]
        for row in labeling.get("single_lines", [])
    }
    doubles = {
        (row["edge_index"], row["occurrence"]): tuple(row["particle_ids"])
        for row in labeling.get("double_lines", [])
    }
    final_by_vertex: dict[int, list[int]] = {}
    for row in labeling.get("final_particles", []):
        final_by_vertex.setdefault(row["vertex"], []).append(row["particle_id"])
    lines = [
        "### 标记图与连接表",
        "",
        f"- 初始顶点：`v{topology.get('initial_vertex', '?')}`；终态粒子："
        + ", ".join(
            f"`p{particle}@v{vertex}`"
            for vertex in sorted(final_by_vertex)
            for particle in sorted(final_by_vertex[vertex])
        ),
        "- 图编码来自本类代表 fixed 记录；边索引与 occurrence 沿用输入卡。",
        "",
        "| edge | source → target | 类型 | occurrence 标记 |",
        "|---:|---|---|---|",
    ]
    for edge_index, edge in enumerate(topology.get("edges", [])):
        marked: list[str] = []
        for occurrence in range(edge.get("multiplicity", 0)):
            if edge.get("edge_type") == "double":
                particles = doubles.get((edge_index, occurrence), ())
                label = "{" + ",".join(f"p{particle}" for particle in particles) + "}"
            else:
                particle = singles.get((edge_index, occurrence))
                label = f"p{particle}" if particle is not None else "?"
            marked.append(f"o{occurrence}: {label}")
        lines.append(
            f"| e{edge_index} | v{edge.get('source')} → v{edge.get('target')} | "
            f"{edge.get('edge_type')} × {edge.get('multiplicity')} | {'; '.join(marked)} |"
        )
    lines.extend([
        "",
        f"- 有序 IKV：`{', '.join(variable_labels)}`。",
        f"- 输入测度约定：`{measure_convention}`；公式来源状态：`not_checked`。",
        "",
    ])
    return lines


def _render_step_one(trace: Mapping[str, Any]) -> list[str]:
    f0 = _field(trace, "f0", {})
    factors = _items(_field(f0, "factors", []))
    double_factors = _items(_field(f0, "double_line_factors", []))
    double_latex = _items(_field(f0, "double_line_factors_latex", []))
    scope = _field(trace, "scope", {})
    d = _field(scope, "d")
    vertices = _field(scope, "vertex_count")
    if not isinstance(d, int) or not isinstance(vertices, int):
        raise ValueError("tree trace scope must state d and the vertex count")
    if len(factors) != vertices or len(double_factors) != d or len(double_latex) != d:
        raise ValueError("tree trace does not contain all actual vertices and d ds factors")
    required = (
        "expanded_expression_latex", "conservation_equation_latex", "incoming_momentum_latex",
        "outgoing_integration_momenta_latex", "outgoing_support_momenta_latex",
        "momentum_delta_latex", "normalization_2pi_power",
    )
    lines = [
        f"## 1. 原图表达式：{vertices} 个二体块与 {d} 条双线测度",
        "",
        "逐个顶点列出真实局部因子、出射测度、顶点守恒和原始 delta。",
        "",
    ]
    for factor in factors:
        missing = [key for key in required if _field(factor, key) is None]
        if missing:
            raise ValueError(f"vertex {_field(factor, 'vertex', '?')} lacks trace fields {missing}")
        lines.extend([
            f"### 顶点 $v_{{{_field(factor, 'vertex', '?')}}}$",
            "",
            _display_latex(_field(factor, "expanded_expression_latex")),
            f"- 守恒方程：{_inline_math(_field(factor, 'conservation_equation_latex'))}",
            f"- 输入动量：{_inline_math(_field(factor, 'incoming_momentum_latex'))}",
            f"- 出射积分动量：{_inline_math(_field(factor, 'outgoing_integration_momenta_latex'))}",
            f"- 壳支撑动量：{_inline_math(_field(factor, 'outgoing_support_momenta_latex'))}",
            f"- 原顶点 delta：{_inline_math(_field(factor, 'momentum_delta_latex'))}",
            f"- 局部 $2\\pi$ 次幂：`{_field(factor, 'normalization_2pi_power')}`",
            "",
        ])
    lines.extend(["### 双线 $ds$ 因子", "", "| edge / occurrence | 实际因子 |", "|---|---|"])
    for factor, latex in zip(double_factors, double_latex):
        lines.append(
            f"| e{_field(factor, 'edge_index', '?')}/o{_field(factor, 'occurrence', '?')} | "
            f"{_inline_math(latex)} |"
        )
    cancellation = _field(trace, "cancellation", {})
    cancelled = _items(_field(cancellation, "cancelled_measures", []))
    retained = _items(_field(cancellation, "retained_external_measures", []))
    lines.extend([
        "",
        "### D0 重复末态测度账目",
        "",
        f"- 策略：`{_field(cancellation, 'policy', '?')}`；实际取消 {len(cancelled)} 个内部单线副本，保留 {len(retained)} 个外线单粒子测度。",
    ])
    for row in cancelled:
        locator = _field(row, "locator", {})
        measure = _field(row, "measure", {})
        lines.append(
            f"- 取消 internal e{_field(locator, 'edge_index', '?')}/o{_field(locator, 'occurrence', '?')}："
            f"粒子 `{_field(measure, 'particle_ids', [])}`。"
        )
    for row in retained:
        source = _field(row, "source_reference", {})
        lines.append(
            f"- 保留外线粒子 `{_field(row, 'particle_ids', [])}`，位于 "
            f"v{_field(source, 'vertex', '?')}/slot{_field(source, 'slot', '?')}。"
        )
    normalization = _field(trace, "normalization", {})
    lines.extend(["", "### 原图局部归一化账", ""])
    for key in (
        "normalization_identity_latex", "per_tree_integration_balance_latex",
        "final_external_measure_factor_latex",
    ):
        expression = _field(normalization, key)
        if not expression:
            raise ValueError(f"tree trace lacks normalization field {key}")
        lines.append(_display_latex(expression))
    return lines


def _render_step_two(trace: Mapping[str, Any]) -> list[str]:
    elimination = _field(trace, "elimination", {})
    steps = _items(_field(elimination, "steps", []))
    transfers = _items(_field(trace, "q_shell_transfers", []))
    scope = _field(trace, "scope", {})
    d = _field(scope, "d")
    if not isinstance(d, int) or len(steps) != d or len(transfers) != d:
        raise ValueError("tree trace must contain exactly d actual leaf eliminations and shell transfers")
    cancellation = _field(trace, "cancellation", {})
    lines = [
        "## 2. 实际逐叶消元、根守恒与壳转移",
        "",
        f"- 从实际双线树执行 {d} 次四维 delta 消元；D0 取消 {len(_items(_field(cancellation, 'cancelled_measures', [])))} 个内部单线副本。",
        "",
    ]
    for index, step in enumerate(steps, start=1):
        fields = (
            "delta4_source_latex", "current_equation_latex", "integrated_q",
            "substitution_latex", "abs_4d_jacobian_latex",
        )
        missing = [key for key in fields if _field(step, key) is None]
        if missing:
            raise ValueError(f"tree leaf step {index} lacks actual replay fields {missing}")
        lines.extend([
            f"### q 消元 {index}：v{_field(step, 'leaf_vertex', '?')} → v{_field(step, 'parent_vertex', '?')}",
            "",
            f"- 原顶点 delta：{_inline_math(_field(step, 'delta4_source_latex'))}",
            f"- 经先前代入后的当前约束：{_inline_math(_field(step, 'current_equation_latex'))}",
            f"- 积分变量：{_inline_math(_field(step, 'integrated_q'))}",
            f"- 实际代入：{_inline_math(_field(step, 'substitution_latex'))}",
            f"- 四维绝对 Jacobian：{_inline_math(_field(step, 'abs_4d_jacobian_latex'))}",
            "",
        ])
    for key, label in (("root_equation_latex", "根方程"), ("root_residual_latex", "根残差")):
        value = _field(elimination, key)
        if not value:
            raise ValueError(f"tree trace lacks {label}")
        lines.append(f"- {label}：{_inline_math(value)}")
    lines.extend([
        f"- 剩余 q 变量：{_inline_math(_field(elimination, 'remaining_q_variables'))}",
        "",
        "### q 壳转移与保留 ds",
        "",
    ])
    for transfer in transfers:
        fields = (
            "derived_solution_latex", "global_support_solution_latex", "shell_delta_latex",
            "positive_energy_transfer_latex", "ds_factor_in_final_expression_latex",
        )
        missing = [key for key in fields if not _field(transfer, key)]
        if missing:
            raise ValueError(f"q-shell transfer lacks actual fields {missing}")
        lines.extend([
            f"#### {_inline_math(_field(transfer, 'integrated_q', '?'))}",
            "",
            f"- 树内解：{_inline_math(_field(transfer, 'derived_solution_latex'))}",
            f"- 用根守恒得到的支撑：{_inline_math(_field(transfer, 'global_support_solution_latex'))}",
            f"- 在壳 delta：{_inline_math(_field(transfer, 'shell_delta_latex'))}",
            f"- 正能量条件：{_inline_math(_field(transfer, 'positive_energy_transfer_latex'))}",
            f"- 保留 ds：{_inline_math(_field(transfer, 'ds_factor_in_final_expression_latex'))}",
            "",
        ])
    constraints = _items(_field(trace, "remaining_constraints_latex", []))
    if len(constraints) != d + 1:
        raise ValueError("trace must retain the global constraint and d ordered IKV constraints")
    lines.extend(["### 剩余整体守恒与原图 IKV 约束", ""])
    lines.extend(_display_latex(constraint) for constraint in constraints)
    lines.append("")
    return lines


def build_all_loop_formula_evidence(
    marked_class: AllLoopMarkedClass,
    *,
    normalization: str = "jing_psf",
    benchmark_masses: Sequence[float] | None = None,
    benchmark_total_mass: float | None = None,
) -> dict[str, Any]:
    """Build one class's trace and direct chart evidence without running a benchmark."""

    if normalization not in {"jing_psf", "standard_2pi4"}:
        raise ValueError("normalization must be 'jing_psf' or 'standard_2pi4'")
    if (benchmark_masses is None) != (benchmark_total_mass is None):
        raise ValueError("benchmark_masses and benchmark_total_mass must be provided together")
    if benchmark_masses is not None:
        benchmark_masses = tuple(float(value) for value in benchmark_masses)
        benchmark_total_mass = float(benchmark_total_mass)
        if len(benchmark_masses) != 4:
            raise ValueError("benchmark_masses must contain exactly four values")
        if any(not math.isfinite(value) for value in benchmark_masses) or not math.isfinite(benchmark_total_mass):
            raise ValueError("benchmark masses and total mass must be finite")
    if not isinstance(marked_class, AllLoopMarkedClass):
        raise TypeError("build_all_loop_formula_evidence expects an AllLoopMarkedClass")

    identity_record = marked_class.to_dict()
    if marked_class.loop_count == 3:
        d5 = build_d5_formula_evidence(
            marked_class,
            normalization=normalization,
            benchmark_masses=tuple(benchmark_masses) if benchmark_masses is not None else None,
            benchmark_total_mass=benchmark_total_mass,
        )
        artifact = d5["formula_artifact"]
        chart = d5["d5_chart_data"]
        return {
            "artifact_type": "four_body_marked_class_formula_evidence",
            "schema_version": ALL_LOOP_EXPORT_SCHEMA_VERSION,
            "class_id": marked_class.class_id,
            "loop_count": marked_class.loop_count,
            "d": 2 + marked_class.loop_count,
            "topology_id": marked_class.topology_id,
            "identity_class": identity_record,
            "representative_fixed_digest": marked_class.representative_fixed_digest,
            "normalization": normalization,
            "source_status": "not_checked",
            "benchmark": artifact.get("internal_benchmark") or {
                "status": "pending", "reason": "benchmark was not requested for this export"
            },
            "formula_build_status": _field(artifact.get("statuses", {}), "direct", "unknown"),
            "formula_statuses": artifact.get("statuses", {}),
            "input_card": d5["input_card"],
            "tree_covariant_trace": d5["tree_covariant_trace"],
            "chart_data": chart,
            "direct_formula": None,
            "d5_formula_evidence": d5,
        }

    card = prepare_formula_input(
        marked_class.representative_record,
        measure_convention=normalization,
    )
    from .low_loop_direct import build_low_loop_chart_data, build_low_loop_formula
    from .records import FormulaOptions

    trace_object = build_tree_covariant_trace(card)
    trace = trace_object.to_dict()
    chart = build_low_loop_chart_data(card)
    if not isinstance(chart, dict):
        raise TypeError("build_low_loop_chart_data must return a dictionary")
    options = FormulaOptions(normalization=normalization)
    formula_object = build_low_loop_formula(card, options=options)
    formula = formula_object.to_dict() if hasattr(formula_object, "to_dict") else formula_object
    if not isinstance(formula, dict):
        raise TypeError("build_low_loop_formula must return a DirectFormula or dictionary")
    chart_status = chart.get("status", "unknown")
    formula_status = formula.get("status", "unknown")
    if chart_status != "built":
        raise ValueError(f"low-loop chart status is {chart_status!r}, expected 'built'")
    if formula_status != "built":
        raise ValueError(f"low-loop formula status is {formula_status!r}, expected 'built'")
    if benchmark_masses is None:
        benchmark = {"status": "pending", "reason": "benchmark was not requested for this export"}
    else:
        # This lazy path becomes active for the low-loop chart when the shared
        # benchmark dispatcher supports direct_gram_low_loop. Its report is
        # recorded verbatim, including pending/unsupported outcomes.
        from .benchmark import compare_paths
        from .reference import build_bridge_formula, build_reference_chart

        reference_chart = build_reference_chart(card, options=options)
        bridge = build_bridge_formula(card, reference_chart, options=options)
        report = compare_paths(
            formula_object,
            bridge,
            masses=benchmark_masses,
            total_mass=benchmark_total_mass,
        )
        benchmark = report.to_dict() if hasattr(report, "to_dict") else _jsonable(report)
    return {
        "artifact_type": "four_body_marked_class_formula_evidence",
        "schema_version": ALL_LOOP_EXPORT_SCHEMA_VERSION,
        "class_id": marked_class.class_id,
        "loop_count": marked_class.loop_count,
        "d": 2 + marked_class.loop_count,
        "topology_id": marked_class.topology_id,
        "identity_class": identity_record,
        "representative_fixed_digest": marked_class.representative_fixed_digest,
        "normalization": normalization,
        "source_status": "not_checked",
        "benchmark": benchmark,
        "formula_build_status": formula_status,
        "formula_statuses": {"covariant": "built", "direct": formula_status},
        "input_card": card.to_dict(),
        "tree_covariant_trace": trace,
        "chart_data": chart,
        "direct_formula": formula,
    }


def _render_low_loop_step_three(chart: Mapping[str, Any]) -> list[str]:
    required = (
        "profile", "target_variables", "angle_variables", "internal_variables",
        "h", "t", "U", "K_latex", "det_K", "Z_variables", "Z_forms",
        "A_Z_latex", "inverse_A_Z_latex", "D", "angular_expansion_latex",
    )
    missing = [key for key in required if key not in chart]
    if missing:
        raise ValueError(f"low-loop chart lacks actual step-3 data fields {missing}")
    target_variables = tuple(chart["target_variables"])
    dimension = chart.get("dimension", len(target_variables))
    if not isinstance(dimension, int) or dimension != len(target_variables) or dimension not in (2, 3, 4):
        raise ValueError("low-loop target-variable count disagrees with d=2+L")
    angular = chart["angular_expansion_latex"]
    if not isinstance(angular, Mapping):
        raise ValueError("low-loop angular_expansion_latex must be a mapping")
    for key in ("Jx", "Jy", "v", "coefficients", "auxiliary"):
        if key not in angular:
            raise ValueError(f"low-loop angular expansion lacks LaTeX field {key}")
    b_variables = chart.get("B_variables")
    if not isinstance(b_variables, (tuple, list)) or len(b_variables) != len(target_variables):
        raise ValueError("low-loop B variables must span the d original IKV")
    z_forms = chart["Z_forms"]
    if not isinstance(z_forms, (tuple, list)) or len(z_forms) != 5:
        raise ValueError("low-loop chart must provide all five actual Z affine forms")
    selected = chart.get("angular_expansion", {}).get("selected_labels", {})
    if not isinstance(selected, Mapping) or any(key not in selected for key in ("h", "t", "Jx", "Jy", "v")):
        raise ValueError("low-loop chart lacks selected subsystem labels for h,t,Jx,Jy,v")
    selected_latex = (
        r"h=" + _math_identifier(selected["h"])
        + r",\quad t=" + _math_identifier(selected["t"])
        + r",\quad J_x=" + _math_identifier(selected["Jx"])
        + r",\quad J_y=" + _math_identifier(selected["Jy"])
        + r",\quad v=" + _math_identifier(selected["v"])
    )
    geometry = chart.get("geometry")
    if not isinstance(geometry, Mapping):
        raise ValueError("low-loop chart lacks its selected local geometry")
    mode = geometry.get("mode")
    particle_order = geometry.get("particles")
    if mode not in {"pair", "nested"} or not isinstance(particle_order, (tuple, list)) or len(particle_order) != 4:
        raise ValueError("low-loop geometry must declare pair/nested mode and particle order (a,b,c,d)")
    try:
        a, b, c, d = (int(value) for value in particle_order)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("low-loop geometry particle order must contain four particle ids") from exc
    if any(value not in (1, 2, 3, 4) for value in (a, b, c, d)) or len({a, b, c, d}) != 4:
        raise ValueError("low-loop geometry particle order must be a permutation of 1,2,3,4")
    particle_map_latex = (
        rf"(a,b,c,d)=({a},{b},{c},{d}),\quad{{}}"
        rf"(m_a,m_b,m_c,m_d)=(m_{{{a}}},m_{{{b}}},m_{{{c}}},m_{{{d}}})"
    )
    if mode == "pair":
        local_geometry_latex = (
            r"\text{pair mode in the total }P\text{ rest frame:}\quad{}"
            r"q_{ab}\parallel +\hat z,\quad q_{cd}\parallel-\hat z;\quad{}"
            r"x=\cos\angle(\mathbf p_a^{(ab)\,\mathrm{rest}},+\hat z),\quad{}"
            r"y=\cos\angle(\mathbf p_c^{(cd)\,\mathrm{rest}},+\hat z),\quad{}"
            r"\varphi=\varphi_c-\varphi_a,\quad{}\varphi_a=0,\ \varphi_c=\varphi"
        )
    else:
        local_geometry_latex = (
            r"\text{nested mode in the }Q_{abc}\text{ rest frame:}\quad{}"
            r"p_c\parallel +\hat z,\quad q_{ab}\parallel-\hat z;\quad{}"
            r"x=\cos\angle(\mathbf p_a^{(ab)\,\mathrm{rest}},+\hat z),\quad{}"
            r"y=\cos\angle(\mathbf p_d^{(abc)},\mathbf p_c^{(abc)}),\quad{}"
            r"\varphi=\varphi_d-\varphi_a,\quad{}\varphi_a=0,\ \varphi_d=\varphi"
        )
    lines = [
        "## 3. 以子系统角定义的低圈坐标与精确线性换元",
        "",
        f"- chart profile/status：`{chart['profile']}` / `{chart.get('status', 'unknown')}`。",
        f"- 原图变量：{_inline_math(r'\mathbf S=(' + r',\;'.join(_math_identifier(value) for value in target_variables) + r')^{\mathsf T}')}。",
        f"- 保留内部角：{_inline_math(_variable_list_latex(chart['internal_variables']))}；总转动角：{_inline_math(_variable_list_latex(chart.get('euler_variables', [])))}。",
        "",
        "### 子系统质量与补足坐标",
        "",
        f"- 显式 $h(S)$：{_inline_math(_affine_equation(chart['h']['label'], chart['h']['S_coefficients'], chart['target_variables'], chart['h']['mass_offset_coefficients']))}",
        f"- 显式 $t(S)$：{_inline_math(_affine_equation(chart['t']['label'], chart['t']['S_coefficients'], chart['target_variables'], chart['t']['mass_offset_coefficients']))}",
        *[
            f"- 显式 $U(S)$：{_inline_math(_affine_equation(row['label'], row['S_coefficients'], chart['target_variables'], row['mass_offset_coefficients']))}"
            for row in chart["U"]
        ],
        "",
        "### 原图 IKV 到 $(h,t,U)$ 的可逆仿射换元",
        "",
        f"- 补足变量次序：{_inline_math(r'\mathbf B=(' + r',\;'.join(_math_identifier(value) for value in b_variables) + r')^{\mathsf T}')}。",
        _display_latex(r"\mathbf B=K\mathbf S+\mathbf b_B"),
        _display_latex(chart["K_latex"]),
        f"- $\\det K={chart['det_K']}$；固定质量的仿射偏移逐行保存在机器证据 JSON。",
        "",
        "### 五个完整不变量与回到基底变量的映射",
        "",
        f"- { _inline_math(r'\mathbf Z=(' + r',\;'.join(_math_identifier(value) for value in chart['Z_variables']) + r')^{\mathsf T}') }。",
        f"- 基底变量：{_inline_math(r'\mathbf X=(' + r',\;'.join(_math_identifier(value) for value in chart['basis_variables']) + r')^{\mathsf T}')}。",
        _display_latex(r"\mathbf Z=A_Z\mathbf X+\mathbf b_Z,\qquad{}\mathbf X=A_Z^{-1}(\mathbf Z-\mathbf b_Z)"),
        _display_latex(chart["A_Z_latex"]),
        _display_latex(chart["inverse_A_Z_latex"]),
        f"- $D=|\\det A_Z|={chart['D']}$。",
        *[
            f"- 实际 affine 形式：{_inline_math(_affine_equation(row['label'], row['X_coefficients'], chart['basis_variables'], row['mass_offset_coefficients']))}"
            for row in z_forms
        ],
        "",
        "### 实际 $J_x,J_y,v$ 角展开系数与运动学辅助量",
        "",
        "- 实际局部粒子次序及质量符号：",
        _display_latex(particle_map_latex),
        f"- 所选模式：`{mode}`；其局部轴、余弦角和有向方位角定义为：",
        _display_latex(local_geometry_latex),
        "",
        _display_latex(selected_latex),
        *_formula_lines([angular["Jx"], angular["Jy"], angular["v"]]),
        "系数：",
        *_formula_lines(angular["coefficients"]),
        "辅助量：",
        *_formula_lines(angular["auxiliary"]),
    ]
    return lines


def _render_low_loop_step_four(chart: Mapping[str, Any]) -> list[str]:
    required = (
        "C", "constraint_equations", "solved_columns",
        "internal_variables", "elimination", "C_latex", "C_E_latex",
    )
    missing = [key for key in required if key not in chart]
    if missing:
        raise ValueError(f"low-loop chart lacks actual step-4 data fields {missing}")
    equations = chart["constraint_equations"]
    if not isinstance(equations, (tuple, list)):
        raise ValueError("constraint_equations must preserve each actual row")
    if any(not row.get("equation_latex") or not row.get("u0_latex") for row in equations):
        raise ValueError("every actual constraint row must provide equation_latex and u0_latex")
    determinant = chart.get("det_C_E_scale", {})
    shape_names = ("x", "y", "v")
    retained_shape_columns = tuple(
        column for column in range(3) if shape_names[column] not in chart["solved_columns"]
    )
    eliminated_count = len(chart["solved_columns"])
    eliminated = bool(chart["elimination"].get("phi_eliminated"))
    root_latex = chart["elimination"].get("phi_roots_latex")
    if not root_latex:
        raise ValueError("low-loop elimination lacks the actual phi-root or retained-phi LaTeX")
    lines = [
        "## 4. 实际角约束、解析消元与全部方位角根",
        "",
        "### 由剩余原图 IKV 得到的全部 $C$ 约束",
        "",
    ]
    if equations:
        numbered_equations = [
            rf"{_math_identifier(row['U_label'])}-u_{{0,{index}}}={row['C_equation_latex']}"
            for index, row in enumerate(equations, start=1)
        ]
        lines.extend([
            _display_latex(chart["C_latex"]),
            *_formula_lines(numbered_equations),
            *[
                f"- {_inline_math(rf'u_{{0,{index}}}=' + str(row['u0_latex']))}。"
                for index, row in enumerate(equations, start=1)
            ],
        ])
    else:
        lines.extend([
            r"- 本类没有剩余内部 $C$ 行（$d=2$）；$x,y,\varphi$ 三个内部坐标均显式保留。",
            f"- $C\\in\\mathbb{{R}}^{{0\\times3}}$，空子式行列式按 $1$ 处理。",
        ])
    lines.extend([
        "",
        "### 实际消元列、保留角与 Jacobian",
        "",
        *([_display_latex(chart["C_E_latex"])] if equations else []),
        f"- 消元列：{_inline_math(_variable_list_latex(chart['solved_columns']))}；保留内部坐标：{_inline_math(_variable_list_latex(chart['internal_variables']))}。",
        f"- 基底行列式：{_inline_math(r'\det(C_E)_{\rm base}=' + _fraction_latex(Fraction(str(chart.get('det_C_E_base', '1')))))}。",
        f"- 尺度化行列式：{_inline_math(_det_c_e_latex(chart))}。",
        "- 选列先依据精确结构有理矩阵；实际约束 determinant 在该结构 determinant 上乘已列出的 $j_{x1},j_{y1}$ 次幂。",
        "",
        "### 逐行求出被消去变量",
        "",
    ])
    if eliminated_count == 0:
        lines.append("- 没有内部约束行；$x,y,v$ 对应坐标保留，不做线性消元。")
    else:
        retained_labels = tuple(shape_names[column] for column in retained_shape_columns)
        u_labels = tuple(_math_identifier(row["U_label"]) for row in equations)
        lines.extend([
            _display_latex(r"\mathbf r=\mathbf U-\mathbf u_0-C_R\mathbf z_R,\qquad{}C_E\mathbf z_E=\mathbf r"),
            f"- $\\mathbf U={_variable_list_latex(u_labels)}^{{\\mathsf T}}$；$\\mathbf u_0=({',\\;'.join(rf'u_{{0,{i}}}' for i in range(1, len(equations) + 1))})^{{\\mathsf T}}$。",
            f"- 保留的 $z_R={_variable_list_latex(retained_labels)}$；对应的 $C_R$ 系数矩阵：",
            _display_latex(_constraint_matrix_from_columns(chart, retained_shape_columns)),
        ])
        if eliminated_count == 1:
            lines.append(
                f"- 单列方程的实际解：${_math_identifier(chart['solved_columns'][0])}=r_1/c_{{11}}$；"
                "$c_{11}$ 是上方实际 $C_E$ 的唯一元素。"
            )
        elif eliminated_count == 2:
            lines.append(
                f"- 二列方程的实际解（$z_E={_variable_list_latex(chart['solved_columns'])}^{{\\mathsf T}}$）："
                r"$z_{E,1}=\frac{r_1c_{22}-c_{12}r_2}{\det C_E},\quad{}"
                r"z_{E,2}=\frac{c_{11}r_2-r_1c_{21}}{\det C_E}$。"
            )
    lines.extend([
        "",
        "### 方位角根或保留角输入",
        "",
        _display_latex(root_latex),
    ])
    if not eliminated:
        retained_phi = chart["elimination"].get("retained_phi_input_latex")
        if not retained_phi:
            raise ValueError("retained internal_phi coordinate lacks its LaTeX input domain")
        lines.append(_display_latex(retained_phi))
    lines.extend([
        "",
    ])
    return lines


def _render_low_loop_step_five(chart: Mapping[str, Any]) -> list[str]:
    required = ("domain_latex", "momenta", "gram_formula_latex")
    missing = [key for key in required if key not in chart]
    if missing:
        raise ValueError(f"low-loop chart lacks actual step-5 data fields {missing}")
    momentum_equations = chart["momenta"].get("equations_latex")
    if not momentum_equations:
        raise ValueError("low-loop momenta lack explicit reconstruction LaTeX")
    geometry = chart.get("geometry")
    if not isinstance(geometry, Mapping):
        raise ValueError("low-loop chart lacks the selected subsystem geometry")
    elimination = chart.get("elimination")
    if not isinstance(elimination, Mapping) or "phi_eliminated" not in elimination:
        raise ValueError("low-loop chart must state whether internal phi was eliminated")
    mode = geometry.get("mode")
    if mode not in {"pair", "nested"}:
        raise ValueError("low-loop chart mode must be pair or nested")
    if mode == "pair":
        threshold_latex = (
            r"H>m_a+m_b,\quad{}T>m_c+m_d,\quad{}M>H+T,\qquad{}"
            r"\lambda(h,m_a^2,m_b^2)>0,\quad{}\lambda(t,m_c^2,m_d^2)>0,\quad{}"
            r"\lambda(M^2,h,t)>0"
        )
        native_frame = [
            r"- pair 模式中的构造动量已在总质心系 $P$；两条子系统轴分别为 $q_{ab}\parallel+\hat z$ 和 $q_{cd}\parallel-\hat z$。",
        ]
    else:
        threshold_latex = (
            r"H>m_a+m_b,\quad{}T>H+m_c,\quad{}M>T+m_d,\qquad{}"
            r"\lambda(h,m_a^2,m_b^2)>0,\quad{}\lambda(t,h,m_c^2)>0,\quad{}"
            r"\lambda(M^2,t,m_d^2)>0"
        )
        native_frame = [
            _display_latex(
                r"\boldsymbol\beta=-\frac{\mathbf p_d^{(abc)}}{T+E_d},\quad{}"
                r"p_i^{(P)}=\operatorname{Boost}_{\boldsymbol\beta}\!\left(p_i^{(abc)}\right)\quad(i=a,b,c,d)"
            ),
            _display_latex(
                r"E'=\gamma(E+\boldsymbol\beta\!\cdot\mathbf p),\quad{}"
                r"\mathbf p'=\mathbf p+\left[\frac{(\gamma-1)(\boldsymbol\beta\!\cdot\mathbf p)}{|\boldsymbol\beta|^2}+\gamma E\right]\boldsymbol\beta,\quad{}"
                r"\gamma=(1-|\boldsymbol\beta|^2)^{-1/2},\quad{}"
                r"\boldsymbol\beta=0\Rightarrow\operatorname{Boost}_{\boldsymbol\beta}=I"
            ),
        ]
    phi_eliminated = bool(elimination["phi_eliminated"])
    internal_phi_indicator = (
        r"B_{(-1,1)}(c_\varphi)"
        if phi_eliminated else
        r"B_{(0,2\pi)}(\varphi)\,\mathbf 1_{\varphi\ne\pi}"
    )
    open_indicator_latex = (
        r"\chi_{\rm open}=\mathbf 1_{\mathcal D}"
        r"=\mathbf 1_{\mathcal D_{\rm rest}}"
        r"B_{(-1,1)}(x)B_{(-1,1)}(y)\,"
        + internal_phi_indicator
    )
    lines = [
        "## 5. 实际动量支、正能条件与开域",
        "",
        "### Gram 正定与坐标开域",
        "",
        *_formula_lines(chart["domain_latex"]),
        "",
        "### 所选子系统的严格阈值",
        "",
        _display_latex(
            r"\lambda(x,y,z)=x^2+y^2+z^2-2xy-2xz-2yz"
        ),
        _display_latex(threshold_latex),
        "",
        "### 已列开域条件对应的显式指示函数",
        "",
        "- $\\mathcal D$ 是本节列出的严格质量阈值、正能、Gram 正定、非零坐标系数与消元主元、Euler 角以及内部角条件的交集；$\\mathcal D_{\\rm rest}$ 的指标是其余已列条件指标的乘积。",
        _display_latex(r"\Theta_+(u)=\mathbf 1_{u>0},\quad{}B_{(a,b)}(u)=\Theta_+(u-a)\Theta_+(b-u)"),
        _display_latex(open_indicator_latex),
        "",
        "### 质心系动量重建、提升与整体转动",
        "",
        *_formula_lines(momentum_equations),
        *native_frame,
        "- 下面的正交架将 $p_1$ 定为 $+z$，并使 $p_2$ 落在 $xz$ 平面且 $p_{2x}>0$。",
        _display_latex(
            r"\mathbf e_z=\frac{\mathbf p_1^{(P)}}{|\mathbf p_1^{(P)}|},\quad{}"
            r"\mathbf e_x=\frac{\mathbf p_2^{(P)}-(\mathbf p_2^{(P)}\!\cdot\mathbf e_z)\mathbf e_z}"
            r"{|\mathbf p_2^{(P)}-(\mathbf p_2^{(P)}\!\cdot\mathbf e_z)\mathbf e_z|},\quad{}"
            r"\mathbf e_y=\mathbf e_z\times\mathbf e_x"
        ),
        _display_latex(
            r"\mathbf k_i^{\rm body}=(\mathbf p_i^{(P)}\!\cdot\mathbf e_x,"
            r"\mathbf p_i^{(P)}\!\cdot\mathbf e_y,\mathbf p_i^{(P)}\!\cdot\mathbf e_z),\quad{}"
            r"p_i^{\rm body}=(E_i,\mathbf k_i^{\rm body})"
        ),
        _display_latex(
            r"R_{zyz}=R_z(\phi)R_y(\theta)R_z(\psi),\quad{}"
            r"\begin{pmatrix}E_i\\\mathbf p_i^{\rm lab}\end{pmatrix}="
            r"\begin{pmatrix}1&0\\0&R_{zyz}\end{pmatrix}"
            r"\begin{pmatrix}E_i\\\mathbf k_i^{\rm body}\end{pmatrix}"
        ),
        _display_latex(
            r"\text{the shape-dependent body frame only reselects the }SO(3)\text{ fiber base point; }"
            r"dR\text{ is Haar invariant, so }J_{\rm rot}=1\text{ and no extra rotation Jacobian is included}"
        ),
        "",
        "### Gram 重建公式",
        "",
        *_formula_lines(chart["gram_formula_latex"]),
        "",
        "### 构造状态",
        "",
        f"- 精确仿射秩：`{chart.get('verification', {}).get('affine_rank', 'not supplied')}`；pivot：`{chart.get('verification', {}).get('constraint_pivot', 'not supplied')}`。",
        f"- 数值事件与独立公式核验：`{chart.get('verification', {}).get('numeric_event', 'pending')}` / `{chart.get('verification', {}).get('independent_formula_check', 'pending')}`。",
    ]
    return lines


def _render_low_loop_step_six(
    chart: Mapping[str, Any],
    formula: Mapping[str, Any],
    normalization: str,
) -> list[str]:
    density = chart.get("density")
    if density is None:
        raise ValueError("low-loop chart lacks the density and Gram pullback Jacobian")
    differential = formula.get("differential_measure_latex") or chart.get("differential_measure_latex")
    mode = str(formula.get("density_mode", "per_branch"))
    suffix = "summed" if mode == "summed" else "per_branch"
    keyed_expressions = chart.get("density_expression_latex", {})
    expression = formula.get("density_expression_latex") or keyed_expressions.get(f"{normalization}_{suffix}")
    if expression is None:
        nested = density.get(
            "expression_by_normalization_summed_latex" if mode == "summed" else "expression_by_normalization_latex",
            {},
        )
        expression = nested.get(normalization)
    if differential is None or expression is None:
        raise ValueError("low-loop formula lacks its final measure or density expression")
    density_equation = str(expression)
    if not density_equation.lstrip().startswith(r"\rho_{\rm low}="):
        density_equation = r"\rho_{\rm low}=" + density_equation
    jacobian = chart.get("jacobian", {})
    required_jacobian = ("rho_X_latex", "pullback_latex", "full_density_latex")
    missing = [key for key in required_jacobian if not jacobian.get(key)]
    if missing:
        raise ValueError(f"low-loop chart lacks LaTeX Jacobian fields {missing}")
    powers = jacobian.get("p_by_normalization")
    power = powers.get(normalization) if isinstance(powers, Mapping) else None
    if isinstance(power, bool) or not isinstance(power, int) or power <= 0:
        raise ValueError(f"low-loop chart lacks the 2pi power for normalization {normalization!r}")
    jacobian_latex = {
        key: str(jacobian[key]).replace(r"(2\pi)^p", rf"(2\pi)^{{{power}}}")
        for key in required_jacobian
    }
    w_definition = density.get("W_definition_latex")
    if not w_definition:
        raise ValueError("low-loop chart lacks the actual W Jacobian branch definition")
    return [
        "## 6. Gram 拉回后的原图单支密度与完整微分",
        "",
        "### 实际密度系数与坐标 Jacobian",
        "",
        f"- 归一化 `{normalization}` 对应 $p={power}$。",
        *[_display_latex(jacobian_latex[key]) for key in required_jacobian],
        _display_latex(w_definition),
        "",
        "### 单支密度与微分测度",
        "",
        _display_latex(differential),
        _display_latex(density_equation),
        "",
        f"- direct formula status：`{formula.get('status', 'unknown')}`；来源状态仍为 `not_checked`。",
    ]


def _render_steps_three_to_six(evidence: Mapping[str, Any], loop_count: int) -> list[str]:
    chart = evidence.get("chart_data")
    if not isinstance(chart, Mapping):
        raise ValueError("formula evidence has no chart-data mapping")
    if loop_count == 3:
        return [
            *_render_step_three(dict(chart)),
            *_render_step_four(dict(chart)),
            *_render_step_five(dict(chart)),
            *_render_step_six(dict(chart)),
        ]
    formula = evidence.get("direct_formula")
    if not isinstance(formula, Mapping):
        raise ValueError("low-loop formula evidence has no direct-formula mapping")
    return [
        *_render_low_loop_step_three(chart),
        *_render_low_loop_step_four(chart),
        *_render_low_loop_step_five(chart),
        *_render_low_loop_step_six(chart, formula, str(evidence.get("normalization", "jing_psf"))),
    ]


def _render_markdown(marked_class: AllLoopMarkedClass, evidence: Mapping[str, Any]) -> str:
    trace = evidence.get("tree_covariant_trace")
    input_card = evidence.get("input_card")
    if not isinstance(trace, Mapping) or not isinstance(input_card, Mapping):
        raise ValueError("formula evidence must include the actual input card and tree trace")
    benchmark = evidence.get("benchmark", {})
    lines = [
        f"# {marked_class.class_id} 四体 L={marked_class.loop_count} 标记公式",
        "",
        "## 标记来源与验证状态",
        "",
        f"- 拓扑：`{marked_class.topology_id}`；loop：L={marked_class.loop_count}；d={2 + marked_class.loop_count}。",
        f"- 代表 fixed digest：`{marked_class.representative_fixed_digest}`；particle-relabeling digest：`{marked_class.particle_identity.digest}`。",
        f"- 类内 fixed 身份：{len(marked_class.members)} 条；fixed → representative 的逐项映射见 `.evidence.json`。",
        f"- 公式构建：`{evidence.get('formula_build_status', 'unknown')}`；树 trace：`{_field(trace, 'status', 'unknown')}`。",
        f"- benchmark：`{_field(benchmark, 'status', 'pending')}`；本次没有把 pending 升级为通过。",
        "- 论文来源对照状态：`not_checked`；生成档案不构成外部来源验证。",
        "",
        f"- 代表 fixed 记录的完整图、变量、局部因子与成员映射保存在 [{marked_class.class_id} evidence]({marked_class.class_id}_{marked_class.topology_id}.evidence.json)。",
        "",
        *_render_marker_graph(input_card),
        *_render_step_one(trace),
        *_render_step_two(trace),
        *_render_steps_three_to_six(evidence, marked_class.loop_count),
        "## 可追溯记录",
        "",
        "- 机器证据保留代表记录、所有 fixed 成员、粒子/顶点映射、输入卡、实际树消元、坐标数据与公式状态。",
        "- 记录粒子换名只描述类成员到代表的显式映射，不合并或丢弃 fixed 身份。",
        "",
    ]
    return "\n".join(lines)


def _normalize_loops(loops: int | Iterable[int] | None) -> tuple[int, ...] | None:
    if loops is None:
        return None
    if isinstance(loops, bool):
        raise TypeError("loops must be an integer or iterable of integer loop counts")
    if isinstance(loops, int):
        values = (loops,)
    else:
        if isinstance(loops, (str, bytes)):
            raise TypeError("loops must contain integer loop counts")
        try:
            values = tuple(loops)
        except TypeError as exc:
            raise TypeError("loops must be an integer or iterable of integer loop counts") from exc
    if not values:
        raise ValueError("loops must not be empty")
    if any(isinstance(value, bool) or not isinstance(value, int) or value not in (0, 1, 2, 3)
           for value in values):
        raise ValueError("loop counts must be integers from 0 through 3")
    if len(set(values)) != len(values):
        raise ValueError("loop counts must not contain duplicates")
    return tuple(sorted(values))


def _selection_mode(
    *,
    all_classes: bool,
    topology: str | None,
    class_selector: str | None,
    loops: tuple[int, ...] | None,
) -> str:
    if not isinstance(all_classes, bool):
        raise TypeError("all_classes must be bool")
    if all_classes and (topology is not None or class_selector is not None):
        raise ValueError("all_classes cannot be combined with a topology or class selector")
    if not all_classes and topology is None and class_selector is None and loops is None:
        raise ValueError("select all_classes, a topology, a class, or one or more loop counts")
    if all_classes:
        return "all"
    if class_selector is not None:
        return "class" if topology is None else "class_and_topology"
    if topology is not None:
        return "topology"
    return "loops"


def _index_entry(marked_class: AllLoopMarkedClass, evidence: Mapping[str, Any], stem: str) -> dict[str, Any]:
    return {
        "class_id": marked_class.class_id,
        "loop_count": marked_class.loop_count,
        "d": 2 + marked_class.loop_count,
        "topology_id": marked_class.topology_id,
        "topology_digest": marked_class.topology_digest,
        "representative_fixed_digest": marked_class.representative_fixed_digest,
        "particle_relabeling_digest": marked_class.particle_identity.digest,
        "fixed_member_count": len(marked_class.members),
        "markdown": f"{stem}.md",
        "evidence": f"{stem}.evidence.json",
        "formula_build_status": evidence.get("formula_build_status", "unknown"),
        "tree_trace_status": _field(evidence.get("tree_covariant_trace", {}), "status", "unknown"),
        "chart_status": _field(evidence.get("chart_data", {}), "status", "unknown"),
        "benchmark_status": _field(evidence.get("benchmark", {}), "status", "pending"),
        "source_status": "not_checked",
        "export_status": "exported",
    }


def _render_readme(
    *,
    catalog: AllLoopCatalog,
    selected: tuple[AllLoopMarkedClass, ...],
    entries: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    mode: str,
    loops: tuple[int, ...] | None,
    benchmark_requested: bool,
) -> str:
    source_counts = {
        loop: sum(item.loop_count == loop for item in catalog.classes)
        for loop in range(4)
    }
    exported_counts = {
        loop: sum(item.get("loop_count") == loop for item in entries)
        for loop in range(4)
    }
    failed_counts = {
        loop: sum(item.get("loop_count") == loop for item in failures)
        for loop in range(4)
    }
    benchmark_counts: dict[str, int] = {}
    for item in (*entries, *failures):
        status = str(item.get("benchmark_status", "pending"))
        benchmark_counts[status] = benchmark_counts.get(status, 0) + 1
    lines = [
        "# 四体全圈标记公式导出",
        "",
        f"- 筛选：`{mode}`；loop 范围：`{list(loops) if loops is not None else 'all'}`。",
        f"- catalog 中的类：{len(catalog.classes)}；本次选择：{len(selected)}；成功导出：{len(entries)}；构建失败：{len(failures)}。",
        f"- 来源固定身份：{catalog.fixed_record_count}；来源拓扑：{catalog.topology_count}；particle-relabeling 类：{len(catalog.classes)}。",
        "- 四体全圈范围按 L=0..3、d=2+L；来源由已保存的 G08 有效标记和三圈完整身份归档重建，不启动枚举。",
        f"- 所有来源对照均为 `not_checked`；benchmark requested：`{benchmark_requested}`。",
        f"- benchmark 状态计数：`{benchmark_counts}`。",
        "- benchmark 的每类原始结果保存在 `.evidence.json`；本次导出不会把 pending/unsupported 自动升级为通过。",
        "",
        "## 各圈覆盖",
        "",
        "| L | catalog 类 | selected | exported | failed |",
        "|---:|---:|---:|---:|---:|",
    ]
    for loop in range(4):
        selected_count = sum(item.loop_count == loop for item in selected)
        lines.append(
            f"| {loop} | {source_counts[loop]} | {selected_count} | "
            f"{exported_counts[loop]} | {failed_counts[loop]} |"
        )
    lines.extend(["", "## 公式档案", ""])
    for entry in entries:
        lines.append(
            f"- [L{entry['loop_count']} {entry['class_id']} · {entry['topology_id']}]({entry['markdown']}) — "
            f"代表 `{entry['representative_fixed_digest']}`；benchmark `{entry['benchmark_status']}`。"
        )
    if failures:
        lines.extend(["", "## 构建失败", ""])
        for failure in failures:
            lines.append(
                f"- L{failure['loop_count']} {failure['class_id']} · {failure['topology_id']}: "
                f"{failure['error_type']} — {failure['error']} ([failure JSON]({failure['failure_json']}))"
            )
    lines.extend([
        "",
        "机器索引：`index.json`；覆盖统计：`coverage.json`；类身份和映射：`all_loop_identity_catalog.json`。",
        "",
    ])
    return "\n".join(lines)


def default_all_loop_output_dir() -> Path:
    demo_root = Path(__file__).resolve().parents[3]
    return demo_root / ALL_LOOP_EXPORT_RELATIVE_PATH


def export_all_four_body_formulas(
    *,
    all_classes: bool = False,
    topology: str | None = None,
    class_selector: str | None = None,
    loops: int | Iterable[int] | None = None,
    output_dir: str | Path | None = None,
    archive_paths: Mapping[int, str | Path] | None = None,
    normalization: str = "jing_psf",
    run_benchmark: bool = False,
    benchmark_masses: Sequence[float] | None = None,
    benchmark_total_mass: float | None = None,
) -> dict[str, Any]:
    """Export selected classes as six-step Markdown and JSON without overwriting files."""

    selected_loops = _normalize_loops(loops)
    mode = _selection_mode(
        all_classes=all_classes,
        topology=topology,
        class_selector=class_selector,
        loops=selected_loops,
    )
    if normalization not in {"jing_psf", "standard_2pi4"}:
        raise ValueError("normalization must be 'jing_psf' or 'standard_2pi4'")
    if not isinstance(run_benchmark, bool):
        raise TypeError("run_benchmark must be bool")
    if not run_benchmark and (benchmark_masses is not None or benchmark_total_mass is not None):
        raise ValueError("benchmark masses require run_benchmark=True")
    if run_benchmark and (benchmark_masses is None or benchmark_total_mass is None):
        raise ValueError("run_benchmark=True requires benchmark_masses and benchmark_total_mass")
    if benchmark_masses is not None:
        try:
            benchmark_masses = tuple(float(value) for value in benchmark_masses)
            benchmark_total_mass = float(benchmark_total_mass)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("benchmark masses and total mass must be finite real numbers") from exc
        if len(benchmark_masses) != 4:
            raise ValueError("benchmark_masses must contain exactly four values")
        if any(not math.isfinite(value) for value in benchmark_masses) or not math.isfinite(benchmark_total_mass):
            raise ValueError("benchmark masses and total mass must be finite")
    catalog: AllLoopCatalog = build_all_loop_catalog(
        loops=selected_loops,
        archive_paths=archive_paths,
    )
    selected = select_all_loop_classes(
        catalog,
        loops=selected_loops,
        topology=topology,
        class_selector=class_selector,
    )
    root = Path(output_dir) if output_dir is not None else default_all_loop_output_dir()
    planned_names = ["README.md", "all_loop_identity_catalog.json", "index.json", "coverage.json"]
    for marked_class in selected:
        stem = f"{marked_class.class_id}_{marked_class.topology_id}"
        planned_names.extend([
            f"{stem}.md", f"{stem}.evidence.json", f"{stem}.failure.json",
        ])
    root.mkdir(parents=True, exist_ok=True)
    collisions = sorted(root / name for name in planned_names if (root / name).exists())
    if collisions:
        listing = ", ".join(str(path) for path in collisions[:8])
        if len(collisions) > 8:
            listing += f", … (+{len(collisions) - 8} more)"
        raise FileExistsError(f"refusing to overwrite existing all-loop export files: {listing}")

    rendered: list[tuple[AllLoopMarkedClass, dict[str, Any], str, str]] = []
    failures: list[dict[str, Any]] = []
    partial_evidence: dict[str, dict[str, Any]] = {}
    for marked_class in selected:
        stem = f"{marked_class.class_id}_{marked_class.topology_id}"
        evidence: dict[str, Any] | None = None
        try:
            evidence = build_all_loop_formula_evidence(
                marked_class,
                normalization=normalization,
                benchmark_masses=benchmark_masses if run_benchmark else None,
                benchmark_total_mass=benchmark_total_mass if run_benchmark else None,
            )
            markdown = _render_markdown(marked_class, evidence)
            rendered.append((marked_class, evidence, markdown, stem))
        except Exception as exc:
            failure = {
                "class_id": marked_class.class_id,
                "loop_count": marked_class.loop_count,
                "topology_id": marked_class.topology_id,
                "representative_fixed_digest": marked_class.representative_fixed_digest,
                "particle_relabeling_digest": marked_class.particle_identity.digest,
                "source_status": "not_checked",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "failure_json": f"{stem}.failure.json",
                "evidence_json": f"{stem}.evidence.json" if evidence is not None else None,
                "formula_build_status": evidence.get("formula_build_status", "unknown") if evidence else "unknown",
                "tree_trace_status": _field(evidence.get("tree_covariant_trace", {}), "status", "unknown") if evidence else "unknown",
                "chart_status": _field(evidence.get("chart_data", {}), "status", "unknown") if evidence else "unknown",
                "benchmark_status": _field(evidence.get("benchmark", {}), "status", "pending") if evidence else "pending",
            }
            failures.append(failure)
            if evidence is not None:
                partial_evidence[marked_class.class_id] = evidence

    entries = [
        _index_entry(marked_class, evidence, stem)
        for marked_class, evidence, _, stem in rendered
    ]
    exported_by_topology = {
        topology_id: sum(entry["topology_id"] == topology_id for entry in entries)
        for topology_id in sorted({item.topology_id for item in catalog.classes})
    }
    failed_by_topology = {
        topology_id: sum(item["topology_id"] == topology_id for item in failures)
        for topology_id in sorted({item.topology_id for item in catalog.classes})
    }
    by_topology = {
        topology_id: {
            "catalog_classes": sum(item.topology_id == topology_id for item in catalog.classes),
            "selected_classes": sum(item.topology_id == topology_id for item in selected),
            "exported_classes": exported_by_topology[topology_id],
            "failed_classes": failed_by_topology[topology_id],
        }
        for topology_id in sorted({item.topology_id for item in catalog.classes})
    }
    by_loop = {
        str(loop): {
            "catalog_classes": sum(item.loop_count == loop for item in catalog.classes),
            "selected_classes": sum(item.loop_count == loop for item in selected),
            "exported_classes": sum(item.get("loop_count") == loop for item in entries),
            "failed_classes": sum(item.get("loop_count") == loop for item in failures),
        }
        for loop in range(4)
    }
    benchmark_statuses = [
        str(item.get("benchmark_status", "pending")) for item in entries
    ] + [str(_field(item, "benchmark_status", "pending")) for item in failures]
    benchmark_status_counts: dict[str, int] = {}
    for status in benchmark_statuses:
        benchmark_status_counts[status] = benchmark_status_counts.get(status, 0) + 1
    if not run_benchmark:
        aggregate_benchmark_status = "pending"
    elif len(benchmark_status_counts) == 1:
        aggregate_benchmark_status = next(iter(benchmark_status_counts))
    elif benchmark_status_counts:
        aggregate_benchmark_status = "mixed"
    else:
        aggregate_benchmark_status = "pending"
    coverage = {
        "artifact_type": "four_body_all_loop_export_coverage",
        "schema_version": ALL_LOOP_EXPORT_SCHEMA_VERSION,
        "source_fixed_identity_count": catalog.fixed_record_count,
        "source_topology_count": catalog.topology_count,
        "source_particle_relabeling_class_count": len(catalog.classes),
        "selected_mode": mode,
        "selected_loop_counts": list(selected_loops) if selected_loops is not None else [0, 1, 2, 3],
        "selected_topology": topology,
        "selected_class_id": class_selector,
        "selected_class_count": len(selected),
        "exported_class_count": len(rendered),
        "failed_class_count": len(failures),
        "benchmark_requested": run_benchmark,
        "benchmark_status": aggregate_benchmark_status,
        "benchmark_status_counts": benchmark_status_counts,
        "source_status": "not_checked",
        "by_loop": by_loop,
        "by_topology": by_topology,
        "failures": failures,
    }
    catalog_payload = catalog.to_dict()
    index = {
        "artifact_type": "four_body_all_loop_formula_export_index",
        "schema_version": ALL_LOOP_EXPORT_SCHEMA_VERSION,
        "output_directory": str(root.resolve()),
        "sources": catalog_payload.get("sources", []),
        "source_status": "not_checked",
        "benchmark_requested": run_benchmark,
        "benchmark_status": aggregate_benchmark_status,
        "built_artifacts_do_not_imply_verified": True,
        "selected_mode": mode,
        "selected_loop_counts": coverage["selected_loop_counts"],
        "classes": entries,
        "failures": failures,
    }

    outputs: dict[Path, str] = {
        root / "README.md": _render_readme(
            catalog=catalog,
            selected=selected,
            entries=entries,
            failures=failures,
            mode=mode,
            loops=selected_loops,
            benchmark_requested=run_benchmark,
        ),
        root / "all_loop_identity_catalog.json": _canonical_json(catalog_payload, indent=2) + "\n",
        root / "index.json": _canonical_json(index, indent=2) + "\n",
        root / "coverage.json": _canonical_json(coverage, indent=2) + "\n",
    }
    for marked_class, evidence, markdown, stem in rendered:
        outputs[root / f"{stem}.md"] = markdown
        outputs[root / f"{stem}.evidence.json"] = _canonical_json(evidence, indent=2) + "\n"
    for failure in failures:
        evidence = partial_evidence.get(failure["class_id"])
        if evidence is not None:
            outputs[root / failure["evidence_json"]] = _canonical_json(evidence, indent=2) + "\n"
        outputs[root / failure["failure_json"]] = _canonical_json({
            "artifact_type": "four_body_all_loop_formula_export_failure",
            "schema_version": ALL_LOOP_EXPORT_SCHEMA_VERSION,
            **failure,
        }, indent=2) + "\n"
    for path, contents in outputs.items():
        with path.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(contents)

    return {
        "output_directory": str(root.resolve()),
        "selected_class_count": len(selected),
        "markdown_count": len(rendered),
        "failed_class_count": len(failures),
        "coverage": coverage,
        "index": index,
    }


__all__ = [
    "ALL_LOOP_EXPORT_RELATIVE_PATH",
    "ALL_LOOP_EXPORT_SCHEMA_VERSION",
    "build_all_loop_formula_evidence",
    "default_all_loop_output_dir",
    "export_all_four_body_formulas",
]
