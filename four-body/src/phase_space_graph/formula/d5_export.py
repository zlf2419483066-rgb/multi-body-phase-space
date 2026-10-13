"""Six-step Markdown and machine-evidence export for archived D5 classes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..formula_inputs import FormulaInputCard
from ..physical_labels import PhysicalLabelRecord
from .d5_catalog import D5IdentityCatalog, D5MarkedClass, build_d5_catalog
from .generate import generate_formula
from .inputs import prepare_formula_input


D5_EXPORT_RELATIVE_PATH = Path(
    "文档/验证与测试/误差分析/三圈106类公式导出_2026-10-08/run01"
)
D5_EXPORT_SCHEMA_VERSION = "D5-six-step-export-v1"


def _jsonable(value: Any) -> Any:
    if hasattr(value, "to_dict"):
        return _jsonable(value.to_dict())
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return value


def _canonical_json(value: Any, *, indent: int | None = None) -> str:
    return json.dumps(
        _jsonable(value), ensure_ascii=False, sort_keys=True,
        separators=None if indent is not None else (",", ":"),
        indent=indent, allow_nan=False,
    )


def _field(data: Any, key: str, default: Any = None) -> Any:
    return data.get(key, default) if isinstance(data, dict) else default


def _formula_field(data: Any, key: str, default: Any = None) -> Any:
    """Prefer an explicit LaTeX field; retain raw ASCII in machine evidence."""

    if isinstance(data, dict):
        latex = data.get(f"{key}_latex")
        if latex is not None:
            return latex
        return data.get(key, default)
    return default


def _latex(value: Any) -> str:
    """Wrap a short expression as inline math, protecting Markdown table pipes."""

    if value is None or value == "":
        return "未提供"
    if isinstance(value, (tuple, list)):
        if not value:
            return r"$\varnothing$"
        value = r"\; , \;".join(str(item) for item in value)
    elif isinstance(value, dict):
        raise TypeError("machine-evidence dictionaries must not be rendered as formula text")
    safe_value = str(value).replace("|", r"\vert ")
    return f"${safe_value}$"


def _display_latex(value: Any) -> str:
    if value is None or value == "":
        return "未提供"
    if isinstance(value, (tuple, list)):
        if not value:
            raise ValueError("an empty formula sequence cannot be displayed as LaTeX")
        value = r"\\[4pt]".join(str(item) for item in value)
    elif isinstance(value, dict):
        raise TypeError("machine-evidence dictionaries must not be rendered as formula text")
    value = str(value).replace("|", r"\vert ")
    return f"\n\n$$\n{value}\n$$\n\n"


def _inline_math(value: Any) -> str:
    if value is None or value == "":
        return "未提供"
    if isinstance(value, (tuple, list)):
        if not value:
            return r"$\varnothing$"
        value = r"\; , \;".join(str(item) for item in value)
    elif isinstance(value, dict):
        raise TypeError("machine-evidence dictionaries must not be rendered as formula text")
    safe_value = str(value).replace("|", r"\vert ")
    return f"${safe_value}$"


def _formula_blocks(data: Any, key: str, *, default: Any = None) -> list[str]:
    value = _formula_field(data, key, default)
    if value is None or value == "" or value == [] or value == ():
        raise ValueError(f"required LaTeX formula {key!r} is missing")
    if isinstance(value, dict):
        if not value or any(not isinstance(expression, str) or not expression for expression in value.values()):
            raise ValueError(f"LaTeX formula mapping {key!r} contains a missing expression")
        return [f"- `{label}`：{_display_latex(expression)}" for label, expression in value.items()]
    if isinstance(value, (tuple, list)):
        if any(not isinstance(expression, str) or not expression for expression in value):
            raise ValueError(f"LaTeX formula sequence {key!r} contains a missing expression")
        return [_display_latex(expression) for expression in value]
    if not isinstance(value, str):
        raise TypeError(f"formula field {key!r} must contain LaTeX text")
    return [_display_latex(value)]


def _items(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _render_marker_graph(card: FormulaInputCard) -> list[str]:
    record = card.record if isinstance(card.record, dict) else {}
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
        "- 图编码直接取自本条 fixed 代表；边索引与出现序号沿用输入卡。",
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
        f"- 有序 IKV：`{', '.join(card.variable_labels)}`。",
        f"- 输入测度约定：`{card.measure_convention}`；公式来源状态：`not_checked`。",
        "",
    ])
    return lines


def _render_step_one(trace: dict[str, Any]) -> list[str]:
    f0 = _field(trace, "f0", {})
    factors = _items(_field(f0, "factors", []))
    double_factors = _items(_field(f0, "double_line_factors", []))
    double_latex = _items(_field(f0, "double_line_factors_latex", []))
    if len(factors) != 6 or len(double_factors) != 5 or len(double_latex) != 5:
        raise ValueError("three-loop four-body D5 trace must show six vertices and all five ds factors")
    if any(not _field(item, "expanded_expression_latex") for item in factors):
        raise ValueError("tree trace must provide the six expanded local vertex expressions in LaTeX")
    required_factor_fields = (
        "conservation_equation_latex", "incoming_momentum_latex",
        "outgoing_integration_momenta_latex", "outgoing_support_momenta_latex",
        "momentum_delta_latex", "normalization_2pi_power",
    )
    for factor in factors:
        missing = [key for key in required_factor_fields if _field(factor, key) is None]
        if missing:
            raise ValueError(f"vertex {_field(factor, 'vertex', '?')} lacks required trace fields: {missing}")
    lines = [
        "## 1. 原图表达式：六个二体块与五条双线测度",
        "",
        "每行对应本图一个顶点因子；保留 trace 中的顶点动量、出射测度与 delta 来源。",
        "",
    ]
    for factor in factors:
        vertex = _field(factor, "vertex", "?")
        lines.extend([
            f"### 顶点 $v_{{{vertex}}}$",
            "",
            _display_latex(_formula_field(factor, "expanded_expression", "未提供")),
            "",
            f"- 顶点方程：{_inline_math(_formula_field(factor, 'conservation_equation', '未提供'))}",
            f"- 输入动量：{_inline_math(_formula_field(factor, 'incoming_momentum', '未提供'))}",
            f"- 出射积分动量：{_inline_math(_formula_field(factor, 'outgoing_integration_momenta', []))}",
            f"- 出射壳支撑动量：{_inline_math(_formula_field(factor, 'outgoing_support_momenta', []))}",
            f"- 顶点守恒 delta：{_inline_math(_formula_field(factor, 'momentum_delta', '未提供'))}",
            f"- 局部 $2\\pi$ 次幂：`{_field(factor, 'normalization_2pi_power', '未提供')}`",
            "",
        ])
    lines.extend(["### 双线 $ds$ 因子", ""])
    if double_factors:
        lines.append("| edge / occurrence | 原图微分因子 | 来源 |")
        lines.append("|---|---|---|")
        for index, factor in enumerate(double_factors):
            formula = double_latex[index] if index < len(double_latex) else None
            lines.append(
                f"| e{_field(factor, 'edge_index', '?')}/o{_field(factor, 'occurrence', '?')} | "
                f"{_inline_math(formula) if formula else 'LaTeX trace missing'} | "
                f"`FormulaInputCard.double_measures` |"
            )
        if len(double_latex) != len(double_factors):
            raise ValueError("tree trace lacks the LaTeX expression for every double-line factor")
        if len(double_factors) != 5:
            lines.extend(["", f"> 输入卡当前记录 {len(double_factors)} 个双线出现，预期三圈四体图为 5 个。"])
    else:
        lines.append("trace 未给出可逐项读取的双线因子。")
    cancellation = _field(trace, "cancellation", {})
    cancelled = _items(_field(cancellation, "cancelled_measures", []))
    retained = _items(_field(cancellation, "retained_external_measures", []))
    lines.extend([
        "",
        "### 重复末态测度账目",
        "",
        f"- D0/LM 输入的取消策略：`{_field(cancellation, 'policy', '未提供')}`。",
        f"- 实际取消 {len(cancelled)} 个内部单线重复出现；逐项如下：",
        "",
        "| occurrence | 粒子 | 原图测度 |",
        "|---|---|---|",
    ])
    for row in cancelled:
        locator = _field(row, "locator", {})
        measure = _field(row, "measure", {})
        particle_ids = _field(measure, "particle_ids", [])
        location = (
            f"internal e{_field(locator, 'edge_index', '?')}/o{_field(locator, 'occurrence', '?')}"
        )
        measure_formula = r"\mathrm{d}\mu(p_{" + ",".join(str(value) for value in particle_ids) + "})"
        lines.append(
            f"| `{location}` | `{', '.join(f'p{value}' for value in particle_ids)}` | "
            f"{_inline_math(measure_formula)} |"
        )
    lines.extend([
        "",
        f"- D0 保留 {len(retained)} 个外线单粒子测度：",
        "",
        "| 粒子 | 外线出现 | 测度 |",
        "|---|---|---|",
    ])
    for row in retained:
        particle_ids = _field(row, "particle_ids", [])
        reference = _field(row, "source_reference", {})
        measure_formula = r"\mathrm{d}\mu(p_{" + ",".join(str(value) for value in particle_ids) + "})"
        lines.append(
            f"| `{', '.join(f'p{value}' for value in particle_ids)}` | "
            f"`v{_field(reference, 'vertex', '?')}/slot{_field(reference, 'slot', '?')}` | "
            f"{_inline_math(measure_formula)} |"
        )
    normalization = _field(trace, "normalization", {})
    normalization_formulas = (
        _field(normalization, "normalization_identity_latex"),
        _field(normalization, "per_tree_integration_balance_latex"),
        _field(normalization, "final_external_measure_factor_latex"),
    )
    if any(not item for item in normalization_formulas):
        raise ValueError("tree trace lacks its actual local 2π normalization balance in LaTeX")
    lines.extend([
        "",
        "### 原图局部归一化账",
        "",
        *[_display_latex(item) for item in normalization_formulas],
        "",
        "- 本步抵消的是重复测度副本；在壳约束由后续 delta 积分实施。",
        "",
    ])
    return lines


def _render_step_two(trace: dict[str, Any]) -> list[str]:
    elimination = _field(trace, "elimination", {})
    steps = _items(_field(elimination, "steps", []))
    cancellation = _field(trace, "cancellation", {})
    cancelled = _items(_field(cancellation, "cancelled_measures", []))
    retained = _items(_field(cancellation, "retained_external_measures", []))
    if len(steps) != 5:
        raise ValueError(f"three-loop D5 tree elimination must contain five q steps, got {len(steps)}")
    lines = [
        "## 2. 实际取消与协变 q 消元",
        "",
        "以下按 trace 实际执行顺序记录每个顶点 $\\delta^4$ 的当前约束、积分变量、代入式和绝对 Jacobian。",
        "",
        f"- D0 策略：`{_field(cancellation, 'policy', '?')}`；trace 实际取消 {len(cancelled)} 个内部单线重复测度并保留 {len(retained)} 个外线测度。",
    ]
    if cancelled:
        lines.append(
            "- 实际取消项：" + "; ".join(
                f"e{_field(_field(row, 'locator', {}), 'edge_index', '?')}/o{_field(_field(row, 'locator', {}), 'occurrence', '?')} "
                f"p{','.join(str(particle) for particle in _field(_field(row, 'measure', {}), 'particle_ids', []))}"
                for row in cancelled
            ) + "."
        )
    lines.append("")
    for index, step in enumerate(steps, start=1):
        source_latex = _field(step, "delta4_source_latex", None)
        current_equation = _field(step, "current_equation_latex", None)
        substitution = _field(step, "substitution_latex")
        jacobian_latex = _field(step, "abs_4d_jacobian_latex")
        if source_latex is None or current_equation is None or substitution is None or jacobian_latex is None:
            raise ValueError("tree trace is missing LaTeX for the actual delta source or current equation")
        lines.extend([
            f"### q 消元 {index}：顶点 `v{_field(step, 'leaf_vertex', '?')}` → `v{_field(step, 'parent_vertex', '?')}`",
            "",
            f"- 使用的原顶点 delta：{_inline_math(source_latex)}",
            f"- 先前消元代入后的当前约束：{_inline_math(current_equation)}",
            f"- 积分变量：{_inline_math(_field(step, 'integrated_q', '未提供'))}",
            f"- 根代入：{_inline_math(substitution)}",
            f"- 四维绝对 Jacobian：{_inline_math(jacobian_latex)}",
            "",
        ])
    root_equation = _field(elimination, "root_equation_latex")
    root_residual = _field(elimination, "root_residual_latex")
    remaining_q_variables = _field(elimination, "remaining_q_variables")
    if not root_equation or not root_residual or remaining_q_variables is None:
        raise ValueError("tree trace lacks the root equation, residual, or remaining-q list")
    lines.extend([
        "",
        f"- 根顶点：`v{_field(elimination, 'root_vertex', '?')}`；根方程：{_latex(root_equation)}",
        f"- 剩余方程：{_latex(root_residual)}",
        f"- 剩余 $q$ 变量：{_inline_math(remaining_q_variables)}",
        "",
        "### 壳层 delta 转移与保留 $ds$",
        "",
    ])
    transfers = _items(_field(trace, "q_shell_transfers", []))
    if len(transfers) != 5:
        raise ValueError(f"D5 tree trace must transfer five q-shell constraints, got {len(transfers)}")
    if transfers:
        for transfer in transfers:
            if not all(_field(transfer, key) for key in (
                "derived_solution_latex", "global_support_solution_latex", "shell_delta_latex",
                "positive_energy_transfer_latex", "ds_factor_in_final_expression_latex",
            )):
                raise ValueError("q-shell transfer lacks required LaTeX solution, support, or ds factor")
            lines.extend([
                f"#### {_inline_math(_field(transfer, 'integrated_q', '?'))}",
                "",
                f"- 图内树消元解：{_inline_math(_field(transfer, 'derived_solution_latex'))}",
                f"- 用根顶点整体守恒后的支撑解：{_inline_math(_field(transfer, 'global_support_solution_latex'))}",
                f"- 转移后的在壳 delta：{_inline_math(_field(transfer, 'shell_delta_latex'))}",
                f"- 正能量支撑：{_inline_math(_field(transfer, 'positive_energy_transfer_latex'))}",
                f"- 原图保留微分：{_inline_math(_field(transfer, 'ds_factor_in_final_expression_latex'))}",
                "",
            ])
    lines.extend([
        "",
        "- 消元后的整体守恒和五个 IKV delta：",
        "",
        *[_display_latex(item) for item in _items(_field(trace, "remaining_constraints_latex", []))],
        "",
        "- 此处 delta 是分布；不得将其当作普通数值相除。",
        "",
    ])
    if len(_items(_field(trace, "remaining_constraints_latex", []))) != 6:
        raise ValueError("tree trace must retain the global delta and five ordered IKV constraints in LaTeX")
    return lines


def _render_step_three(chart: dict[str, Any]) -> list[str]:
    required_fields = (
        "mass_offset", "L", "inverse_L", "det_L", "abs_det_L", "forward_equations",
        "inverse_equations", "energy_expressions", "completed_pair_invariants",
        "gram_matrix", "angle_variables", "angle_measure", "rotation",
    )
    missing = [key for key in required_fields if _formula_field(chart, key) is None]
    if missing:
        raise ValueError(f"D5 chart lacks step-3 LaTeX fields: {missing}")
    lines = [
        "## 3. 总 COM、角坐标与本图 IKV 映射",
        "",
        f"- chart profile/status：`{_field(chart, 'profile', '?')}` / `{_field(chart, 'status', '?')}`。",
        f"- 本图有序 IKV：`{', '.join(_field(chart, 'ordered_s', []))}`。",
        f"- 公共基底：`{', '.join(_field(chart, 'basis_variables', []))}`。",
        "",
        "共同基底别名与分量次序：",
        _display_latex(r"\mathbf{X}=(u,v,w,t,z)^{\mathsf T}=(s_{234},s_{13},s_{24},s_{12},s_{34})^{\mathsf T}"),
        "### 固定总 COM 能量的映射",
        "",
    ]
    lines.extend(_formula_blocks(chart, "mass_offset"))
    lines.extend(["", "### 线性映射矩阵", ""])
    lines.extend(_formula_blocks(chart, "L"))
    lines.extend(_formula_blocks(chart, "inverse_L"))
    lines.extend(_formula_blocks(chart, "det_L"))
    lines.extend(_formula_blocks(chart, "abs_det_L"))
    lines.extend(["", "### 正向 IKV 映射方程", ""])
    lines.extend(_formula_blocks(chart, "forward_equations"))
    lines.extend(["", "### 逆映射方程", ""])
    lines.extend(_formula_blocks(chart, "inverse_equations"))
    coordinate_map = _field(chart, "coordinate_map", {})
    delta_change = _field(coordinate_map, "constraint_delta_change_latex")
    if not delta_change:
        raise ValueError("D5 chart lacks the fixed-mass constraint delta change formula")
    lines.extend(["", "### 固定质量下的约束 delta 换元", "", _display_latex(delta_change)])
    lines.extend(["", "### 总 COM 能量及空间动量的 Gram 重建", ""])
    for title, key in (
        ("能量表达式", "energy_expressions"),
        ("完成的两体不变量", "completed_pair_invariants"),
        ("Gram 矩阵", "gram_matrix"),
        ("角变量与角测度", "angle_variables"),
        ("角测度恒等式", "angle_measure"),
    ):
        lines.extend([f"#### {title}", "", *_formula_blocks(chart, key), ""])
    root = _field(chart, "root", {})
    lines.extend(["#### 体轴动量", "", *_formula_blocks(root, "body_frame_momenta"), ""])
    lines.extend(["#### 旋转到 parent COM 系", "", *_formula_blocks(chart, "rotation"), ""])
    return lines


def _render_step_four(chart: dict[str, Any]) -> list[str]:
    scalar = _field(chart, "scalar_elimination", {})
    scalar_rows = _items(scalar)
    jacobian = _field(chart, "scalar_elimination_jacobian", {})
    lines = [
        "## 4. 根、delta 积分与 Jacobian",
        "",
        "原图 IKV 线性换元的绝对行列式：",
        _display_latex(_formula_field(chart, "abs_det_L", "未提供")),
        "",
        "### 六个标量约束、消元变量、根与根上导数",
        "",
    ]
    for index, row in enumerate(scalar_rows, start=1):
        constraint = _field(row, "constraint_latex")
        root_formula = _field(row, "root_latex")
        derivative = _field(row, "root_derivative_latex", _field(row, "absolute_derivative_latex"))
        if not (constraint and root_formula and derivative):
            raise ValueError(f"D5 scalar-elimination row {index} lacks its LaTeX constraint/root/derivative")
        lines.extend([
            f"#### 消去 `{_field(row, 'eliminated_variable', '?')}`",
            "",
            "约束：",
            _display_latex(constraint),
            "根：",
            _display_latex(root_formula),
            "根上绝对导数：",
            _display_latex(derivative),
        ])
    if len(scalar_rows) != 6:
        raise ValueError(f"D5 scalar elimination must contain six rows, got {len(scalar_rows)}")
    lines.extend([
        "",
        "六个标量 delta 的总 Jacobian：",
        _display_latex(_field(jacobian, "total_jacobian_latex", _field(jacobian, "total_latex"))),
        "约束组合变换：",
        _display_latex(_field(jacobian, "constraint_group_change_latex")),
        "约束组合行列式：",
        *[_display_latex(item) for item in _items(_field(jacobian, "constraint_group_determinants_latex"))],
        "每个镜像支的角换元 Jacobian：",
        _display_latex(_field(jacobian, "angle_change_per_mirror_branch_latex")),
        "",
        "### Gram 根及镜像动量",
        "",
    ])
    root = _field(chart, "root", {})
    mirror_constraint = _field(root, "mirror_root_constraint_latex")
    mirror_derivative = _field(root, "mirror_root_absolute_derivative_latex")
    mirror_root = _field(root, "mirror_root_latex")
    if not (mirror_constraint and mirror_derivative and mirror_root):
        raise ValueError("D5 chart lacks the Gram mirror-root constraint, root, or derivative")
    jacobian_latex_fields = (
        "total_jacobian_latex", "constraint_group_change_latex",
        "constraint_group_determinants_latex", "angle_change_per_mirror_branch_latex",
    )
    if any(not _field(jacobian, key) for key in jacobian_latex_fields):
        raise ValueError("D5 scalar-elimination Jacobian is missing a required LaTeX field")
    lines.extend([
        "Gram 约束：",
        _display_latex(mirror_constraint),
        "两个根：",
        _display_latex(mirror_root),
        "根上绝对导数：",
        _display_latex(mirror_derivative),
        "- 几何重建中的 $y$ 符号与上述两个根对应；不得再将同一镜像根计作额外 delta Jacobian。",
        "",
        "### 两个镜像支的动量重建",
        "",
    ])
    branches = _items(_field(chart, "branches", []))
    if not branches:
        raise ValueError("D5 chart must retain both mirror branches")
    if len(branches) != 2 or {item.get("sigma") for item in branches} != {-1, 1}:
        raise ValueError("D5 chart must retain exactly the sigma=±1 mirror branches")
    for branch in branches:
        reconstruction = _field(branch, "momentum_reconstruction_latex", None)
        if not reconstruction or not _field(branch, "root_latex"):
            raise ValueError("D5 mirror branch lacks its LaTeX root or momentum reconstruction")
        branch_label = _field(branch, "label")
        if branch_label is None:
            branch_label = f"sigma={_field(branch, 'sigma', '?')}"
        lines.extend([
            f"#### 支 `{branch_label}`",
            "",
            "根：",
            _display_latex(_field(branch, "root_latex")),
            *[_display_latex(item) for item in reconstruction],
            f"- 单支密度权重：`{_field(branch, 'density_weight', 'one branch')}`",
            "",
        ])
    return lines


def _render_step_five(chart: dict[str, Any]) -> list[str]:
    root_conditions = _field(chart, "root_conditions_latex")
    domain = _field(chart, "domain_latex")
    gram = _formula_field(chart, "gram_matrix")
    chi_open = _field(chart, "chi_open_definition_latex")
    if not root_conditions or not domain or not gram or not chi_open:
        raise ValueError("D5 chart lacks LaTeX for positive-energy conditions, domain, open support, or Gram matrix")
    formula_groups = {
        "positive-energy conditions": root_conditions,
        "chart domain": domain,
        "strict open-domain support": chi_open,
        "Gram matrix": gram,
    }
    for label, value in formula_groups.items():
        expressions = list(value.values()) if isinstance(value, dict) else _items(value)
        if not expressions or any(not isinstance(item, str) or not item for item in expressions):
            raise ValueError(f"D5 chart has an empty LaTeX expression in {label}")
    return [
        "## 5. 正能与 Gram 物理域",
        "",
        "### 正能与根条件",
        "",
        *[_display_latex(item) for item in _items(root_conditions)],
        "",
        "### 联合 chart 域",
        "",
        *[_display_latex(item) for item in _items(domain)],
        "",
        "### 严格开域支撑函数",
        "",
        *[_display_latex(item) for item in _items(chi_open)],
        "",
        "### Gram 矩阵与正定域",
        "",
        *(_display_latex(item) for item in _items(gram)),
        "",
    ]


def _render_step_six(chart: dict[str, Any]) -> list[str]:
    normalizations = _field(chart, "normalization_expressions_latex")
    derivation = _field(chart, "normalization_derivation_latex")
    density = _field(chart, "density_expression_latex")
    differential = _field(chart, "differential_measure_latex")
    if not isinstance(normalizations, dict) or not derivation or not density or not differential:
        raise ValueError("D5 chart lacks final density or both normalization derivations in LaTeX")
    if any(not isinstance(item, str) or not item for item in _items(derivation)):
        raise ValueError("D5 normalization derivation contains a missing LaTeX formula")
    lines = [
        "## 6. 原图变量上的单支密度",
        "",
        "微分变量采用本图有序五 IKV 与 $dR$；密度按两个镜像支分别给出。",
        "",
        _display_latex(differential),
        _display_latex(density),
        "",
        "### 从六个标量 delta 到单支密度",
        "",
        *[_display_latex(item) for item in _items(derivation)],
        "",
        "### 两种归一化约定与求和式",
        "",
    ]
    expected = {
        "jing_psf_per_branch", "standard_2pi4_per_branch",
        "jing_psf_summed", "standard_2pi4_summed",
    }
    if not expected.issubset(normalizations):
        raise ValueError(f"D5 normalization LaTeX fields are incomplete: {sorted(expected - set(normalizations))}")
    for key in sorted(expected):
        if not isinstance(normalizations[key], str) or not normalizations[key]:
            raise ValueError(f"D5 normalization LaTeX field {key!r} is empty")
        lines.extend([f"#### `{key}`", "", _display_latex(normalizations[key])])
    lines.extend([
        "",
        f"- 支数：{len(_items(_field(chart, 'branches', [])))}；各支标签见第 4 步与 evidence JSON。",
        "- 公式来源状态：`not_checked`；直接 chart 和 tree trace 的程序状态保持 `built`，本次导出不升级为 `verified`。",
        "",
    ])
    return lines


def _render_markdown(
    marked_class: D5MarkedClass,
    card: FormulaInputCard,
    trace: dict[str, Any],
    chart: dict[str, Any],
    formula_artifact: dict[str, Any],
) -> str:
    rep = marked_class.representative
    formula_statuses = _field(formula_artifact, "statuses", {})
    benchmark = _field(formula_artifact, "internal_benchmark")
    benchmark_status = _field(benchmark, "status", "pending")
    lines = [
        f"# {marked_class.class_id} 三圈四体标记公式",
        "",
        "## 标记来源与状态",
        "",
        f"- 拓扑：`{marked_class.topology_id}`；拓扑 digest：`{marked_class.topology_digest}`。",
        f"- 代表 fixed 标记：`{rep.program_id}`；fixed digest：`{rep.digest}`。",
        f"- particle_relabeling 类：`{marked_class.particle_identity.digest}`；类内 fixed 记录数：{len(marked_class.members)}。",
        f"- 代表策略：从历史 fixed 档案选择真实记录；本代表的 fixed key 已保留在独立 catalog JSON。",
        "- generate_formula statuses：`"
        + "; ".join(f"{key}={value}" for key, value in sorted(formula_statuses.items()))
        + "`。",
        f"- 局部数值 benchmark：`{benchmark_status}`；详细点位与残差见 `.evidence.json`。",
        "- `source_status = not_checked`；trace/chart 为 `built` 时保持原状态，不作为物理验证结论。",
        "",
        f"- 完整 {len(marked_class.members)} 条 fixed key/digest 与逐项映射见 [身份目录](d5_identity_catalog.json)；本类机器证书见 [{marked_class.class_id} evidence]({marked_class.class_id}_{marked_class.topology_id}.evidence.json)。",
        "",
        *_render_marker_graph(card),
        *_render_step_one(trace),
        *_render_step_two(trace),
        *_render_step_three(chart),
        *_render_step_four(chart),
        *_render_step_five(chart),
        *_render_step_six(chart),
        "## 可追溯记录",
        "",
        "- fixed key、fixed digest、逐项 vertex/particle/edge-occurrence 映射和原始 trace 保存在对应机器 JSON。",
        "- 映射方向为 fixed 来源记录 → 本类 fixed 代表；质量按 $m_{rep}[target]=m_{source}[source]$ 替换，$p$ 与 $s$ 下标使用同一粒子映射。回到来源公式时，对代表式作反向下标替换。",
        "",
    ]
    return "\n".join(lines)


def build_d5_formula_evidence(
    marked_class: D5MarkedClass,
    *,
    normalization: str = "jing_psf",
    benchmark_masses: tuple[float, float, float, float] | None = None,
    benchmark_total_mass: float | None = None,
) -> dict[str, Any]:
    """Build one class's generated formula, raw tree trace, and D5 chart data."""

    if normalization not in {"jing_psf", "standard_2pi4"}:
        raise ValueError("normalization must be 'jing_psf' or 'standard_2pi4'")
    if (benchmark_masses is None) != (benchmark_total_mass is None):
        raise ValueError("benchmark_masses and benchmark_total_mass must be provided together")
    record: PhysicalLabelRecord = marked_class.representative.canonical_record
    card = prepare_formula_input(record, measure_convention=normalization)
    artifact = generate_formula(
        card,
        run_benchmark=benchmark_masses is not None,
        masses=benchmark_masses,
        total_mass=benchmark_total_mass,
    )

    # Imports are deliberately local until the collaborating implementation
    # defines these D5-only interfaces.
    from .d5_direct import build_d5_chart_data
    from .tree_covariant import build_tree_covariant_trace

    trace_object = build_tree_covariant_trace(card)
    trace = trace_object.to_dict() if hasattr(trace_object, "to_dict") else trace_object
    if not isinstance(trace, dict):
        raise TypeError("build_tree_covariant_trace must return a dict or an object with to_dict()")
    chart = build_d5_chart_data(card)
    if not isinstance(chart, dict):
        raise TypeError("build_d5_chart_data must return a dict")

    evidence = {
        "artifact_type": "d5_three_loop_formula_evidence",
        "schema_version": D5_EXPORT_SCHEMA_VERSION,
        "class_id": marked_class.class_id,
        "topology_id": marked_class.topology_id,
        "representative": {
            "program_id": marked_class.representative.program_id,
            "fixed_digest": marked_class.representative.digest,
            "particle_relabeling_digest": marked_class.particle_identity.digest,
            "member_fixed_digests": [member.archived.digest for member in marked_class.members],
        },
        "source_status": "not_checked",
        "normalization": normalization,
        "formula_artifact": {
            "statuses": dict(artifact.statuses),
            "diagnostics": list(artifact.diagnostics),
            "input_digest": artifact.input_digest,
            "internal_benchmark": (
                artifact.internal_benchmark.to_dict() if artifact.internal_benchmark else None
            ),
        },
        "input_card": card.to_dict(),
        "tree_covariant_trace": trace,
        "d5_chart_data": chart,
    }
    return evidence


def _compact_index_entry(marked_class: D5MarkedClass, evidence: dict[str, Any], filename: str) -> dict[str, Any]:
    benchmark = _field(evidence["formula_artifact"], "internal_benchmark")
    formula_statuses = _field(evidence["formula_artifact"], "statuses", {})
    return {
        "class_id": marked_class.class_id,
        "topology_id": marked_class.topology_id,
        "topology_digest": marked_class.topology_digest,
        "representative_program_id": marked_class.representative.program_id,
        "representative_fixed_digest": marked_class.representative.digest,
        "particle_relabeling_digest": marked_class.particle_identity.digest,
        "fixed_member_count": len(marked_class.members),
        "markdown": filename,
        "evidence": filename.removesuffix(".md") + ".evidence.json",
        "source_status": evidence["source_status"],
        "formula_build_status": _field(formula_statuses, "direct", "unknown"),
        "formula_statuses": formula_statuses,
        "tree_trace_status": _field(evidence["tree_covariant_trace"], "status", "unknown"),
        "chart_status": _field(evidence["d5_chart_data"], "status", "unknown"),
        "benchmark_status": _field(benchmark, "status", "pending"),
        "benchmark_compared": _field(benchmark, "compared", 0),
        "benchmark_failures": len(_items(_field(benchmark, "failures", []))),
        "export_status": "exported",
    }


def _benchmark_status_counts(entries: list[dict[str, Any]], failures: list[dict[str, Any]]) -> dict[str, int]:
    statuses = [entry.get("benchmark_status", "pending") for entry in entries]
    statuses.extend(item.get("benchmark_status", "pending") for item in failures)
    return {
        "pass": sum(status == "verified_local" for status in statuses),
        "fail": sum(status == "failed" for status in statuses),
        "pending": sum(status not in {"verified_local", "failed"} for status in statuses),
    }


def _render_export_readme(
    *,
    catalog: D5IdentityCatalog,
    selected_mode: str,
    selected_selector: str | None,
    entries: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    coverage: dict[str, Any],
) -> str:
    benchmark_counts = _benchmark_status_counts(entries, failures)
    validation_rows = entries + failures
    validation_counts = {
        "direct_built": sum(entry.get("formula_build_status") == "built" for entry in validation_rows),
        "tree_trace_built": sum(entry.get("tree_trace_status") == "built" for entry in validation_rows),
        "chart_built": sum(entry.get("chart_status") == "built" for entry in validation_rows),
        "source_not_checked": coverage["selected_class_count"],
    }
    lines = [
        "# 三圈四体 D5 标记公式导出",
        "",
        f"- 筛选：`{selected_mode}`" + (f" / `{selected_selector}`" if selected_selector else ""),
        f"- 固定标记来源：{catalog.fixed_record_count}；拓扑：{catalog.topology_count}；particle_relabeling 类：{len(catalog.classes)}。",
        f"- 本次选择：{coverage['selected_class_count']}；已导出 Markdown：{coverage['exported_class_count']}；构建失败：{coverage['failed_class_count']}。",
        f"- 全量来源共 106 个标记类；按每类两条镜像动量支计为 212 条支记录，这不代表 212 个互不等价公式或物理贡献。",
        f"- 构建状态：direct built {validation_counts['direct_built']}，tree trace built {validation_counts['tree_trace_built']}，chart built {validation_counts['chart_built']}；source not_checked {validation_counts['source_not_checked']}。",
        f"- benchmark：pass {benchmark_counts['pass']}，fail {benchmark_counts['fail']}，pending {benchmark_counts['pending']}。`verified_local` 只表示有限数值样本检查。",
        "- 来源状态保持 `not_checked`；公式及来源不会因导出或本地 benchmark 自动变为 `verified`。",
        "",
        "## 公式档案",
        "",
    ]
    for entry in entries:
        lines.append(
            f"- [{entry['class_id']} · {entry['topology_id']}]({entry['markdown']}) — "
            f"代表 `{entry['representative_program_id']}`，类内 fixed 记录 {entry['fixed_member_count']} 条，"
            f"benchmark `{entry['benchmark_status']}`。"
        )
    if failures:
        lines.extend(["", "## 构建失败", ""])
        for failure in failures:
            evidence_link = (
                f"；已有部分机器证据 [{failure['evidence_json']}]({failure['evidence_json']})"
                if failure.get("evidence_json") else ""
            )
            lines.append(
                f"- {failure['class_id']} · {failure['topology_id']}：{failure['error_type']} — "
                f"{failure['error']} ([failure JSON]({failure['failure_json']})){evidence_link}"
            )
    lines.extend([
        "",
        "机器索引：`index.json`；覆盖统计：`coverage.json`；固定身份与映射：`d5_identity_catalog.json`。",
        "",
    ])
    return "\n".join(lines)


def default_d5_output_dir() -> Path:
    demo_root = Path(__file__).resolve().parents[3]
    return demo_root / D5_EXPORT_RELATIVE_PATH


def export_d5_formulas(
    *,
    all_classes: bool = False,
    topology: str | None = None,
    class_selector: str | None = None,
    output_dir: str | Path | None = None,
    archive_path: str | Path | None = None,
    normalization: str = "jing_psf",
    benchmark_masses: tuple[float, float, float, float] | None = None,
    benchmark_total_mass: float | None = None,
) -> dict[str, Any]:
    """Export all classes, one topology, or one class without replacing files."""

    selectors = int(all_classes) + int(topology is not None) + int(class_selector is not None)
    if selectors != 1:
        raise ValueError("select exactly one of all_classes=True, topology=<Txx/digest>, or class_selector=<D5/digest>")
    if (benchmark_masses is None) != (benchmark_total_mass is None):
        raise ValueError("benchmark_masses and benchmark_total_mass must be provided together")
    if benchmark_masses is not None and len(benchmark_masses) != 4:
        raise ValueError("benchmark_masses must contain exactly four values")
    catalog: D5IdentityCatalog = build_d5_catalog(archive_path)
    if all_classes:
        selected = catalog.classes
        selected_mode = "all"
        selected_selector = None
    elif topology is not None:
        selected = catalog.select_topology(topology)
        selected_mode = "topology"
        selected_selector = topology
    else:
        selected = catalog.select_class(class_selector or "")
        selected_mode = "class"
        selected_selector = class_selector
    root = Path(output_dir) if output_dir is not None else default_d5_output_dir()

    planned_names = ["README.md", "d5_identity_catalog.json", "index.json", "coverage.json"]
    for item in selected:
        stem = f"{item.class_id}_{item.topology_id}"
        planned_names.extend([f"{stem}.md", f"{stem}.evidence.json", f"{stem}.failure.json"])
    root.mkdir(parents=True, exist_ok=True)
    collisions = sorted(root / name for name in planned_names if (root / name).exists())
    if collisions:
        listing = ", ".join(str(path) for path in collisions[:8])
        if len(collisions) > 8:
            listing += f", … (+{len(collisions) - 8} more)"
        raise FileExistsError(f"refusing to overwrite existing export files: {listing}")

    rendered: list[tuple[D5MarkedClass, dict[str, Any], str, str]] = []
    failures: list[dict[str, Any]] = []
    failed_evidence: dict[str, dict[str, Any]] = {}
    for marked_class in selected:
        filename = f"{marked_class.class_id}_{marked_class.topology_id}.md"
        evidence: dict[str, Any] | None = None
        try:
            evidence = build_d5_formula_evidence(
                marked_class,
                normalization=normalization,
                benchmark_masses=benchmark_masses,
                benchmark_total_mass=benchmark_total_mass,
            )
            markdown = _render_markdown(
                marked_class,
                prepare_formula_input(marked_class.representative.canonical_record,
                                      measure_convention=normalization),
                evidence["tree_covariant_trace"],
                evidence["d5_chart_data"],
                evidence["formula_artifact"],
            )
            rendered.append((marked_class, evidence, markdown, filename))
        except Exception as exc:
            failure = {
                "class_id": marked_class.class_id,
                "topology_id": marked_class.topology_id,
                "representative_program_id": marked_class.representative.program_id,
                "representative_fixed_digest": marked_class.representative.digest,
                "particle_relabeling_digest": marked_class.particle_identity.digest,
                "source_status": "not_checked",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "failure_json": filename.removesuffix(".md") + ".failure.json",
                "benchmark_status": (
                    _field(_field(evidence.get("formula_artifact", {}), "internal_benchmark"), "status", "pending")
                    if evidence is not None else "pending"
                ),
                "formula_build_status": (
                    _field(_field(evidence.get("formula_artifact", {}), "statuses", {}), "direct", "unknown")
                    if evidence is not None else "unknown"
                ),
                "tree_trace_status": (
                    _field(evidence.get("tree_covariant_trace", {}), "status", "unknown")
                    if evidence is not None else "unknown"
                ),
                "chart_status": (
                    _field(evidence.get("d5_chart_data", {}), "status", "unknown")
                    if evidence is not None else "unknown"
                ),
                "evidence_json": (
                    filename.removesuffix(".md") + ".evidence.json"
                    if evidence is not None else None
                ),
            }
            failures.append(failure)
            if evidence is not None:
                failed_evidence[marked_class.class_id] = evidence

    entries = [
        _compact_index_entry(marked_class, evidence, filename)
        for marked_class, evidence, _, filename in rendered
    ]
    exported_by_topology = {
        topology_id: sum(item["topology_id"] == topology_id for item in entries)
        for topology_id in sorted({item.topology_id for item in catalog.classes})
    }
    failed_by_topology = {
        topology_id: sum(item["topology_id"] == topology_id for item in failures)
        for topology_id in sorted({item.topology_id for item in catalog.classes})
    }
    topology_coverage = {
        topology_id: {
            "total_classes": sum(item.topology_id == topology_id for item in catalog.classes),
            "selected_classes": sum(item.topology_id == topology_id for item in selected),
            "exported_classes": exported_by_topology[topology_id],
            "failed_classes": failed_by_topology[topology_id],
            "selected_complete": (
                exported_by_topology[topology_id]
                == sum(item.topology_id == topology_id for item in selected)
                and failed_by_topology[topology_id] == 0
                if sum(item.topology_id == topology_id for item in selected) > 0
                else None
            ),
            "all_classes_exported": exported_by_topology[topology_id]
            == sum(item.topology_id == topology_id for item in catalog.classes),
        }
        for topology_id in sorted({item.topology_id for item in catalog.classes})
    }
    coverage = {
        "artifact_type": "d5_export_coverage",
        "schema_version": "D5-coverage-v1",
        "source_fixed_identity_count": catalog.fixed_record_count,
        "source_topology_count": catalog.topology_count,
        "source_particle_relabeling_class_count": len(catalog.classes),
        "selected_mode": selected_mode,
        "selected_topology": topology,
        "selected_class_id": class_selector,
        "selected_class_count": len(selected),
        "selected_markdown_count": len(rendered),
        "exported_class_count": len(rendered),
        "failed_class_count": len(failures),
        "failures": failures,
        "topologies": topology_coverage,
    }
    index = {
        "artifact_type": "d5_formula_export_index",
        "schema_version": "D5-index-v1",
        "output_directory": str(root.resolve()),
        "source_archive": str(catalog.archive_path),
        "source_archive_sha256": catalog.archive_sha256,
        "source_status": "not_checked",
        "built_artifacts_do_not_imply_verified": True,
        "classes": entries,
        "failures": failures,
    }

    outputs: dict[Path, str] = {
        root / "README.md": _render_export_readme(
            catalog=catalog,
            selected_mode=selected_mode,
            selected_selector=selected_selector,
            entries=entries,
            failures=failures,
            coverage=coverage,
        ),
        root / "d5_identity_catalog.json": _canonical_json(catalog.to_dict(), indent=2) + "\n",
        root / "index.json": _canonical_json(index, indent=2) + "\n",
        root / "coverage.json": _canonical_json(coverage, indent=2) + "\n",
    }
    for marked_class, evidence, markdown, filename in rendered:
        outputs[root / filename] = markdown
        outputs[root / (filename.removesuffix(".md") + ".evidence.json")] = (
            _canonical_json(evidence, indent=2) + "\n"
        )
    for failure in failures:
        filename = failure["failure_json"]
        evidence = failed_evidence.get(failure["class_id"])
        if evidence is not None:
            outputs[root / failure["evidence_json"]] = _canonical_json(evidence, indent=2) + "\n"
        outputs[root / filename] = _canonical_json({
            "artifact_type": "d5_formula_export_failure",
            "schema_version": "D5-failure-v1",
            **failure,
            "source_status": "not_checked",
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
    "D5_EXPORT_RELATIVE_PATH",
    "D5_EXPORT_SCHEMA_VERSION",
    "build_d5_formula_evidence",
    "default_d5_output_dir",
    "export_d5_formulas",
]
