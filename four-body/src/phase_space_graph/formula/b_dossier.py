"""Standalone B/F03 dossier assembly and Markdown rendering.

This presentation layer composes existing input, expansion, cancellation,
local-reduction, covariant-trace, and optional direct-sample records.  It does
not change formula-generation or verification state.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

from ..domain import Labeling, Topology
from ..formula_inputs import FormulaInputCard
from ..invariants import analyze_invariants
from ..labeling import MomentumCertificate
from ..physical_labels import PhysicalLabelRecord, review_physical_record
from .b_covariant import BFCovariantCertificate, build_b_covariant_certificate
from .b_direct import (
    BDirectSample,
    B_F03_ANGLE_DEFINITIONS,
    B_F03_DIFFERENTIAL_MEASURE,
    B_F03_DENSITY_EXPRESSION,
    B_F03_NORMALIZATION_IDENTITY,
    B_F03_STANDARD_DIFFERENTIAL_MEASURE,
    B_F03_STANDARD_DENSITY_EXPRESSION,
    B_F03_STANDARD_NORMALIZATION_IDENTITY,
    sample_b_f03,
)
from .b_helicity import B_HELICITY_ANGLE_DEFINITIONS, B_HELICITY_MEASURE_IDENTITY
from .cancellations import CancellationPlan, CancellationPlanOverride, OccurrenceLocator, build_cancellation_plan
from .expanded import ExpandedF0Expression, build_expanded_f0_expression
from .local_reduction import LocalReductionTrace, reduce_local_measures


B_F03_DOSSIER_SCHEMA_VERSION = "FM-BF03-dossier-v1"
_DIRECT_REQUIRED = frozenset({
    "total_mass", "masses", "s12", "s34", "s23",
    "theta0", "phi0", "theta1", "phi1", "phi4_star",
})
_DIRECT_ALLOWED = _DIRECT_REQUIRED | {"normalization"}


def _canonical_json(value: Any, *, indent: int | None = None) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=None if indent is not None else (",", ":"),
                      indent=indent, allow_nan=False)


def _record_from_card(card: FormulaInputCard) -> PhysicalLabelRecord:
    """Rebuild and recheck the typed record embedded in an already-ready card."""

    stored = card.record
    if not isinstance(stored, dict):
        raise ValueError("direct B/F03 sampling requires a complete record in the formula card")
    topology = Topology.from_dict(stored["topology"])
    labeling = Labeling.from_dict(stored["complete_labeling"])
    momentum_data = stored["momentum"]
    momentum = MomentumCertificate(
        tuple((row["edge_index"], row["occurrence"], tuple(row["coefficients"]))
              for row in momentum_data["double_momenta"]),
        tuple((row["vertex"], tuple(row["residual"]))
              for row in momentum_data["vertex_residuals"]),
        bool(momentum_data["verified"]),
    )
    independence = analyze_invariants(momentum.variables, topology.n)
    groups = []
    for group in stored["single_groups"]:
        occurrences = []
        for item in group["occurrences"]:
            if item.get("kind") == "external":
                occurrences.append(("external", item["vertex"], item["slot"]))
            elif item.get("kind") == "internal":
                occurrences.append(("internal", item["edge_index"], item["occurrence"]))
            else:
                raise ValueError("formula card contains an unknown single occurrence kind")
        groups.append((group["particle_id"], tuple(occurrences)))
    cancellations = tuple(
        (row["particle_id"], row["count"])
        for row in stored["measure_cancellations"]
    )
    record = PhysicalLabelRecord(
        topology=topology,
        complete_labeling=labeling,
        momentum=momentum,
        independence=independence,
        single_groups=tuple(groups),
        measure_cancellations=cancellations,
        schema_version=stored["schema_version"],
    )
    review_physical_record(record)
    if record.to_dict() != stored:
        raise ValueError("reconstructed direct-sample record disagrees with the formula card")
    return record


def _prepare_direct_sample(
    card: FormulaInputCard,
    chart_parameters: Mapping[str, Any] | None,
) -> tuple[dict[str, Any] | None, BDirectSample | None]:
    if chart_parameters is None:
        return None, None
    if not isinstance(chart_parameters, Mapping):
        raise TypeError("direct_chart_parameters must be a mapping")
    keys = set(chart_parameters)
    missing = _DIRECT_REQUIRED - keys
    extra = keys - _DIRECT_ALLOWED
    if missing or extra:
        raise ValueError(
            f"direct chart parameters missing {sorted(missing)!r} or contain unsupported keys {sorted(extra)!r}"
        )
    parameters = dict(chart_parameters)
    normalization = parameters.pop("normalization", card.measure_convention)
    if normalization != card.measure_convention:
        raise ValueError("direct sample normalization must match the formula card measure convention")
    record = _record_from_card(card)
    sample = sample_b_f03(record, **parameters, normalization=normalization)
    # Round-trip the supplied values once so the dossier only retains JSON-safe
    # values and rejects NaN/Infinity deterministically.
    encoded = _canonical_json({**parameters, "normalization": normalization})
    return json.loads(encoded), sample


@dataclass(frozen=True, slots=True)
class BF03Dossier:
    """Typed composition of source-bound B/F03 derivation artifacts."""

    input_card: FormulaInputCard
    expanded_f0: ExpandedF0Expression
    cancellation_plan: CancellationPlan
    local_reduction: LocalReductionTrace
    covariant_certificate: BFCovariantCertificate
    direct_chart_parameters: dict[str, Any] | None = None
    direct_sample: BDirectSample | None = None
    schema_version: str = B_F03_DOSSIER_SCHEMA_VERSION
    status: str = "assembled"

    def __post_init__(self) -> None:
        if self.schema_version != B_F03_DOSSIER_SCHEMA_VERSION or self.status != "assembled":
            raise ValueError("unsupported B/F03 dossier schema or status")
        if self.covariant_certificate.status != "built" or self.covariant_certificate.formula_verified:
            raise ValueError("a dossier must preserve the unverified built covariant certificate")
        if self.covariant_certificate.source_match_status != "not_checked":
            raise ValueError("a dossier cannot claim source matching")
        expected_digest = self.expanded_f0.input_card_digest
        if not (
            self.cancellation_plan.input_card_digest
            == self.local_reduction.input_card_digest
            == self.covariant_certificate.input_card_digest
            == expected_digest
        ):
            raise ValueError("dossier components must share one input-card digest")
        if (self.direct_chart_parameters is None) != (self.direct_sample is None):
            raise ValueError("direct chart parameters and direct sample must be present together")
        if self.direct_sample is not None and self.direct_sample.normalization != self.input_card.measure_convention:
            raise ValueError("direct sample normalization disagrees with the input card")
        if self.direct_chart_parameters is not None:
            encoded = _canonical_json(self.direct_chart_parameters)
            object.__setattr__(self, "direct_chart_parameters", json.loads(encoded))

    @property
    def formula_verified(self) -> bool:
        return False

    @property
    def source_match_status(self) -> str:
        return "not_checked"

    def to_dict(self) -> dict[str, object]:
        return {
            "artifact_type": "b_f03_dossier",
            "schema_version": self.schema_version,
            "status": self.status,
            "input_card": self.input_card.to_dict(),
            "expanded_f0": self.expanded_f0.to_dict(),
            "cancellation_plan": self.cancellation_plan.to_dict(),
            "local_reduction": self.local_reduction.to_dict(),
            "covariant_certificate": self.covariant_certificate.to_dict(),
            "direct_chart_parameters": self.direct_chart_parameters,
            "direct_sample": self.direct_sample.to_dict() if self.direct_sample else None,
            "formula_verified": self.formula_verified,
            "source_match_status": self.source_match_status,
        }

    def to_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_markdown(self) -> str:
        return render_b_f03_dossier_markdown(self)

    def write_markdown_new_file(self, path: str | Path) -> Path:
        """Write Markdown with exclusive creation; an existing file is never replaced."""

        target = Path(path)
        with target.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(self.to_markdown())
        return target


def build_b_f03_dossier(
    card: FormulaInputCard,
    *,
    direct_chart_parameters: Mapping[str, Any] | None = None,
) -> BF03Dossier:
    """Compose the B/F03 plan, traces, and optional local chart sample."""

    expanded = build_expanded_f0_expression(card)
    plan = build_cancellation_plan(
        card,
        overrides=(CancellationPlanOverride(
            particle_id=2,
            cancellation_occurrences=(OccurrenceLocator("external", 3, 0),),
            reason="B/F03 dossier choice: retain internal p2 and cancel terminal external p2.",
        ),),
    )
    local = reduce_local_measures(plan, expanded)
    certificate = build_b_covariant_certificate(card, expanded, plan, local)
    safe_parameters, direct_sample = _prepare_direct_sample(card, direct_chart_parameters)
    return BF03Dossier(
        input_card=card,
        expanded_f0=expanded,
        cancellation_plan=plan,
        local_reduction=local,
        covariant_certificate=certificate,
        direct_chart_parameters=safe_parameters,
        direct_sample=direct_sample,
    )


def _json_block(value: Any) -> str:
    return "```json\n" + _canonical_json(value, indent=2) + "\n```"


def _domain_markdown() -> str:
    rows = (
        "`M > 0`, `m_i >= 0`, and `M >= m_1 + m_2 + m_3 + m_4`.",
        "`s12 >= (m1+m2)^2`, `s34 >= (m3+m4)^2`, `s23 >= (m2+m3)^2`; the q12/q34 rest-frame invariants must be positive.",
        "`sqrt(s12) + sqrt(s34) <= M`, with the Källén domains and nonzero two-body momenta checked for the sequential blocks.",
        "`theta0, theta1 in [0, pi]`; `phi0, phi1, phi4_star in [0, 2*pi)`.",
        "The analytic cosine root must lie in `[-1,1]` and satisfy its unsquared on-shell equation; reconstructed energies, shells, and conservation are checked.",
        "Thresholds, polar-angle poles, a non-timelike Q, and `|c0|=1` are reported as chart degeneracies where applicable.",
    )
    return "\n".join(f"- {row}" for row in rows)


def render_b_f03_dossier_markdown(dossier: BF03Dossier) -> str:
    """Render the complete deterministic Markdown review dossier."""

    if not isinstance(dossier, BF03Dossier):
        raise TypeError("render_b_f03_dossier_markdown expects a BF03Dossier")
    card = dossier.input_card
    identity = card.identity or {}
    lines = [
        "# B/F03 推导档案",
        "",
        "## 状态",
        "",
        f"- 档案：`{dossier.status}`",
        f"- 全局协变证书：`{dossier.covariant_certificate.status}`；`formula_verified = false`",
        "- 来源匹配：`not_checked`",
        f"- 局部约简：`{dossier.local_reduction.status}`",
        f"- 直接数值 chart：`{dossier.direct_sample.status if dossier.direct_sample else 'not_sampled'}`",
        "",
        "## 输入身份与约定",
        "",
        f"- 身份 digest：`{identity.get('digest', 'missing')}`",
        f"- 输入卡 SHA-256：`{dossier.expanded_f0.input_card_digest}`",
        f"- 身份模式：`{identity.get('mode', 'missing')}`",
        f"- 测度约定：`{card.measure_convention}`",
        f"- 有序变量：`{', '.join(card.variable_labels)}`",
        "- 输入卡约束：",
        "",
        _json_block({
            "constraints": list(card.constraints),
            "formula_status": dict(card.formula_status),
            "missing_steps": list(card.missing_steps),
        }),
        "",
        "- 卡片来源信息：",
        "",
        _json_block(card.provenance),
        "",
        "## FM-03 展开态：抵消前的局部二体因子",
        "",
    ]
    for factor in dossier.expanded_f0.factors:
        lines.extend([
            f"### 顶点 {factor.vertex}",
            "",
            f"原始局部表达式：`{factor.expression}`",
            "",
            f"动量守恒 delta：`{factor.momentum_delta}`",
            "",
            "入射来源：",
            "",
            _json_block([item.to_dict() for item in factor.incoming_references]),
            "",
            "出射测度与来源：",
            "",
        ])
        for measure in factor.outgoing_measures:
            lines.extend([
                f"- `{measure.expression}`",
                "",
                _json_block(measure.source_reference.to_dict()),
                "",
            ])
    lines.extend([
        "## 出现级抵消计划",
        "",
        f"- 计划策略：`{dossier.cancellation_plan.selection_policy}`",
        "- p2 明确保留内部出现 e2/o0，取消末顶点外部出现 v3/slot0；LM 输入卡和图身份不改写。",
        "",
        _json_block(dossier.cancellation_plan.to_dict()),
        "",
        "## 局部操作、条件与残余约束",
        "",
    ])
    for factor in dossier.local_reduction.factors:
        lines.extend([
            f"### 顶点 {factor.vertex}：`{factor.status}`",
            "",
            f"操作结果：`{factor.after_expression}`",
            "",
            f"说明：{factor.note}",
            "",
            f"积分变量：`{factor.integrated_variable or 'none'}`；局部 delta⁴ Jacobian：`{factor.delta4_jacobian or 'not used'}`",
            "",
            "取消出现、被移除测度、来源和残余约束：",
            "",
            _json_block({
                "canceled_occurrences": list(factor.canceled_occurrences),
                "removed_measures": list(factor.removed_measures),
                "integrated_measure": factor.integrated_measure,
                "remaining_constraint": factor.remaining_constraint.to_dict() if factor.remaining_constraint else None,
                "source_references": list(factor.source_references),
                "two_pi_accounting": factor.two_pi_accounting.to_dict(),
            }),
            "",
        ])
    lines.extend([
        "## 全局协变消元证书（built）",
        "",
        f"总动量残余式：`{dossier.covariant_certificate.residual_global_equation}`；`{dossier.covariant_certificate.residual_global_delta4}`",
        "",
        "### 四个顶点守恒式及来源",
        "",
    ])
    for equation in dossier.covariant_certificate.conservation_equations:
        lines.extend([
            f"- v{equation.vertex}: `{equation.equation}`",
            "",
            _json_block({"incoming_sources": list(equation.incoming_sources), "outgoing_sources": list(equation.outgoing_sources)}),
            "",
        ])
    lines.extend(["### 四维 delta⁴ 的唯一用途", ""])
    for disposition in dossier.covariant_certificate.delta4_dispositions:
        lines.extend([
            f"- v{disposition.vertex}: `{disposition.disposition}`；消费方：`{disposition.consumed_by or '保留为全局守恒'}`；来源：`{disposition.source_delta4}`",
            "",
        ])
    lines.extend(["### q 变量消元与壳层 delta 转移", ""])
    for step in dossier.covariant_certificate.elimination_steps:
        transfer = step.shell_transfer
        lines.extend([
            f"#### {step.pair_momentum_label}: `{step.relation}`",
            "",
            f"- 图内变量：`{step.q_momentum}`；操作：`{step.operation}`；绝对 Jacobian：`{step.absolute_jacobian}`。",
            f"- Jacobian 依据：{step.jacobian_reason}",
            f"- delta⁴ 使用：{step.delta4_consumption}",
            f"- 壳层转移：`{transfer.source_shell_delta}` → `{transfer.result_delta}`；正能量：`{transfer.result_positive_energy}`（由正能粒子能量推出：`{transfer.positive_energy_implied}`）。",
            f"- 保留双线微分：`{transfer.double_line_differential}`。",
            "",
            "来源和代换记录：",
            "",
            _json_block(step.to_dict()),
            "",
        ])
    lines.extend([
        "### 保留粒子测度与归一化账本",
        "",
        _json_block({
            "retained_particle_measures": [item.to_dict() for item in dossier.covariant_certificate.retained_particle_measures],
            "preserved_double_line_factors": list(dossier.covariant_certificate.preserved_double_line_factors),
            "invariant_deltas": list(dossier.covariant_certificate.invariant_deltas),
            "normalization": dossier.covariant_certificate.normalization.to_dict(),
        }),
        "",
        "## 直接密度与 chart 定义域",
        "",
    ])
    if card.measure_convention == "standard_2pi4":
        density, measure, normalization = (
            B_F03_STANDARD_DENSITY_EXPRESSION,
            B_F03_STANDARD_DIFFERENTIAL_MEASURE,
            B_F03_STANDARD_NORMALIZATION_IDENTITY,
        )
    else:
        density, measure, normalization = (
            B_F03_DENSITY_EXPRESSION,
            B_F03_DIFFERENTIAL_MEASURE,
            B_F03_NORMALIZATION_IDENTITY,
        )
    lines.extend([
        f"- 密度：`{density}`",
        f"- 微分：{measure}",
        f"- 归一化恒等式：`{normalization}`",
        "- 显式支撑：`boxcar(c0; -1, 1) * theta(M-sqrt(s12)-sqrt(s34)) * theta(s12-(m1+m2)^2) * theta(s34-(m3+m4)^2) * theta(s23-(m2+m3)^2) * product_i theta(p_i^0)`；其余非零动量与坐标轴条件见下方 chart 条件。",
        "- 角变量定义：",
        "",
        *[f"  - {item}" for item in B_F03_ANGLE_DEFINITIONS],
        "",
        "由 `sample_b_f03` 实施的 chart 条件：",
        "",
        _domain_markdown(),
        "",
        "### 三整体角＋两内部角的 helicity 换图册",
        "",
        "- 整体角：`theta0, phi0, psi`；内部角：`u=cos(alpha), phi4_star`。",
        f"- 角测度：`{B_HELICITY_MEASURE_IDENTITY}`。",
        "- 两图册使用同一 B/F03 密度；数值正逆映射由 `helicity_to_fixed_angles` 与 `fixed_to_helicity_angles` 提供。",
        "- 该 helicity 选择尚未转换成导师举例的粒子 1/2 欧拉角。",
        "",
        *[f"- {item}" for item in B_HELICITY_ANGLE_DEFINITIONS],
        "",
    ])
    if dossier.direct_sample is None:
        lines.extend([
            "未提供数值 chart 参数，因此没有计算数值样本或密度。",
            "",
        ])
    else:
        lines.extend([
            f"数值样本状态：`{dossier.direct_sample.status}`；诊断：{'；'.join(dossier.direct_sample.diagnostics) or '无'}。",
            "",
            "输入参数与完整样本记录：",
            "",
            _json_block({
                "parameters": dossier.direct_chart_parameters,
                "sample": dossier.direct_sample.to_dict(),
            }),
            "",
        ])
    lines.extend([
        "### 完整机器可读协变证书",
        "",
        _json_block(dossier.covariant_certificate.to_dict()),
        "",
    ])
    unverified = [
        "covariant certificate 仅为 built trace；formula_verified 保持 false。",
        "外部资料来源匹配保持 not_checked。",
        "B/F03 直接密度是局部 chart 结果；particle-1/2 overall Euler chart 转换尚未实现。",
    ]
    if dossier.direct_sample is not None:
        unverified.append("直接数值样本自身明确标记为局部样本，不作为全局协变证书。")
    else:
        unverified.append("没有传入数值 chart 参数，直接公式未作数值域点评估。")
    lines.extend([
        "## 未验证事项",
        "",
        *[f"- {item}" for item in unverified],
        "",
    ])
    return "\n".join(lines)


__all__ = [
    "B_F03_DOSSIER_SCHEMA_VERSION",
    "BF03Dossier",
    "build_b_f03_dossier",
    "render_b_f03_dossier_markdown",
]
