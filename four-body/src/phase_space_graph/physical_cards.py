"""Chinese, machine-readable physical record cards.

LM-09 is deliberately a presentation boundary.  It consumes a normalized
record and the certificates already produced by G02/G06; it never guesses a
momentum, reruns a rank test, or imports the Notebook UI.  The structured
card is the source of truth and the Markdown/HTML renderers are deterministic
views of that same content.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from html import escape
from typing import Any, Iterable, Mapping

from .canonical import canonicalize_topology
from .domain import CountMode, EdgeType, Labeling, Topology
from .invariants import invariant_label
from .label_identity import MarkedIdentity, canonicalize_physical_labeling
from .physical_labels import NormalizationResult, PhysicalLabelRecord, Provenance, review_physical_record


CARD_SCHEMA_VERSION = "IM-4-card-v1"


def _jsonable(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if hasattr(value, "numerator") and hasattr(value, "denominator"):
        return _format_fraction(value)
    if hasattr(value, "value") and isinstance(value.value, str):
        return value.value
    return value


def _format_fraction(value: Any) -> str:
    """Format an exact integer/Fraction without introducing floating point."""

    numerator = getattr(value, "numerator", value)
    denominator = getattr(value, "denominator", 1)
    if denominator == 1:
        return str(numerator)
    return f"{numerator}/{denominator}"


def _particle_symbol(particle: int) -> str:
    return f"p_{particle}"


def _vector_expression(vector: Iterable[Any]) -> str:
    """Render a particle-basis vector, retaining negative/repeated terms."""

    terms: list[str] = []
    for particle, coefficient in enumerate(vector, 1):
        if not coefficient:
            continue
        magnitude = abs(coefficient)
        factor = _particle_symbol(particle) if magnitude == 1 else f"{_format_fraction(magnitude)} p_{particle}"
        if not terms:
            terms.append(f"-{factor}" if coefficient < 0 else factor)
        else:
            terms.append(("- " if coefficient < 0 else "+ ") + factor)
    return " ".join(terms) or "0"


def _subset_symbol(subset: Iterable[int]) -> str:
    return "s_{" + "".join(str(item) for item in subset) + "}"


def _line_type(edge_type: EdgeType) -> str:
    return "双线" if edge_type is EdgeType.DOUBLE else "单线"


def _stage_label(stage: str) -> str:
    return {
        "field_rejected": "字段检查未通过",
        "structure_rejected": "结构检查未通过",
        "momentum_rejected": "结构通过，但动量／守恒检查未通过",
        "dependent": "结构通过；动量通过；变量不独立",
        "accepted": "结构通过；动量通过；变量独立",
    }.get(stage, f"阶段未知：{stage}")


def _formula_status(stage: str) -> str:
    if stage in {"accepted", "dependent"}:
        return "未核验（本记录卡不执行相空间公式推导）"
    return "未进入公式核验"


def _source_dict(provenance: Provenance | Mapping[str, Any] | None) -> dict[str, Any] | None:
    if provenance is None:
        return None
    if isinstance(provenance, Provenance):
        return provenance.to_dict()
    if isinstance(provenance, Mapping):
        return {str(key): _jsonable(value) for key, value in provenance.items()}
    raise TypeError("provenance must be Provenance, a mapping, or None")


@dataclass(frozen=True, slots=True)
class ConnectionRow:
    """One expanded internal line; compact multiplicity never hides an occurrence."""

    edge_index: int
    occurrence: int
    source: int
    target: int
    line_type: str
    particle_ids: tuple[int, ...]
    coefficients: tuple[int, ...] | None
    momentum_expression: str | None
    invariant: tuple[int, ...] | None
    invariant_label: str | None

    def to_dict(self) -> dict[str, object]:
        return {
            "edge_index": self.edge_index,
            "occurrence": self.occurrence,
            "source": self.source,
            "target": self.target,
            "line_type": self.line_type,
            "particle_ids": list(self.particle_ids),
            "coefficients": list(self.coefficients) if self.coefficients is not None else None,
            "momentum_expression": self.momentum_expression,
            "invariant": list(self.invariant) if self.invariant is not None else None,
            "invariant_label": self.invariant_label,
        }


@dataclass(frozen=True, slots=True)
class ExternalLineRow:
    vertex: int
    slot: int
    particle_id: int

    def to_dict(self) -> dict[str, int]:
        return {"vertex": self.vertex, "slot": self.slot, "particle_id": self.particle_id}


@dataclass(frozen=True, slots=True)
class ParticleOccurrenceRow:
    particle_id: int
    occurrences: tuple[tuple[str, int, int], ...]
    measure_cancellations: int

    def to_dict(self) -> dict[str, object]:
        return {
            "particle_id": self.particle_id,
            "occurrences": [
                (
                    {"kind": "external", "vertex": first, "slot": second}
                    if kind == "external"
                    else {"kind": "internal", "edge_index": first, "occurrence": second}
                )
                for kind, first, second in self.occurrences
            ],
            "measure_cancellations": self.measure_cancellations,
        }


@dataclass(frozen=True, slots=True)
class VertexEquationRow:
    vertex: int
    incoming_terms: tuple[str, ...]
    outgoing_terms: tuple[str, ...]
    incoming_recovered: str
    outgoing_recovered: str
    residual: tuple[int, ...]

    @property
    def equation(self) -> str:
        left = " + ".join(self.incoming_terms) or "0"
        right = " + ".join(self.outgoing_terms) or "0"
        return f"{left} = {right}"

    def to_dict(self) -> dict[str, object]:
        return {
            "vertex": self.vertex,
            "incoming_terms": list(self.incoming_terms),
            "outgoing_terms": list(self.outgoing_terms),
            "equation": self.equation,
            "incoming_recovered": self.incoming_recovered,
            "outgoing_recovered": self.outgoing_recovered,
            "residual": list(self.residual),
        }


@dataclass(frozen=True, slots=True)
class InvariantRow:
    edge_index: int
    occurrence: int
    subset: tuple[int, ...]
    coefficients: tuple[int, ...]
    variable: tuple[int, ...]
    variable_label: str
    momentum_expression: str
    expression: str

    def to_dict(self) -> dict[str, object]:
        return {
            "edge_index": self.edge_index,
            "occurrence": self.occurrence,
            "subset": list(self.subset),
            "coefficients": list(self.coefficients),
            "variable": list(self.variable),
            "variable_label": self.variable_label,
            "momentum_expression": self.momentum_expression,
            "expression": self.expression,
        }


@dataclass(frozen=True, slots=True)
class PhysicalCard:
    """A complete Chinese physical record card or a readable failure card."""

    schema_version: str
    stage: str
    stage_label: str
    formula_status: str
    n: int | None
    loop_count: int | None
    mode: str | None
    topology_key: tuple[Any, ...] | None
    identity_key: tuple[Any, ...] | None
    identity_digest: str | None
    connections: tuple[ConnectionRow, ...]
    external_lines: tuple[ExternalLineRow, ...]
    particles: tuple[ParticleOccurrenceRow, ...]
    vertex_equations: tuple[VertexEquationRow, ...]
    invariants: tuple[InvariantRow, ...]
    independence: dict[str, object] | None
    diagnostics: tuple[dict[str, object], ...]
    record: dict[str, object] | None
    identity: dict[str, object] | None
    provenance: dict[str, object] | None

    def __post_init__(self) -> None:
        if self.schema_version != CARD_SCHEMA_VERSION:
            raise ValueError(f"unsupported physical-card schema: {self.schema_version!r}")
        if not isinstance(self.stage, str) or not self.stage:
            raise TypeError("PhysicalCard.stage must be a non-empty string")
        object.__setattr__(self, "connections", tuple(self.connections))
        object.__setattr__(self, "external_lines", tuple(self.external_lines))
        object.__setattr__(self, "particles", tuple(self.particles))
        object.__setattr__(self, "vertex_equations", tuple(self.vertex_equations))
        object.__setattr__(self, "invariants", tuple(self.invariants))
        object.__setattr__(self, "diagnostics", tuple(dict(item) for item in self.diagnostics))

    @property
    def identity_mode(self) -> str | None:
        return self.mode

    def to_dict(self) -> dict[str, object]:
        """Serialize the structured card; presentation text is intentionally omitted."""

        return {
            "artifact_type": "physical_record_card",
            "schema_version": self.schema_version,
            "status": {
                "stage": self.stage,
                "label": self.stage_label,
                "formula_status": self.formula_status,
                "n": self.n,
                "loop_count": self.loop_count,
                "mode": self.mode,
            },
            "topology_key": _jsonable(self.topology_key),
            "identity_key": _jsonable(self.identity_key),
            "identity_digest": self.identity_digest,
            "connections": [row.to_dict() for row in self.connections],
            "external_lines": [row.to_dict() for row in self.external_lines],
            "particles": [row.to_dict() for row in self.particles],
            "vertex_equations": [row.to_dict() for row in self.vertex_equations],
            "invariants": [row.to_dict() for row in self.invariants],
            "independence": _jsonable(self.independence),
            "diagnostics": _jsonable(self.diagnostics),
            "record": _jsonable(self.record),
            "identity": _jsonable(self.identity),
            "provenance": _jsonable(self.provenance),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


# A descriptive alias used in early design notes.
PhysicalRecordCard = PhysicalCard


def _diagnostics(normalization: NormalizationResult) -> tuple[dict[str, object], ...]:
    rows: list[dict[str, object]] = []
    for issue in normalization.issues:
        rows.append({"code": issue.code, "message": issue.message, "path": issue.path})
    report = normalization.structure_report
    if report is not None:
        for failure in report.failures:
            rows.append({"code": failure.code, "message": failure.message, "details": failure.details})
    evaluation = normalization.legacy_evaluation
    if evaluation is not None:
        for code, message in zip(evaluation.failure_codes, evaluation.failure_messages):
            row = {"code": code, "message": message}
            if row not in rows:
                rows.append(row)
    if not rows and normalization.stage not in {"accepted", "dependent"}:
        rows.append({"code": normalization.stage, "message": "该记录没有可用的进一步诊断证据。"})
    return tuple(rows)


def _build_connections(record: PhysicalLabelRecord) -> tuple[ConnectionRow, ...]:
    labeling = record.complete_labeling
    double_labels = {(item.edge_index, item.occurrence): tuple(item.particle_ids) for item in labeling.double_lines}
    single_labels = {(item.edge_index, item.occurrence): item.particle_id for item in labeling.single_lines}
    momenta = {(edge, occurrence): tuple(vector) for edge, occurrence, vector in record.momentum.double_momenta}
    rows: list[ConnectionRow] = []
    for edge_index, edge in enumerate(record.topology.edges):
        if edge.multiplicity <= 0:
            continue
        for occurrence in range(edge.multiplicity):
            ref = (edge_index, occurrence)
            vector = momenta.get(ref) if edge.edge_type is EdgeType.DOUBLE else None
            if edge.edge_type is EdgeType.DOUBLE:
                particles = tuple(index + 1 for index, coefficient in enumerate(vector or ()) if coefficient == 1)
                if not particles:
                    particles = tuple(double_labels.get(ref, ()))
                proper = vector is not None and all(value in (0, 1) for value in vector)
                subset = tuple(particles) if proper and 2 <= len(particles) < record.topology.n else None
                variable = invariant_label(subset) if subset is not None else None
                rows.append(ConnectionRow(
                    edge_index=edge_index,
                    occurrence=occurrence,
                    source=edge.source,
                    target=edge.target,
                    line_type="双线",
                    particle_ids=particles,
                    coefficients=tuple(vector) if vector is not None else None,
                    momentum_expression=_vector_expression(vector) if vector is not None else None,
                    invariant=subset,
                    invariant_label=variable,
                ))
            else:
                particle = single_labels.get(ref)
                rows.append(ConnectionRow(
                    edge_index=edge_index,
                    occurrence=occurrence,
                    source=edge.source,
                    target=edge.target,
                    line_type="单线",
                    particle_ids=(particle,) if particle is not None else (),
                    coefficients=None,
                    momentum_expression=_particle_symbol(particle) if particle is not None else None,
                    invariant=None,
                    invariant_label=None,
                ))
    return tuple(rows)


def _build_external_lines(record: PhysicalLabelRecord) -> tuple[ExternalLineRow, ...]:
    return tuple(
        ExternalLineRow(item.vertex, item.slot, item.particle_id)
        for item in sorted(record.complete_labeling.final_particles, key=lambda item: (item.vertex, item.slot, item.particle_id))
    )


def _build_particles(record: PhysicalLabelRecord) -> tuple[ParticleOccurrenceRow, ...]:
    counts = dict(record.measure_cancellations)
    return tuple(
        ParticleOccurrenceRow(particle, tuple(occurrences), counts.get(particle, 0))
        for particle, occurrences in record.single_groups
    )


def _build_vertex_equations(record: PhysicalLabelRecord, connections: tuple[ConnectionRow, ...]) -> tuple[VertexEquationRow, ...]:
    labeling = record.complete_labeling
    final_by_vertex: dict[int, list[int]] = {vertex: [] for vertex in record.topology.vertices}
    for item in labeling.final_particles:
        final_by_vertex.setdefault(item.vertex, []).append(item.particle_id)
    row_by_ref = {(row.edge_index, row.occurrence): row for row in connections}
    incoming: dict[int, list[str]] = {vertex: [] for vertex in record.topology.vertices}
    outgoing: dict[int, list[str]] = {vertex: [] for vertex in record.topology.vertices}
    incoming_vectors: dict[int, list[int]] = {vertex: [0] * record.topology.n for vertex in record.topology.vertices}
    outgoing_vectors: dict[int, list[int]] = {vertex: [0] * record.topology.n for vertex in record.topology.vertices}

    def add_vector(target: list[int], vector: Iterable[int]) -> None:
        for index, value in enumerate(vector):
            target[index] += value

    initial_vector = [1] * record.topology.n
    incoming[record.topology.initial_vertex].append("P")
    add_vector(incoming_vectors[record.topology.initial_vertex], initial_vector)
    for row in connections:
        symbol = f"q_{{{row.edge_index + 1},{row.occurrence + 1}}}" if row.line_type == "双线" else (
            _particle_symbol(row.particle_ids[0]) if row.particle_ids else f"x_{{{row.edge_index + 1},{row.occurrence + 1}}}"
        )
        vector = row.coefficients if row.line_type == "双线" else tuple(
            1 if index + 1 == row.particle_ids[0] else 0 for index in range(record.topology.n)
        ) if row.particle_ids else (0,) * record.topology.n
        if row.target in incoming:
            incoming[row.target].append(symbol)
            add_vector(incoming_vectors[row.target], vector)
        if row.source in outgoing:
            outgoing[row.source].append(symbol)
            add_vector(outgoing_vectors[row.source], vector)
    for vertex, particles in final_by_vertex.items():
        for particle in sorted(particles):
            outgoing[vertex].append(_particle_symbol(particle))
            add_vector(outgoing_vectors[vertex], (1 if index + 1 == particle else 0 for index in range(record.topology.n)))
    residuals = dict(record.momentum.vertex_residuals)
    return tuple(
        VertexEquationRow(
            vertex=vertex,
            incoming_terms=tuple(incoming[vertex]),
            outgoing_terms=tuple(outgoing[vertex]),
            incoming_recovered=_vector_expression(incoming_vectors[vertex]),
            outgoing_recovered=_vector_expression(outgoing_vectors[vertex]),
            residual=tuple(residuals.get(vertex, (0,) * record.topology.n)),
        )
        for vertex in record.topology.vertices
    )


def _build_invariants(record: PhysicalLabelRecord, connections: tuple[ConnectionRow, ...]) -> tuple[InvariantRow, ...]:
    by_ref = {(row.edge_index, row.occurrence): row for row in connections if row.line_type == "双线"}
    result: list[InvariantRow] = []
    variables = record.independence.variables
    expressions = record.independence.expressions
    # MomentumCertificate rows and the selected variable list have the same
    # order.  Index by edge occurrence so repeated variables (D) remain two
    # rows rather than being expanded into a Cartesian product.
    evidence_by_ref = {
        (edge, occurrence): (tuple(variable), expression)
        for (edge, occurrence, _), variable, expression in zip(
            record.momentum.double_momenta, variables, expressions
        )
    }
    used: set[tuple[int, int]] = set()
    for row in connections:
        if row.line_type != "双线":
            continue
        variable, expression = evidence_by_ref.get(
            (row.edge_index, row.occurrence),
            (tuple(row.invariant or ()), None),
        )
        if not variable:
            continue
        used.add((row.edge_index, row.occurrence))
        result.append(InvariantRow(
            edge_index=row.edge_index,
            occurrence=row.occurrence,
            subset=tuple(variable),
            coefficients=tuple(row.coefficients or ()),
            variable=tuple(variable),
            variable_label=invariant_label(variable),
            momentum_expression=row.momentum_expression or "未恢复",
            expression=expression.format() if expression is not None else "未恢复",
        ))
    for index, (variable, expression) in enumerate(zip(variables, expressions)):
        if not any(item.variable == tuple(variable) and item.expression == expression.format() for item in result):
            # Keep a variable whose edge was non-standard or whose occurrence
            # was deliberately left unlabelled; never silently drop it.
            result.append(InvariantRow(
                edge_index=-1,
                occurrence=index,
                subset=tuple(variable),
                coefficients=tuple(int(value) if getattr(value, "denominator", 1) == 1 else value for value in expression.coefficients),
                variable=tuple(variable),
                variable_label=invariant_label(variable),
                momentum_expression="未恢复",
                expression=expression.format(),
            ))
    return tuple(result)


def _independence_payload(record: PhysicalLabelRecord) -> dict[str, object]:
    result = record.independence
    certificate = result.certificate
    payload: dict[str, object] = {
        "variables": [list(variable) for variable in result.variables],
        "labels": [invariant_label(variable) for variable in result.variables],
        "rank": result.rank,
        "variable_count": result.variable_count,
        "independent": result.independent,
        "certificate": certificate.to_dict() if certificate else None,
        "relation": certificate.format() if certificate else None,
    }
    return payload


def build_physics_card(
    normalization: NormalizationResult,
    *,
    identity: MarkedIdentity | None = None,
    provenance: Provenance | Mapping[str, Any] | None = None,
) -> PhysicalCard:
    """Build one deterministic card from an LM-02 normalization result.

    ``identity`` is optional so rejected records can still be displayed.  For
    a complete record an omitted identity is computed once using the existing
    LM-04 fixed-label convention; this is a view convenience and does not
    alter the record or perform a new search.
    """

    if not isinstance(normalization, NormalizationResult):
        raise TypeError("build_physics_card expects a NormalizationResult")
    if identity is not None and not isinstance(identity, MarkedIdentity):
        raise TypeError("identity must be a MarkedIdentity or None")
    record = normalization.record
    if record is not None:
        checked = review_physical_record(record)
        if checked.stage != normalization.stage:
            raise ValueError("normalization stage disagrees with the physical record")
    if identity is not None:
        if record is None:
            raise ValueError("an identity cannot supply a missing physical record")
        expected = canonicalize_physical_labeling(record, identity.mode)
        if (identity.key, identity.digest, identity.canonical_record) != (expected.key, expected.digest, expected.canonical_record):
            raise ValueError("identity does not belong to the supplied physical record")
        record = identity.canonical_record
    if record is not None and identity is None:
        identity = canonicalize_physical_labeling(record, CountMode.FIXED_LABELS)
        record = identity.canonical_record
    topology = record.topology if record is not None else None
    report = normalization.structure_report
    if topology is not None:
        topology_key = identity.topology_key if identity is not None else canonicalize_topology(topology).key
        loop_count = report.metrics.loop_count if report is not None else None
        # Use the existing G03 analyzer only when a hand-built normalization
        # result omitted its structural report.
        if report is None:
            from .structure import analyze_topology
            loop_count = analyze_topology(topology).metrics.loop_count
    else:
        topology_key = None
        loop_count = report.metrics.loop_count if report is not None else None

    mode = identity.mode.value if identity is not None else None
    source = _source_dict(provenance)
    if record is None:
        return PhysicalCard(
            schema_version=CARD_SCHEMA_VERSION,
            stage=normalization.stage,
            stage_label=_stage_label(normalization.stage),
            formula_status=_formula_status(normalization.stage),
            n=report.metrics.n if report is not None else None,
            loop_count=loop_count,
            mode=mode,
            topology_key=topology_key,
            identity_key=identity.key if identity is not None else None,
            identity_digest=identity.digest if identity is not None else None,
            connections=(), external_lines=(), particles=(), vertex_equations=(), invariants=(),
            independence=None,
            diagnostics=_diagnostics(normalization),
            record=None,
            identity=identity.to_dict() if identity is not None else None,
            provenance=source,
        )

    connections = _build_connections(record)
    return PhysicalCard(
        schema_version=CARD_SCHEMA_VERSION,
        stage=normalization.stage,
        stage_label=_stage_label(normalization.stage),
        formula_status=_formula_status(normalization.stage),
        n=topology.n,
        loop_count=loop_count,
        mode=mode,
        topology_key=topology_key,
        identity_key=identity.key if identity is not None else None,
        identity_digest=identity.digest if identity is not None else None,
        connections=connections,
        external_lines=_build_external_lines(record),
        particles=_build_particles(record),
        vertex_equations=_build_vertex_equations(record, connections),
        invariants=_build_invariants(record, connections),
        independence=_independence_payload(record),
        diagnostics=_diagnostics(normalization),
        record=record.to_dict(),
        identity=identity.to_dict() if identity is not None else None,
        provenance=source,
    )


def _markdown_table(headers: Iterable[str], rows: Iterable[Iterable[Any]]) -> list[str]:
    headers = tuple(headers)
    output = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    output.extend("| " + " | ".join(str(value).replace("|", "\\|") for value in row) + " |" for row in rows)
    return output


def render_card_markdown(card: PhysicalCard) -> str:
    """Render a card in a compact, study-oriented Chinese Markdown view."""

    if not isinstance(card, PhysicalCard):
        raise TypeError("render_card_markdown expects a PhysicalCard")
    lines = [
        f"## 物理记录卡（{card.stage_label}）",
        "",
        f"- 体系：n={card.n if card.n is not None else '未知'}，圈数 l={card.loop_count if card.loop_count is not None else '未知'}",
        f"- 标记模式：{card.mode or '未提供身份模式'}",
        f"- 公式状态：{card.formula_status}",
    ]
    if card.identity_digest:
        lines.append(f"- 独立身份摘要：`{card.identity_digest}`")
    if card.connections:
        lines += ["", "### 1. 编号连接表", "", *_markdown_table(
            ("边", "出现", "起点→终点", "线型", "粒子／动量", "不变质量"),
            (
                (
                    row.edge_index + 1,
                    row.occurrence + 1,
                    f"{row.source}→{row.target}",
                    row.line_type,
                    ",".join(map(str, row.particle_ids)) if row.line_type == "双线" else (f"p_{row.particle_ids[0]}" if row.particle_ids else "未标记"),
                    row.invariant_label or "—",
                )
                for row in card.connections
            ),
        )]
        lines += ["", "外线：", *_markdown_table(
            ("顶点", "槽位", "粒子"),
            ((row.vertex, row.slot + 1, f"p_{row.particle_id}") for row in card.external_lines),
        )]
        lines += ["", "### 2. 单线出现与测度抵消", "", *_markdown_table(
            ("粒子", "全部出现", "需抵消次数"),
            (
                (
                    f"p_{row.particle_id}",
                    "; ".join(
                        f"外线(v{first},槽{second + 1})" if kind == "external" else f"内部单线(边{first + 1},出现{second + 1})"
                        for kind, first, second in row.occurrences
                    ) or "无",
                    row.measure_cancellations,
                )
                for row in card.particles
            ),
        )]
        lines += ["", "### 3. 双线动量与质量平方变量", "", *_markdown_table(
            ("双线", "系数向量", "动量表达式", "子集", "变量"),
            (
                (
                    f"边{row.edge_index + 1}，出现{row.occurrence + 1}",
                    row.coefficients or (),
                    row.momentum_expression or "未恢复",
                    row.subset or "—",
                    row.variable_label,
                )
                for row in card.invariants
            ),
        )]
        lines += ["", "### 4. 逐顶点守恒", ""]
        for row in card.vertex_equations:
            lines.append(f"- 顶点 {row.vertex}：`{row.equation}`；回代：`{row.incoming_recovered} = {row.outgoing_recovered}`；残差 `{row.residual}`")
        lines += ["", "### 5. 独立性", ""]
        if card.independence:
            independent = "相互独立" if card.independence.get("independent") else "存在依赖关系"
            lines.append(f"变量数 d={card.independence.get('variable_count')}，精确秩={card.independence.get('rank')}：{independent}。")
            relation = card.independence.get("relation")
            if relation:
                lines.append(f"依赖证书：`{relation}`")
        else:
            lines.append("尚未得到可用的独立性证书。")
    if card.diagnostics:
        lines += ["", "### 诊断", "", *_markdown_table(
            ("代码", "说明"), ((row.get("code", ""), row.get("message", "")) for row in card.diagnostics)
        )]
    lines += ["", "<details><summary>身份、来源与机器记录（默认折叠）</summary>", "", "```json", card.to_json(), "```", "</details>"]
    return "\n".join(lines)


def render_card_html(card: PhysicalCard) -> str:
    """Render the same card as escaped standalone HTML (no source execution)."""

    if not isinstance(card, PhysicalCard):
        raise TypeError("render_card_html expects a PhysicalCard")
    markdown = render_card_markdown(card)
    visible, _, _ = markdown.partition("<details><summary>")
    # Keep this dependency-free.  The visible card is escaped and the raw
    # machine record is a real collapsed HTML disclosure.  Both the source
    # text and the JSON are escaped, so hostile provenance is inert HTML.
    return (
        '<article class="physical-record-card"><pre>'
        + escape(visible.rstrip())
        + '</pre><details><summary>身份、来源与机器记录（默认折叠）</summary><pre>'
        + escape(card.to_json())
        + '</pre></details></article>'
    )


__all__ = [
    "CARD_SCHEMA_VERSION",
    "ConnectionRow",
    "ExternalLineRow",
    "ParticleOccurrenceRow",
    "VertexEquationRow",
    "InvariantRow",
    "PhysicalCard",
    "PhysicalRecordCard",
    "build_physics_card",
    "render_card_markdown",
    "render_card_html",
]
