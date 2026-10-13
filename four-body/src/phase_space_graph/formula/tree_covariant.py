"""Replayable covariant reduction along four-body double-line trees.

This module builds a graph-derived certificate for n=4 and L=0..3, with
``d=2+L`` double-line integration variables, ``d+1`` vertices, and the fixed
D0 particle-measure policy.  It deliberately stops before direct IKV charts,
angular Jacobians, branch analysis, domain construction, or source matching.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Iterable, Mapping

from ..formula_inputs import FormulaInputCard
from .cancellations import (
    LM12_DEFAULT_POLICY,
    CancellationPlan,
    OccurrenceLocator,
    build_cancellation_plan,
)
from .expanded import ExpandedF0Expression, build_expanded_f0_expression
from .inputs import validate_formula_card
from .records import RuleStep


TREE_COVARIANT_SCHEMA_VERSION = "FM-tree-covariant-v1"


def _jsonable(value: Any) -> Any:
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    return value


@dataclass(frozen=True, slots=True)
class TreeCovariantTrace:
    """Serializable evidence for the executed D0 double-tree eliminations."""

    schema_version: str
    status: str
    scope: dict[str, Any]
    f0: dict[str, Any]
    cancellation: dict[str, Any]
    elimination: dict[str, Any]
    q_shell_transfers: tuple[dict[str, Any], ...]
    remaining_constraints: tuple[str, ...]
    remaining_constraints_latex: tuple[str, ...]
    normalization: dict[str, Any]
    expression: str
    expression_latex: str
    rule_steps: tuple[RuleStep, ...]
    formula_verified: bool = False
    source_match_status: str = "not_assessed"
    diagnostics: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.schema_version != TREE_COVARIANT_SCHEMA_VERSION:
            raise ValueError(f"unsupported tree-covariant schema: {self.schema_version!r}")
        if self.status != "built":
            raise ValueError("a TreeCovariantTrace represents only a completed tree replay")
        if self.formula_verified or self.source_match_status != "not_assessed":
            raise ValueError("tree elimination cannot claim full formula verification or source matching")
        object.__setattr__(self, "q_shell_transfers", tuple(dict(item) for item in self.q_shell_transfers))
        object.__setattr__(self, "remaining_constraints", tuple(self.remaining_constraints))
        object.__setattr__(self, "remaining_constraints_latex", tuple(self.remaining_constraints_latex))
        object.__setattr__(self, "rule_steps", tuple(self.rule_steps))
        object.__setattr__(self, "diagnostics", tuple(self.diagnostics))

    def to_dict(self) -> dict[str, Any]:
        return _jsonable({
            "artifact_type": "tree_covariant_trace",
            "schema_version": self.schema_version,
            "status": self.status,
            "scope": self.scope,
            "f0": self.f0,
            "cancellation": self.cancellation,
            "elimination": self.elimination,
            "q_shell_transfers": self.q_shell_transfers,
            "remaining_constraints": self.remaining_constraints,
            "remaining_constraints_latex": self.remaining_constraints_latex,
            "normalization": self.normalization,
            "expression": self.expression,
            "expression_latex": self.expression_latex,
            "rule_steps": self.rule_steps,
            "formula_verified": self.formula_verified,
            "source_match_status": self.source_match_status,
            "diagnostics": self.diagnostics,
        })


def _q_name(key: tuple[int, int]) -> str:
    return f"q_{{{key[0] + 1},{key[1] + 1}}}"


def _symbol_order(symbol: str) -> tuple[int, int, int]:
    if symbol == "P":
        return (0, 0, 0)
    if symbol.startswith("p_"):
        return (1, int(symbol[2:]), 0)
    if symbol.startswith("q_{") and symbol.endswith("}"):
        edge, occurrence = symbol[3:-1].split(",")
        return (2, int(edge), int(occurrence))
    return (3, 0, 0)


def _render_terms(poly: Mapping[str, int]) -> str:
    pieces: list[str] = []
    for symbol in sorted((key for key, value in poly.items() if value), key=_symbol_order):
        coefficient = poly[symbol]
        magnitude = abs(coefficient)
        term = symbol if magnitude == 1 else f"{magnitude} {symbol}"
        if not pieces:
            pieces.append(f"-{term}" if coefficient < 0 else term)
        else:
            pieces.append((" - " if coefficient < 0 else " + ") + term)
    return "".join(pieces) or "0"


def _render_balance(poly: Mapping[str, int]) -> str:
    positive = {key: value for key, value in poly.items() if value > 0}
    negative = {key: -value for key, value in poly.items() if value < 0}
    return f"{_render_terms(positive)} = {_render_terms(negative)}"


def _render_residual(poly: Mapping[str, int]) -> str:
    return f"{_render_terms(poly)} = 0"


def _add_scaled(target: dict[str, int], source: Mapping[str, int], scale: int) -> None:
    for symbol, coefficient in source.items():
        value = target.get(symbol, 0) + scale * coefficient
        if value:
            target[symbol] = value
        else:
            target.pop(symbol, None)


def _sum_expression(symbols: Iterable[str]) -> str:
    values = tuple(symbols)
    return " + ".join(values) if values else "0"


def _latex_symbol(symbol: str) -> str:
    if symbol.startswith("p_"):
        return rf"p_{{{symbol[2:]}}}"
    if symbol.startswith("q_{") and symbol.endswith("}"):
        return rf"q_{{{symbol[3:-1].replace(',', ',')}}}"
    return symbol


def _latex_atom(expression: str) -> str:
    """Render the symbol forms emitted by FormulaInputCard as math markup."""
    value = expression.strip()
    value = re.sub(r"\bs_(\d+)\b", r"s_{\1}", value)
    value = re.sub(r"\bm_(\d+)\^2\b", r"m_{\1}^{2}", value)
    value = re.sub(r"\bp_(\d+)\b", r"p_{\1}", value)
    value = re.sub(r"\bq_\{(\d+),(\d+)\}", r"q_{\1,\2}", value)
    value = value.replace("delta^4", r"\delta^{(4)}")
    value = value.replace("delta(", r"\delta(")
    value = value.replace("theta(", r"\theta(")
    value = value.replace("(2*pi)", r"(2\pi)")
    value = value.replace("^0", r"^{0}").replace("^2", r"^{2}")
    return value


def _latex_terms(poly: Mapping[str, int]) -> str:
    pieces: list[str] = []
    for symbol in sorted((key for key, value in poly.items() if value), key=_symbol_order):
        coefficient = poly[symbol]
        magnitude = abs(coefficient)
        token = _latex_symbol(symbol)
        term = token if magnitude == 1 else rf"{magnitude}\,{token}"
        if not pieces:
            pieces.append(rf"-{term}" if coefficient < 0 else term)
        else:
            pieces.append((" - " if coefficient < 0 else " + ") + term)
    return "".join(pieces) or "0"


def _latex_balance(poly: Mapping[str, int]) -> str:
    positive = {key: value for key, value in poly.items() if value > 0}
    negative = {key: -value for key, value in poly.items() if value < 0}
    return rf"{_latex_terms(positive)} = {_latex_terms(negative)}"


def _latex_residual(poly: Mapping[str, int]) -> str:
    return rf"{_latex_terms(poly)} = 0"


def _latex_shell_measure(measure: Any) -> str:
    momentum = _latex_atom(measure.momentum_expression)
    mass_squared = _latex_atom(measure.mass_squared)
    return (
        rf"\frac{{\mathrm{{d}}^4 {momentum}\,"
        rf"\delta\!\left(({momentum})^2-{mass_squared}\right)\,"
        rf"\theta\!\left({momentum}^0\right)}}{{(2\pi)^3}}"
    )


def _latex_local_factor(factor: Any) -> str:
    incoming = _latex_atom(factor.incoming_momentum)
    outgoing = [_latex_atom(item) for item in factor.outgoing_integration_momenta]
    delta = rf"\delta^{{(4)}}\!\left(({incoming})-({outgoing[0]})-({outgoing[1]})\right)"
    measures = r"\,".join(_latex_shell_measure(item) for item in factor.outgoing_measures)
    prefix = r"(2\pi)^4\," if factor.normalization_2pi_power == 4 else ""
    return prefix + delta + r"\," + measures


def _latex_ds_factor(ds_factor: Any) -> str:
    label = "".join(str(item) for item in ds_factor.variable)
    if ds_factor.convention == "jing_psf":
        return rf"(2\pi)^3\,\mathrm{{d}}s_{{{label}}}"
    return rf"\frac{{\mathrm{{d}}s_{{{label}}}}}{{2\pi}}"


def _equation_rows(equations: Mapping[int, Mapping[str, int]], vertices: Iterable[int]) -> list[dict[str, Any]]:
    return [
        {"vertex": vertex, "residual": _render_residual(equations[vertex]),
         "residual_latex": _latex_residual(equations[vertex]),
         "coefficients": dict(sorted(equations[vertex].items(), key=lambda item: _symbol_order(item[0])))}
        for vertex in sorted(vertices)
    ]


def _locator_for_measure(measure: Any) -> OccurrenceLocator | None:
    source = measure.source_reference
    if source.role == "external":
        if source.vertex is None or source.slot is None:
            raise ValueError("expanded external measure is missing its vertex/slot locator")
        return OccurrenceLocator("external", source.vertex, source.slot)
    if source.role == "single":
        if source.edge_index is None or source.occurrence is None:
            raise ValueError("expanded internal single measure is missing its edge locator")
        return OccurrenceLocator("internal", source.edge_index, source.occurrence)
    return None


def _validate_scope(card: FormulaInputCard) -> int:
    if card.n != 4 or card.loop_count not in (0, 1, 2, 3):
        raise ValueError("tree-covariant replay supports only n=4 and loop_count=0..3")
    d = 2 + card.loop_count
    if len(card.double_measures) != d:
        raise ValueError(f"L={card.loop_count} requires d={d} double measures")
    if card.record is None or not isinstance(card.record, dict):
        raise ValueError("tree-covariant replay requires a complete physical record")
    if card.identity is None:
        raise ValueError("tree-covariant replay requires a complete formula identity for the D0 ledger")
    if len(card.independent_variables) != d or len(set(card.independent_variables)) != d:
        raise ValueError(f"d={d} tree replay requires {d} distinct ordered independent variables")
    return d


def _build_cancellation(card: FormulaInputCard) -> tuple[CancellationPlan, set[OccurrenceLocator]]:
    plan = build_cancellation_plan(card)
    if plan.selection_policy != LM12_DEFAULT_POLICY:
        raise ValueError("tree-covariant replay requires the fixed D0 cancellation policy")
    canceled: set[OccurrenceLocator] = set()
    for particle in plan.particles:
        if particle.retained_occurrence.kind != "external":
            raise ValueError("D0 must retain each particle's external measure occurrence")
        if any(item.kind != "internal" for item in particle.cancellation_occurrences):
            raise ValueError("D0 may cancel internal single measures only")
        canceled.update(item.locator for item in particle.cancellation_occurrences)
    return plan, canceled


def _build_vertex_equations(
    card: FormulaInputCard,
    expanded: ExpandedF0Expression,
) -> tuple[dict[int, dict[str, int]], dict[int, dict[str, Any]], dict[tuple[int, int], tuple[int, int]]]:
    factor_by_vertex = {factor.vertex: factor for factor in card.factors}
    expanded_by_vertex = {factor.vertex: factor for factor in expanded.factors}
    if len(factor_by_vertex) != len(card.factors) or set(factor_by_vertex) != set(expanded_by_vertex):
        raise ValueError("expanded factors do not cover each formula vertex exactly once")

    single_lines: dict[tuple[int, int], tuple[int, int]] = {}
    q_edges: dict[tuple[int, int], tuple[int, int]] = {}
    for factor in card.factors:
        for line in factor.outgoing:
            if line.edge_index is None or line.occurrence is None:
                if line.role == "external":
                    continue
                raise ValueError(f"vertex {factor.vertex} has an internal line without occurrence coordinates")
            key = (line.edge_index, line.occurrence)
            if line.role == "single":
                if len(line.particle_ids) != 1:
                    raise ValueError(f"single line {key} must carry exactly one particle identity")
                if key in single_lines:
                    raise ValueError(f"single line {key} is emitted by multiple vertices")
                if line.target is None:
                    raise ValueError(f"single line {key} is missing its target vertex")
                single_lines[key] = (line.particle_ids[0], line.source)
            elif line.role == "double":
                if key in q_edges:
                    raise ValueError(f"double line {key} is emitted by multiple vertices")
                if line.target is None:
                    raise ValueError(f"double line {key} is missing its target vertex")
                q_edges[key] = (line.source, line.target)
            else:
                raise ValueError(f"unsupported outgoing connection role {line.role!r}")

    equations: dict[int, dict[str, int]] = {}
    factor_source: dict[int, dict[str, Any]] = {}
    for vertex, factor in factor_by_vertex.items():
        incoming_symbols: list[str] = []
        incoming_poly: dict[str, int] = {}
        for reference in factor.incoming_references:
            role = reference.get("role")
            if role == "initial":
                if vertex != card.record["topology"]["initial_vertex"]:
                    raise ValueError("the initial momentum reference is not on the declared root")
                symbol = "P"
            else:
                key = (reference.get("edge_index"), reference.get("occurrence"))
                if not all(isinstance(item, int) for item in key):
                    raise ValueError(f"vertex {vertex} has an incoming line without an occurrence key")
                typed_key = (int(key[0]), int(key[1]))
                endpoints = q_edges.get(typed_key)
                if role == "double":
                    if endpoints is None or endpoints[1] != vertex:
                        raise ValueError(f"incoming double line {typed_key} is not bound to vertex {vertex}")
                    symbol = _q_name(typed_key)
                elif role == "single":
                    line = factor_by_vertex.get(int(reference.get("source", -1)))
                    source_line = None if line is None else next((item for item in line.outgoing
                        if item.role == "single" and (item.edge_index, item.occurrence) == typed_key), None)
                    if source_line is None or source_line.target != vertex or len(source_line.particle_ids) != 1:
                        raise ValueError(f"incoming single line {typed_key} is not bound to vertex {vertex}")
                    symbol = f"p_{source_line.particle_ids[0]}"
                else:
                    raise ValueError(f"vertex {vertex} has unsupported incoming role {role!r}")
            incoming_symbols.append(symbol)
            incoming_poly[symbol] = incoming_poly.get(symbol, 0) + 1

        incoming_expression = _sum_expression(incoming_symbols)
        incoming_latex = r" + ".join(_latex_symbol(item) for item in incoming_symbols) or "0"
        if factor.incoming_momentum != incoming_expression:
            raise ValueError(
                f"vertex {vertex} incoming momentum does not match its occurrence references: "
                f"{factor.incoming_momentum!r} != {incoming_expression!r}"
            )

        expanded_factor = expanded_by_vertex[vertex]
        outgoing_symbols: list[str] = []
        outgoing_poly: dict[str, int] = {}
        for line in factor.outgoing:
            if line.role == "double":
                assert line.edge_index is not None and line.occurrence is not None
                symbol = _q_name((line.edge_index, line.occurrence))
            else:
                if len(line.particle_ids) != 1:
                    raise ValueError(f"vertex {vertex} outgoing {line.role} must carry one particle identity")
                symbol = f"p_{line.particle_ids[0]}"
            outgoing_symbols.append(symbol)
            outgoing_poly[symbol] = outgoing_poly.get(symbol, 0) + 1

        if tuple(outgoing_symbols) != tuple(expanded_factor.outgoing_integration_momenta):
            raise ValueError(f"vertex {vertex} expanded outgoing integration variables do not match its input card")
        outgoing_latex = r" + ".join(_latex_symbol(item) for item in outgoing_symbols) or "0"
        residual = dict(incoming_poly)
        _add_scaled(residual, outgoing_poly, -1)
        equations[vertex] = residual
        factor_source[vertex] = {
            "vertex": vertex,
            "incoming_momentum": incoming_expression,
            "incoming_references": [item.to_dict() for item in expanded_factor.incoming_references],
            "outgoing_integration_momenta": list(expanded_factor.outgoing_integration_momenta),
            "outgoing_support_momenta": list(expanded_factor.outgoing_support_momenta),
            "momentum_delta": expanded_factor.momentum_delta,
            "conservation_equation": f"{incoming_expression} = {_sum_expression(outgoing_symbols)}",
            "incoming_momentum_latex": incoming_latex,
            "outgoing_integration_momenta_latex": [_latex_symbol(item) for item in outgoing_symbols],
            "outgoing_support_momenta_latex": [_latex_atom(item) for item in expanded_factor.outgoing_support_momenta],
            "conservation_equation_latex": f"{incoming_latex} = {outgoing_latex}",
            "momentum_delta_latex": (
                rf"\delta^{{(4)}}\!\left(({incoming_latex})-({outgoing_latex.split(' + ')[0]})"
                + (rf"-({outgoing_latex.split(' + ')[1]})" if len(outgoing_symbols) == 2 else "")
                + r"\right)"
            ),
            "expression_latex": _latex_local_factor(expanded_factor),
            "expanded_expression_latex": _latex_local_factor(expanded_factor),
            "residual_coefficients": dict(sorted(residual.items(), key=lambda item: _symbol_order(item[0]))),
            "normalization_2pi_power": expanded_factor.normalization_2pi_power,
            "expression": expanded_factor.expression,
        }
    return equations, factor_source, q_edges


def _validate_double_tree(
    vertices: tuple[int, ...],
    root: int,
    q_edges: Mapping[tuple[int, int], tuple[int, int]],
    expanded: ExpandedF0Expression,
) -> dict[tuple[int, int], Any]:
    d = len(vertices) - 1
    if root not in vertices:
        raise ValueError("the declared root is missing from the double-line tree vertices")
    if len(q_edges) != d:
        raise ValueError(f"a d={d} double-line tree requires exactly d={d} double-line edges")
    if len(expanded.double_line_factors) != d:
        raise ValueError(f"the expanded F0 expression does not contain d={d} double-line measures")
    ds_by_key = {(item.edge_index, item.occurrence): item for item in expanded.double_line_factors}
    if len(ds_by_key) != len(expanded.double_line_factors) or set(ds_by_key) != set(q_edges):
        raise ValueError(f"the d={d} q integration variables and ds double-line factors do not match exactly")

    adjacency: dict[int, set[int]] = {vertex: set() for vertex in vertices}
    for key, (source, target) in q_edges.items():
        if source not in adjacency or target not in adjacency or source == target:
            raise ValueError(f"double line {key} has invalid endpoints {source}->{target}")
        adjacency[source].add(target)
        adjacency[target].add(source)
    visited: set[int] = set()
    pending = [root]
    while pending:
        vertex = pending.pop()
        if vertex in visited:
            continue
        visited.add(vertex)
        pending.extend(adjacency[vertex] - visited)
    if visited != set(vertices):
        raise ValueError("the double-line graph is disconnected and cannot be reduced as one spanning tree")
    if sum(len(items) for items in adjacency.values()) // 2 != len(vertices) - 1:
        raise ValueError("the connected double-line graph is not a tree")
    return ds_by_key


def _substitute(poly: Mapping[str, int], symbol: str, replacement: Mapping[str, int]) -> dict[str, int]:
    result = dict(poly)
    coefficient = result.pop(symbol, 0)
    _add_scaled(result, replacement, coefficient)
    return result


def _expand_q_solution(
    symbol: str,
    substitutions: Mapping[str, Mapping[str, int]],
    memo: dict[str, dict[str, int]],
    active: set[str],
) -> dict[str, int]:
    if symbol in memo:
        return dict(memo[symbol])
    if symbol in active:
        raise ValueError(f"cyclic q substitutions encountered at {symbol}")
    if symbol not in substitutions:
        raise ValueError(f"q integration variable {symbol} has no executed elimination step")
    active.add(symbol)
    result: dict[str, int] = {}
    for term, coefficient in substitutions[symbol].items():
        if term.startswith("q_{"):
            _add_scaled(result, _expand_q_solution(term, substitutions, memo, active), coefficient)
        else:
            result[term] = result.get(term, 0) + coefficient
            if not result[term]:
                del result[term]
    active.remove(symbol)
    memo[symbol] = result
    return dict(result)


def _global_support(poly: Mapping[str, int], external_ids: tuple[int, ...]) -> dict[str, int]:
    result = dict(poly)
    coefficient = result.pop("P", 0)
    for particle_id in external_ids:
        symbol = f"p_{particle_id}"
        result[symbol] = result.get(symbol, 0) + coefficient
        if not result[symbol]:
            del result[symbol]
    return result


def _particle_ids_from_support(poly: Mapping[str, int]) -> tuple[int, ...]:
    result: list[int] = []
    for symbol, coefficient in poly.items():
        if not symbol.startswith("p_"):
            raise ValueError(f"global support still contains a non-particle symbol {symbol}")
        if coefficient != 1:
            raise ValueError(f"global support coefficient for {symbol} is {coefficient}, expected +1")
        result.append(int(symbol[2:]))
    return tuple(sorted(result))


def _build_elimination(
    vertices: tuple[int, ...],
    root: int,
    initial_equations: Mapping[int, Mapping[str, int]],
    factor_sources: Mapping[int, Mapping[str, Any]],
    q_edges: Mapping[tuple[int, int], tuple[int, int]],
) -> tuple[dict[str, Any], dict[str, dict[str, int]], tuple[str, ...]]:
    equations = {vertex: dict(poly) for vertex, poly in initial_equations.items()}
    adjacency: dict[int, set[int]] = {vertex: set() for vertex in vertices}
    key_between: dict[frozenset[int], tuple[int, int]] = {}
    for key, (source, target) in q_edges.items():
        adjacency[source].add(target)
        adjacency[target].add(source)
        key_between[frozenset((source, target))] = key

    active = set(vertices)
    substitutions: dict[str, dict[str, int]] = {}
    elimination_order: list[str] = []
    steps: list[dict[str, Any]] = []

    while len(active) > 1:
        leaves = sorted(vertex for vertex in active if vertex != root and len(adjacency[vertex] & active) == 1)
        if not leaves:
            raise ValueError("double-line tree has no removable non-root leaf")
        leaf = leaves[0]
        parent = next(iter(adjacency[leaf] & active))
        key = key_between[frozenset((leaf, parent))]
        q = _q_name(key)
        current = equations[leaf]
        q_coefficient = current.get(q, 0)
        if q_coefficient not in (-1, 1):
            raise ValueError(f"leaf vertex {leaf} has q coefficient {q_coefficient}; expected +1 or -1")
        rest = dict(current)
        rest.pop(q)
        solution = {symbol: -coefficient // q_coefficient for symbol, coefficient in rest.items()}
        # Integer division above is exact because q_coefficient is ±1.
        before_rows = _equation_rows(equations, active)
        for other in tuple(active):
            if other != leaf:
                equations[other] = _substitute(equations[other], q, solution)
        active.remove(leaf)
        del equations[leaf]
        substitutions[q] = solution
        elimination_order.append(q)
        after_rows = _equation_rows(equations, active)
        remaining_q = sorted(
            (
                _q_name(edge_key) for edge_key, endpoints in q_edges.items()
                if endpoints[0] in active and endpoints[1] in active
            ),
            key=_symbol_order,
        )
        factor = factor_sources[leaf]
        equation_text = _render_residual(current)
        solution_text = _render_terms(solution)
        step = {
            "leaf_vertex": leaf,
            "parent_vertex": parent,
            "delta4_source": factor["momentum_delta"],
            "delta4_source_latex": factor["momentum_delta_latex"],
            "current_equation": equation_text,
            "current_equation_latex": _latex_residual(current),
            "current_equation_coefficients": dict(sorted(current.items(), key=lambda item: _symbol_order(item[0]))),
            "equations_before": before_rows,
            "integrated_q": q,
            "q_coefficient": q_coefficient,
            "substitution": f"{q} = {solution_text}",
            "substitution_latex": rf"{_latex_symbol(q)} = {_latex_terms(solution)}",
            "substitution_coefficients": dict(sorted(solution.items(), key=lambda item: _symbol_order(item[0]))),
            "abs_4d_jacobian": "1",
            "abs_4d_jacobian_latex": r"\left|\det(\pm I_4)\right|=1",
            "jacobian_reason": f"delta^4中{q}的线性系数为{q_coefficient} I_4，绝对行列式 |{q_coefficient}|^4=1。",
            "remaining_q_variables": remaining_q,
            "equations_after": after_rows,
        }
        steps.append(step)

    if set(active) != {root}:
        raise ValueError("tree elimination did not leave exactly the initial vertex")
    root_poly = equations[root]
    root_equation = _render_balance(root_poly)
    root_residual = _render_residual(root_poly)
    return {
        "root_vertex": root,
        "steps": steps,
        "remaining_q_variables": [],
        "root_equation": root_equation,
        "root_equation_latex": _latex_balance(root_poly),
        "root_residual": root_residual,
        "root_residual_latex": _latex_residual(root_poly),
        "root_residual_coefficients": dict(sorted(root_poly.items(), key=lambda item: _symbol_order(item[0]))),
    }, substitutions, tuple(elimination_order)


def _tree_rule_steps(elimination: Mapping[str, Any]) -> list[RuleStep]:
    # The structured tree step dictionaries remain the serialization source;
    # this helper makes the same exact sequence available to CovariantResult.
    result: list[RuleStep] = []
    for step in elimination["steps"]:
        result.append(RuleStep(
            rule_id="FM-tree-leaf-q-elimination",
            before="; ".join(f"v{row['vertex']}: {row['residual']}" for row in step["equations_before"]),
            after=(f"integrate d^4({step['integrated_q']}) against v{step['leaf_vertex']} delta^4; "
                   f"{step['substitution']}; "
                   + "; ".join(f"v{row['vertex']}: {row['residual']}" for row in step["equations_after"])),
            eliminated=(step["integrated_q"], f"delta^4(v{step['leaf_vertex']})"),
            jacobian=step["abs_4d_jacobian"],
            conditions=(f"leaf={step['leaf_vertex']}, parent={step['parent_vertex']}, q coefficient={step['q_coefficient']}",
                        "unit absolute 4D Jacobian"),
            evidence="current graph vertex equation replayed from FormulaInputCard occurrences",
        ))
    return result


def _format_ds_without_coefficient(ds_factor: Any) -> str:
    label = ds_factor.variable_label
    return f"d{label}"


def _build_normalization(card: FormulaInputCard, expanded: ExpandedF0Expression, cancellation_count: int) -> dict[str, Any]:
    q_count = len(expanded.double_line_factors)
    if card.loop_count is None:
        raise ValueError("normalization ledger requires a known loop count")
    d = q_count
    vertex_count = d + 1
    expected_cancellations = card.loop_count
    external_count = sum(
        measure.source_reference.role == "external"
        for factor in expanded.factors for measure in factor.outgoing_measures
    )
    outgoing_count = sum(len(factor.outgoing_measures) for factor in expanded.factors)
    local_powers = [factor.normalization_2pi_power for factor in expanded.factors]
    if len(set(local_powers)) != 1 or local_powers[0] not in (0, 4):
        raise ValueError("the expanded local factors have inconsistent (2*pi) normalization")
    local_total = sum(local_powers)
    if card.measure_convention == "jing_psf":
        ds_power_per_line = 3
    elif card.measure_convention == "standard_2pi4":
        ds_power_per_line = -1
    else:
        raise ValueError(f"unsupported measure convention {card.measure_convention!r}")
    ds_total = q_count * ds_power_per_line
    q_measure_power = 3 * q_count
    external_measure_power = 3 * external_count
    canceled_measure_power = 3 * cancellation_count
    initial_measure_power = 3 * outgoing_count
    remaining_measure_power = 3 * (outgoing_count - cancellation_count)
    residual_power = local_total - initial_measure_power + canceled_measure_power + ds_total
    expected_power = (-12 if card.measure_convention == "jing_psf" else -8)
    if (
        q_count != 2 + card.loop_count
        or len(expanded.factors) != vertex_count
        or external_count != 4
        or outgoing_count != 2 * vertex_count
        or cancellation_count != expected_cancellations
        or residual_power != expected_power
    ):
        raise ValueError(
            "the D0 (2*pi) ledger does not reduce to the specified four-external-measure normalization"
        )
    per_tree_q = {
        "jing_psf": {
            "consumed_local_delta4_power": 0,
            "q_shell_measure_denominator_power": -3,
            "double_line_factor_power": 3,
            "net_power_per_tree_integration": 0,
        },
        "standard_2pi4": {
            "consumed_local_delta4_power": 4,
            "q_shell_measure_denominator_power": -3,
            "double_line_factor_power": -1,
            "net_power_per_tree_integration": 0,
        },
    }[card.measure_convention]
    per_tree_latex = (
        r"(2\pi)^3(2\pi)^{-3}=1"
        if card.measure_convention == "jing_psf" else
        r"(2\pi)^4(2\pi)^{-3}(2\pi)^{-1}=1"
    )
    normalization_latex = (
        rf"\left[(2\pi)^3(2\pi)^{{-3}}\right]^{{{d}}}=1,\qquad "
        r"\prod_{i=1}^{4}\frac{\mathrm{d}^4p_i\,\delta(p_i^2-m_i^2)\,\theta(p_i^0)}{(2\pi)^3}"
        if card.measure_convention == "jing_psf" else
        rf"\left[(2\pi)^4(2\pi)^{{-3}}(2\pi)^{{-1}}\right]^{{{d}}}=1,\qquad "
        r"(2\pi)^4\delta^{(4)}\!\left(P-\sum_{i=1}^{4}p_i\right)"
        r"\prod_{i=1}^{4}\frac{\mathrm{d}^4p_i\,\delta(p_i^2-m_i^2)\,\theta(p_i^0)}{(2\pi)^3}"
    )
    return {
        "measure_convention": card.measure_convention,
        "local_factor_count": len(expanded.factors),
        "local_delta4_prefactor_power_per_factor": local_powers[0],
        "local_delta4_prefactor_power_total": local_total,
        "outgoing_measure_count_before_D0": outgoing_count,
        "outgoing_measure_denominator_power_before_D0": -initial_measure_power,
        "D0_cancelled_internal_single_measure_count": cancellation_count,
        "D0_cancelled_denominator_power_relief": canceled_measure_power,
        "remaining_measure_count_after_D0": outgoing_count - cancellation_count,
        "remaining_measure_denominator_power_after_D0": -remaining_measure_power,
        "q_shell_measure_count": q_count,
        "q_shell_measure_denominator_power": -q_measure_power,
        "retained_external_measure_count": external_count,
        "retained_external_measure_denominator_power": -external_measure_power,
        "double_line_factor_power_per_line": ds_power_per_line,
        "double_line_factor_power_total": ds_total,
        "tree_integrations": q_count,
        "per_tree_integration_balance": per_tree_q,
        "per_tree_integration_balance_latex": per_tree_latex,
        "root_global_delta4_prefactor": "(2*pi)^4" if card.measure_convention == "standard_2pi4" else "1",
        "root_global_delta4_prefactor_power": 4 if card.measure_convention == "standard_2pi4" else 0,
        "net_residual_power_including_four_external_measures": residual_power,
        "expected_residual_power_including_four_external_measures": expected_power,
        "final_measure_statement": (
            f"four retained dmu(p_i) factors contribute (2*pi)^-12; each of the {d} q-shell/ds pairs cancels locally"
            if card.measure_convention == "jing_psf" else
            f"the {d} tree integrations leave one global (2*pi)^4 delta prefactor; four retained dmu(p_i) factors contribute (2*pi)^-12"
        ),
        "normalization_identity_latex": normalization_latex,
        "final_external_measure_factor_latex": r"\prod_{i=1}^{4}\mathrm{d}\mu(p_i)=\frac{\prod_i\mathrm{d}^4p_i\,\delta(p_i^2-m_i^2)\theta(p_i^0)}{(2\pi)^{12}}",
        "standard_2pi4_global_factor_applied_once": card.measure_convention == "standard_2pi4",
    }


def build_tree_covariant_trace(card: FormulaInputCard) -> TreeCovariantTrace:
    """Build an exact symbolic replay of D0 cancellation and d q integrations.

    The q variables are created from each double-line occurrence and remain
    independent throughout the vertex equations.  Only after all d q
    integrations and the final root equation are built is ``P`` replaced by
    the sum of the four recorded external momenta for the support checks.
    """

    if not isinstance(card, FormulaInputCard):
        raise TypeError("build_tree_covariant_trace expects a FormulaInputCard")
    validate_formula_card(card)
    d = _validate_scope(card)
    assert card.record is not None
    topology = card.record["topology"]
    root = int(topology["initial_vertex"])
    vertices = tuple(sorted(int(vertex) for vertex in topology["vertices"]))

    expanded = build_expanded_f0_expression(card)
    cancellation_plan, canceled_locators = _build_cancellation(card)
    equations, factor_sources, q_edges = _build_vertex_equations(card, expanded)
    ds_by_key = _validate_double_tree(vertices, root, q_edges, expanded)
    if set(equations) != set(vertices):
        raise ValueError("formula vertex equations do not cover the declared topology vertices")

    q_measure_by_key: dict[tuple[int, int], Any] = {}
    for factor in expanded.factors:
        for measure in factor.outgoing_measures:
            source = measure.source_reference
            if source.role != "double":
                continue
            if source.edge_index is None or source.occurrence is None:
                raise ValueError("an outgoing q measure lacks its double-line occurrence")
            key = (source.edge_index, source.occurrence)
            if key in q_measure_by_key:
                raise ValueError(f"expanded F0 repeats q integration variable {key}")
            q_measure_by_key[key] = measure
    if set(q_measure_by_key) != set(q_edges):
        raise ValueError("every double-line q must have exactly one expanded four-momentum measure")

    # Bind every D0 cancellation to the concrete expanded outgoing measure.
    expanded_measure_by_locator: dict[OccurrenceLocator, Any] = {}
    for factor in expanded.factors:
        for measure in factor.outgoing_measures:
            locator = _locator_for_measure(measure)
            if locator is not None:
                if locator in expanded_measure_by_locator:
                    raise ValueError(f"expanded F0 repeats measure occurrence {locator}")
                expanded_measure_by_locator[locator] = measure
    if not canceled_locators <= set(expanded_measure_by_locator):
        raise ValueError("D0 cancellation plan refers to a missing expanded single measure")
    canceled_measures = [
        {
            "locator": locator.to_dict(),
            "measure": expanded_measure_by_locator[locator].to_dict(),
            "reason": "D0 retains the unique external occurrence and cancels this internal single-line copy.",
        }
        for locator in sorted(canceled_locators, key=OccurrenceLocator.sort_key)
    ]
    retained_external_measures = [
        measure.to_dict()
        for locator, measure in sorted(expanded_measure_by_locator.items(), key=lambda item: item[0].sort_key())
        if locator.kind == "external"
    ]
    if len(retained_external_measures) != 4:
        raise ValueError("D0 must leave exactly four external particle measures")

    elimination, substitutions, elimination_order = _build_elimination(
        vertices, root, equations, factor_sources, q_edges,
    )
    external_ids = tuple(sorted(
        int(item["particle_id"]) for item in card.record["complete_labeling"]["final_particles"]
    ))
    if external_ids != (1, 2, 3, 4):
        raise ValueError(f"four-body external particle records must be p_1 through p_4, got {external_ids}")
    expected_root = {"P": 1, **{f"p_{particle}": -1 for particle in external_ids}}
    if elimination["root_residual_coefficients"] != expected_root:
        raise ValueError(
            "leaf replay did not produce the required root equation P-sum(p_i)=0: "
            f"{elimination['root_residual']}"
        )

    memo: dict[str, dict[str, int]] = {}
    transfers: list[dict[str, Any]] = []
    transfer_by_variable: dict[tuple[int, ...], str] = {}
    support_rule_steps: list[RuleStep] = []
    ordered_keys = sorted(q_edges)
    for key in ordered_keys:
        q = _q_name(key)
        q_measure = q_measure_by_key[key]
        ds_factor = ds_by_key[key]
        derived = _expand_q_solution(q, substitutions, memo, set())
        on_global_support = _global_support(derived, external_ids)
        derived_ids = _particle_ids_from_support(on_global_support)
        recorded_ids = tuple(ds_factor.variable)
        if derived_ids != recorded_ids:
            raise ValueError(
                f"tree-derived support for {q} is {derived_ids}, but its recorded double-line subset is {recorded_ids}"
            )
        if tuple(q_measure.particle_ids) != recorded_ids:
            raise ValueError(f"q shell support for {q} disagrees with the actual ds occurrence")
        variable = tuple(ds_factor.variable)
        if variable in transfer_by_variable:
            raise ValueError(f"two q lines use the same ds variable {ds_factor.variable_label}")
        transfer_by_variable[variable] = q
        support_expression = _render_terms(on_global_support)
        shell_delta = f"delta(({support_expression})^2 - {q_measure.mass_squared})"
        positive_energy = f"theta(({support_expression})^0)"
        transfer = {
            "edge_index": key[0],
            "occurrence": key[1],
            "integrated_q": q,
            "elimination_order": elimination_order.index(q) + 1,
            "derived_solution": f"{q} = {_render_terms(derived)}",
            "derived_solution_latex": rf"{_latex_symbol(q)} = {_latex_terms(derived)}",
            "derived_solution_coefficients": dict(sorted(derived.items(), key=lambda item: _symbol_order(item[0]))),
            "global_support_solution": f"{q} = {support_expression}",
            "global_support_solution_latex": rf"{_latex_symbol(q)} = {_latex_terms(on_global_support)}",
            "global_support_coefficients": dict(sorted(on_global_support.items(), key=lambda item: _symbol_order(item[0]))),
            "global_support_uses_root_equation": elimination["root_equation"],
            "derived_particle_ids": list(derived_ids),
            "recorded_particle_ids": list(recorded_ids),
            "support_match": True,
            "source_q_shell_delta": q_measure.shell_delta,
            "source_q_shell_delta_latex": (
                rf"\delta\!\left(({_latex_symbol(q)})^2-{_latex_atom(q_measure.mass_squared)}\right)"
            ),
            "shell_delta": shell_delta,
            "shell_delta_latex": rf"\delta\!\left(({_latex_terms(on_global_support)})^2-{_latex_atom(q_measure.mass_squared)}\right)",
            "source_positive_energy": q_measure.positive_energy,
            "source_positive_energy_latex": rf"\theta\!\left(({_latex_symbol(q)})^0\right)",
            "positive_energy_transfer": positive_energy,
            "positive_energy_transfer_latex": rf"\theta\!\left(({_latex_terms(on_global_support)})^0\right)",
            "source_ds_factor": ds_factor.to_dict(),
            "source_ds_factor_latex": _latex_ds_factor(ds_factor),
            "ds_factor_in_final_expression": _format_ds_without_coefficient(ds_factor),
            "ds_factor_in_final_expression_latex": rf"\mathrm{{d}}s_{{{''.join(str(item) for item in ds_factor.variable)}}}",
            "measure_denominator": q_measure.denominator,
            "transfer_rule": "substitute only after the global root equation establishes P=sum_i p_i",
        }
        transfers.append(transfer)
        support_rule_steps.append(RuleStep(
            rule_id="FM-q-shell-positive-energy-transfer",
            before=f"{ds_factor.differential} * {q_measure.shell_delta} * {q_measure.positive_energy}",
            after=f"{_format_ds_without_coefficient(ds_factor)} * {shell_delta} * {positive_energy}",
            eliminated=(),
            jacobian="1",
            conditions=(f"{q} support checked against the recorded subset {list(recorded_ids)}",
                        "substitution is on the derived global-conservation support"),
            evidence="all leaf delta^4 eliminations plus the remaining root equation",
        ))

    if set(transfer_by_variable) != set(card.independent_variables):
        raise ValueError(
            f"the {d} transferred q-shell constraints do not match the formula card's ordered independent-variable set"
        )
    if len(transfers) != d or any(not item["support_match"] for item in transfers):
        raise ValueError("not every q-shell transfer passed the graph-derived support check")

    normalization = _build_normalization(card, expanded, len(canceled_locators))
    root_constraint = f"delta^4(P - p_1 - p_2 - p_3 - p_4)"
    q_constraints: list[str] = []
    for variable in card.independent_variables:
        q = transfer_by_variable[tuple(variable)]
        transfer = next(item for item in transfers if item["integrated_q"] == q)
        label = "s" + "".join(str(item) for item in variable)
        q_constraints.append(
            f"delta({label} - ({transfer['global_support_solution'].split(' = ', 1)[1]})^2) "
            f"{transfer['positive_energy_transfer']}"
        )
    remaining_constraints = (root_constraint, *q_constraints)
    external_measure_product = " * ".join(f"dmu(p_{particle})" for particle in external_ids)
    ds_product = " * ".join(item["ds_factor_in_final_expression"] for item in sorted(
        transfers, key=lambda row: (tuple(row["recorded_particle_ids"]), row["edge_index"], row["occurrence"])
    ))
    transfer_product = " * ".join(
        f"delta(({item['global_support_solution'].split(' = ', 1)[1]})^2 - "
        f"{next(ds.variable_label for ds in expanded.double_line_factors if ds.edge_index == item['edge_index'] and ds.occurrence == item['occurrence'])}) "
        f"{item['positive_energy_transfer']}"
        for item in transfers
    )
    global_prefactor = normalization["root_global_delta4_prefactor"]
    expression = (
        f"{global_prefactor} * {root_constraint} * ({external_measure_product}) * "
        f"({ds_product}) * ({transfer_product})"
    )
    root_constraint_latex = r"\delta^{(4)}\!\left(P-p_1-p_2-p_3-p_4\right)"
    q_constraints_latex: list[str] = []
    ds_factors_latex: list[str] = []
    shell_transfers_latex: list[str] = []
    for variable in card.independent_variables:
        transfer = next(item for item in transfers if tuple(item["recorded_particle_ids"]) == tuple(variable))
        label = "".join(str(item) for item in variable)
        support_latex = _latex_terms(transfer["global_support_coefficients"])
        q_constraints_latex.append(
            rf"\delta\!\left(s_{{{label}}}-({support_latex})^2\right) "
            + transfer["positive_energy_transfer_latex"]
        )
        ds_factors_latex.append(rf"\mathrm{{d}}s_{{{label}}}")
        shell_transfers_latex.append(
            rf"\delta\!\left(({support_latex})^2-s_{{{label}}}\right) "
            + transfer["positive_energy_transfer_latex"]
        )
    remaining_constraints_latex = (root_constraint_latex, *q_constraints_latex)
    dmu_product_latex = r"\prod_{i=1}^{4}\mathrm{d}\mu(p_i)"
    ds_product_latex = r"\,".join(ds_factors_latex)
    transfer_product_latex = r"\," + r"\,".join(shell_transfers_latex)
    global_prefactor_latex = r"(2\pi)^4" if card.measure_convention == "standard_2pi4" else "1"
    expression_latex = (
        rf"{global_prefactor_latex}\,{root_constraint_latex}\,"
        rf"{dmu_product_latex}\,{ds_product_latex}{transfer_product_latex}"
    )

    factor_rows = [factor_sources[vertex] for vertex in sorted(factor_sources)]
    double_factor_rows = [
        {**item.to_dict(), "expression_latex": _latex_ds_factor(item)}
        for item in expanded.double_line_factors
    ]
    original_expression = expanded.expression
    f0 = {
        "status": expanded.status,
        "measure_convention": expanded.measure_convention,
        "expression": expanded.expression,
        "factors": factor_rows,
        "double_line_factors": double_factor_rows,
        "expanded_f0_expression": original_expression,
        "expression_latex": r"\,".join(
            [_latex_local_factor(item) for item in expanded.factors]
            + [_latex_ds_factor(item) for item in expanded.double_line_factors]
        ),
        "expanded_local_factors_latex": [_latex_local_factor(item) for item in expanded.factors],
        "double_line_factors_latex": [_latex_ds_factor(item) for item in expanded.double_line_factors],
        "input_card_digest": expanded.input_card_digest,
    }
    cancellation = {
        "policy": "D0: retain each unique external particle measure; cancel every repeated internal single-line measure",
        "plan": cancellation_plan.to_dict(),
        "cancelled_measures": canceled_measures,
        "retained_external_measures": retained_external_measures,
        "retained_q_measures": [
            {**q_measure_by_key[key].to_dict(), "expression_latex": _latex_shell_measure(q_measure_by_key[key])}
            for key in sorted(q_measure_by_key)
        ],
        "cancelled_measure_count": len(canceled_measures),
        "retained_external_measure_count": len(retained_external_measures),
        "retained_q_measure_count": len(q_measure_by_key),
        "double_line_measures_preserved": len(expanded.double_line_factors),
        "post_D0_measure_inventory": {
            "external_particle_measures": len(retained_external_measures),
            "independent_q_shell_measures": len(q_measure_by_key),
            "internal_single_copies_cancelled": len(canceled_measures),
        },
    }
    elimination["rule_steps"] = [item.to_dict() for item in _tree_rule_steps(elimination)]
    elimination["root_global_constraint"] = root_constraint
    elimination["root_global_constraint_latex"] = root_constraint_latex
    rule_steps = [
        RuleStep(
            rule_id="FM-F0-expanded-local-delta4",
            before=card_to_local_f0_summary(card),
            after=expanded.expression,
            evidence="build_expanded_f0_expression: explicit independent q measures, shell deltas and theta factors",
        ),
        RuleStep(
            rule_id="FM-D0-occurrence-measure-cancellation",
            before=f"{sum(len(factor.outgoing_measures) for factor in expanded.factors)} outgoing measures plus {d} ds factors",
            after=(f"cancel {len(canceled_measures)} internal single-line copies; retain four external measures and "
                   f"all {len(expanded.double_line_factors)} q-shell/ds pairs"),
            eliminated=tuple(str(item["locator"]) for item in canceled_measures),
            jacobian="1",
            conditions=("D0 keeps the unique external occurrence for each final particle",),
            evidence="occurrence-bound CancellationPlan and expanded outgoing measures",
        ),
    ]
    rule_steps.extend(_tree_rule_steps(elimination))
    rule_steps.extend(support_rule_steps)
    rule_steps.append(RuleStep(
        rule_id="FM-root-global-conservation-and-normalization",
        before=elimination["root_residual"],
        after=expression,
        eliminated=(),
        jacobian="1",
        conditions=("initial vertex retained as root", f"{d} independent q-shell/IKV constraints retained",
                    f"net (2*pi) power {normalization['net_residual_power_including_four_external_measures']}"),
        evidence=f"sum of the {d} executed tree-leaf substitutions with all internal single lines paired",
    ))
    return TreeCovariantTrace(
        schema_version=TREE_COVARIANT_SCHEMA_VERSION,
        status="built",
        scope={
            "n": 4,
            "loop_count": card.loop_count,
            "d": d,
            "vertex_count": len(vertices),
            "double_line_tree_edge_count": len(q_edges),
            "double_line_tree_connected": True,
            "root_vertex": root,
            "root_retained": True,
            "cancellation_policy": "D0",
        },
        f0=f0,
        cancellation=cancellation,
        elimination={
            key: value for key, value in elimination.items() if key != "rule_steps"
        },
        q_shell_transfers=tuple(transfers),
        remaining_constraints=remaining_constraints,
        remaining_constraints_latex=remaining_constraints_latex,
        normalization=normalization,
        expression=expression,
        expression_latex=expression_latex,
        rule_steps=tuple(rule_steps),
        formula_verified=False,
        source_match_status="not_assessed",
        diagnostics=(
            "Covariant tree cancellation and q elimination are built from the supplied graph occurrences.",
            "Direct IKV density, later Jacobians, branches, physical domain, source matching, and full formula verification are outside the scope of this trace.",
        ),
    )


def card_to_local_f0_summary(card: FormulaInputCard) -> str:
    """Readable pre-expansion local-factor product for the first RuleStep."""
    return " * ".join(factor.expression for factor in card.factors)


__all__ = [
    "TREE_COVARIANT_SCHEMA_VERSION",
    "TreeCovariantTrace",
    "build_tree_covariant_trace",
]
