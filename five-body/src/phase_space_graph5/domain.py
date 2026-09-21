"""Five-body domain records shared by the structure and enumeration layers.

The data model follows the four-body G01 vocabulary.  It is deliberately
agnostic about invariant independence: a topology and a labeling are field
objects, while structural validity is decided by :mod:`structure`.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import json
from typing import Any, Iterable, Mapping, Sequence


class _StringEnum(str, Enum):
    def __str__(self) -> str:
        return self.value


class EdgeType(_StringEnum):
    SINGLE = "single"
    DOUBLE = "double"


class CountMode(_StringEnum):
    FIXED_LABELS = "fixed_labels"
    PARTICLE_RELABELING = "particle_relabeling"


class RunStatus(_StringEnum):
    COMPLETE = "complete"
    INCOMPLETE = "incomplete"
    FAILED = "failed"


class _Severity(_StringEnum):
    ERROR = "error"
    WARNING = "warning"


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    code: str
    message: str
    path: str = ""
    severity: str = _Severity.ERROR.value

    def __post_init__(self) -> None:
        if not isinstance(self.code, str) or not self.code:
            raise TypeError("ValidationIssue.code must be a non-empty string")
        if not isinstance(self.message, str) or not self.message:
            raise TypeError("ValidationIssue.message must be a non-empty string")
        if not isinstance(self.path, str):
            raise TypeError("ValidationIssue.path must be a string")
        if self.severity not in {item.value for item in _Severity}:
            raise ValueError(f"Unsupported validation severity: {self.severity!r}")

    def to_dict(self) -> dict[str, str]:
        result = {"code": self.code, "message": self.message, "severity": self.severity}
        if self.path:
            result["path"] = self.path
        return result


@dataclass(frozen=True, slots=True)
class ValidationReport:
    issues: tuple[ValidationIssue, ...] = ()

    @property
    def valid(self) -> bool:
        return not any(issue.severity == _Severity.ERROR.value for issue in self.issues)

    def error_codes(self) -> tuple[str, ...]:
        return tuple(issue.code for issue in self.issues if issue.severity == _Severity.ERROR.value)

    def to_dict(self) -> dict[str, Any]:
        return {"valid": self.valid, "issues": [issue.to_dict() for issue in self.issues]}


def issue_codes(report: ValidationReport | Iterable[ValidationIssue]) -> tuple[str, ...]:
    issues = report.issues if isinstance(report, ValidationReport) else tuple(report)
    return tuple(issue.code for issue in issues)


def _is_bool(value: Any) -> bool:
    return isinstance(value, bool)


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not _is_bool(value)


def _edge_type(value: EdgeType | str) -> EdgeType:
    if isinstance(value, EdgeType):
        return value
    try:
        return EdgeType(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Unknown edge type: {value!r}") from exc


def _count_mode(value: CountMode | str) -> CountMode:
    if isinstance(value, CountMode):
        return value
    try:
        return CountMode(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Unknown count mode: {value!r}") from exc


# Keep the source module's private helper names available to sibling migration
# modules that were copied from the four-body implementation.
_coerce_edge_type = _edge_type
_coerce_count_mode = _count_mode


@dataclass(frozen=True, slots=True)
class DiagramSpec:
    n: int = 5
    loop_count: int = 0
    count_mode: CountMode = CountMode.FIXED_LABELS

    def __post_init__(self) -> None:
        if not _is_int(self.n):
            raise TypeError("DiagramSpec.n must be an integer")
        if not _is_int(self.loop_count):
            raise TypeError("DiagramSpec.loop_count must be an integer")
        object.__setattr__(self, "count_mode", _count_mode(self.count_mode))

    @property
    def supported(self) -> bool:
        return self.n == 5 and 0 <= self.loop_count <= 5

    def validate_fields(self) -> ValidationReport:
        issues: list[ValidationIssue] = []
        if self.n != 5:
            issues.append(ValidationIssue("unsupported_n", "The five-body package accepts n=5.", "n"))
        if self.loop_count < 0:
            issues.append(ValidationIssue("negative_loop_count", "loop_count must be non-negative.", "loop_count"))
        elif self.n == 5 and self.loop_count > 5:
            issues.append(ValidationIssue("unsupported_loop_count", "For n=5, loop_count must be between 0 and 5.", "loop_count"))
        return ValidationReport(tuple(issues))

    def to_dict(self) -> dict[str, Any]:
        return {"n": self.n, "loop_count": self.loop_count, "count_mode": self.count_mode.value}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DiagramSpec":
        if not isinstance(data, Mapping):
            raise TypeError("DiagramSpec data must be a mapping")
        return cls(data.get("n"), data.get("loop_count"), data.get("count_mode", CountMode.FIXED_LABELS.value))


@dataclass(frozen=True, slots=True)
class InternalEdge:
    source: int
    target: int
    edge_type: EdgeType = EdgeType.SINGLE
    multiplicity: int = 1

    def __post_init__(self) -> None:
        if not _is_int(self.source) or not _is_int(self.target):
            raise TypeError("InternalEdge endpoints must be integers")
        if not _is_int(self.multiplicity):
            raise TypeError("InternalEdge.multiplicity must be an integer")
        object.__setattr__(self, "edge_type", _edge_type(self.edge_type))

    @property
    def expanded_count(self) -> int:
        return max(self.multiplicity, 0)

    def to_dict(self) -> dict[str, Any]:
        return {"source": self.source, "target": self.target, "edge_type": self.edge_type.value, "multiplicity": self.multiplicity}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "InternalEdge":
        if not isinstance(data, Mapping):
            raise TypeError("InternalEdge data must be a mapping")
        return cls(data.get("source"), data.get("target"), data.get("edge_type"), data.get("multiplicity", 1))


@dataclass(frozen=True, slots=True)
class Topology:
    n: int
    vertices: tuple[int, ...]
    edges: tuple[InternalEdge, ...]
    initial_vertex: int
    final_slots: tuple[tuple[int, int], ...]

    def __init__(self, n: int, vertices: Iterable[int], edges: Iterable[InternalEdge], initial_vertex: int, final_slots: Mapping[int, int] | Iterable[tuple[int, int]]) -> None:
        if not _is_int(n):
            raise TypeError("Topology.n must be an integer")
        raw_vertices = tuple(vertices)
        vertex_tuple = tuple(sorted(raw_vertices)) if all(_is_int(v) for v in raw_vertices) else raw_vertices
        edge_tuple = tuple(edges)
        if not all(isinstance(edge, InternalEdge) for edge in edge_tuple):
            raise TypeError("Topology.edges must contain InternalEdge objects")
        edge_tuple = tuple(sorted(edge_tuple, key=lambda e: (e.source, e.target, e.edge_type.value, e.multiplicity)))
        slots = tuple(final_slots.items()) if isinstance(final_slots, Mapping) else tuple(final_slots)
        object.__setattr__(self, "n", n)
        object.__setattr__(self, "vertices", vertex_tuple)
        object.__setattr__(self, "edges", edge_tuple)
        object.__setattr__(self, "initial_vertex", initial_vertex)
        object.__setattr__(self, "final_slots", tuple(sorted(slots, key=_stable_pair_sort_key)))

    def slot_count(self, vertex: int) -> int | None:
        for pair in self.final_slots:
            if isinstance(pair, (tuple, list)) and len(pair) == 2 and pair[0] == vertex:
                return pair[1]
        return None

    @property
    def edge_count(self) -> int:
        return sum(edge.expanded_count for edge in self.edges)

    def validate_fields(self) -> ValidationReport:
        issues: list[ValidationIssue] = []
        if not _is_int(self.n):
            issues.append(ValidationIssue("invalid_n_type", "n must be an integer.", "n"))
        elif self.n < 1:
            issues.append(ValidationIssue("invalid_n", "n must be positive.", "n"))
        vertices = tuple(self.vertices)
        if not all(_is_int(v) for v in vertices):
            issues.append(ValidationIssue("invalid_vertex_id_type", "Every vertex identifier must be an integer.", "vertices"))
        duplicates = _duplicates(vertices)
        if duplicates:
            issues.append(ValidationIssue("duplicate_vertex_id", f"Vertex identifiers must be unique: {duplicates!r}.", "vertices"))
        vertex_set = {v for v in vertices if _is_int(v)}
        if not _is_int(self.initial_vertex) or self.initial_vertex not in vertex_set:
            issues.append(ValidationIssue("unknown_initial_vertex", "initial_vertex must refer to one of vertices.", "initial_vertex"))
        for index, edge in enumerate(self.edges):
            path = f"edges[{index}]"
            if edge.source not in vertex_set:
                issues.append(ValidationIssue("unknown_edge_vertex", "Edge source must refer to one of vertices.", f"{path}.source"))
            if edge.target not in vertex_set:
                issues.append(ValidationIssue("unknown_edge_vertex", "Edge target must refer to one of vertices.", f"{path}.target"))
            if edge.multiplicity < 0:
                issues.append(ValidationIssue("negative_edge_multiplicity", "Edge multiplicity must be non-negative.", f"{path}.multiplicity"))
        slot_vertices = tuple(pair[0] for pair in self.final_slots if isinstance(pair, (tuple, list)) and len(pair) == 2 and _is_int(pair[0]))
        duplicates = _duplicates(slot_vertices)
        if duplicates:
            issues.append(ValidationIssue("duplicate_final_slot_vertex", f"final_slots contains duplicate vertices: {duplicates!r}.", "final_slots"))
        if set(slot_vertices) != vertex_set:
            missing = sorted(vertex_set - set(slot_vertices))
            unknown = sorted(set(slot_vertices) - vertex_set)
            if missing:
                issues.append(ValidationIssue("missing_final_slot_vertex", f"Every vertex needs a final-slot count; missing {missing!r}.", "final_slots"))
            if unknown:
                issues.append(ValidationIssue("unknown_final_slot_vertex", f"final_slots contains unknown vertices: {unknown!r}.", "final_slots"))
        total_slots = 0
        for index, pair in enumerate(self.final_slots):
            path = f"final_slots[{index}]"
            if not isinstance(pair, (tuple, list)) or len(pair) != 2:
                issues.append(ValidationIssue("invalid_final_slot_entry", "Final-slot entries must be (integer vertex, integer count) pairs.", path))
                continue
            vertex, count = pair
            if not _is_int(vertex) or not _is_int(count):
                issues.append(ValidationIssue("invalid_final_slot_type", "Final-slot entries must be (integer vertex, integer count).", path))
                continue
            if count < 0:
                issues.append(ValidationIssue("negative_final_slot_count", "Final-slot count must be non-negative.", f"{path}.count"))
            else:
                total_slots += count
        if _is_int(self.n) and total_slots != self.n:
            issues.append(ValidationIssue("final_slot_count_mismatch", f"The final-slot total is {total_slots}, but n={self.n}.", "final_slots"))
        return ValidationReport(tuple(issues))

    def to_dict(self) -> dict[str, Any]:
        return {"n": self.n, "vertices": list(self.vertices), "edges": [e.to_dict() for e in self.edges], "initial_vertex": self.initial_vertex, "final_slots": [{"vertex": v, "count": c} for v, c in self.final_slots]}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Topology":
        if not isinstance(data, Mapping):
            raise TypeError("Topology data must be a mapping")
        raw = data.get("final_slots", ())
        slots = raw if isinstance(raw, Mapping) else tuple((item["vertex"], item["count"]) if isinstance(item, Mapping) else tuple(item) for item in raw)
        return cls(data.get("n"), tuple(data.get("vertices", ())), tuple(InternalEdge.from_dict(e) for e in data.get("edges", ())), data.get("initial_vertex"), slots)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_json(cls, text: str) -> "Topology":
        if not isinstance(text, str):
            raise TypeError("Topology JSON must be a string")
        return cls.from_dict(json.loads(text))


@dataclass(frozen=True, slots=True)
class FinalParticleLabel:
    vertex: int
    slot: int
    particle_id: int

    def to_dict(self) -> dict[str, int]:
        return {"vertex": self.vertex, "slot": self.slot, "particle_id": self.particle_id}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "FinalParticleLabel":
        return cls(data.get("vertex"), data.get("slot"), data.get("particle_id"))


@dataclass(frozen=True, slots=True)
class SingleLineLabel:
    edge_index: int
    occurrence: int
    particle_id: int

    def to_dict(self) -> dict[str, int]:
        return {"edge_index": self.edge_index, "occurrence": self.occurrence, "particle_id": self.particle_id}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SingleLineLabel":
        return cls(data.get("edge_index"), data.get("occurrence", 0), data.get("particle_id"))


@dataclass(frozen=True, slots=True)
class DoubleLineLabel:
    edge_index: int
    occurrence: int
    particle_ids: tuple[int, ...]

    def __init__(self, edge_index: int, occurrence: int, particle_ids: Iterable[int]) -> None:
        object.__setattr__(self, "edge_index", edge_index)
        object.__setattr__(self, "occurrence", occurrence)
        object.__setattr__(self, "particle_ids", tuple(particle_ids))

    def to_dict(self) -> dict[str, Any]:
        return {"edge_index": self.edge_index, "occurrence": self.occurrence, "particle_ids": list(self.particle_ids)}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DoubleLineLabel":
        return cls(data.get("edge_index"), data.get("occurrence", 0), data.get("particle_ids", ()))


@dataclass(frozen=True, slots=True)
class Labeling:
    final_particles: tuple[FinalParticleLabel, ...] = ()
    single_lines: tuple[SingleLineLabel, ...] = ()
    double_lines: tuple[DoubleLineLabel, ...] = ()

    def __init__(self, final_particles: Iterable[FinalParticleLabel] = (), single_lines: Iterable[SingleLineLabel] = (), double_lines: Iterable[DoubleLineLabel] = ()) -> None:
        final = tuple(final_particles)
        single = tuple(single_lines)
        double = tuple(double_lines)
        if not all(isinstance(x, FinalParticleLabel) for x in final):
            raise TypeError("final_particles must contain FinalParticleLabel objects")
        if not all(isinstance(x, SingleLineLabel) for x in single):
            raise TypeError("single_lines must contain SingleLineLabel objects")
        if not all(isinstance(x, DoubleLineLabel) for x in double):
            raise TypeError("double_lines must contain DoubleLineLabel objects")
        object.__setattr__(self, "final_particles", tuple(sorted(final, key=lambda x: (x.vertex, x.slot))))
        object.__setattr__(self, "single_lines", tuple(sorted(single, key=lambda x: (x.edge_index, x.occurrence))))
        object.__setattr__(self, "double_lines", tuple(sorted(double, key=lambda x: (x.edge_index, x.occurrence))))

    def validate_fields(self, topology: Topology) -> ValidationReport:
        issues: list[ValidationIssue] = []
        vertices = set(topology.vertices)
        final_keys: list[tuple[int, int]] = []
        particles: list[int] = []
        for index, label in enumerate(self.final_particles):
            path = f"final_particles[{index}]"
            final_keys.append((label.vertex, label.slot))
            particles.append(label.particle_id)
            if not _is_int(label.vertex) or label.vertex not in vertices:
                issues.append(ValidationIssue("unknown_label_vertex", "Final label vertex is unknown.", f"{path}.vertex"))
            count = topology.slot_count(label.vertex)
            if count is not None and (not _is_int(label.slot) or label.slot < 0 or label.slot >= count):
                issues.append(ValidationIssue("slot_out_of_range", "Final label slot is outside its vertex slots.", f"{path}.slot"))
            if not _is_int(label.particle_id) or not 1 <= label.particle_id <= topology.n:
                issues.append(ValidationIssue("particle_id_out_of_range", "Particle IDs must be in 1..n.", f"{path}.particle_id"))
        for duplicate in _duplicates(final_keys):
            issues.append(ValidationIssue("duplicate_final_slot_label", f"Final slot is labeled more than once: {duplicate!r}.", "final_particles"))
        if len(particles) != len(_unique_values(particles)):
            issues.append(ValidationIssue("duplicate_final_particle_label", "Each final-state particle must be assigned to one slot.", "final_particles"))
        expected = set(range(1, topology.n + 1)) if _is_int(topology.n) else set()
        if {p for p in particles if _is_int(p)} != expected:
            issues.append(ValidationIssue("final_particle_bijection_mismatch", "Final labels must contain every particle ID exactly once.", "final_particles"))
        for name, collection, expected_type in (("single_lines", self.single_lines, EdgeType.SINGLE), ("double_lines", self.double_lines, EdgeType.DOUBLE)):
            seen: list[tuple[int, int]] = []
            for index, label in enumerate(collection):
                path = f"{name}[{index}]"
                edge_index, occurrence = label.edge_index, label.occurrence
                seen.append((edge_index, occurrence))
                if not _is_int(edge_index) or not 0 <= edge_index < len(topology.edges):
                    issues.append(ValidationIssue("edge_index_out_of_range", "Line label edge index is unknown.", f"{path}.edge_index"))
                    continue
                edge = topology.edges[edge_index]
                if edge.edge_type is not expected_type:
                    issues.append(ValidationIssue("line_type_mismatch", f"Expected a {expected_type.value} edge.", path))
                if not _is_int(occurrence) or occurrence < 0 or occurrence >= edge.multiplicity:
                    issues.append(ValidationIssue("edge_occurrence_out_of_range", "Line occurrence is outside edge multiplicity.", f"{path}.occurrence"))
                values = (label.particle_id,) if name == "single_lines" else label.particle_ids
                if name == "double_lines" and not values:
                    issues.append(ValidationIssue("empty_composite_subset", "A double-line subset cannot be empty.", f"{path}.particle_ids"))
                if any(not _is_int(p) or not 1 <= p <= topology.n for p in values):
                    issues.append(ValidationIssue("particle_id_out_of_range", "Composite particle IDs must be in 1..n.", path))
                if name == "double_lines" and len(values) != len(_unique_values(values)):
                    issues.append(ValidationIssue("duplicate_composite_particle", "A composite subset cannot repeat a particle.", f"{path}.particle_ids"))
            for duplicate in _duplicates(seen):
                issues.append(ValidationIssue("duplicate_line_label", f"Line occurrence is labeled more than once: {duplicate!r}.", name))
        return ValidationReport(tuple(issues))

    def to_dict(self) -> dict[str, Any]:
        return {"final_particles": [x.to_dict() for x in self.final_particles], "single_lines": [x.to_dict() for x in self.single_lines], "double_lines": [x.to_dict() for x in self.double_lines]}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Labeling":
        if not isinstance(data, Mapping):
            raise TypeError("Labeling data must be a mapping")
        return cls(tuple(FinalParticleLabel.from_dict(x) for x in data.get("final_particles", ())), tuple(SingleLineLabel.from_dict(x) for x in data.get("single_lines", ())), tuple(DoubleLineLabel.from_dict(x) for x in data.get("double_lines", ())))

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_json(cls, text: str) -> "Labeling":
        if not isinstance(text, str):
            raise TypeError("Labeling JSON must be a string")
        return cls.from_dict(json.loads(text))


def _duplicates(values: Sequence[Any]) -> tuple[Any, ...]:
    seen: list[Any] = []
    duplicates: list[Any] = []
    for value in values:
        if value in seen and value not in duplicates:
            duplicates.append(value)
        seen.append(value)
    return tuple(duplicates)


def _unique_values(values: Iterable[Any]) -> tuple[Any, ...]:
    result: list[Any] = []
    for value in values:
        if value not in result:
            result.append(value)
    return tuple(result)


def _stable_pair_sort_key(pair: Any) -> tuple[str, str]:
    if isinstance(pair, (tuple, list)) and pair and _is_int(pair[0]):
        return ("0", f"{pair[0]:020d}")
    return ("1", repr(pair))
