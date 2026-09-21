"""Immutable domain objects and field validation for graphic-method diagrams.

This module owns the G01 vocabulary and field validation.  G03 and G04 consume
its ``Topology`` and ``Labeling`` objects for graph-rule checks and canonical
encoding; this module itself still does not decide whether a graph is a valid
complete-expansion diagram.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import json
from typing import Any, Iterable, Mapping, Sequence


class _StringEnum(str, Enum):
    """Enum whose JSON representation is its stable string value."""

    def __str__(self) -> str:
        return self.value


class EdgeType(_StringEnum):
    """The two line types in a phase-space graph."""

    SINGLE = "single"
    DOUBLE = "double"


class CountMode(_StringEnum):
    """Particle-equivalence convention used by a later counting run."""

    FIXED_LABELS = "fixed_labels"
    PARTICLE_RELABELING = "particle_relabeling"


class RunStatus(_StringEnum):
    """Status values used by later enumeration/reporting code."""

    COMPLETE = "complete"
    INCOMPLETE = "incomplete"
    FAILED = "failed"


class _Severity(_StringEnum):
    ERROR = "error"
    WARNING = "warning"


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    """One field-validation result with a stable machine-readable code."""

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
        result = {
            "code": self.code,
            "message": self.message,
            "severity": self.severity,
        }
        if self.path:
            result["path"] = self.path
        return result


@dataclass(frozen=True, slots=True)
class ValidationReport:
    """Immutable collection of issues returned by field-level validation."""

    issues: tuple[ValidationIssue, ...] = ()

    @property
    def valid(self) -> bool:
        return not any(issue.severity == _Severity.ERROR.value for issue in self.issues)

    def error_codes(self) -> tuple[str, ...]:
        return tuple(
            issue.code for issue in self.issues if issue.severity == _Severity.ERROR.value
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "issues": [issue.to_dict() for issue in self.issues],
        }


def issue_codes(report: ValidationReport | Iterable[ValidationIssue]) -> tuple[str, ...]:
    """Return issue codes without making callers depend on internal containers."""

    issues = report.issues if isinstance(report, ValidationReport) else tuple(report)
    return tuple(issue.code for issue in issues)


def _is_bool(value: Any) -> bool:
    # bool is an int subclass, but accepting True as a vertex, count, or mass
    # would make malformed input look valid.
    return isinstance(value, bool)


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not _is_bool(value)


def _coerce_edge_type(value: EdgeType | str) -> EdgeType:
    if isinstance(value, EdgeType):
        return value
    try:
        return EdgeType(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Unknown edge type: {value!r}") from exc


def _coerce_count_mode(value: CountMode | str) -> CountMode:
    if isinstance(value, CountMode):
        return value
    try:
        return CountMode(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Unknown count mode: {value!r}") from exc


@dataclass(frozen=True, slots=True)
class DiagramSpec:
    """Scope of one enumeration run.

    G01 supports four-body runs and the three-body low-dimensional fixtures.
    The loop bound follows the complete-expansion relation, but the class does
    not validate graph topology itself.
    """

    n: int = 4
    loop_count: int = 0
    count_mode: CountMode = CountMode.FIXED_LABELS

    def __post_init__(self) -> None:
        if not _is_int(self.n):
            raise TypeError("DiagramSpec.n must be an integer")
        if not _is_int(self.loop_count):
            raise TypeError("DiagramSpec.loop_count must be an integer")
        object.__setattr__(self, "count_mode", _coerce_count_mode(self.count_mode))

    @property
    def supported(self) -> bool:
        return self.n in (3, 4) and 0 <= self.loop_count <= 2 * self.n - 5

    def validate_fields(self) -> ValidationReport:
        issues: list[ValidationIssue] = []
        if self.n < 3:
            issues.append(
                ValidationIssue(
                    "unsupported_n",
                    "The current G01 model supports n=3 fixtures and n=4 runs.",
                    "n",
                )
            )
        elif self.n > 4:
            issues.append(
                ValidationIssue(
                    "unsupported_n",
                    "General n-body enumeration is not implemented in G01.",
                    "n",
                )
            )
        if self.loop_count < 0:
            issues.append(
                ValidationIssue(
                    "negative_loop_count",
                    "loop_count must be non-negative.",
                    "loop_count",
                )
            )
        elif self.n in (3, 4) and self.loop_count > 2 * self.n - 5:
            issues.append(
                ValidationIssue(
                    "unsupported_loop_count",
                    f"For n={self.n}, loop_count must be between 0 and {2 * self.n - 5}.",
                    "loop_count",
                )
            )
        return ValidationReport(tuple(issues))

    def to_dict(self) -> dict[str, Any]:
        return {
            "n": self.n,
            "loop_count": self.loop_count,
            "count_mode": self.count_mode.value,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DiagramSpec":
        if not isinstance(data, Mapping):
            raise TypeError("DiagramSpec data must be a mapping")
        return cls(
            n=data.get("n"),
            loop_count=data.get("loop_count"),
            count_mode=data.get("count_mode", CountMode.FIXED_LABELS.value),
        )


@dataclass(frozen=True, slots=True)
class InternalEdge:
    """A directed internal edge; multiplicity preserves parallel edges."""

    source: int
    target: int
    edge_type: EdgeType = EdgeType.SINGLE
    multiplicity: int = 1

    def __post_init__(self) -> None:
        if not _is_int(self.source) or not _is_int(self.target):
            raise TypeError("InternalEdge endpoints must be integers")
        if not _is_int(self.multiplicity):
            raise TypeError("InternalEdge.multiplicity must be an integer")
        object.__setattr__(self, "edge_type", _coerce_edge_type(self.edge_type))

    @property
    def expanded_count(self) -> int:
        """Number of graph edges represented by this compact record."""

        return max(self.multiplicity, 0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "edge_type": self.edge_type.value,
            "multiplicity": self.multiplicity,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "InternalEdge":
        if not isinstance(data, Mapping):
            raise TypeError("InternalEdge data must be a mapping")
        return cls(
            source=data.get("source"),
            target=data.get("target"),
            edge_type=data.get("edge_type"),
            multiplicity=data.get("multiplicity", 1),
        )


@dataclass(frozen=True, slots=True)
class Topology:
    """A field-level graph description without particle labels.

    ``final_slots`` is stored as sorted ``(vertex_id, slot_count)`` pairs.
    Slot positions are zero-based and are local to a vertex.  The constructor
    accepts a mapping for convenience and normalizes its order so valid objects
    have deterministic dictionary/JSON representations.
    """

    n: int
    vertices: tuple[int, ...]
    edges: tuple[InternalEdge, ...]
    initial_vertex: int
    final_slots: tuple[tuple[int, int], ...]

    def __init__(
        self,
        n: int,
        vertices: Iterable[int],
        edges: Iterable[InternalEdge],
        initial_vertex: int,
        final_slots: Mapping[int, int] | Iterable[tuple[int, int]],
    ) -> None:
        if not _is_int(n):
            raise TypeError("Topology.n must be an integer")
        raw_vertices = tuple(vertices)
        # Valid vertex IDs are integers.  Sort those for a stable representation
        # while leaving malformed mixed-type input constructible for validation.
        vertex_tuple = (
            tuple(sorted(raw_vertices))
            if all(_is_int(vertex) for vertex in raw_vertices)
            else raw_vertices
        )
        edge_tuple = tuple(edges)
        slot_pairs = tuple(final_slots.items()) if isinstance(final_slots, Mapping) else tuple(final_slots)
        if not all(isinstance(edge, InternalEdge) for edge in edge_tuple):
            raise TypeError("Topology.edges must contain InternalEdge objects")
        # Sorting is a representation choice only.  Duplicate edge records are
        # retained because parallel edges are meaningful to later topology code.
        edge_tuple = tuple(
            sorted(
                edge_tuple,
                key=lambda edge: (
                    edge.source,
                    edge.target,
                    edge.edge_type.value,
                    edge.multiplicity,
                ),
            )
        )
        slot_pairs = tuple(sorted(slot_pairs, key=_stable_pair_sort_key))
        object.__setattr__(self, "n", n)
        object.__setattr__(self, "vertices", vertex_tuple)
        object.__setattr__(self, "edges", edge_tuple)
        object.__setattr__(self, "initial_vertex", initial_vertex)
        object.__setattr__(self, "final_slots", slot_pairs)

    def slot_count(self, vertex: int) -> int | None:
        for pair in self.final_slots:
            if not isinstance(pair, (tuple, list)) or len(pair) != 2:
                continue
            slot_vertex, count = pair
            if slot_vertex == vertex:
                return count
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

        if not all(_is_int(vertex) for vertex in self.vertices):
            issues.append(
                ValidationIssue(
                    "invalid_vertex_id_type",
                    "Every vertex identifier must be an integer.",
                    "vertices",
                )
            )
        duplicate_vertices = _duplicates(self.vertices)
        if duplicate_vertices:
            issues.append(
                ValidationIssue(
                    "duplicate_vertex_id",
                    f"Vertex identifiers must be unique: {duplicate_vertices!r}.",
                    "vertices",
                )
            )
        vertex_set = {vertex for vertex in self.vertices if _is_int(vertex)}
        if not _is_int(self.initial_vertex) or self.initial_vertex not in vertex_set:
            issues.append(
                ValidationIssue(
                    "unknown_initial_vertex",
                    "initial_vertex must refer to one of vertices.",
                    "initial_vertex",
                )
            )

        for index, edge in enumerate(self.edges):
            path = f"edges[{index}]"
            if edge.source not in vertex_set:
                issues.append(
                    ValidationIssue(
                        "unknown_edge_vertex",
                        "Edge source must refer to one of vertices.",
                        f"{path}.source",
                    )
                )
            if edge.target not in vertex_set:
                issues.append(
                    ValidationIssue(
                        "unknown_edge_vertex",
                        "Edge target must refer to one of vertices.",
                        f"{path}.target",
                    )
                )
            if edge.multiplicity < 0:
                issues.append(
                    ValidationIssue(
                        "negative_edge_multiplicity",
                        "Edge multiplicity must be non-negative.",
                        f"{path}.multiplicity",
                    )
                )

        slot_vertices = tuple(
            pair[0]
            for pair in self.final_slots
            if isinstance(pair, (tuple, list)) and len(pair) == 2 and _is_int(pair[0])
        )
        duplicate_slot_vertices = _duplicates(slot_vertices)
        if duplicate_slot_vertices:
            issues.append(
                ValidationIssue(
                    "duplicate_final_slot_vertex",
                    f"final_slots contains duplicate vertices: {duplicate_slot_vertices!r}.",
                    "final_slots",
                )
            )
        if set(slot_vertices) != vertex_set:
            missing = sorted(vertex_set - set(slot_vertices))
            unknown = sorted(set(slot_vertices) - vertex_set)
            if missing:
                issues.append(
                    ValidationIssue(
                        "missing_final_slot_vertex",
                        f"Every vertex needs a final-slot count; missing {missing!r}.",
                        "final_slots",
                    )
                )
            if unknown:
                issues.append(
                    ValidationIssue(
                        "unknown_final_slot_vertex",
                        f"final_slots contains unknown vertices: {unknown!r}.",
                        "final_slots",
                    )
                )
        total_slots = 0
        for index, pair in enumerate(self.final_slots):
            path = f"final_slots[{index}]"
            if not isinstance(pair, (tuple, list)) or len(pair) != 2:
                issues.append(
                    ValidationIssue(
                        "invalid_final_slot_entry",
                        "Final-slot entries must be (integer vertex, integer count) pairs.",
                        path,
                    )
                )
                continue
            vertex, count = pair
            if not _is_int(vertex) or not _is_int(count):
                issues.append(
                    ValidationIssue(
                        "invalid_final_slot_type",
                        "Final-slot entries must be (integer vertex, integer count).",
                        path,
                    )
                )
                continue
            if count < 0:
                issues.append(
                    ValidationIssue(
                        "negative_final_slot_count",
                        "Final-slot count must be non-negative.",
                        f"{path}.count",
                    )
                )
            else:
                total_slots += count
        if _is_int(self.n) and total_slots != self.n:
            issues.append(
                ValidationIssue(
                    "final_slot_count_mismatch",
                    f"The final-slot total is {total_slots}, but n={self.n}.",
                    "final_slots",
                )
            )
        return ValidationReport(tuple(issues))

    def to_dict(self) -> dict[str, Any]:
        return {
            "n": self.n,
            "vertices": list(self.vertices),
            "edges": [edge.to_dict() for edge in self.edges],
            "initial_vertex": self.initial_vertex,
            "final_slots": [
                {"vertex": vertex, "count": count} for vertex, count in self.final_slots
            ],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Topology":
        if not isinstance(data, Mapping):
            raise TypeError("Topology data must be a mapping")
        raw_slots = data.get("final_slots", ())
        if isinstance(raw_slots, Mapping):
            slots: Mapping[int, int] | Iterable[tuple[int, int]] = raw_slots
        else:
            slots = tuple(
                (entry["vertex"], entry["count"])
                if isinstance(entry, Mapping)
                else tuple(entry)
                for entry in raw_slots
            )
        return cls(
            n=data.get("n"),
            vertices=tuple(data.get("vertices", ())),
            edges=tuple(InternalEdge.from_dict(edge) for edge in data.get("edges", ())),
            initial_vertex=data.get("initial_vertex"),
            final_slots=slots,
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_json(cls, text: str) -> "Topology":
        if not isinstance(text, str):
            raise TypeError("Topology JSON must be a string")
        data = json.loads(text)
        return cls.from_dict(data)


@dataclass(frozen=True, slots=True)
class FinalParticleLabel:
    """Assignment of one physical final-state particle to a local slot."""

    vertex: int
    slot: int
    particle_id: int

    def to_dict(self) -> dict[str, int]:
        return {
            "vertex": self.vertex,
            "slot": self.slot,
            "particle_id": self.particle_id,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "FinalParticleLabel":
        return cls(data.get("vertex"), data.get("slot"), data.get("particle_id"))


@dataclass(frozen=True, slots=True)
class SingleLineLabel:
    """Particle label for one occurrence of an internal single edge."""

    edge_index: int
    occurrence: int
    particle_id: int

    def to_dict(self) -> dict[str, int]:
        return {
            "edge_index": self.edge_index,
            "occurrence": self.occurrence,
            "particle_id": self.particle_id,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SingleLineLabel":
        return cls(data.get("edge_index"), data.get("occurrence", 0), data.get("particle_id"))


@dataclass(frozen=True, slots=True)
class DoubleLineLabel:
    """Particle subset carried by one internal double edge occurrence."""

    edge_index: int
    occurrence: int
    particle_ids: tuple[int, ...]

    def __init__(self, edge_index: int, occurrence: int, particle_ids: Iterable[int]) -> None:
        object.__setattr__(self, "edge_index", edge_index)
        object.__setattr__(self, "occurrence", occurrence)
        object.__setattr__(self, "particle_ids", tuple(particle_ids))

    def to_dict(self) -> dict[str, Any]:
        return {
            "edge_index": self.edge_index,
            "occurrence": self.occurrence,
            "particle_ids": list(self.particle_ids),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DoubleLineLabel":
        return cls(data.get("edge_index"), data.get("occurrence", 0), data.get("particle_ids", ()))


@dataclass(frozen=True, slots=True)
class Labeling:
    """Particle labels attached to a topology.

    Final-state labels must form a bijection between slots and particles.
    Repeated particle IDs on distinct internal single-line occurrences are
    intentionally allowed: they encode the repeated measure that later G06
    will cancel with dashed-line factors.
    """

    final_particles: tuple[FinalParticleLabel, ...] = ()
    single_lines: tuple[SingleLineLabel, ...] = ()
    double_lines: tuple[DoubleLineLabel, ...] = ()

    def __init__(
        self,
        final_particles: Iterable[FinalParticleLabel] = (),
        single_lines: Iterable[SingleLineLabel] = (),
        double_lines: Iterable[DoubleLineLabel] = (),
    ) -> None:
        final_tuple = tuple(final_particles)
        single_tuple = tuple(single_lines)
        double_tuple = tuple(double_lines)
        if not all(isinstance(item, FinalParticleLabel) for item in final_tuple):
            raise TypeError("final_particles must contain FinalParticleLabel objects")
        if not all(isinstance(item, SingleLineLabel) for item in single_tuple):
            raise TypeError("single_lines must contain SingleLineLabel objects")
        if not all(isinstance(item, DoubleLineLabel) for item in double_tuple):
            raise TypeError("double_lines must contain DoubleLineLabel objects")
        object.__setattr__(self, "final_particles", tuple(sorted(final_tuple, key=lambda item: (item.vertex, item.slot))))
        object.__setattr__(self, "single_lines", tuple(sorted(single_tuple, key=lambda item: (item.edge_index, item.occurrence))))
        object.__setattr__(self, "double_lines", tuple(sorted(double_tuple, key=lambda item: (item.edge_index, item.occurrence))))

    def validate_fields(self, topology: Topology) -> ValidationReport:
        issues: list[ValidationIssue] = []
        vertex_set = set(topology.vertices)
        final_keys: list[tuple[int, int]] = []
        final_particles: list[int] = []
        for index, label in enumerate(self.final_particles):
            path = f"final_particles[{index}]"
            final_keys.append((label.vertex, label.slot))
            final_particles.append(label.particle_id)
            if not _is_int(label.vertex) or label.vertex not in vertex_set:
                issues.append(ValidationIssue("unknown_label_vertex", "Final label vertex is unknown.", f"{path}.vertex"))
            slot_count = topology.slot_count(label.vertex)
            if slot_count is not None and (not _is_int(label.slot) or label.slot < 0 or label.slot >= slot_count):
                issues.append(ValidationIssue("slot_out_of_range", "Final label slot is outside its vertex slots.", f"{path}.slot"))
            if not _is_int(label.particle_id) or not 1 <= label.particle_id <= topology.n:
                issues.append(ValidationIssue("particle_id_out_of_range", "Particle IDs must be in 1..n.", f"{path}.particle_id"))
        for duplicate in _duplicates(final_keys):
            issues.append(ValidationIssue("duplicate_final_slot_label", f"Final slot is labeled more than once: {duplicate!r}.", "final_particles"))
        if len(final_particles) != len(_unique_values(final_particles)):
            issues.append(ValidationIssue("duplicate_final_particle_label", "Each final-state particle must be assigned to one slot.", "final_particles"))
        expected_particles = set(range(1, topology.n + 1)) if _is_int(topology.n) else set()
        if set(particle for particle in final_particles if _is_int(particle)) != expected_particles:
            issues.append(ValidationIssue("final_particle_bijection_mismatch", "Final labels must contain every particle ID exactly once.", "final_particles"))

        for collection_name, collection, expected_type in (
            ("single_lines", self.single_lines, EdgeType.SINGLE),
            ("double_lines", self.double_lines, EdgeType.DOUBLE),
        ):
            seen_refs: list[tuple[int, int]] = []
            for index, label in enumerate(collection):
                path = f"{collection_name}[{index}]"
                edge_index = label.edge_index
                occurrence = label.occurrence
                seen_refs.append((edge_index, occurrence))
                if not _is_int(edge_index) or not 0 <= edge_index < len(topology.edges):
                    issues.append(ValidationIssue("edge_index_out_of_range", "Line label edge index is unknown.", f"{path}.edge_index"))
                    continue
                edge = topology.edges[edge_index]
                if edge.edge_type != expected_type:
                    issues.append(ValidationIssue("line_type_mismatch", f"Expected a {expected_type.value} edge.", path))
                if not _is_int(occurrence) or occurrence < 0 or occurrence >= edge.multiplicity:
                    issues.append(ValidationIssue("edge_occurrence_out_of_range", "Line occurrence is outside edge multiplicity.", f"{path}.occurrence"))
                if collection_name == "single_lines":
                    if not _is_int(label.particle_id) or not 1 <= label.particle_id <= topology.n:
                        issues.append(ValidationIssue("particle_id_out_of_range", "Particle IDs must be in 1..n.", f"{path}.particle_id"))
                else:
                    if not label.particle_ids:
                        issues.append(ValidationIssue("empty_composite_subset", "A double-line subset cannot be empty.", f"{path}.particle_ids"))
                    if any(not _is_int(particle) or not 1 <= particle <= topology.n for particle in label.particle_ids):
                        issues.append(ValidationIssue("particle_id_out_of_range", "Composite particle IDs must be in 1..n.", f"{path}.particle_ids"))
                    if len(label.particle_ids) != len(_unique_values(label.particle_ids)):
                        issues.append(ValidationIssue("duplicate_composite_particle", "A composite subset cannot repeat a particle ID.", f"{path}.particle_ids"))
            for duplicate in _duplicates(seen_refs):
                issues.append(ValidationIssue("duplicate_line_label", f"Line occurrence is labeled more than once: {duplicate!r}.", collection_name))

        # A topology can be deliberately malformed for G03 testing.  Its own
        # field issues are reported by Topology.validate_fields(); labeling
        # checks remain separately inspectable so later modules can combine the
        # two reports without losing a useful diagnostic.
        return ValidationReport(tuple(issues))

    def to_dict(self) -> dict[str, Any]:
        return {
            "final_particles": [item.to_dict() for item in self.final_particles],
            "single_lines": [item.to_dict() for item in self.single_lines],
            "double_lines": [item.to_dict() for item in self.double_lines],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Labeling":
        if not isinstance(data, Mapping):
            raise TypeError("Labeling data must be a mapping")
        return cls(
            final_particles=tuple(FinalParticleLabel.from_dict(item) for item in data.get("final_particles", ())),
            single_lines=tuple(SingleLineLabel.from_dict(item) for item in data.get("single_lines", ())),
            double_lines=tuple(DoubleLineLabel.from_dict(item) for item in data.get("double_lines", ())),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

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
    unique: list[Any] = []
    for value in values:
        if value not in unique:
            unique.append(value)
    return tuple(unique)


def _stable_pair_sort_key(pair: Any) -> tuple[str, str]:
    """Sort valid integer pairs numerically without crashing on bad input."""

    if isinstance(pair, (tuple, list)) and pair and _is_int(pair[0]):
        return ("0", f"{pair[0]:020d}")
    return ("1", repr(pair))
