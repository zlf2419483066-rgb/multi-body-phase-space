"""Candidate records to drawing semantics for the five body package.

The drawing layer deliberately consumes the candidate contract instead of the
four body ``PhysicalLabelRecord``.  In particular, a momentum candidate has
no independence result and therefore must remain drawable while its E status
is ``not_evaluated``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from typing import Any, Iterable

from ..physical_labels import CandidateMomentumRecord, review_candidate_record


def _value(value: Any) -> Any:
    return getattr(value, "value", value)


def _edge_type(edge: Any) -> str:
    return str(_value(getattr(edge, "edge_type", ""))).lower()


def _topology_dict(topology: Any) -> dict[str, Any]:
    if hasattr(topology, "to_dict"):
        value = topology.to_dict()
        if isinstance(value, dict):
            return value
    edges = []
    for edge in getattr(topology, "edges", ()):
        edges.append({
            "source": getattr(edge, "source", None),
            "target": getattr(edge, "target", None),
            "edge_type": _edge_type(edge),
            "multiplicity": getattr(edge, "multiplicity", 1),
        })
    return {
        "n": getattr(topology, "n", None),
        "vertices": list(getattr(topology, "vertices", ())),
        "initial_vertex": getattr(topology, "initial_vertex", None),
        "final_slots": [list(item) for item in getattr(topology, "final_slots", ())],
        "edges": edges,
    }


def _stable_digest(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class EdgeRef:
    """Stable reference to one logical edge occurrence or external slot."""

    kind: str
    edge_index: int | None = None
    occurrence: int | None = None
    vertex: int | None = None
    slot: int | None = None

    def __post_init__(self) -> None:
        if self.kind not in {"internal", "external", "initial"}:
            raise ValueError(f"unknown EdgeRef kind: {self.kind!r}")
        if self.kind == "internal" and (self.edge_index is None or self.occurrence is None):
            raise ValueError("internal EdgeRef needs edge_index and occurrence")
        if self.kind == "external" and (self.vertex is None or self.slot is None):
            raise ValueError("external EdgeRef needs vertex and slot")

    def key(self) -> tuple[Any, ...]:
        if self.kind == "internal":
            return self.kind, self.edge_index, self.occurrence
        if self.kind == "external":
            return self.kind, self.vertex, self.slot
        return (self.kind,)


@dataclass(frozen=True, slots=True)
class DrawingEdge:
    ref: EdgeRef
    source: int | None
    target: int | None
    edge_type: str
    particle_id: int | None = None
    particle_ids: tuple[int, ...] = ()
    dashed: bool = False
    auxiliary: bool = False


@dataclass(frozen=True, slots=True)
class DrawingInput:
    mode: str
    topology: Any
    edges: tuple[DrawingEdge, ...]
    record_digest: str | None = None
    identity_key: tuple[Any, ...] | None = None
    evidence_stage: str = "structure_only"
    display_convention: str = "external_retained_internal_cancelled"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.mode not in {"topology", "marked"}:
            raise ValueError("mode must be topology or marked")
        if self.mode == "marked" and not self.record_digest:
            raise ValueError("marked input requires a candidate record digest")


def _iter_labels(labeling: Any, name: str) -> tuple[Any, ...]:
    return tuple(getattr(labeling, name, ()) or ())


def _label_attr(label: Any, *names: str, default: Any = None) -> Any:
    for name in names:
        if hasattr(label, name):
            return getattr(label, name)
    return default


def _complete_double_labels(record: CandidateMomentumRecord) -> dict[tuple[int, int], tuple[int, ...]]:
    labeling = getattr(record, "complete_labeling", None)
    result: dict[tuple[int, int], tuple[int, ...]] = {}
    for label in _iter_labels(labeling, "double_lines"):
        edge = int(_label_attr(label, "edge_index", default=-1))
        occurrence = int(_label_attr(label, "occurrence", default=0))
        values = _label_attr(label, "particle_ids", "particles", default=())
        result[(edge, occurrence)] = tuple(sorted(int(value) for value in values))

    return result


def _record_payload(record: Any) -> dict[str, Any]:
    if hasattr(record, "to_dict"):
        value = record.to_dict()
        if isinstance(value, dict):
            return value
    topology = getattr(record, "topology", None)
    labeling = getattr(record, "complete_labeling", None)
    return {
        "topology": _topology_dict(topology),
        "labeling": getattr(labeling, "to_dict", lambda: repr(labeling))(),
        "momentum": getattr(getattr(record, "momentum", None), "to_dict", lambda: repr(getattr(record, "momentum", None)))(),
    }


def candidate_record_digest(record: Any) -> str:
    """Return a deterministic digest of the complete candidate payload."""

    return _stable_digest(_record_payload(record))


def _candidate_identity(record: Any, identity: Any) -> tuple[str, tuple[Any, ...] | None]:
    digest = getattr(identity, "digest", None) if identity is not None else None
    key = getattr(identity, "key", None) if identity is not None else None
    if digest:
        return str(digest), tuple(key) if key is not None else None
    return candidate_record_digest(record), tuple(key) if key is not None else None


def _single_group_refs(record: Any) -> set[tuple[int, int]]:
    refs: set[tuple[int, int]] = set()
    for group in getattr(record, "single_groups", ()) or ():
        try:
            _, occurrences = group
        except (TypeError, ValueError):
            continue
        for occurrence in occurrences or ():
            if isinstance(occurrence, dict):
                if occurrence.get("kind") == "internal":
                    refs.add((int(occurrence["edge_index"]), int(occurrence["occurrence"])))
            elif isinstance(occurrence, (tuple, list)) and len(occurrence) >= 3 and occurrence[0] == "internal":
                refs.add((int(occurrence[1]), int(occurrence[2])))
    return refs


def _build_marked_edges(record: Any) -> tuple[DrawingEdge, ...]:
    topology = getattr(record, "topology")
    labeling = getattr(record, "complete_labeling")
    finals = {
        (int(_label_attr(label, "vertex")), int(_label_attr(label, "slot"))): int(_label_attr(label, "particle_id", "particle"))
        for label in _iter_labels(labeling, "final_particles")
    }
    singles = {
        (int(_label_attr(label, "edge_index")), int(_label_attr(label, "occurrence", default=0))): int(_label_attr(label, "particle_id", "particle"))
        for label in _iter_labels(labeling, "single_lines")
    }
    doubles = _complete_double_labels(record)
    cancelled = _single_group_refs(record)
    edges: list[DrawingEdge] = []
    for index, edge in enumerate(getattr(topology, "edges", ())):
        source, target = int(edge.source), int(edge.target)
        kind = _edge_type(edge)
        for occurrence in range(max(0, int(getattr(edge, "multiplicity", 1)))):
            ref = EdgeRef("internal", index, occurrence)
            if kind == "single":
                edges.append(DrawingEdge(ref, source, target, kind, particle_id=singles.get((index, occurrence)),
                                         dashed=(index, occurrence) in cancelled))
            else:
                edges.append(DrawingEdge(ref, source, target, kind,
                                         particle_ids=doubles.get((index, occurrence), ())))
    for (vertex, slot), particle in sorted(finals.items()):
        edges.append(DrawingEdge(EdgeRef("external", vertex=vertex, slot=slot), vertex, None,
                                 "single", particle_id=particle, auxiliary=True))
    edges.append(DrawingEdge(EdgeRef("initial"), None, int(topology.initial_vertex), "single", auxiliary=True))
    return tuple(edges)


def _build_topology_edges(topology: Any) -> tuple[DrawingEdge, ...]:
    edges: list[DrawingEdge] = []
    for index, edge in enumerate(getattr(topology, "edges", ())):
        for occurrence in range(max(0, int(getattr(edge, "multiplicity", 1)))):
            edges.append(DrawingEdge(EdgeRef("internal", index, occurrence), int(edge.source), int(edge.target), _edge_type(edge)))
    for vertex, count in getattr(topology, "final_slots", ()):
        for slot in range(max(0, int(count))):
            edges.append(DrawingEdge(EdgeRef("external", vertex=int(vertex), slot=slot), int(vertex), None,
                                     "single", auxiliary=True))
    edges.append(DrawingEdge(EdgeRef("initial"), None, int(topology.initial_vertex), "single", auxiliary=True))
    return tuple(edges)


def prepare_drawing(record: CandidateMomentumRecord | None = None, *, topology: Any = None,
                    identity: Any = None,
                    mode: str = "marked") -> DrawingInput:
    """Build a drawing input without searching or running independence checks."""

    if identity is not None and record is None:
        record = getattr(identity, "canonical_record", None)
        if record is None:
            raise ValueError("candidate identity lacks canonical_record")
    if record is not None:
        if not isinstance(record, CandidateMomentumRecord):
            raise TypeError("record must be CandidateMomentumRecord")
        checked = review_candidate_record(record)
        if checked.record is None or checked.stage != "momentum_candidate":
            raise ValueError(f"candidate record review failed: {checked.stage}")
        if identity is not None and getattr(identity, "canonical_record", None) is None:
            raise ValueError("candidate identity lacks canonical_record")
        if identity is not None:
            canonical = getattr(identity, "canonical_record")
            checked_identity = review_candidate_record(canonical)
            if checked_identity.record is None or checked_identity.stage != "momentum_candidate":
                raise ValueError("candidate identity canonical record failed review")
            # Identity canonicalization is validated by the identity module;
            # this local check ensures the supplied record and identity agree
            # on the complete physical payload without replacing the source
            # record's vertex numbers.
            if getattr(identity, "schema_version", "IM-5-candidate-v1") != "IM-5-candidate-v1":
                raise ValueError("unsupported candidate identity schema")
            # Rebuild the identity from the supplied source record.  This is
            # the integrity boundary between canonical index data and the
            # original-numbered record used for drawing.
            from ..label_identity import canonicalize_candidate_labeling
            rebuilt = canonicalize_candidate_labeling(record, getattr(identity, "mode", "fixed_labels"))
            if rebuilt.key != getattr(identity, "key", None) or rebuilt.digest != getattr(identity, "digest", None):
                raise ValueError("candidate identity does not match supplied source record")
            if rebuilt.canonical_record.to_dict() != canonical.to_dict():
                raise ValueError("candidate identity canonical record differs from supplied identity")
    if record is not None:
        topology = getattr(record, "topology", None)
        if topology is None or getattr(record, "complete_labeling", None) is None:
            raise ValueError("candidate record must contain topology and complete_labeling")
        digest, key = _candidate_identity(record, identity)
        metadata = {
            "candidate_schema": "IM-5-candidate-v1", "identity_digest": digest,
            "independence": "not_evaluated",
            "identity_key": list(key) if key is not None else None,
        }
        if identity is not None:
            metadata["original_vertex_map"] = [list(item) for item in getattr(identity, "vertex_map", ())]
        return DrawingInput("marked", topology, _build_marked_edges(record), digest, key,
                            evidence_stage=str(getattr(record, "stage", "momentum_candidate")),
                            metadata=metadata)
    if topology is None:
        raise ValueError("topology or candidate record is required")
    return DrawingInput("topology", topology, _build_topology_edges(topology),
                        evidence_stage="structure_only",
                        metadata={"candidate_schema": "IM-5-candidate-v1", "independence": "not_evaluated"})


__all__ = [
    "EdgeRef", "DrawingEdge", "DrawingInput", "candidate_record_digest", "prepare_drawing",
]
