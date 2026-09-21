"""GD-01: convert an existing topology/marked record into drawing semantics."""
from __future__ import annotations
from dataclasses import dataclass, field
import hashlib, json
from typing import Any

from ..domain import EdgeType, Topology
from ..physical_labels import PhysicalLabelRecord, review_physical_record


@dataclass(frozen=True, slots=True)
class EdgeRef:
    kind: str
    edge_index: int | None = None
    occurrence: int | None = None
    vertex: int | None = None
    slot: int | None = None

    def __post_init__(self):
        if self.kind not in {"internal", "external", "initial"}:
            raise ValueError("unknown EdgeRef kind")
        if self.kind == "internal" and (self.edge_index is None or self.occurrence is None):
            raise ValueError("internal EdgeRef needs edge_index and occurrence")
        if self.kind == "external" and (self.vertex is None or self.slot is None):
            raise ValueError("external EdgeRef needs vertex and slot")

    def key(self) -> tuple[Any, ...]:
        if self.kind == "internal":
            return (self.kind, self.edge_index, self.occurrence)
        if self.kind == "external":
            return (self.kind, self.vertex, self.slot)
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
    topology: Topology
    edges: tuple[DrawingEdge, ...]
    record_digest: str | None = None
    identity_key: tuple[Any, ...] | None = None
    evidence_stage: str = "structure_only"
    display_convention: str = "external_retained_internal_cancelled"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.mode not in {"topology", "marked"}:
            raise ValueError("mode must be topology or marked")
        if self.mode == "marked" and not self.record_digest:
            raise ValueError("marked input requires a physical record digest")


def prepare_drawing(record: PhysicalLabelRecord | None = None, *, topology: Topology | None = None,
                    identity: Any = None, mode: str = "marked") -> DrawingInput:
    """Build a read-only drawing input without running any physical search."""
    if identity is not None:
        if record is not None or topology is not None:
            raise ValueError("pass identity or record/topology, not both")
        record = identity.canonical_record
    if record is not None:
        if not isinstance(record, PhysicalLabelRecord):
            raise TypeError("record must be PhysicalLabelRecord")
        checked = review_physical_record(record)
        if checked.record is None or checked.stage not in {"accepted", "dependent"}:
            raise ValueError(f"physical record review failed: {checked.stage}")
        topology = record.topology
        final = {(x.vertex, x.slot): x.particle_id for x in record.complete_labeling.final_particles}
        singles = {(x.edge_index, x.occurrence): x.particle_id for x in record.complete_labeling.single_lines}
        doubles = {(x.edge_index, x.occurrence): tuple(x.particle_ids) for x in record.complete_labeling.double_lines}
        edges: list[DrawingEdge] = []
        for i, edge in enumerate(topology.edges):
            for occ in range(max(0, edge.multiplicity)):
                ref = EdgeRef("internal", edge_index=i, occurrence=occ)
                if edge.edge_type == EdgeType.SINGLE:
                    pid = singles.get((i, occ))
                    edges.append(DrawingEdge(ref, edge.source, edge.target, edge.edge_type.value,
                                             particle_id=pid, dashed=any(k == "internal" and a == i and b == occ
                                             for _, group in record.single_groups for k, a, b in group)))
                else:
                    edges.append(DrawingEdge(ref, edge.source, edge.target, edge.edge_type.value,
                                             particle_ids=doubles.get((i, occ), ())))
        for (vertex, slot), pid in sorted(final.items()):
            edges.append(DrawingEdge(EdgeRef("external", vertex=vertex, slot=slot), vertex, None,
                                     "single", particle_id=pid, auxiliary=True))
        edges.append(DrawingEdge(EdgeRef("initial"), None, topology.initial_vertex, "single", auxiliary=True))
        digest = identity.digest if identity is not None else hashlib.sha256(json.dumps(record.to_dict(), ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
        return DrawingInput("marked", topology, tuple(edges), digest,
                            identity.key if identity is not None else None, "accepted" if checked.stage == "accepted" else checked.stage)
    if topology is None:
        raise ValueError("topology or record is required")
    edges = []
    for i, edge in enumerate(topology.edges):
        for occ in range(max(0, edge.multiplicity)):
            edges.append(DrawingEdge(EdgeRef("internal", i, occ), edge.source, edge.target, edge.edge_type.value))
    for vertex, count in topology.final_slots:
        for slot in range(max(0, count)):
            edges.append(DrawingEdge(EdgeRef("external", vertex=vertex, slot=slot), vertex, None, "single", auxiliary=True))
    edges.append(DrawingEdge(EdgeRef("initial"), None, topology.initial_vertex, "single", auxiliary=True))
    return DrawingInput("topology", topology, tuple(edges))
