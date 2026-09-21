"""Hand-calculated physical fixtures for the independent-labeling layers."""

from __future__ import annotations

from dataclasses import dataclass

from phase_space_graph import (
    DoubleLineLabel,
    EdgeType,
    FinalParticleLabel,
    InternalEdge,
    Labeling,
    MomentumCertificate,
    SingleLineLabel,
    Topology,
    analyze_invariants,
)
from phase_space_graph.physical_labels import (
    PhysicalLabelRecord,
    measure_cancellation_counts,
    single_occurrence_groups,
)

Vector = tuple[int, ...]


@dataclass(frozen=True, slots=True)
class LabelFixture:
    name: str
    topology: Topology
    labeling: Labeling
    record: PhysicalLabelRecord | None
    expected_variables: tuple[tuple[int, ...], ...]
    expected_rank: int | None
    expected_cancellations: tuple[tuple[int, int], ...]


def _topology(
    edges: list[InternalEdge],
    final_slots: dict[int, int],
    *,
    vertices: tuple[int, ...],
    n: int = 4,
) -> Topology:
    return Topology(n, vertices, edges, 0, final_slots)


def _record(
    topology: Topology,
    labeling: Labeling,
    vectors: dict[tuple[int, int, int], Vector],
) -> PhysicalLabelRecord:
    complete_doubles: list[DoubleLineLabel] = []
    certificate_rows: list[tuple[int, int, Vector]] = []
    for edge_index, edge in enumerate(topology.edges):
        if edge.edge_type is not EdgeType.DOUBLE:
            continue
        for occurrence in range(edge.multiplicity):
            vector = vectors[(edge.source, edge.target, occurrence)]
            certificate_rows.append((edge_index, occurrence, vector))
            complete_doubles.append(
                DoubleLineLabel(
                    edge_index,
                    occurrence,
                    tuple(index + 1 for index, coefficient in enumerate(vector) if coefficient == 1),
                )
            )
    complete = Labeling(labeling.final_particles, labeling.single_lines, complete_doubles)
    zero = (0,) * topology.n
    momentum = MomentumCertificate(
        tuple(certificate_rows),
        tuple((vertex, zero) for vertex in topology.vertices),
        True,
    )
    independence = analyze_invariants(momentum.variables, topology.n)
    return PhysicalLabelRecord(
        topology=topology,
        complete_labeling=complete,
        momentum=momentum,
        independence=independence,
        single_groups=single_occurrence_groups(topology, complete),
        measure_cancellations=measure_cancellation_counts(topology, complete),
    )


def fixture_a() -> LabelFixture:
    topology = _topology(
        [InternalEdge(0, 1, EdgeType.DOUBLE), InternalEdge(1, 2, EdgeType.DOUBLE)],
        {0: 1, 1: 1, 2: 2},
        vertices=(0, 1, 2),
    )
    labeling = Labeling(
        [
            FinalParticleLabel(0, 0, 4),
            FinalParticleLabel(1, 0, 3),
            FinalParticleLabel(2, 0, 1),
            FinalParticleLabel(2, 1, 2),
        ]
    )
    record = _record(
        topology,
        labeling,
        {(0, 1, 0): (1, 1, 1, 0), (1, 2, 0): (1, 1, 0, 0)},
    )
    return LabelFixture(
        "A",
        topology,
        labeling,
        record,
        ((1, 2, 3), (1, 2)),
        2,
        ((1, 0), (2, 0), (3, 0), (4, 0)),
    )


def fixture_b() -> LabelFixture:
    topology = _topology(
        [
            InternalEdge(0, 1, EdgeType.DOUBLE),
            InternalEdge(0, 2, EdgeType.DOUBLE),
            InternalEdge(2, 3, EdgeType.DOUBLE),
            InternalEdge(1, 2, EdgeType.SINGLE),
        ],
        {0: 0, 1: 1, 2: 1, 3: 2},
        vertices=(0, 1, 2, 3),
    )
    labeling = Labeling(
        [
            FinalParticleLabel(1, 0, 1),
            FinalParticleLabel(2, 0, 4),
            FinalParticleLabel(3, 0, 2),
            FinalParticleLabel(3, 1, 3),
        ],
        [SingleLineLabel(2, 0, 2)],
    )
    record = _record(
        topology,
        labeling,
        {
            (0, 1, 0): (1, 1, 0, 0),
            (0, 2, 0): (0, 0, 1, 1),
            (2, 3, 0): (0, 1, 1, 0),
        },
    )
    return LabelFixture(
        "B",
        topology,
        labeling,
        record,
        ((1, 2), (3, 4), (2, 3)),
        3,
        ((1, 0), (2, 1), (3, 0), (4, 0)),
    )


def _fixture_c(single_particle: int) -> LabelFixture:
    topology = _topology(
        [
            InternalEdge(0, 1, EdgeType.DOUBLE),
            InternalEdge(1, 2, EdgeType.DOUBLE),
            InternalEdge(2, 3, EdgeType.DOUBLE),
            InternalEdge(0, 1, EdgeType.SINGLE),
        ],
        {0: 0, 1: 1, 2: 1, 3: 2},
        vertices=(0, 1, 2, 3),
    )
    labeling = Labeling(
        [
            FinalParticleLabel(1, 0, 4),
            FinalParticleLabel(2, 0, 3),
            FinalParticleLabel(3, 0, 1),
            FinalParticleLabel(3, 1, 2),
        ],
        [SingleLineLabel(1, 0, single_particle)],
    )
    record = _record(
        topology,
        labeling,
        {
            (0, 1, 0): (1, 1, 0, 1) if single_particle == 3 else (1, 0, 1, 1),
            (1, 2, 0): (1, 1, 1, 0),
            (2, 3, 0): (1, 1, 0, 0),
        },
    )
    variables = (
        ((1, 2, 4), (1, 2, 3), (1, 2))
        if single_particle == 3
        else ((1, 3, 4), (1, 2, 3), (1, 2))
    )
    cancellations = (
        ((1, 0), (2, 0), (3, 1), (4, 0))
        if single_particle == 3
        else ((1, 0), (2, 1), (3, 0), (4, 0))
    )
    return LabelFixture(f"C{single_particle}", topology, labeling, record, variables, 3, cancellations)


def fixture_d() -> LabelFixture:
    topology = _topology(
        [
            InternalEdge(0, 1, EdgeType.DOUBLE),
            InternalEdge(1, 2, EdgeType.DOUBLE),
            InternalEdge(2, 3, EdgeType.DOUBLE),
            InternalEdge(0, 2, EdgeType.SINGLE),
        ],
        {0: 0, 1: 1, 2: 1, 3: 2},
        vertices=(0, 1, 2, 3),
    )
    labeling = Labeling(
        [
            FinalParticleLabel(1, 0, 3),
            FinalParticleLabel(2, 0, 1),
            FinalParticleLabel(3, 0, 2),
            FinalParticleLabel(3, 1, 4),
        ],
        [SingleLineLabel(1, 0, 1)],
    )
    record = _record(
        topology,
        labeling,
        {
            (0, 1, 0): (0, 1, 1, 1),
            (1, 2, 0): (0, 1, 0, 1),
            (2, 3, 0): (0, 1, 0, 1),
        },
    )
    return LabelFixture(
        "D",
        topology,
        labeling,
        record,
        ((2, 3, 4), (2, 4), (2, 4)),
        2,
        ((1, 1), (2, 0), (3, 0), (4, 0)),
    )


def fixture_e() -> LabelFixture:
    topology = _topology(
        [
            InternalEdge(0, 1, EdgeType.DOUBLE),
            InternalEdge(1, 2, EdgeType.DOUBLE),
        ],
        {0: 2, 1: 0, 2: 2},
        vertices=(0, 1, 2),
    )
    labeling = Labeling(
        [
            FinalParticleLabel(0, 0, 3),
            FinalParticleLabel(0, 1, 4),
            FinalParticleLabel(2, 0, 1),
            FinalParticleLabel(2, 1, 2),
        ]
    )
    return LabelFixture(
        "E",
        topology,
        labeling,
        None,
        (),
        None,
        ((1, 0), (2, 0), (3, 0), (4, 0)),
    )


def fixture_f() -> LabelFixture:
    topology = _topology(
        [
            InternalEdge(0, 1, EdgeType.DOUBLE),
            InternalEdge(1, 2, EdgeType.DOUBLE),
            InternalEdge(1, 3, EdgeType.DOUBLE),
            InternalEdge(3, 4, EdgeType.DOUBLE),
            InternalEdge(3, 5, EdgeType.DOUBLE),
            InternalEdge(0, 1, EdgeType.SINGLE),
            InternalEdge(2, 3, EdgeType.SINGLE, multiplicity=2),
        ],
        {0: 0, 1: 0, 2: 0, 3: 0, 4: 2, 5: 2},
        vertices=(0, 1, 2, 3, 4, 5),
    )
    labeling = Labeling(
        [
            FinalParticleLabel(4, 0, 1),
            FinalParticleLabel(4, 1, 3),
            FinalParticleLabel(5, 0, 2),
            FinalParticleLabel(5, 1, 4),
        ],
        [SingleLineLabel(1, 0, 1), SingleLineLabel(4, 0, 1), SingleLineLabel(4, 1, 2)],
    )
    record = _record(
        topology,
        labeling,
        {
            (0, 1, 0): (0, 1, 1, 1),
            (1, 2, 0): (1, 1, 0, 0),
            (1, 3, 0): (0, 0, 1, 1),
            (3, 4, 0): (1, 0, 1, 0),
            (3, 5, 0): (0, 1, 0, 1),
        },
    )
    return LabelFixture(
        "F",
        topology,
        labeling,
        record,
        ((2, 3, 4), (1, 2), (3, 4), (1, 3), (2, 4)),
        5,
        ((1, 2), (2, 1), (3, 0), (4, 0)),
    )


def all_fixtures() -> tuple[LabelFixture, ...]:
    return (fixture_a(), fixture_b(), _fixture_c(3), _fixture_c(2), fixture_d(), fixture_e(), fixture_f())
