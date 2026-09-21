"""Hand-written five-body fixtures used by the first structural/enum checks."""

from __future__ import annotations

from .domain import EdgeType, FinalParticleLabel, InternalEdge, Labeling, SingleLineLabel, Topology


def fixture_a() -> tuple[Topology, Labeling]:
    """Zero-loop chain: 0:p1, 1:p2, 2:p3, 3:p4/p5."""

    topology = Topology(
        n=5,
        vertices=(0, 1, 2, 3),
        edges=(
            InternalEdge(0, 1, EdgeType.DOUBLE),
            InternalEdge(1, 2, EdgeType.DOUBLE),
            InternalEdge(2, 3, EdgeType.DOUBLE),
        ),
        initial_vertex=0,
        final_slots={0: 1, 1: 1, 2: 1, 3: 2},
    )
    labeling = Labeling(
        final_particles=(
            FinalParticleLabel(0, 0, 1),
            FinalParticleLabel(1, 0, 2),
            FinalParticleLabel(2, 0, 3),
            FinalParticleLabel(3, 0, 4),
            FinalParticleLabel(3, 1, 5),
        )
    )
    return topology, labeling


def fixture_b(a: int = 3) -> tuple[Topology, Labeling]:
    """One-loop chain with single line ``0->2`` carrying particle ``a``.

    The final state is deliberately placed at vertices 1, 2, 3, and 4;
    vertex 0 has no final-state slot.  For ``a=3`` the recovered double-line
    subsets are p1245, p245, p345, p45.  The fixture itself stores only the
    declared external and single labels; a downstream momentum layer fills
    double labels from its certificate.
    """

    if isinstance(a, bool) or not isinstance(a, int) or not 1 <= a <= 5:
        raise ValueError("a must be an integer in 1..5")
    topology = Topology(
        n=5,
        vertices=(0, 1, 2, 3, 4),
        edges=(
            InternalEdge(0, 1, EdgeType.DOUBLE),
            InternalEdge(1, 2, EdgeType.DOUBLE),
            InternalEdge(2, 3, EdgeType.DOUBLE),
            InternalEdge(3, 4, EdgeType.DOUBLE),
            InternalEdge(0, 2, EdgeType.SINGLE),
        ),
        initial_vertex=0,
        final_slots={0: 0, 1: 1, 2: 1, 3: 1, 4: 2},
    )
    labeling = Labeling(
        final_particles=(
            FinalParticleLabel(1, 0, 1),
            FinalParticleLabel(2, 0, 2),
            FinalParticleLabel(3, 0, 3),
            FinalParticleLabel(4, 0, 4),
            FinalParticleLabel(4, 1, 5),
        ),
        single_lines=(SingleLineLabel(1, 0, a),),
    )
    return topology, labeling


__all__ = ["fixture_a", "fixture_b"]

