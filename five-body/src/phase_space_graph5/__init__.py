"""Five-body structure package for the staged candidate workflow."""

from .domain import (
    CountMode,
    DiagramSpec,
    DoubleLineLabel,
    EdgeType,
    FinalParticleLabel,
    InternalEdge,
    Labeling,
    RunStatus,
    SingleLineLabel,
    Topology,
    ValidationIssue,
    ValidationReport,
    issue_codes,
)
from .structure import StructuralCheck, TopologyMetrics, TopologyStructureReport, analyze_topology
from .canonical import CanonicalizationResult, canonicalize_labeling, canonicalize_labeling_reference, canonicalize_topology, canonicalize_topology_reference
from .enumeration import EnumerationLimits, EnumerationReport, PruningLedger, enumerate_topologies, prufer_labeled_trees, tree_skeletons
from .fixtures import fixture_a, fixture_b
from .flow_labeling import FLOW_STRATEGY, FlowLabelingReport, enumerate_flow_labelings

__all__ = [
    "CountMode", "DiagramSpec", "DoubleLineLabel", "EdgeType", "FinalParticleLabel",
    "InternalEdge", "Labeling", "RunStatus", "SingleLineLabel", "Topology",
    "ValidationIssue", "ValidationReport", "issue_codes", "StructuralCheck",
    "TopologyMetrics", "TopologyStructureReport", "analyze_topology", "CanonicalizationResult",
    "canonicalize_labeling", "canonicalize_labeling_reference", "canonicalize_topology",
    "canonicalize_topology_reference", "EnumerationLimits", "EnumerationReport", "PruningLedger",
    "enumerate_topologies", "prufer_labeled_trees", "tree_skeletons", "fixture_a", "fixture_b",
    "FLOW_STRATEGY", "FlowLabelingReport", "enumerate_flow_labelings",
]
