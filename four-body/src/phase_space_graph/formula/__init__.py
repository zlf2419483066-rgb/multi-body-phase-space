"""Formula generation, direct D5 charts, covariant traces, and cross-checks.

The direct IKV route and recursive reference route remain separately typed.
The analytic direct profiles include the two-pair chart and the three-loop
d=5 Gram chart; other invariant sets remain explicitly unsupported.
"""

from .records import (
    FormulaOptions, RuleStep, BranchRecord, CoordinateMap,
    RawExpression, CovariantResult, DirectFormula, BridgeFormula,
    BenchmarkReport, FormulaArtifact,
)
from .raw import build_f0_expression, build_raw_expression, reduce_covariant
from .expressions import ExpressionNode, symbol, differential, delta, product
from .direct_charts import (
    DirectChart, select_direct_chart, reconstruct_direct_momenta,
    build_direct_formula, build_direct_density,
)
from .reference import (
    ReferenceChart, build_reference_chart, reconstruct_reference_momenta,
    build_bridge_formula, build_reference_density,
)
from .benchmark import compare_paths, benchmark_formula, map_reference_to_target
from .generate import (
    generate_formula, generate_formulas,
    generate_complete_formula, generate_complete_formulas,
)
from .export import FormulaSelection, select_artifacts, export_markdown
from .validation import ValidationFinding, validate_artifact, render_validation_markdown
from .batch import BatchItem, FormulaBatchReport, run_formula_batch
from .expanded import ExpandedF0Expression, build_expanded_f0_expression
from .b_direct import BDirectSample, check_b_f03_eligibility, sample_b_f03
from .cancellations import (
    CANCELLATION_PLAN_SCHEMA_VERSION,
    LM12_DEFAULT_POLICY,
    FORMULA_OVERRIDE_POLICY,
    OccurrenceLocator,
    ParticleOccurrence,
    CancellationPlanOverride,
    ParticleCancellationPlan,
    CancellationPlan,
    build_cancellation_plan,
)
from .local_reduction import (
    LOCAL_REDUCTION_SCHEMA_VERSION,
    TwoPiAccounting,
    RemainingConstraintTrace,
    LocalFactorTrace,
    LocalReductionTrace,
    reduce_local_measures,
)
from .b_covariant import (
    B_F03_CERTIFICATE_SCHEMA_VERSION,
    B_F03_IDENTITY_DIGEST,
    B_F03_VARIABLES,
    VertexConservationTrace,
    ShellDeltaTransfer,
    MomentumEliminationStep,
    ParticleMeasureSource,
    Delta4Disposition,
    CovariantNormalizationLedger,
    BFCovariantCertificate,
    build_b_covariant_certificate,
)
from .b_dossier import (
    B_F03_DOSSIER_SCHEMA_VERSION,
    BF03Dossier,
    build_b_f03_dossier,
    render_b_f03_dossier_markdown,
)
from .b_helicity import (
    B_HELICITY_ANGLE_DEFINITIONS,
    B_HELICITY_MEASURE_IDENTITY,
    B_HELICITY_PROFILE,
    B_HELICITY_VARIABLES,
    BHelicityMapResult,
    BHelicitySample,
    fixed_to_helicity_angles,
    helicity_to_fixed_angles,
    sample_b_f03_helicity,
)
from .d5_direct import build_d5_chart_data, build_d5_formula, reconstruct_d5_momenta
from .tree_covariant import TreeCovariantTrace, build_tree_covariant_trace
from .d5_catalog import build_d5_catalog
from .d5_export import export_d5_formulas
from .all_loop_catalog import (
    AllLoopCatalogMember, AllLoopMarkedClass, AllLoopCatalog,
    build_all_loop_catalog, select_all_loop_classes,
)
from .all_loop_export import (
    ALL_LOOP_EXPORT_RELATIVE_PATH, ALL_LOOP_EXPORT_SCHEMA_VERSION,
    build_all_loop_formula_evidence, default_all_loop_output_dir,
    export_all_four_body_formulas,
)

__all__ = [
    "FormulaOptions", "RuleStep", "BranchRecord", "CoordinateMap",
    "RawExpression", "CovariantResult", "DirectFormula", "BridgeFormula",
    "BenchmarkReport", "FormulaArtifact", "ExpressionNode", "symbol", "differential", "delta", "product", "build_f0_expression", "build_raw_expression",
    "reduce_covariant", "DirectChart", "select_direct_chart",
    "reconstruct_direct_momenta", "build_direct_formula", "build_direct_density", "ReferenceChart",
    "build_reference_chart", "reconstruct_reference_momenta",
    "build_bridge_formula", "build_reference_density", "compare_paths", "benchmark_formula", "map_reference_to_target",
    "generate_formula", "generate_formulas", "generate_complete_formula",
    "generate_complete_formulas", "FormulaSelection",
    "select_artifacts", "export_markdown",
    "ValidationFinding", "validate_artifact", "render_validation_markdown",
    "BatchItem", "FormulaBatchReport", "run_formula_batch",
    "ExpandedF0Expression", "build_expanded_f0_expression",
    "BDirectSample", "check_b_f03_eligibility", "sample_b_f03",
    "CANCELLATION_PLAN_SCHEMA_VERSION", "LM12_DEFAULT_POLICY",
    "FORMULA_OVERRIDE_POLICY", "OccurrenceLocator", "ParticleOccurrence",
    "CancellationPlanOverride", "ParticleCancellationPlan", "CancellationPlan",
    "build_cancellation_plan",
    "LOCAL_REDUCTION_SCHEMA_VERSION", "TwoPiAccounting", "RemainingConstraintTrace",
    "LocalFactorTrace", "LocalReductionTrace", "reduce_local_measures",
    "B_F03_CERTIFICATE_SCHEMA_VERSION", "B_F03_IDENTITY_DIGEST", "B_F03_VARIABLES",
    "VertexConservationTrace", "ShellDeltaTransfer", "MomentumEliminationStep",
    "ParticleMeasureSource", "Delta4Disposition", "CovariantNormalizationLedger",
    "BFCovariantCertificate", "build_b_covariant_certificate",
    "B_F03_DOSSIER_SCHEMA_VERSION", "BF03Dossier", "build_b_f03_dossier",
    "render_b_f03_dossier_markdown",
    "B_HELICITY_MEASURE_IDENTITY", "B_HELICITY_PROFILE", "B_HELICITY_VARIABLES",
    "B_HELICITY_ANGLE_DEFINITIONS",
    "BHelicityMapResult", "BHelicitySample", "fixed_to_helicity_angles",
    "helicity_to_fixed_angles", "sample_b_f03_helicity",
    "build_d5_chart_data", "build_d5_formula", "reconstruct_d5_momenta",
    "TreeCovariantTrace", "build_tree_covariant_trace", "build_d5_catalog",
    "export_d5_formulas",
    "AllLoopCatalogMember", "AllLoopMarkedClass", "AllLoopCatalog",
    "build_all_loop_catalog", "select_all_loop_classes",
    "ALL_LOOP_EXPORT_RELATIVE_PATH", "ALL_LOOP_EXPORT_SCHEMA_VERSION",
    "build_all_loop_formula_evidence", "default_all_loop_output_dir",
    "export_all_four_body_formulas",
]
