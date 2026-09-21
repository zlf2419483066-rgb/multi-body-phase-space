"""Formula generation and cross-checking for the four-body demo.

The package keeps the direct IKV route and the recursive reference route as
typed, separately versioned artefacts.  It intentionally supports a small
analytic chart first; unsupported invariant sets remain explicit rather than
being silently replaced by the reference calculation.
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
from .generate import generate_formula, generate_formulas
from .export import FormulaSelection, select_artifacts, export_markdown
from .validation import ValidationFinding, validate_artifact, render_validation_markdown
from .batch import BatchItem, FormulaBatchReport, run_formula_batch

__all__ = [
    "FormulaOptions", "RuleStep", "BranchRecord", "CoordinateMap",
    "RawExpression", "CovariantResult", "DirectFormula", "BridgeFormula",
    "BenchmarkReport", "FormulaArtifact", "ExpressionNode", "symbol", "differential", "delta", "product", "build_f0_expression", "build_raw_expression",
    "reduce_covariant", "DirectChart", "select_direct_chart",
    "reconstruct_direct_momenta", "build_direct_formula", "build_direct_density", "ReferenceChart",
    "build_reference_chart", "reconstruct_reference_momenta",
    "build_bridge_formula", "build_reference_density", "compare_paths", "benchmark_formula", "map_reference_to_target",
    "generate_formula", "generate_formulas", "FormulaSelection",
    "select_artifacts", "export_markdown",
    "ValidationFinding", "validate_artifact", "render_validation_markdown",
    "BatchItem", "FormulaBatchReport", "run_formula_batch",
]
