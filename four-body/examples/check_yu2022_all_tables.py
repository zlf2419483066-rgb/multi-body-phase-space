"""Run the independent 22-table Yu (2022) density pullback audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping


DEMO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = DEMO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from phase_space_graph.drawing.archive import _record_from_dict  # noqa: E402
from phase_space_graph.formula.benchmark import _default_points  # noqa: E402
from phase_space_graph.formula.inputs import prepare_formula_input  # noqa: E402
from phase_space_graph.formula.records import FormulaOptions  # noqa: E402
from phase_space_graph.formula.reference import build_bridge_formula, build_reference_chart  # noqa: E402
from phase_space_graph.formula.yu2022_all_tables import run_check  # noqa: E402
from phase_space_graph.data_paths import default_output  # noqa: E402


RUN04_FIRST_EVIDENCE = (
    DEMO_ROOT / "文档" / "验证与测试" / "误差分析" / "四体全圈143类公式_2026-10-09"
    / "run04" / "L0-C001_L0-T01.evidence.json"
)
DEFAULT_OUTPUT = default_output("Yu2022全22表核对_2026-10-09", "external_numeric_run01")
DEFAULT_CASES = (
    {"masses": (1.0, 1.5, 0.8, 1.2), "total_mass": 6.0},
    {"masses": (0.7, 1.1, 1.6, 0.9), "total_mass": 9.0},
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare all 22 Yu (2022) table rows against the saved sequential "
            "reference, with both mirror sheets and two finite-difference steps."
        )
    )
    parser.add_argument("--normalization", choices=("jing_psf", "standard_2pi4"), default="jing_psf")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def _build_bridge(normalization: str):
    with RUN04_FIRST_EVIDENCE.open("r", encoding="utf-8") as handle:
        evidence = json.load(handle)
    record_payload = evidence["input_card"]["record"]
    record = _record_from_dict(record_payload)
    card = prepare_formula_input(record, measure_convention=normalization)
    options = FormulaOptions(normalization=normalization)
    chart = build_reference_chart(card, options=options)
    return build_bridge_formula(card, chart, options=options)


def _max_residual(table: Mapping[str, Any]) -> float | None:
    residuals = [
        row["relative_residual"]
        for row in table.get("comparisons", ())
        if isinstance(row, Mapping) and isinstance(row.get("relative_residual"), (int, float))
    ]
    return max(residuals) if residuals else None


def _render_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Yu (2022) 22-table external density pullback check",
        "",
        "This report records finite numerical comparisons. Status values describe evaluation completeness only; they do not promote source status or imply a global proof.",
        "",
    ]
    for case_index, case in enumerate(report.get("cases", ()), 1):
        lines.extend([
            f"## Case {case_index}",
            "",
            f"- Masses: `{case.get('masses')}`",
            f"- Total mass: `{case.get('total_mass')}`",
            f"- Normalization: `{case.get('normalization')}`",
            f"- Points: `{case.get('point_count', 8)}` (both mirrors; steps `{case.get('finite_difference_steps')}`)",
            "",
            "| Table | Variant | Evaluation status | Rows | Errors | Maximum relative residual | Source density expression without 2π factors |",
            "|---:|---|---|---:|---:|---:|---|",
        ])
        for table in case.get("tables", ()):
            maximum = _max_residual(table)
            max_text = "—" if maximum is None else f"{maximum:.8g}"
            expression = str(table.get("source_expression", "")).replace("|", "\\|")
            lines.append(
                f"| {table.get('table')} | {table.get('variant')} | {table.get('status')} | "
                f"{table.get('comparison_count', 0)} | {table.get('error_count', 0)} | {max_text} | `{expression}` |"
            )
        lines.append("")
    lines.extend([
        "## Protocol",
        "",
        "- Each row maps the fixed reference coordinates to five table coordinates and three ZYZ Euler angles; invariant-square coordinates are output as s/M².",
        "- Central-difference Jacobians use the requested steps and wrap only the output Euler phi and psi rows.",
        "- Both reflection mirrors are evaluated from their own momentum vectors.",
        "- Table 23 and 24 row-variable and caption-variable evaluations are listed independently.",
        "- An error is retained on its comparison row and is not silently skipped.",
        "",
    ])
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite existing output path: {args.output}")
    bridge = _build_bridge(args.normalization)
    cases = []
    for item in DEFAULT_CASES:
        masses = tuple(item["masses"])
        parent_mass = float(item["total_mass"])
        points = _default_points(masses, parent_mass)
        if points is None or len(points) < 8:
            raise RuntimeError("the default sequential reference grid must provide at least eight points")
        case_report = run_check(
            bridge,
            masses=masses,
            total_mass=parent_mass,
            points=points,
            normalization=args.normalization,
        )["cases"][0]
        case_report["point_count"] = len(points)
        cases.append(case_report)
    report = {
        "schema_version": "yu2022-all-tables-pullback-v1",
        "input_evidence": str(RUN04_FIRST_EVIDENCE.relative_to(DEMO_ROOT)),
        "input_record_source": "run04 first saved evidence input_card.record, reconstructed with drawing.archive._record_from_dict",
        "normalization": args.normalization,
        "cases": cases,
    }
    payload = json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
    markdown = _render_markdown(report)
    args.output.mkdir(parents=True, exist_ok=False)
    json_path = args.output / "yu2022_all_tables_external_check.json"
    markdown_path = args.output / "yu2022_all_tables_external_check.md"
    with json_path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(payload)
    with markdown_path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(markdown)
    print(f"Wrote {json_path}")
    print(f"Wrote {markdown_path}")
    return 0 if all(
        table.get("status") == "evaluated"
        for case in cases for table in case.get("tables", ())
    ) else 1


if __name__ == "__main__":
    raise SystemExit(main())
