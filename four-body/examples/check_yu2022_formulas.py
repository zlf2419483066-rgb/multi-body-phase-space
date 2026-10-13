"""Write a local Yu2022 Table 5/7 density pullback audit from the fixed reference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


DEMO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = DEMO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from phase_space_graph.formula.all_loop_catalog import build_all_loop_catalog  # noqa: E402
from phase_space_graph.formula.all_loop_export import default_all_loop_output_dir  # noqa: E402
from phase_space_graph.formula.inputs import prepare_formula_input  # noqa: E402
from phase_space_graph.formula.records import FormulaOptions  # noqa: E402
from phase_space_graph.formula.reference import build_bridge_formula, build_reference_chart  # noqa: E402
from phase_space_graph.formula.yu2022_check import (  # noqa: E402
    check_yu2022_reference,
    compare_all_loop_export_to_yu2022,
    render_yu2022_markdown,
)


DEFAULT_RESULT_DIR = (
    DEMO_ROOT / "文档" / "验证与测试" / "误差分析"
    / "Yu2022外部密度校对_2026-10-09" / "run01"
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Check the Yu2022 Table 5 and Table 7 per-mirror density coefficients "
            "against the fixed sequential 12/34 reference, then correlate any saved "
            "143-class benchmark evidence without rebuilding formulas."
        )
    )
    parser.add_argument("--normalization", choices=("jing_psf", "standard_2pi4"), default="jing_psf")
    parser.add_argument("--masses", type=float, nargs=4, default=(1.0, 1.5, 0.8, 1.2), metavar=("M1", "M2", "M3", "M4"))
    parser.add_argument("--total-mass", type=float, default=6.0)
    parser.add_argument("--all-loop-export", type=Path, default=default_all_loop_output_dir())
    parser.add_argument("--output", type=Path, default=DEFAULT_RESULT_DIR)
    parser.add_argument("--relative-tolerance", type=float, default=1.0e-9)
    parser.add_argument("--fd-relative-tolerance", type=float, default=5.0e-6)
    parser.add_argument("--kinematic-tolerance", type=float, default=1.0e-9)
    return parser


def _write_new(path: Path, contents: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(contents)


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    catalog = build_all_loop_catalog()
    if not catalog.classes:
        raise RuntimeError("frozen all-loop identity catalog is empty")
    card = prepare_formula_input(
        catalog.classes[0].representative_record,
        measure_convention=args.normalization,
    )
    options = FormulaOptions(normalization=args.normalization)
    bridge = build_bridge_formula(card, build_reference_chart(card, options=options), options=options)
    paper_report = check_yu2022_reference(
        bridge,
        masses=tuple(args.masses),
        total_mass=args.total_mass,
        normalization=args.normalization,
        relative_tolerance=args.relative_tolerance,
        fd_relative_tolerance=args.fd_relative_tolerance,
        kinematic_tolerance=args.kinematic_tolerance,
    )
    all_loop_report = compare_all_loop_export_to_yu2022(args.all_loop_export, paper_report)
    paper_report["all_loop_export_comparison"] = all_loop_report
    markdown = render_yu2022_markdown(paper_report)
    markdown += "\n## 全圈导出 direct pullback 关联\n\n"
    markdown += (
        f"- Export status: `{all_loop_report.get('status')}`；"
        f"classes: {all_loop_report.get('class_count', 0)} / 143；"
        f"direct pullback comparisons: {all_loop_report.get('actual_pullback_comparisons', 0)} / "
        f"{all_loop_report.get('expected_pullback_comparisons_all_143', 4576)}。\n"
    )
    markdown += "- 原全圈导出 `source_status` 保留原值；此文件只添加独立外部对照摘要。\n\n"
    markdown += "| class ID | L | topology | direct benchmark | paper comparison | pullbacks | source status |\n"
    markdown += "|---|---:|---|---|---|---:|---|\n"
    for item in all_loop_report.get("class_results", ()):
        markdown += (
            f"| {item.get('class_id')} | {item.get('loop_count')} | {item.get('topology_id')} | "
            f"{item.get('benchmark_status')} | {item.get('status')} | {item.get('compared_pullbacks', 0)} | "
            f"{item.get('source_status', 'not_checked')} |\n"
        )
    if all_loop_report.get("diagnostics"):
        markdown += "\n" + "\n".join(f"- {item}" for item in all_loop_report["diagnostics"]) + "\n"

    args.output.mkdir(parents=True, exist_ok=True)
    json_path = args.output / "yu2022_external_density_check.json"
    markdown_path = args.output / "yu2022_external_density_check.md"
    if json_path.exists() or markdown_path.exists():
        raise FileExistsError(f"refusing to overwrite an existing Yu2022 report in {args.output}")
    _write_new(json_path, json.dumps(paper_report, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n")
    _write_new(markdown_path, markdown)
    print(f"Fixed-reference status: {paper_report['status']}")
    print(f"All-loop association: {all_loop_report['status']} ({all_loop_report.get('actual_pullback_comparisons', 0)} pullbacks)")
    print(f"Wrote {json_path} and {markdown_path}")
    return 0 if (
        paper_report["status"] == "verified_local"
        and all_loop_report["status"] == "verified_local"
    ) else 1


if __name__ == "__main__":
    raise SystemExit(main())
