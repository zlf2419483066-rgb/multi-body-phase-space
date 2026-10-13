"""Export selected four-body L=0..3 marked classes as six-step dossiers."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


DEMO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = DEMO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from phase_space_graph.formula.all_loop_export import (  # noqa: E402
    default_all_loop_output_dir,
    export_all_four_body_formulas,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Export six-step four-body L=0..3 marked-class formula dossiers from the "
            "frozen G08 and three-loop identity sources. It does not enumerate candidates "
            "and never replaces existing files."
        )
    )
    selection = parser.add_mutually_exclusive_group(required=False)
    selection.add_argument("--all", action="store_true", help="export all classes (143 in the full catalog)")
    selection.add_argument(
        "--topology",
        help="export one topology by Txx ID or topology-digest prefix",
    )
    selection.add_argument(
        "--class",
        dest="class_selector",
        help="export one Ln-Cxxx class or particle-relabeling digest prefix",
    )
    parser.add_argument(
        "--loops", type=int, nargs="+", choices=(0, 1, 2, 3),
        help="restrict to one or more loop counts; can be combined with --all, --topology, or --class",
    )
    parser.add_argument("--output", type=Path, default=default_all_loop_output_dir(),
                        help="single export directory (default: the four-body all-loop run01 directory)")
    parser.add_argument("--normalization", choices=("jing_psf", "standard_2pi4"),
                        default="jing_psf", help="measure convention used to build each formula card")
    parser.add_argument(
        "--benchmark", action="store_true",
        help="run and save an explicit per-class numerical benchmark report",
    )
    parser.add_argument(
        "--masses", type=float, nargs=4, metavar=("M1", "M2", "M3", "M4"),
        help="four final-particle masses; required with --benchmark",
    )
    parser.add_argument(
        "--total-mass", type=float, metavar="M",
        help="parent mass; required with --benchmark",
    )
    parser.add_argument("--g08-l0", type=Path, help="override the frozen L0 G08 audit path")
    parser.add_argument("--g08-l1", type=Path, help="override the frozen L1 G08 audit path")
    parser.add_argument("--g08-l2", type=Path, help="override the frozen L2 G08 audit path")
    parser.add_argument("--l3-archive", type=Path, help="override the frozen L3 fixed-identity archive path")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if not any((args.all, args.topology is not None, args.class_selector is not None, args.loops is not None)):
        parser.error("choose --all, --topology, --class, or --loops")
    if args.benchmark and (args.masses is None or args.total_mass is None):
        parser.error("--benchmark requires both --masses M1 M2 M3 M4 and --total-mass M")
    if not args.benchmark and (args.masses is not None or args.total_mass is not None):
        parser.error("--masses and --total-mass require --benchmark")
    archive_paths = {
        loop: path
        for loop, path in (
            (0, args.g08_l0), (1, args.g08_l1), (2, args.g08_l2), (3, args.l3_archive),
        )
        if path is not None
    }
    result = export_all_four_body_formulas(
        all_classes=args.all,
        topology=args.topology,
        class_selector=args.class_selector,
        loops=tuple(args.loops) if args.loops is not None else None,
        output_dir=args.output,
        archive_paths=archive_paths,
        normalization=args.normalization,
        run_benchmark=args.benchmark,
        benchmark_masses=tuple(args.masses) if args.masses is not None else None,
        benchmark_total_mass=args.total_mass,
    )
    print(f"Exported {result['markdown_count']} Markdown dossier(s) to {result['output_directory']}")
    print(f"Selected {result['selected_class_count']} class(es); build failures: {result['failed_class_count']}")
    return 1 if result["failed_class_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
