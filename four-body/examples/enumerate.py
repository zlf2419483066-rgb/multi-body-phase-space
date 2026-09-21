"""Run a small four-body topology-enumeration example."""

from __future__ import annotations

import argparse

from phase_space_graph import EnumerationLimits, enumerate_topologies


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--loop-count", type=int, default=0)
    parser.add_argument("--max-seconds", type=float, default=30.0)
    args = parser.parse_args()

    report = enumerate_topologies(
        n=4,
        loop_count=args.loop_count,
        limits=EnumerationLimits(max_seconds=args.max_seconds),
    )
    print(report.to_dict())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

