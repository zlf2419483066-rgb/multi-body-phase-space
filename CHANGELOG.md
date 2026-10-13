# Changelog

## Unreleased — 2026-10-13

- Fixed six formula-export f-string expressions that could not be parsed by Python 3.10/3.11. Added an early syntax check to CI and disabled matrix fail-fast so both supported Python versions finish independently.
- Protected POSIX worker cleanup from signalling the caller's process group. Added real isolated-process regression coverage and an early cleanup check before the full CI suites.

- Published four-body direct formulas for loops 0–3, covariant tree reduction, 143-class catalog/export, and local benchmark and source-audit code.
- Added frozen catalog inputs and regression tests needed to reproduce the formula workflow without private run directories.
- Added the five-body invariant-independence library and optional SymPy installation support.
- Updated research scope, numerical evidence and unresolved source/physical-domain limitations. Historical counts are distinguished from checks rerun in this checkout.

## 0.1.0 — 2026-09-21

- Published the first curated public repository containing the four-body and five-body demos.
- Added package installation metadata, test instructions, CI, license, contribution guidance, and validation-scope documentation.
- Kept long-running generated catalogs and internal research records outside the public source tree.

