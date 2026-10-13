# Publication checks — 2026-10-13

## Python 3.10 compatibility follow-up

The first published commit failed Linux CI on Python 3.10: six LaTeX expressions inside f-strings used syntax accepted by the local Python 3.13 interpreter but rejected by Python 3.10/3.11. The Python 3.12 job was cancelled by the matrix's fail-fast behavior. The initial local checks below therefore did not establish cross-version compatibility.

The expressions now use ordinary string concatenation or precomputed LaTeX. A real Python 3.10.11 interpreter successfully compiled both packages' sources, examples and tests and passed all 12 formula-export regression tests, including the 37 low-loop marked classes. CI now compiles the source tree first and allows both matrix jobs to finish independently. The corrected remote CI result is the release gate.

## Initial local checks

Validation was run against the curated public checkout on Windows with Python 3.13.13. Optional checks used SymPy 1.14.0, Matplotlib 3.11.2, ipywidgets 8.1.9 and IPython 9.17.1. CI separately targets Python 3.10 and 3.12 on Linux; its result must be checked after publication.

- The initial four-body discovery ran 371 tests. Six errors were missing optional widget dependencies; one was a missing source-audit specification. After installing widgets and shipping the specification, the affected suites passed: browser adapter 5, review regressions 17, Notebook 9 and source-table audit 7 tests. The initial missing-module entry is not counted as successful verification.
- The initial five-body discovery ran 255 tests, with two errors and three skips. One error was a missing optional drawing dependency; the other was an internal L5 tool relying on an unshipped production codec. The internal tool and its public discovery entry were excluded. After installing Matplotlib, all 2 drawing and 9 renderer tests passed, including actual SVG/PNG output. The other discovery entries had no failures; skipped checks are not reported as passes.
- The documented tree-formula command exported both L0 classes with zero build failures and `verified_local` benchmark status for both.
- All shipped frozen fixture bytes were compared with their original local sources. Git attributes preserve these files without newline conversion. Staged changes passed `git diff --check`.

These checks validate the public packaging and tested software behavior. Historical 143-class and five-body high-loop research results are documented separately in [validation scope](validation.md); they were not recomputed in full for this publication.
