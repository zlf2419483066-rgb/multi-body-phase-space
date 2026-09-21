# Multi-body phase-space graphic method

This repository contains two small, installable Python demos for enumerating and auditing graph representations of multi-body phase space in particle-physics calculations.

The demos are deliberately separate:

- [`four-body/`](four-body/) is the more complete four-body prototype. It includes canonical topology enumeration, momentum-label recovery, invariant-independence checks, identity normalization, reporting, formula-input records, and optional drawing helpers.
- [`five-body/`](five-body/) is a staged five-body prototype. Its public enumeration scope is currently the structural and momentum-candidate workflow for five-body diagrams at loop counts 0–3. It does not claim completed invariant-independence, effective-topology, or full-formula evaluation.

The repository is intended to be a maintained research-software starting point, not a dump of the internal experiment archive. Large generated catalogs, machine-specific run logs, private working notes, and source-paper copies are kept outside this public tree.

## Requirements

- Python 3.10 or newer
- No mandatory third-party runtime dependency for the core enumerators
- Optional drawing support: Matplotlib and a Graphviz `dot` executable

Install the two demos independently:

```powershell
python -m pip install -e ./four-body
python -m pip install -e ./five-body
```

Run the deterministic unit suites from the repository root:

```powershell
python -m unittest discover -s four-body/tests -p "test_*.py"
python -m unittest discover -s five-body/tests -p "test_*.py"
```

The first command exercises the four-body API and its layered tests. The second exercises the five-body staged candidate workflow. Tests that require Graphviz or Matplotlib skip their integration portion when those optional tools are unavailable.

## Quick examples

```powershell
python four-body/examples/enumerate.py --loop-count 0
python five-body/examples/run_candidates.py --loop-count 0 --no-render --max-seconds 30
```

Generated data is written outside the repository by default. Set `PSF_DATA_ROOT` to choose another data volume. The exact scope, status, and limitations of each demo are documented in its own README and in [`docs/validation.md`](docs/validation.md).

## Research and JOSS status

The code is being prepared for a future Journal of Open Source Software submission. This public repository is the beginning of the open maintenance history; it is not yet a claim that the JOSS submission gates have been met. In particular, the project still needs sustained public development, a final author list, a JOSS paper, and additional research-use evidence before submission.

The repository includes an OSI-approved MIT license, tests, contribution guidance, and continuous integration. See [`JOSS_READINESS.md`](JOSS_READINESS.md) for the current checklist and explicit non-claims.

## License

The software is released under the MIT License; see [`LICENSE`](LICENSE).

