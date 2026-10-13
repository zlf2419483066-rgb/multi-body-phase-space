# Four-body demo

This package is the more complete of the two public demos. It provides an auditable graph-domain model for four-body phase-space diagrams and a layered workflow for topology enumeration, momentum labeling, invariant analysis, identity normalization, reporting, formula-input records, and optional drawing.

## Install and test

From the repository root:

```powershell
python -m pip install -e ./four-body
python -m unittest discover -s four-body/tests -p "test_*.py"
```

The core package uses the Python standard library. Drawing integration additionally uses Matplotlib and Graphviz. The tests skip external drawing checks when `dot` is unavailable.

## Example

```powershell
python four-body/examples/enumerate.py --loop-count 0
```

## Direct formula generation

The formula module covers loop counts 0–3, with 54 topologies, 143 particle-relabeling classes and 3,363 fixed identities. It provides graph-derived covariant tree reduction, mixed invariant/angular coordinates, mirror branches, physical-domain checks, and six-step formula dossiers.

Export and benchmark the two tree classes:

```powershell
python four-body/examples/export_all_four_body_formulas.py --all --loops 0 --benchmark --masses 1 1.5 0.8 1.2 --total-mass 6
```

Use `--all` without `--loops` for all 143 classes. Outputs use the external data root; existing export directories are not overwritten.

Run the independent source-density audit with an explicit output directory:

```powershell
python four-body/examples/check_yu2022_all_tables.py --output ../yu2022-audit
```

The formula domain currently assumes four dimensions, positive masses, a parent above threshold, and a nondegenerate noncoplanar open region. Finite-point agreement is not a global proof. Source discrepancies and remaining branch/boundary questions are documented in [`../docs/validation.md`](../docs/validation.md).

Small frozen identity sources and the low-loop audit inputs required for catalog regression tests are shipped under the original `文档/` paths to preserve source hashes. Full generated dossiers and the internal experiment archive are not shipped.

