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

For the research scope and evidence boundary, see [`../docs/validation.md`](../docs/validation.md). The historical internal audit archives are not required for the public unit suite and are not shipped here.

