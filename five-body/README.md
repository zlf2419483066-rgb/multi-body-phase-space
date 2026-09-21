# Five-body demo

This package is a staged five-body extension of the graph-method prototype. It separates structural candidates, momentum-qualified candidates, identity handling, checkpoint/catalog support, and optional drawing/reporting so that later physical filters can be added without silently changing earlier counts.

The public enumeration API currently supports five-body loop counts 0–3. Higher-loop production catalogs are intentionally not part of this checkout. Invariant-independence evaluation, final effective-topology selection, and a complete five-body phase-space formula remain future work.

## Install and test

From the repository root:

```powershell
python -m pip install -e ./five-body
python -m unittest discover -s five-body/tests -p "test_*.py"
```

## Examples

```powershell
python five-body/examples/run_candidates.py --loop-count 0 --no-render --max-seconds 30
python five-body/examples/run_candidates.py --fixture A --no-render --max-seconds 30
```

Generated catalogs and run reports are data products. Keep them outside Git unless a future release explicitly defines a small, reproducible fixture or archive asset.

