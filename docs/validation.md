# Validation scope

This page separates executable software checks from research claims. A passing unit test is evidence about the tested software contract; it is not, by itself, a proof that the complete physical model or a literature transcription is correct.

## Four-body demo

The four-body package currently exposes a layered workflow:

1. validate the graph-domain objects and local graph rules;
2. enumerate canonical topology representatives;
3. recover and normalize momentum labels;
4. classify invariant combinations and record dependence certificates;
5. preserve particle-label identity and formula-input records;
6. optionally create validated drawing data.

The project’s current internal convention distinguishes momentum candidates from effective topologies. The working research record uses 43 momentum-qualified candidates and 33 candidates after the invariant-independence stage. These values are scope-specific results of the current convention, not universal constants and not a replacement for literature-by-literature validation.

The public test suite covers the domain, canonicalization, enumeration, labeling, reporting, drawing geometry, and formula-input contracts. Full formula normalization and every historical audit archive are intentionally outside the first public checkout.

## Five-body demo

The five-body package is intentionally staged. The public API supports the bounded enumeration scope stated by the source code: five-body diagrams with loop count 0, 1, 2, or 3. It provides structural candidates, canonicalization, momentum-label recovery, candidate identity handling, checkpoint/catalog helpers, and optional drawing/reporting.

The following are not claimed as complete in this release:

- invariant-independence evaluation;
- final effective-topology selection;
- complete non-equivalent identity enumeration at higher loop counts;
- a validated five-body phase-space formula;
- exhaustive high-loop catalogs shipped in Git.

The five-body tests compare optimized and reference paths on bounded fixtures and low-loop cases. Long-running production catalogs and their machine-specific receipts remain external data products.

## Evidence policy

Reports should state the run scope, stop reason, and whether the result is complete. A timeout, process interruption, or output limit produces a partial result, not an empty solution set. Future releases should add reproducible public fixtures or release assets when a generated catalog is needed for a claim.

