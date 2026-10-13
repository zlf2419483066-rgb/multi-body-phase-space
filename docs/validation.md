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

The public test suite now also covers direct formulas, covariant reduction, mixed-coordinate solutions, catalog selection, export contracts, local numerical comparisons and the Yu2022 source-density audit. The three-loop 43/33 counts above are not the totals across all loops.

### Formula coverage and historical numerical evidence

Work completed on 2026-10-09 covers loops 0/1/2/3 with 2/5/14/33 topologies and 2/6/29/106 marked classes, respectively: 54 topologies, 143 classes and 3,363 fixed identities in total. All 143 graph-derived covariant certificates and all 3,363 identity mappings were replayed in the archived run.

The final internal export run checked eight reference points per class: 1,144 reference evaluations, 2,288 mirror checks and 2,288 density comparisons at two finite-difference steps. The maximum density relative difference was 1.79581e-7 and the maximum momentum relative difference was 3.37549e-14, against a 5e-6 comparison tolerance. These are historical finite-point results, not a new full rerun on publication and not a proof over the entire physical domain.

The implementation is restricted to four-dimensional positive-mass kinematics above threshold in a nondegenerate noncoplanar open region, using the selected D0 convention. Massless/degenerate boundaries, other dashed-edge choices, exhaustive inverse-root coverage, Notebook integration of the new formula entry points, and full manual/literature verification remain open.

### Yu2022 Tables 5–26

The October 9 source audit used two unequal-mass profiles, eight points per profile, both mirrors, two difference steps and two normalization conventions. Eighteen tables matched the printed density directly. Tables 23/24 matched in cosine coordinates; converting to the angle differentials stated in the heading requires explicit sine factors. Tables 21/26 did not pass as printed: Table 21 has a dependent coordinate choice, and an independently derived candidate factor for Table 26 remains a candidate correction, not a verified replacement of the source.

Separate suspected auxiliary kinematic expressions in Tables 5/7/15/18/22 must not be confused with the density comparisons. The audit retains the original failures. Successful program execution or a catalog link does not imply that a formula passed.

### Shipped fixtures

Only the original low-loop G08 audits, the 615 KB compressed three-loop identity source and one reference evidence record are included for reproducible catalog/formula tests. Their original paths and bytes are retained for hash verification. Large generated dossiers, paper copies and private working notes remain outside this tree.

## Five-body demo

The five-body package is intentionally staged. The public API supports the bounded enumeration scope stated by the source code: five-body diagrams with loop count 0, 1, 2, or 3. It provides structural candidates, canonicalization, momentum-label recovery, candidate identity handling, checkpoint/catalog helpers, and optional drawing/reporting.

The following are not claimed as complete in this release:

- complete non-equivalent identity enumeration at higher loop counts;
- a validated five-body phase-space formula;
- exhaustive high-loop catalogs shipped in Git.

The five-body tests compare optimized and reference paths on bounded fixtures and low-loop cases. Long-running production catalogs and their machine-specific receipts remain external data products.

### Generic-positive-mass independence

The new separate independence library supports exact rational algebra, optional SymPy operations, Gram constraints, certificate verification and generic topology aggregation. The existing candidate records are not silently rewritten as accepted effective topologies.

The 2026-09-23 archived research run classified all 18,832 momentum candidates at loops 0–5 for five distinct positive masses, fixed mass parameters and a parent mass above the sum of final masses. Accepted topology counts were 3/18/97/503/2,491/12,102, totaling 15,214; 3,618 were rejected and none remained unknown. Rejection means that no allowed fixed-label assignment produced a qualifying momentum and independence certificate, rather than a failure of a single representative assignment.

These counts are reported from the maintained research record. The full catalog is not included or recomputed by public CI. Special mass cases and a complete five-body formula are not established by this result. The public enumeration API remains limited to loops 0–3.

## Evidence policy

Reports should state the run scope, stop reason, and whether the result is complete. A timeout, process interruption, or output limit produces a partial result, not an empty solution set. Future releases should add reproducible public fixtures or release assets when a generated catalog is needed for a claim.

The [October 13 publication checks](publication-checks-2026-10-13.md) record the actual local test runs, environment repairs and targeted reruns. Catalog and source-audit defaults use the shipped source-checkout fixtures; retain the checkout when installing the demos in editable mode.

