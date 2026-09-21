"""Physical counterexamples found in the 2026-09-16 review (no large search)."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

from im_fixtures import fixture_a, fixture_b, fixture_d, fixture_f, _fixture_c
from phase_space_graph import (
    CountMode, EnumerationLimits, LabelCatalogBundle, Labeling, NormalizationResult,
    SingleLineLabel, SourceDiagram, build_formula_input, build_physics_card,
    build_topology_catalog, canonicalize_physical_labeling, catalog_from_audit,
    enumerate_labelings, match_source_record, normalize_physical_labeling,
    read_label_catalog, run_label_catalog, write_label_catalog,
)
from phase_space_graph.notebook_browser import AuditBrowser


def checked(f):
    return normalize_physical_labeling(f.topology, f.labeling)


def catalog(f, mode="fixed_labels"):
    return build_topology_catalog(f.topology, [checked(f)], mode=mode,
        search_evidence={"search_status": "complete", "records_complete": True, "import_status": "complete"})


class ReviewRegressionTests(unittest.TestCase):
    def test_parallel_occurrence_order_has_one_canonical_record(self):
        f = fixture_f()
        swapped = Labeling(f.labeling.final_particles, [
            SingleLineLabel(x.edge_index, x.occurrence, 3-x.particle_id if x.edge_index == 4 else x.particle_id)
            for x in f.labeling.single_lines])
        a = canonicalize_physical_labeling(f.record)
        b = canonicalize_physical_labeling(normalize_physical_labeling(f.topology, swapped).record)
        self.assertEqual(a.key, b.key)
        self.assertEqual(a.canonical_record, b.canonical_record)

    def test_generic_representative_does_not_cover_twelve_fixed_labels(self):
        f = fixture_a()
        generic = enumerate_labelings(f.topology, count_mode=CountMode.PARTICLE_RELABELING)
        fixed = enumerate_labelings(f.topology, count_mode=CountMode.FIXED_LABELS)
        self.assertEqual((generic.unique_count, fixed.unique_count), (1, 12))
        c = catalog_from_audit({"labeling_reports": [generic.to_dict()]}, mode="fixed_labels").catalogs[0]
        self.assertEqual(c.identity_count, 1)
        self.assertEqual(c.existence, "yes")
        self.assertFalse(c.enumeration_complete)
        reverse = catalog_from_audit({"labeling_reports": [fixed.to_dict()]}, mode="particle_relabeling").catalogs[0]
        self.assertEqual(reverse.identity_count, 1)
        self.assertTrue(reverse.enumeration_complete)

    def test_missing_or_unknown_mode_never_proves_complete(self):
        r = enumerate_labelings(fixture_a().topology).to_dict()
        for value in (None, "unknown"):
            r["count_mode"] = value
            c = catalog_from_audit({"labeling_reports": [r]}, mode="fixed_labels").catalogs[0]
            self.assertFalse(c.enumeration_complete)

    def test_missing_effective_records_are_unknown_not_no_solution(self):
        r = enumerate_labelings(fixture_a().topology).to_dict()
        r["valid_results"] = []
        c = catalog_from_audit({"labeling_reports": [r]}, mode="fixed_labels").catalogs[0]
        self.assertEqual(c.existence, "unknown")
        self.assertFalse(c.enumeration_complete)

    def test_missing_topology_report_is_preserved_as_unsearched(self):
        a, b = fixture_a(), fixture_b()
        audit = {"topology_report": {"n":4, "loop_count":0, "status":"complete",
                 "topologies":[a.topology.to_dict(), b.topology.to_dict()]},
                 "labeling_reports":[enumerate_labelings(a.topology).to_dict()]}
        bundle = catalog_from_audit(audit, mode="fixed_labels")
        self.assertEqual(len(bundle.catalogs), 2)
        self.assertFalse(bundle.enumeration_complete)
        self.assertTrue(any(c.existence == "unknown" for c in bundle.catalogs))

    def test_partial_graph_search_is_not_complete_universe(self):
        run = run_label_catalog(4, 0, mode="fixed_labels", topology_limits=EnumerationLimits(max_topologies=1))
        self.assertEqual(run.legacy_audit.status.value, "incomplete")
        self.assertTrue(run.label_catalog.catalogs[0].enumeration_complete)
        self.assertFalse(run.label_catalog.topology_scope_complete)
        self.assertFalse(run.label_catalog.enumeration_complete)

    def test_real_rejection_witnesses_survive_two_round_trips(self):
        r = enumerate_labelings(fixture_d().topology).to_dict()
        bundle = catalog_from_audit({"labeling_reports":[r]}, mode="fixed_labels")
        with TemporaryDirectory() as tmp:
            for i in range(2):
                path = Path(tmp)/f"{i}.json"
                write_label_catalog(bundle, path)
                bundle = read_label_catalog(path)
                witnesses = bundle.catalogs[0].rejected_witnesses
                codes = {issue.code for w in witnesses for issue in w.issues}
                self.assertTrue(set(r["failure_witnesses"]) <= codes)
                momentum_rejected = next(w for w in witnesses if w.stage == "momentum_rejected")
                self.assertIsNotNone(momentum_rejected.legacy_evaluation)
                self.assertTrue(momentum_rejected.legacy_evaluation.labeling.single_lines)

    def test_catalog_corruption_downgrades_local_status_and_source_match(self):
        f = fixture_b()
        original = LabelCatalogBundle((catalog(f),)).to_dict()
        for corruption in ("digest", "missing_record", "rank"):
            payload = deepcopy(original)
            identity = payload["catalogs"][0]["identities"][0]
            if corruption == "digest":
                identity["digest"] = "wrong"
            elif corruption == "rank":
                identity["canonical_record"]["independence"]["rank"] = 99
            else:
                identity["canonical_record"].pop("complete_labeling")
            with self.subTest(corruption=corruption), TemporaryDirectory() as tmp:
                p = Path(tmp)/"damaged.json"
                p.write_text(json.dumps(payload), encoding="utf-8")
                loaded = read_label_catalog(p)
                c = loaded.catalogs[0]
                self.assertFalse(c.enumeration_complete)
                self.assertEqual(c.import_status, "failed")
                self.assertNotEqual(c.existence, "no")
                src = SourceDiagram("manual",1,"B",f.topology,True,f.labeling)
                self.assertFalse(match_source_record(src, loaded, mode="fixed_labels")["catalog_match"])
                p2 = Path(tmp)/"again.json"
                write_label_catalog(loaded, p2)
                self.assertFalse(read_label_catalog(p2).catalogs[0].enumeration_complete)

    def test_single_complete_topology_does_not_prove_another_is_absent(self):
        c = _fixture_c(3)
        src = SourceDiagram("manual",1,"C",c.topology,True,c.labeling)
        result = match_source_record(src, [catalog(fixture_b())], mode="fixed_labels")
        self.assertEqual(result["status"], "inconclusive")
        self.assertFalse(result["source_only"])

    def test_deleting_nondefault_saved_identity_cannot_remain_complete(self):
        f = fixture_a()
        report = enumerate_labelings(f.topology).to_dict()
        payload = catalog_from_audit({"labeling_reports":[report]}, mode="fixed_labels").to_dict()
        payload["catalogs"][0]["identities"].pop()
        with TemporaryDirectory() as tmp:
            p = Path(tmp)/"missing-row.json"
            p.write_text(json.dumps(payload), encoding="utf-8")
            result = read_label_catalog(p).catalogs[0]
            self.assertEqual(result.identity_count, 11)
            self.assertEqual(result.existence, "yes")
            self.assertFalse(result.enumeration_complete)
            self.assertTrue(any(i.code == "stored_identity_manifest_mismatch" for i in result.issues))

    def test_wrong_search_scope_cannot_certify_no_solution(self):
        result = build_topology_catalog(fixture_a().topology, [], mode="fixed_labels",
            search_evidence={"n":5, "loop_count":1, "search_status":"complete", "records_complete":True})
        self.assertEqual(result.existence, "unknown")
        self.assertFalse(result.enumeration_complete)

    def test_formula_rejects_unverified_or_cross_attached_record(self):
        f = fixture_b()
        bad = replace(f.record, momentum=replace(f.record.momentum, verified=False))
        with self.assertRaises(ValueError):
            build_formula_input(bad, measure_convention="jing_psf")
        other_identity = canonicalize_physical_labeling(_fixture_c(3).record)
        with self.assertRaises(ValueError):
            build_formula_input(checked(f), measure_convention="jing_psf", identity=other_identity)
        with self.assertRaises(ValueError):
            build_physics_card(checked(f), identity=other_identity)

    def test_formula_preserves_all_local_refs_and_pending_steps(self):
        card = build_formula_input(checked(fixture_b()), measure_convention="jing_psf")
        factor = card.factors[2]
        self.assertEqual(len(factor.incoming_references), 2)
        self.assertEqual({r["role"] for r in factor.incoming_references}, {"double", "single"})
        self.assertEqual(factor.to_dict()["incoming_recovered"], "p_2 + p_3 + p_4")
        self.assertEqual([r["slot"] for r in card.factors[3].connection_references], [0, 1])
        for item in card.particle_measures:
            self.assertEqual(len(item.to_dict()["retained_occurrences"]), 1)
            self.assertEqual(len(item.to_dict()["cancellation_occurrences"]), item.cancellation_count)
        self.assertTrue(all(f.reference_frame is None and f.angles is None for f in card.factors))
        self.assertFalse(card.formula_verified)

    def test_repeated_parent_coefficients_remain_an_explicit_square(self):
        # Representation boundary only; no claim that this defines a valid
        # complete graph or a globally valid phase-space parameterization.
        from phase_space_graph.formula_inputs import _mass_symbols
        mass, squared = _mass_symbols((2,1,0,0), 4, squared=False)
        self.assertEqual(squared, "(2 p_1 + p_2)^2")
        self.assertNotEqual(mass, "m_12")


class ReviewBrowserTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.ui = AuditBrowser(self.tmp.name)
        self.addCleanup(self.ui.view.close)
        self.ui.display = Mock()
        self.ui.clear = Mock()

    def test_unsaved_mode_switch_keeps_data(self):
        self.ui.catalog = LabelCatalogBundle((catalog(fixture_b()),))
        previous = self.ui.catalog
        self.ui.source_kind = "fresh_run"
        self.ui.source.options = [("unsaved", None)]
        self.ui.save.disabled = False
        self.ui.mode.value = "particle_relabeling"
        self.assertIs(self.ui.catalog, previous)
        self.assertFalse(self.ui.save.disabled)
        self.ui.mode.value = "fixed_labels"
        self.assertTrue(self.ui.records)

    def test_failed_conversion_keeps_consistent_old_state(self):
        self.ui.audit = {"old": True}
        self.ui.catalog = LabelCatalogBundle((catalog(fixture_b()),))
        previous = (self.ui.audit, self.ui.catalog)
        self.ui.source_kind = "legacy_audit"
        with patch("phase_space_graph.label_catalog.run_audit", return_value=Mock(to_dict=Mock(return_value={"new": True}))), \
             patch("phase_space_graph.label_catalog.catalog_from_audit", side_effect=ValueError("conversion failed")):
            self.ui._run(None)
        self.assertIs(self.ui.audit, previous[0])
        self.assertIs(self.ui.catalog, previous[1])
        self.assertIn("计算失败", self.ui.notice.value)

    def test_mixed_modes_are_not_summed_or_listed_together(self):
        self.ui.catalog = LabelCatalogBundle((catalog(fixture_b()), catalog(fixture_b(), "particle_relabeling")))
        self.ui.source_kind = "label_catalog"
        self.ui._summary()
        self.ui._graphs()
        self.assertEqual(len(self.ui.graph.options), 1)
        self.ui.mode.value = "particle_relabeling"
        self.assertEqual(len(self.ui.graph.options), 1)
        self.assertEqual(self.ui.catalog.catalogs[self.ui.graph.value].mode, CountMode.PARTICLE_RELABELING)
