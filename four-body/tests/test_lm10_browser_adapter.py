import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

from im_fixtures import fixture_b, fixture_d, _fixture_c
from phase_space_graph import CountMode, normalize_physical_labeling
from phase_space_graph.label_catalog import catalog_from_audit, write_label_catalog
from phase_space_graph.notebook_browser import (
    AuditBrowser,
    clear_browser_source_cache,
    load_browser_source,
)


def _legacy_report(fixture, *, status="incomplete"):
    checked = normalize_physical_labeling(fixture.topology, fixture.labeling)
    return {
        "status": status,
        "topology": fixture.topology.to_dict(),
        "valid_results": [dict(checked.legacy_evaluation.to_dict(), valid=True)] if checked.legacy_evaluation else [],
        "failure_counts": {},
        "failure_samples": [],
        "failure_witnesses": {},
    }


def _legacy_audit(*fixtures):
    return {
        "status": "incomplete",
        "topology_report": {"n": 4, "loop_count": 1},
        "labeling_reports": [_legacy_report(fixture) for fixture in fixtures],
    }


class LM10BrowserAdapterTests(unittest.TestCase):
    def setUp(self):
        clear_browser_source_cache()

    def test_selected_file_cache_and_explicit_refresh(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "audit.json"
            path.write_text(json.dumps(_legacy_audit(fixture_b())), encoding="utf-8")
            first = load_browser_source(path)
            second = load_browser_source(path)
            self.assertFalse(first.loaded_from_cache)
            self.assertTrue(second.loaded_from_cache)
            clear_browser_source_cache()
            third = load_browser_source(path)
            self.assertFalse(third.loaded_from_cache)

    def test_old_audit_switches_between_all_valid_labels_and_incomplete_status(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "文档/验证与测试/结果图表/四体图枚举/g08_四体_l1_夹具.json"
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps(_legacy_audit(_fixture_c(3), _fixture_c(2), fixture_d()), ensure_ascii=False), encoding="utf-8")
            browser = AuditBrowser(root)
            browser.display = Mock()
            browser.clear = Mock()
            browser.loop.value = 1
            browser._files()
            self.assertTrue(browser._catalog_view())
            self.assertGreaterEqual(len(browser.label.options), 2)
            self.assertIn("s124", browser.label.options[0][0])
            self.assertIn("尚未找全", browser.graph.options[0][0])
            browser.filter.value = '全部拓扑'
            browser.graph.value = 1  # D: a separate topology with a dependent witness
            self.assertTrue(any("dependent" in title or "依赖" in title for title, _ in browser.records))

    def test_catalog_file_round_trip_keeps_mode_and_default_identity(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            bundle = catalog_from_audit(_legacy_audit(fixture_b()), mode=CountMode.PARTICLE_RELABELING)
            path = root / "文档/验证与测试/结果图表/独立标记/label_catalog_四体_l1_夹具.json"
            write_label_catalog(bundle, path)
            source = load_browser_source(path, mode=CountMode.PARTICLE_RELABELING)
            self.assertEqual(source.source_kind, "label_catalog")
            self.assertEqual(source.catalog.catalogs[0].mode, CountMode.PARTICLE_RELABELING)
            self.assertEqual(source.catalog.catalogs[0].default_label_key, bundle.catalogs[0].default_label_key)

    def test_load_failure_keeps_previous_usable_state(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "文档/验证与测试/结果图表/四体图枚举/g08_四体_l1_ok.json"
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps(_legacy_audit(fixture_b())), encoding="utf-8")
            browser = AuditBrowser(root)
            browser.display = Mock()
            browser.clear = Mock()
            browser.loop.value = 1
            browser._files()
            old_audit, old_catalog = browser.audit, browser.catalog
            path.write_text("{broken", encoding="utf-8")
            browser._load()
            self.assertIs(browser.audit, old_audit)
            self.assertIs(browser.catalog, old_catalog)
            self.assertIn("读取失败", browser.notice.value)

    def test_fresh_result_is_saved_as_exclusive_catalog(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            browser = AuditBrowser(root)
            browser.display = Mock()
            browser.clear = Mock()
            browser.loop.value = 1
            browser.catalog = catalog_from_audit(_legacy_audit(fixture_b()), mode=CountMode.FIXED_LABELS)
            browser.audit = None
            browser.source_kind = "fresh_run"
            browser._save(None)
            saved = list(browser.catalog_directory.glob("label_catalog_四体_l1_交互检查_*.json"))
            self.assertEqual(len(saved), 1)
            self.assertEqual(load_browser_source(saved[0]).source_kind, "label_catalog")
            with patch("phase_space_graph.notebook_browser.write_label_catalog", side_effect=FileExistsError("collision")):
                browser._save(None)
            self.assertIn("另存失败", browser.notice.value)


if __name__ == "__main__":
    unittest.main()
