import tempfile
import unittest
import os
import shutil
from pathlib import Path

from phase_space_graph.drawing.archive import (
    export_representatives, load_identity_archive, select_representatives,
)


ARCHIVE = Path(__file__).parents[1] / "文档" / "验证与测试" / "误差分析" / "三圈来源集合审查_2026-09-16" / "run01" / "2544个固定标记完整身份.json.gz"


class DrawingArchiveTests(unittest.TestCase):
    def test_archive_rebuilds_and_selects_one_per_topology(self):
        entries = load_identity_archive(ARCHIVE)
        self.assertEqual(len(entries), 2544)
        representatives = select_representatives(entries)
        self.assertEqual(len(representatives), 33)
        self.assertEqual(len({item.topology_key for item in representatives}), 33)

    def test_bounded_representative_export_writes_manifest(self):
        dot = os.environ.get("PSF_GRAPHVIZ_DOT") or shutil.which("dot")
        if not dot:
            candidate = os.path.expanduser(r"~\Graphviz-16.1.0\bin\dot.exe")
            dot = candidate if os.path.isfile(candidate) else None
        if not dot:
            self.skipTest("Graphviz dot is unavailable; set PSF_GRAPHVIZ_DOT for archive export")
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "representative"
            manifest = export_representatives(ARCHIVE, output, max_items=1, dot_path=dot)
            self.assertEqual(manifest["selected_count"], 1)
            self.assertEqual(manifest["failed_count"], 0)
            self.assertTrue((output / "manifest.json").is_file())
            self.assertEqual(len(list(output.glob("*.svg"))), 1)


if __name__ == "__main__":
    unittest.main()
