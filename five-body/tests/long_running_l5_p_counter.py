from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest


DEMO = Path(__file__).parents[1]
RUNNER_PATH = DEMO / "examples" / "count_l5_momentum_candidates.py"
sys.path.insert(0, str(DEMO / "src"))
spec = importlib.util.spec_from_file_location("count_l5_momentum_candidates", RUNNER_PATH)
assert spec is not None and spec.loader is not None
runner = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = runner
spec.loader.exec_module(runner)

from phase_space_graph5 import EdgeType, InternalEdge, Topology, enumerate_topologies  # noqa: E402
from phase_space_graph5.flow_labeling import _flow_is_legal, _scalar_double_flows  # noqa: E402


class L5PCountOnlyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        def l5_chain(last_single_target: int) -> Topology:
            edges = [InternalEdge(index, index + 1, EdgeType.DOUBLE) for index in range(8)]
            edges.extend(
                InternalEdge(index, index + 1, EdgeType.SINGLE)
                for index in range(4)
            )
            edges.append(InternalEdge(4, last_single_target, EdgeType.SINGLE))
            outgoing = {vertex: 0 for vertex in range(9)}
            for edge in edges:
                outgoing[edge.source] += edge.multiplicity
            return Topology(
                5,
                range(9),
                edges,
                0,
                {vertex: 2 - outgoing[vertex] for vertex in range(9)},
            )

        cls.legal = l5_chain(5)
        cls.illegal = l5_chain(8)
        assert _flow_is_legal(_scalar_double_flows(cls.legal))
        assert not _flow_is_legal(_scalar_double_flows(cls.illegal))

    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _db(self, name: str, rows: list[tuple[object, int, int]]) -> Path:
        path = self.root / name
        path.unlink(missing_ok=True)
        db = sqlite3.connect(path)
        db.execute(
            "CREATE TABLE canonical_keys (key_blob BLOB PRIMARY KEY, topology_blob BLOB NOT NULL, "
            "skeleton_index INTEGER NOT NULL, orientation_mask INTEGER NOT NULL)"
        )
        for ordinal, (topology, skeleton, mask) in enumerate(rows):
            db.execute(
                "INSERT INTO canonical_keys VALUES(?,?,?,?)",
                (bytes([ordinal]), runner.packed_codec.encode_topology(topology), skeleton, mask),
            )
        db.commit()
        db.close()
        return path

    def _inputs(self) -> tuple[Path, Path]:
        db1 = self._db(
            "one.sqlite3",
            [(self.legal, 0, 0), (self.illegal, 0, 1), (self.legal, 1, 0), (self.illegal, 1, 1)],
        )
        db2 = self._db("two.sqlite3", [(self.legal, 2, 0)])
        old1, old2 = r"C:\old\one.sqlite3", r"D:\old\two.sqlite3"
        inventory = {
            "loop_count": 5,
            "generation_complete": True,
            "total_expected": 4,
            "total_generation_rows": 4,
            "skeletons": [
                {"skeleton_index": 0, "representative_masks": [0, 1], "rows": 2,
                 "components": [{"database": {"path": old1}, "selector": {"skeleton_index_equals": 0, "orientation_mask_in": [0, 1]}, "rows": 2}]},
                {"skeleton_index": 1, "representative_masks": [0], "rows": 1,
                 "components": [{"database": {"path": old1}, "selector": {"skeleton_index_equals": 1, "orientation_mask_in": [0]}, "rows": 1}]},
                {"skeleton_index": 2, "representative_masks": [0], "rows": 1,
                 "components": [{"database": {"path": old2}, "selector": {"skeleton_index_equals": 2, "orientation_mask_in": [0]}, "rows": 1}]},
            ],
        }
        relocation = {
            "entries": [
                {"source": old1, "target": str(db1), "status": "verified", "size_bytes": db1.stat().st_size},
                {"source": old2, "target": str(db2), "status": "verified", "size_bytes": db2.stat().st_size},
            ]
        }
        inv_path, move_path = self.root / "inventory.json", self.root / "relocation.json"
        inv_path.write_text(json.dumps(inventory), encoding="utf-8")
        move_path.write_text(json.dumps(relocation), encoding="utf-8")
        return inv_path, move_path

    def test_exact_selectors_exclude_unselected_rows_and_count_frozen_P(self) -> None:
        inventory, relocation = self._inputs()
        work, scope = runner.load_work(inventory, relocation)
        self.assertEqual(scope["inventory_generation_rows"], 4)
        results = [runner.count_database(item) for item in work]
        self.assertEqual(sum(row["selected_rows"] for row in results), 4)
        self.assertEqual(sum(row["P_rows"] for row in results), 3)

    def test_packed_fast_path_matches_frozen_predicate_and_history_on_l0_l3(self) -> None:
        historical = {0: (3, 3), 1: (69, 18), 2: (1731, 99), 3: (48492, 533)}
        for loop_count, (expected_S, expected_P) in historical.items():
            report = enumerate_topologies(5, loop_count)
            self.assertTrue(report.complete)
            self.assertEqual(report.unique_count, expected_S)
            actual_P = 0
            for topology in report.topologies:
                blob = runner.packed_codec.encode_topology(topology)
                expected = _flow_is_legal(_scalar_double_flows(topology))
                actual_P += int(expected)
                self.assertEqual(
                    runner.packed_flow_is_legal(blob),
                    expected,
                    (loop_count, topology.to_dict()),
                )
            self.assertEqual(actual_P, expected_P)

    def test_checkpoint_resume_and_relocation_binding(self) -> None:
        inventory, relocation = self._inputs()
        output = self.root / "report.json"
        first = runner.run_worker(
            inventory, relocation, output, stop_after_databases=1
        )
        self.assertFalse(first["complete"])
        self.assertEqual(first["totals"]["databases_completed"], 1)
        final = runner.run_worker(inventory, relocation, output)
        self.assertTrue(final["complete"])
        self.assertEqual(final["totals"]["S_rows"], 4)
        self.assertEqual(final["totals"]["P_rows"], 3)
        self.assertEqual(final["totals"]["scalar_fast_P_rows"], 3)
        self.assertEqual(final["totals"]["frozen_P_rows"], 3)
        self.assertEqual(final["totals"]["witness_verified_P_rows"], 3)
        self.assertEqual(final["totals"]["databases_completed"], 2)
        rewritten = json.loads(relocation.read_text(encoding="utf-8"))
        rewritten["extra"] = True
        relocation.write_text(json.dumps(rewritten), encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "resume_binding_mismatch"):
            runner.run_worker(inventory, relocation, output)

    def test_lpt_assignment_and_strict_merge(self) -> None:
        inventory, relocation = self._inputs()
        work, _ = runner.load_work(inventory, relocation)
        assignments = runner.assign_work_lpt(work, 2)
        loads = [sum(c.expected_rows for item in shard for c in item.components) for shard in assignments]
        self.assertEqual(sorted(loads), [1, 3])
        reports = []
        for worker in range(2):
            path = self.root / f"worker-{worker}.json"
            report = runner.run_worker(
                inventory, relocation, path, worker_index=worker, worker_count=2
            )
            self.assertTrue(report["complete"])
            reports.append(path)
        merged = runner.merge_reports(
            inventory, relocation, reports, self.root / "merged.json"
        )
        self.assertEqual(merged["totals"]["S_rows"], 4)
        self.assertEqual(merged["totals"]["P_rows"], 3)
        self.assertEqual(merged["per_loop"]["5"]["P_rows"], 3)
        self.assertEqual(merged["scope"]["exact_loop_count"], 5)
        pristine = reports[1].read_text(encoding="utf-8")
        damaged_loop = json.loads(pristine)
        first = next(iter(damaged_loop["completed_databases"].values()))
        first["per_loop"]["5"]["P_rows"] += 1
        reports[1].write_text(json.dumps(damaged_loop), encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "database_result_invalid"):
            runner.merge_reports(inventory, relocation, reports, self.root / "bad-loop.json")
        reports[1].write_text(pristine, encoding="utf-8")
        damaged = json.loads(reports[1].read_text(encoding="utf-8"))
        damaged["scope"]["assigned_work_ids"] = []
        reports[1].write_text(json.dumps(damaged), encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "assignment_mismatch"):
            runner.merge_reports(inventory, relocation, reports, self.root / "bad.json")

    def test_selected_non_l5_blob_is_rejected(self) -> None:
        inventory, relocation = self._inputs()
        raw = json.loads(relocation.read_text(encoding="utf-8"))
        database = Path(raw["entries"][0]["target"])
        low_loop = enumerate_topologies(5, 1).topologies[0]
        db = sqlite3.connect(database)
        db.execute(
            "UPDATE canonical_keys SET topology_blob=? WHERE key_blob=(SELECT key_blob FROM canonical_keys LIMIT 1)",
            (runner.packed_codec.encode_topology(low_loop),),
        )
        db.commit()
        db.close()
        # Updating an existing row does not change SQLite's file size, so the
        # relocation size binding still passes and the row-level scope check
        # is what rejects this mixed-loop corruption.
        work, _ = runner.load_work(inventory, relocation)
        with self.assertRaisesRegex(RuntimeError, "outside_exact_L5_scope"):
            runner.count_database(next(item for item in work if item.database == database.resolve()))

    def test_selector_overlap_and_nonempty_wal_are_rejected(self) -> None:
        inventory, relocation = self._inputs()
        raw = json.loads(inventory.read_text(encoding="utf-8"))
        raw["skeletons"][1]["components"][0]["selector"]["skeleton_index_equals"] = 0
        inventory.write_text(json.dumps(raw), encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "component_skeleton_mismatch"):
            runner.load_work(inventory, relocation)
        inventory, relocation = self._inputs()
        work, _ = runner.load_work(inventory, relocation)
        Path(str(work[0].database) + "-wal").write_bytes(b"unsafe")
        with self.assertRaisesRegex(RuntimeError, "nonempty_sqlite_wal"):
            runner.count_database(work[0])

    def test_empty_wal_with_stale_nonempty_shm_is_allowed_and_recorded(self) -> None:
        inventory, relocation = self._inputs()
        work, _ = runner.load_work(inventory, relocation)
        database = work[0].database
        Path(str(database) + "-wal").write_bytes(b"")
        Path(str(database) + "-shm").write_bytes(b"stale-shared-memory")
        result = runner.count_database(work[0])
        self.assertTrue(result["sqlite_sidecars"]["stale_shm_ignored"])
        self.assertEqual(result["sqlite_sidecars"]["wal"]["bytes"], 0)
        self.assertEqual(
            result["sqlite_sidecars"]["shm"]["bytes"], len(b"stale-shared-memory")
        )


if __name__ == "__main__":
    unittest.main()
