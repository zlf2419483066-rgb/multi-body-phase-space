"""Worker cleanup must never signal the caller's process group."""
from pathlib import Path
import os
import runpy
import signal
import subprocess
import sys
import unittest
from unittest.mock import Mock, patch


EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
HELPERS = (("run_candidates.py", "_terminate_worker"),
           ("run_parallel_candidates.py", "_terminate_tree"))


class ProcessGroupCleanupTests(unittest.TestCase):
    def test_only_isolated_worker_groups_receive_group_signals(self):
        for filename, name in HELPERS:
            helper = runpy.run_path(str(EXAMPLES / filename))[name]
            for group in (100, 200):
                with self.subTest(helper=name, worker_group=group):
                    process = Mock(pid=200)
                    snapshot = {
                        "psutil_available": True, "snapshot_complete": True,
                        "pid_alive": False, "process_exited": True,
                        "descendant_alive": [], "descendant_records": [],
                    }
                    with patch.object(sys, "platform", "linux"), \
                         patch.object(os, "getpgid", return_value=group, create=True), \
                         patch.object(os, "getpgrp", return_value=100, create=True), \
                         patch.object(signal, "SIGKILL", 9, create=True), \
                         patch.object(os, "killpg", create=True) as kill_group, \
                         patch.dict(helper.__globals__, {"_process_snapshot": lambda *a, **k: snapshot}):
                        helper(process)
                    if group == 100:
                        kill_group.assert_not_called()
                    else:
                        kill_group.assert_called_once_with(200, 9)
                    process.kill.assert_called_once()
                    process.wait.assert_called_once()

    @unittest.skipIf(os.name == "nt", "real POSIX process-group regression")
    def test_shared_group_cleanup_leaves_its_caller_alive(self):
        # The harness has its own session, so even a regression cannot kill
        # the unittest process or the CI runner.
        for filename, name in HELPERS:
            with self.subTest(helper=name):
                code = (
                    "import runpy,subprocess,sys\n"
                    "helper=runpy.run_path(sys.argv[1])[sys.argv[2]]\n"
                    "child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(20)'])\n"
                    "helper(child)\n"
                    "assert child.poll() is not None\n"
                    "print('caller survived')\n"
                )
                result = subprocess.run(
                    [sys.executable, "-c", code, str(EXAMPLES / filename), name],
                    start_new_session=True, capture_output=True, text=True, timeout=10,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("caller survived", result.stdout)
