"""UI state and exact formula checks. Requires optional ipywidgets/IPython."""

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

from phase_space_graph.notebook_browser import (
    AuditBrowser, dependence_equation, linear, stage,
)


class FormulaTests(unittest.TestCase):
    def test_exact_fraction_and_negative_coefficients(self):
        self.assertEqual(linear([('-1/2', 'p_1'), ('2', 'p_2'), (0, 'p_3')]),
                         r'-\frac{1}{2} p_1 +2 p_2')

    def test_hand_derived_triple_mass_relation(self):
        certificate = {
            'variables': [[1, 2], [1, 3], [2, 3], [1, 2, 3]],
            'multipliers': ['1', '1', '1', '-1'],
            'rhs_mass': {'parent_coefficient': '0',
                         'particle_coefficients': ['1', '1', '1', '0']},
        }
        self.assertEqual(dependence_equation(certificate),
                         's_{1,2} +s_{1,3} +s_{2,3} -s_{1,2,3} = m_{1}^2 +m_{2}^2 +m_{3}^2')

    def test_partial_search_is_not_rejected_as_a_topology(self):
        self.assertEqual(stage({'status': 'incomplete', 'valid_results': []}), '待完成／运行失败')


@unittest.skipUnless(importlib.util.find_spec('ipywidgets') and importlib.util.find_spec('IPython'),
                     'Install the optional notebook dependencies to validate the UI')
class BrowserTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.browser = AuditBrowser(root)
        self.addCleanup(self.browser.view.close)
        # UI renderer outputs are not needed for state tests.
        self.browser.display = Mock()
        self.browser.clear = Mock()
        self.browser.directory.mkdir(parents=True)
        self.fixture = {
            'status': 'complete', 'counts': {'topologies': 1},
            'topology_report': {'n': 4, 'loop_count': 0},
            'labeling_reports': [{
                'status': 'complete', 'counts': {'momentum_valid': 2},
                'topology': {'n': 4, 'vertices': [0], 'edges': [],
                             'final_slots': [{'vertex': 0, 'count': 4}], 'initial_vertex': 0},
                'valid_results': [
                    {'labeling': {'final_particles': [], 'single_lines': []},
                     'independence': {'labels': ['s12', 's123']}},
                    {'labeling': {'final_particles': [], 'single_lines': []},
                     'independence': {'labels': ['s13', 's134']}},
                ],
            }],
        }
        self.path = self.browser.directory / 'g08_四体_l0_测试.json'
        self.path.write_text(json.dumps(self.fixture), encoding='utf-8')

    def test_browsing_does_not_run_enumeration_and_exposes_all_labels(self):
        with patch('phase_space_graph.reporting.run_audit') as run:
            self.browser._files()
            run.assert_not_called()
        self.assertEqual(len(self.browser.label.options), 2)
        self.browser.label.value = 1
        self.assertEqual(self.browser.records[1][1]['independence']['labels'], ['s13', 's134'])

    def test_empty_filter_and_old_missing_witness(self):
        self.browser._files()
        self.browser.filter.value = '独立性未通过'
        self.assertIsNone(self.browser.graph.value)
        rejected = deepcopy(self.fixture)
        rejected['labeling_reports'][0]['valid_results'] = []
        self.path.write_text(json.dumps(rejected), encoding='utf-8')
        self.browser._files()
        self.assertEqual(self.browser.graph.value, 0)
        self.assertIsNone(self.browser.label.value)
        self.assertEqual(self.browser.records, [])

    def test_fresh_computation_updates_memory_without_saving(self):
        self.browser._files()
        fresh = deepcopy(self.fixture)
        fresh['labeling_reports'][0]['valid_results'] = fresh['labeling_reports'][0]['valid_results'][:1]
        with patch('phase_space_graph.label_catalog.run_audit') as run:
            run.return_value.to_dict.return_value = fresh
            self.browser._run(None)
        self.assertIs(self.browser.audit, fresh)
        self.assertEqual(len(self.browser.label.options), 1)
        self.assertIsNone(self.browser.source.value)
        self.assertEqual(json.loads(self.path.read_text(encoding='utf-8')), self.fixture)

    def test_save_twice_preserves_original_and_refresh_reads_saved_data(self):
        self.browser._files()
        original_bytes = self.path.read_bytes()
        self.browser._save(None)
        first = self.browser.source.value
        self.browser._save(None)
        second = self.browser.source.value
        self.assertNotEqual(first, second)
        self.assertEqual(self.path.read_bytes(), original_bytes)
        self.browser._files()
        self.assertEqual(self.browser.source.value, second)
        self.assertEqual(self.browser.audit, self.fixture)

    def test_failed_compute_keeps_data_and_reenables_controls(self):
        self.browser._files()
        previous = self.browser.audit
        with patch('phase_space_graph.label_catalog.run_audit', side_effect=RuntimeError('test failure')):
            self.browser._run(None)
        self.assertIs(self.browser.audit, previous)
        self.assertFalse(self.browser.busy)
        self.assertFalse(self.browser.run.disabled)
        self.assertIn('计算失败', self.browser.notice.value)

    def test_invalid_momentum_is_not_presented_as_subset_mass(self):
        topo = {'n': 4, 'vertices': [0, 1], 'initial_vertex': 0,
                'edges': [{'source': 0, 'target': 1, 'edge_type': 'double', 'multiplicity': 1}]}
        momentum = {'double_momenta': [{'edge_index': 0, 'occurrence': 0,
                                       'coefficients': [1, -1, 2, 0]}],
                    'vertex_residuals': [], 'verified': False}
        self.browser._physics(topo, {'final_particles': [], 'single_lines': []}, momentum, None)
        rendered = [getattr(call.args[0], 'data', '') for call in self.browser.display.call_args_list]
        self.assertIn('q_{1,1} = p_{1} -p_{2} +2 p_{3}', rendered)
        self.assertFalse(any('q_{1,1}^2 = s_' in str(item) for item in rendered))


if __name__ == '__main__':
    unittest.main()
