import copy
import json
import sys
import subprocess
import os
import unittest

from aster.benchmark.history_reader import audit, build_cases, feature_vector, read_facts, replay


class HistoryReaderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = build_cases()

    def test_full_suite_replay_and_no_model_dependency(self):
        report = audit(self.rows)
        self.assertEqual(report['counts'], {'train': 80, 'dev': 40})
        self.assertEqual(report['cross_split_overlap'], 0)
        self.assertEqual(report['model_scored'], 0)
        subprocess.run([sys.executable, '-c',
                        'import sys; import aster.benchmark.history_reader; assert \"torch\" not in sys.modules'],
                       env={**os.environ, 'PYTHONPATH': 'src'}, check=True)
        self.assertEqual(report['state_contrasts'], 48)

    def test_same_value_write_revokes_freshness(self):
        by_family = {r['family']: r for r in self.rows if r['leakage_group'] == 'train/task0'}
        self.assertEqual(by_family['normal3']['facts'], {'C': True, 'M': True, 'F': True})
        self.assertEqual(by_family['same_write']['facts'], {'C': True, 'M': True, 'F': False})
        self.assertEqual(by_family['wrong_overwrite']['facts'], {'C': True, 'M': False, 'F': False})
        self.assertEqual(by_family['unrelated_write']['facts'], by_family['normal3']['facts'])

    def test_unknown_preserved_and_other_calculation_ignored(self):
        row = next(r for r in self.rows if r['family'] == 'other_calculation')
        self.assertEqual(row['facts'], {'C': False, 'M': None, 'F': None})
        self.assertEqual(feature_vector(row['facts']), [0, 0, 0])

    def test_label_and_saved_transition_tampering_rejected(self):
        rows = copy.deepcopy(self.rows)
        rows[0]['facts']['C'] = True
        with self.assertRaisesRegex(ValueError, 'label mismatch'):
            audit(rows)
        row = copy.deepcopy(self.rows[1])
        row['trajectory'][0]['observation']['output'] = -777
        with self.assertRaisesRegex(ValueError, 'replay mismatch'):
            replay(row)

    def test_input_excludes_supervision_and_schema_rejects_extra_fields(self):
        data = json.loads(self.rows[1]['input'])
        self.assertEqual(set(data), {'schema', 'task', 'memory', 'events'})
        self.assertEqual(set(data['events'][0]), {'action', 'observation'})
        data['target'] = 'calculator'
        with self.assertRaisesRegex(ValueError, 'schema mismatch'):
            read_facts(json.dumps(data))


if __name__ == '__main__':
    unittest.main()
