import json
from pathlib import Path
import tempfile
import unittest
from src.artifacts.paths import training_run_path, resolve_training_run


class TrainingPathTests(unittest.TestCase):
    def test_tagged_and_untagged_destinations(self):
        root = Path('/project')
        self.assertEqual(training_run_path(root, 'gin_depth_training.ipynb', 'exclusive', timestamp='20260918_123456'),
                         root/'training_results/gin_depth_training/exclusive_20260918_123456')
        self.assertEqual(training_run_path(root, 'fate_masking_training', timestamp='20260918_123456'),
                         root/'training_results/fate_masking_training/20260918_123456')
        for tag in ('../escape', '/absolute', 'two words'):
            with self.assertRaises(ValueError):training_run_path(root, 'gin_depth_training', tag)

    def test_selection_is_explicit_when_multiple_runs_exist(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaisesRegex(ValueError, 'Available runs'):
                resolve_training_run(root, 'gin_depth_training')
            a = training_run_path(root, 'gin_depth_training', timestamp='20260918_123456')
            a.mkdir(parents=True);(a/'settings.json').write_text('{}');(a/'models.json').write_text('[]')
            self.assertEqual(resolve_training_run(root, 'gin_depth_training'), a)
            b = training_run_path(root, 'gin_depth_training', 'second', timestamp='20260918_123457')
            b.mkdir();(b/'settings.json').write_text('{}');(b/'models.json').write_text('[]')
            with self.assertRaisesRegex(ValueError, 'Available runs'):
                resolve_training_run(root, 'gin_depth_training')
            self.assertEqual(resolve_training_run(root, 'gin_depth_training', b.name), b)
            self.assertEqual(resolve_training_run(root, 'gin_depth_training', str(a)), a)
