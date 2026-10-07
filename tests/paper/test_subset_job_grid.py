"""The conservative launcher partitions a configurable grid without loading data."""
import json
from pathlib import Path
import runpy
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]


class SubsetJobGridTests(unittest.TestCase):
    def test_grid_is_partitioned_by_fold_and_model(self):
        module = runpy.run_path(str(ROOT/'scripts/run_paper_training_folds.py'))
        settings = """SETTINGS = dict(
            marker_subsets={'AldoB': ['AldoB'], 'LGR5': ['LGR5']},
            depths=[2], subset_depths={'LGR5': [0, 1, 2, 3, 4]},
            hidden_dims=[16], seeds=[7, 8], edge_loss_params=dict(weighted=False))
        """
        with tempfile.TemporaryDirectory() as directory:
            notebook = Path(directory)/'training.ipynb'
            notebook.write_text(json.dumps({'cells': [{}]*4 + [{'source': settings.splitlines(True)}]}))
            jobs = module['subset_jobs'](notebook, [1, 3])
            self.assertEqual(len(jobs), 24)
            self.assertEqual(len(set(jobs)), 24)
            self.assertEqual(jobs, [job for fold in [1, 3] for job in module['subset_jobs'](notebook, [fold])])
        for fold in [1, 3]:
            keys = {key for f, key in jobs if f == fold}
            self.assertEqual(len(keys), 12)
            for seed in [7, 8]:
                for depth in range(5):
                    self.assertIn(f'gin_LGR5_intact_d{depth}_h16_f{fold}_s{seed}', keys)
                self.assertIn(f'gin_AldoB_intact_d2_h16_f{fold}_s{seed}', keys)


if __name__ == '__main__':
    unittest.main()
