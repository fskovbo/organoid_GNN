"""Selection must cover all held-out folds, not an arbitrary catalog row."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import pandas as pd
from src.artifacts.runs import select_depth_records


class DepthSelectionTests(unittest.TestCase):
    def test_complete_grid_ambiguity_and_missing_checkpoints(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            (directory/'splits.json').write_text(json.dumps([{'fold':0}, {'fold':1}]))
            records = pd.DataFrame([dict(key=f'{name}_{depth}_{fold}_{seed}', name=name,
                depth=depth, hidden_dim=8, fold=fold, seed=seed, subset='all', signal='intact')
                for name in ['gin','film'] for depth in [1,2] for fold in [0,1] for seed in [42,43]])
            run = SimpleNamespace(modern=True, directory=directory, records=records, settings={'seeds':[42,43]})
            selected = select_depth_records(run, 2, model_name='gin')
            self.assertEqual(set(selected.fold), {0,1})
            self.assertEqual(set(selected.seed), {42,43})
            self.assertEqual(set(selected.depth), {2})
            self.assertEqual(len(select_depth_records(run,2,model_name='gin',seeds=[43])),2)
            with self.assertRaisesRegex(ValueError, 'architecture'):
                select_depth_records(run,2)
            run.records = records[records.key != 'gin_2_1_43']
            with self.assertRaisesRegex(ValueError, 'Incomplete'):
                select_depth_records(run,2,model_name='gin')
            with self.assertRaises(ValueError):
                select_depth_records(run,2,model_name='gin',seeds=[999])
            run.records = pd.concat([records,records.query("key == 'gin_2_0_42'")])
            with self.assertRaisesRegex(ValueError, 'duplicate'):
                select_depth_records(run,2,model_name='gin')


if __name__ == '__main__':
    unittest.main()
