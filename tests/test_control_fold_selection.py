"""Fold caps retain original fold IDs and memberships without making a new split."""
import contextlib
import io
import json
from pathlib import Path
import unittest
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


class ControlFoldSelectionTests(unittest.TestCase):
    def test_limits_keep_original_membership_and_validate_values(self):
        notebook=json.loads((ROOT/'experiments/training/graph_controls_training.ipynb').read_text())
        source=''.join(notebook['cells'][6]['source'])
        source=source[source.index("max_folds = CONTROL_SETTINGS"):source.index('cohort_payload =')]
        members={fold:dict(fold=fold,train=[f'train_{fold}'],validation=[f'val_{fold}']) for fold in [2,7,9]}
        records=pd.DataFrame([dict(fold=fold,depth=depth,hidden_dim=8,seed=42) for fold in members for depth in [0,2]])
        for limit, expected in [(1,[2]),(2,[2,7]),(None,[2,7,9]),(10,[2,7,9])]:
            ns=dict(CONTROL_SETTINGS={'max_folds':limit},fold_membership=members,
                    reference_records=records,reference_settings={'n_folds':3},SETTINGS={})
            with contextlib.redirect_stdout(io.StringIO()):exec(source,ns)
            self.assertEqual(ns['membership'],[members[f] for f in expected])
            self.assertEqual(set(ns['reference_records'].fold),set(expected))
            self.assertEqual(ns['SETTINGS']['n_folds'],len(expected))
            self.assertEqual(ns['SETTINGS']['reference_n_folds'],3)
        for limit in [0,-1,True,1.5,'1']:
            with self.assertRaisesRegex(ValueError,'positive integer'):
                exec(source,dict(CONTROL_SETTINGS={'max_folds':limit}))


if __name__=='__main__':unittest.main()
