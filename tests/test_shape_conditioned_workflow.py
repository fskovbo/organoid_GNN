"""Tiny fixed-accommodation fits and independent notebook restoration."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from src.artifacts.bundle import load_bundle
from tests.test_energy_workflow import run_training,run_evaluation


class ConditionalWorkflowTests(unittest.TestCase):
    def test_fixed_models_normalizations_and_artificial_fields(self):
        torch.set_num_threads(2)
        specs=[dict(name='center_only',family='energy',radius=0),
               dict(name='presence_r1',family='energy',radius=1,pairs=[('LGR5','Lysozyme'),('KI67','Agr2')]),
               dict(name='presence_r2',family='energy',radius=2),dict(name='gin_d1',family='gin',depth=1)]
        with tempfile.TemporaryDirectory() as d:
            run=Path(d)/'fixed'
            ns=run_training('shape_conditioned_energy_training.ipynb',run,models=specs,gamma=.5,
                gin_hidden_dim=8,gin_batch_size=4,gin_max_epochs=2,gin_patience=1)
            self.assertEqual(len(ns['records']),8)
            for rec in ns['records']:
                model=load_bundle(run/rec['bundle'])['model']
                if rec['family']=='energy':self.assertEqual(model.fixed_strength,.5)
                if rec['variant']=='presence_r1':
                    self.assertEqual(model.hidden_dim,10)
                    self.assertEqual(model.pair_indices,[[4,5],[3,0]])
                    table=model.coefficients([1])['amplitude'][0]
                    mask=np.ones((8,8),bool);mask[4,5]=False;mask[3,0]=False
                    np.testing.assert_array_equal(table[mask],0)
            ev=run_evaluation('shape_conditioned_energy_evaluation.ipynb',run)
            self.assertLess(max(row['max_difference'] for row in ev['verification']),2e-6)
            view=run_evaluation('energy_model_inspection.ipynb',run,LATTICE_SIDE=15,CHECK_LATTICE_SIDE=21,MAX_DISTANCE=3)
            self.assertLess(view['size_checks'].max_raw_response_difference.max(),1e-5)
            self.assertTrue((view['OUTPUT_DIR']/'artificial_radial_fields.csv').exists())


if __name__=='__main__':unittest.main()
