"""Restricted receivers and exact fixed accommodation, including gamma=0."""
import copy
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from src.models.curvature_energy import MeanCurvatureEnergy
from src.training.coupled_fit import graph_samples
from src.training.energy_fit import EnergyObjective, TensorEnergyObjective, fit_energy_fixed
from src.artifacts.bundle import save_bundle, load_bundle
from tests.test_coupled_fate import graphs


class FixedEnergyTests(unittest.TestCase):
    def settings(self):
        return dict(n_markers=2,interaction='pooled',activation='presence',size_dependent=False,
            center_response=False,pair_constraints='direct',source_indices=[0],recipient_indices=[1,2],interaction_radius=1)

    def test_structural_zeros_and_fixed_strength(self):
        samples=graph_samples(graphs(),2)
        for gamma in [0.,.4]:
            m=MeanCurvatureEnergy(**self.settings(),fixed_strength=gamma).configure(samples)
            self.assertEqual(m.hidden_dim,5)
            m.weights.copy_(torch.arange(5,dtype=torch.float64));m.is_fitted.fill_(True)
            np.testing.assert_array_equal(m.strength([10,1000]),[gamma,gamma])
            table=m.coefficients([1])['amplitude'][0]
            np.testing.assert_array_equal(table[0],0.)
            np.testing.assert_array_equal(table[:,1:],0.)
            for s in samples:
                np.testing.assert_allclose(m.design(s),m.raw_design(s)@m.mapping(),atol=1e-14)
                local=m.design(s)@m.array(m.weights)
                np.testing.assert_array_equal(local[s['identity']==0],0.)
                if gamma==0:np.testing.assert_allclose(m.predict_sample(s),local*float(m.target_scale),atol=1e-14)
            cpu=EnergyObjective(m,samples);self.assertEqual(len(cpu.pack()),0);value,gradient=cpu(cpu.pack());self.assertEqual(len(gradient),0)
            for device in ['cpu']+(['cuda'] if torch.cuda.is_available() else []):
                obj=TensorEnergyObjective(copy.deepcopy(m),samples,device=device);actual,g=obj(obj.pack())
                self.assertAlmostEqual(value,actual,places=10);self.assertEqual(len(g),0)

    def test_fixed_fit_save_and_membership(self):
        samples=graph_samples(graphs(),2)
        for gamma in [0.,.2]:
            model,info,trials=fit_energy_fixed(samples,model_settings=self.settings(),strength=gamma,penalties=[dict(pair_ridge=.001)],device='cpu')
            self.assertEqual(info['optimizer']['iterations'],0)
            self.assertTrue(info['optimizer']['success']);self.assertEqual(model.fixed_strength,gamma)
            self.assertFalse(set(info['inner_train'])&set(info['inner_validation']))
            self.assertEqual(set(info['fit_organoids']),{s['organoid_str'] for s in samples})
            with tempfile.TemporaryDirectory() as d:
                save_bundle(Path(d)/'m',dict(model=model),splits={});restored=load_bundle(Path(d)/'m')['model']
                self.assertEqual(restored.recipient_indices,[1,2]);self.assertEqual(restored.fixed_strength,gamma)
                np.testing.assert_array_equal(model.predict_sample(samples[0]),restored.predict_sample(samples[0]))
        for gamma in [-1,np.inf,np.nan]:
            with self.assertRaises(ValueError):MeanCurvatureEnergy(**self.settings(),fixed_strength=gamma)
        for recipients in [[],[1,1],[-1],[3]]:
            settings=self.settings();settings['recipient_indices']=recipients
            with self.assertRaises(ValueError):MeanCurvatureEnergy(**settings)

if __name__=='__main__':
    torch.set_num_threads(2)
    unittest.main()
