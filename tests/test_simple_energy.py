"""Contracts for restricted-source, uncentered constant curvature energies."""
import copy
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from scipy.optimize._numdiff import approx_derivative
from src.models.curvature_energy import MeanCurvatureEnergy
from src.models.energy_ops import EnergyBatch
from src.training.energy_fit import EnergyObjective, TensorEnergyObjective, fit_energy, fit_energy_partition
from src.training.coupled_fit import graph_samples
from src.artifacts.bundle import save_bundle,load_bundle
from tests.test_coupled_fate import graphs


class SimpleEnergyTests(unittest.TestCase):
    def setup_model(self,activation='linear'):
        samples=graph_samples(graphs(),2)
        for s in samples:
            s['y']=.01*np.sin(np.arange(len(s['identity'])))+.03*(s['identity']==0)-.01*(s['identity']==1)
        settings=dict(n_markers=2,interaction='pooled',size_dependent=False,center_response=False,
            pair_constraints='direct',source_indices=[0],activation=activation,min_pair_organoids=1)
        return MeanCurvatureEnergy(**settings).configure(samples),samples,settings

    def test_direct_terms_no_compensation_and_no_N(self):
        m,s,_=self.setup_model();m.weights[:3].copy_(torch.tensor([.1,.2,.3]));m.weights[3:]=.5;m.is_fitted.fill_(True)
        co=m.coefficients([80,700]);self.assertTrue((co['amplitude'][:,:,0]>0).all())
        np.testing.assert_array_equal(co['amplitude'][:,:,1:],0.)
        np.testing.assert_array_equal(co['center'][0],co['center'][1]);np.testing.assert_array_equal(co['amplitude'][0],co['amplitude'][1])
        np.testing.assert_array_equal(m.predict_sample(s[0],80),m.predict_sample(s[0],700))
        np.testing.assert_array_equal(m.reference,0.)
        np.testing.assert_allclose(m.design(s[0]),m.raw_design(s[0])@m.mapping(),atol=1e-14)
        # Zero chosen sources: only the center term survives. Other fates still enter denominators.
        clean=copy.deepcopy(s[0]);clean['counts'][:,:,0]=0
        np.testing.assert_allclose(m.design(clean)@m.array(m.weights),m.array(m.weights)[clean['identity']])
        changed=copy.deepcopy(s[0]);changed['counts'][:,:,1:]+=3
        self.assertTrue(np.any(m.exposure(changed)[:,0]<m.exposure(s[0])[:,0]))

    def test_gradient_and_cpu_cuda_for_both_activations(self):
        for activation in ['linear','fixed']:
            m,s,_=self.setup_model(activation)
            o=EnergyObjective(m,s);p=o.pack();v,g=o(p)
            np.testing.assert_allclose(g,approx_derivative(lambda p:o(p)[0],p).ravel(),rtol=1e-4,atol=1e-8)
            o(p)  # Restore parameters after finite-difference perturbations.
            for device in ['cpu']+(['cuda'] if torch.cuda.is_available() else []):
                t=TensorEnergyObjective(copy.deepcopy(m),s,device=device);tv,tg=t(p)
                self.assertAlmostEqual(v,tv,places=11);np.testing.assert_allclose(g,tg,rtol=1e-5,atol=1e-10)
                np.testing.assert_allclose(np.concatenate(m.predict_samples(s)),EnergyBatch(s,2,device=device).predict(t.model).cpu().numpy(),atol=1e-10)

    def test_centered_readout_is_exact_without_refitting(self):
        m,s,_=self.setup_model('fixed');m.weights.copy_(torch.linspace(-.2,.4,m.hidden_dim));m.is_fitted.fill_(True)
        centered=copy.deepcopy(m);centered.center_response=True;centered.refresh_reference(s)
        beta=m.coefficients([100])['amplitude'][0]
        shift=(beta*centered.array(centered.reference)[:,0]).sum(1)/float(m.target_scale)
        centered.weights[:3]+=torch.tensor(shift)
        for sample in s:np.testing.assert_allclose(m.predict_sample(sample),centered.predict_sample(sample),atol=1e-12)

    def test_noiseless_planted_recovery(self):
        planted,s,settings=self.setup_model('fixed')
        planted.weights.copy_(torch.tensor([.1,-.2,.05,.4,-.3,.2]));planted.lambda_logits[0]=np.log(np.expm1(.4));planted.is_fitted.fill_(True)
        expected=planted.coefficients([1])
        for sample in s:sample['y']=planted.predict_sample(sample)
        fitted=MeanCurvatureEnergy(**settings).configure(s)
        fitted.lambda_logits[0]=np.log(np.expm1(.2))
        fit_energy_partition(fitted,s,dict(ridge=0.,pair_ridge=0.,lambda_ridge=0.),tolerance=1e-9)
        self.assertAlmostEqual(fitted.strength([1])[0],.4,places=4)
        np.testing.assert_allclose(fitted.coefficients([1])['center'],expected['center'],atol=1e-7)
        np.testing.assert_allclose(fitted.coefficients([1])['amplitude'],expected['amplitude'],atol=1e-7)

    def test_fit_and_reload(self):
        _,s,settings=self.setup_model('fixed')
        m,info,_=fit_energy(s,model_settings=settings,penalties=[dict(pair_ridge=.001)],starts=[(.2,3.,1)],blas_threads=1)
        self.assertTrue(info['optimizer']['success']);self.assertEqual(m.hidden_dim,6)
        with tempfile.TemporaryDirectory() as d:
            save_bundle(Path(d)/'m',dict(model=m),splits={});r=load_bundle(Path(d)/'m')['model']
            self.assertEqual(r.source_indices,[0]);self.assertEqual(r.pair_constraints,'direct')
            np.testing.assert_array_equal(m.predict_sample(s[0]),r.predict_sample(s[0]))
        for invalid in [[0,0],[-1],[3],[]]:
            with self.assertRaises(ValueError):MeanCurvatureEnergy(2,pair_constraints='direct',source_indices=invalid)
        with self.assertRaises(ValueError):MeanCurvatureEnergy(2,source_indices=[0])

if __name__=='__main__':
    torch.set_num_threads(2)
    unittest.main()
