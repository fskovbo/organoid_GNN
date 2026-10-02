"""Presence, interaction-radius and reference contracts, including CPU/CUDA parity."""
import copy
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from scipy.optimize._numdiff import approx_derivative
from src.models.curvature_energy import MeanCurvatureEnergy
from src.models.energy_ops import EnergyBatch
from src.training.energy_fit import EnergyObjective, TensorEnergyObjective
from src.training.coupled_fit import graph_samples
from src.artifacts.bundle import save_bundle, load_bundle
from tests.test_coupled_fate import graphs


class PresenceEnergyTests(unittest.TestCase):
    def test_responses_radius_and_reference(self):
        samples=graph_samples(graphs(),2)
        for radius in [1,2]:
            for activation in ['linear','presence','mixed']:
                choices=[['linear','presence','linear'],['presence','linear','presence'],['linear','linear','presence']]
                m=MeanCurvatureEnergy(2,interaction='pooled',size_dependent=False,center_response=False,
                    pair_constraints='direct',activation=activation,interaction_radius=radius,
                    reference_source=2 if activation=='linear' else None,
                    pair_activations=choices if activation=='mixed' else None).configure(samples)
                s=samples[0];x=m.exposure(s);q=m.raw_response(s)[:,0]
                expected=x if activation=='linear' else (x>0).astype(float)
                if activation=='mixed':expected=np.where(np.array(choices)[s['identity']]=='presence',x>0,x)
                np.testing.assert_array_equal(q,expected)
                other=copy.deepcopy(s);other['counts'][:,1]+=7
                if radius==1:np.testing.assert_array_equal(m.raw_response(s),m.raw_response(other))
                np.testing.assert_allclose(m.design(s),m.raw_design(s)@m.mapping(),atol=1e-14)
                o=EnergyObjective(m,samples);p=o.pack();v,g=o(p)
                np.testing.assert_allclose(g,approx_derivative(lambda p:o(p)[0],p).ravel(),rtol=1e-4,atol=1e-8)
                o(p)
                for device in ['cpu']+(['cuda'] if torch.cuda.is_available() else []):
                    b=EnergyBatch(samples,2,device=device)
                    np.testing.assert_allclose(b.design(m).cpu(),np.concatenate([m.design(s) for s in samples]),atol=1e-14)
                    tensor=TensorEnergyObjective(copy.deepcopy(m),samples,device=device);tv,tg=tensor(p)
                    self.assertAlmostEqual(v,tv,places=10);np.testing.assert_allclose(g,tg,atol=1e-9)
                if activation=='linear':np.testing.assert_array_equal(m.coefficients([1])['amplitude'][:,:,2],0.)
                with tempfile.TemporaryDirectory() as d:
                    save_bundle(Path(d)/'model',{'model':m},splits={})
                    restored=load_bundle(Path(d)/'model')['model']
                    np.testing.assert_array_equal(m.predict_sample(s),restored.predict_sample(s))

    def test_presence_ignores_abundance_and_handles_absence(self):
        s=graph_samples(graphs(),2)[0]
        m=MeanCurvatureEnergy(2,interaction='pooled',activation='presence',pair_constraints='direct',center_response=False)
        changed=copy.deepcopy(s);changed['counts']*=3
        np.testing.assert_array_equal(m.raw_response(s),m.raw_response(changed))
        changed['counts'][:]=0
        np.testing.assert_array_equal(m.raw_response(changed),0.)

    def test_exact_linear_gauge_on_populated_rings(self):
        s=graph_samples(graphs(),2)[0]
        # Ensure both rings populated for this algebraic gauge test.
        s=copy.deepcopy(s);s['counts']+=1
        plain=MeanCurvatureEnergy(2,interaction='pooled',activation='linear',pair_constraints='direct',size_dependent=False,center_response=False).configure([s])
        ref=MeanCurvatureEnergy(2,interaction='pooled',activation='linear',pair_constraints='direct',size_dependent=False,center_response=False,reference_source=2).configure([s])
        w=np.arange(12)*.07;plain.weights.copy_(torch.tensor(w))
        a=w[:3];beta=w[3:].reshape(3,3)
        ref.weights.copy_(torch.tensor(np.r_[a+beta[:,2],(beta[:,:2]-beta[:,2,None]).ravel()]))
        np.testing.assert_allclose(plain.design(s)@w,ref.design(s)@ref.array(ref.weights),atol=1e-14)

if __name__=='__main__':
    torch.set_num_threads(2)
    unittest.main()
