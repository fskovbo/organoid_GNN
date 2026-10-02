"""Conditional target scaling and mean-constrained energy fitting."""
import copy
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from src.artifacts.bundle import save_bundle, load_bundle
from src.data.target_transforms import ObservedGraphMoments, center_graph_values
from src.models.curvature_energy import MeanCurvatureEnergy
from src.training.coupled_fit import graph_samples
from src.training.energy_fit import EnergyObjective, TensorEnergyObjective, fit_energy_fixed
from tests.test_coupled_fate import graphs


class ObservedMomentTests(unittest.TestCase):
    def test_moments_roundtrip_floor_and_affine_equivariance(self):
        y=np.array([-.2,.1,.3,.7]);original=y.copy()
        m=ObservedGraphMoments.from_targets(y,std_floor=.001)
        z=m.transform(y)
        np.testing.assert_allclose(z.mean(),0,atol=1e-14)
        np.testing.assert_allclose(np.var(z),1,atol=1e-14)
        np.testing.assert_allclose(m.inverse(z),y,atol=1e-14)
        other=ObservedGraphMoments.from_targets(3*y+2,std_floor=.001)
        np.testing.assert_allclose(other.transform(3*y+2),z,atol=1e-14)
        np.testing.assert_array_equal(y,original)
        flat=ObservedGraphMoments.from_targets(np.ones(5),std_floor=.03)
        self.assertEqual(flat.scale,.03);np.testing.assert_array_equal(flat.transform(np.ones(5)),0)
        center=ObservedGraphMoments.from_targets(y,std_floor=.001,standardize=False)
        self.assertEqual(center.scale,1.)
        for bad in [[],[np.nan],[np.inf]]:
            with self.assertRaises(ValueError):ObservedGraphMoments.from_targets(bad,std_floor=.001)

    def test_projection_is_per_graph_and_differentiable(self):
        values=torch.tensor([1.,3.,20.,30.,40.],requires_grad=True)
        ids=torch.tensor([0,0,1,1,1]);v=center_graph_values(values,ids)
        torch.testing.assert_close(v,torch.tensor([-1.,1.,-10.,0.,10.]))
        v.square().sum().backward();torch.testing.assert_close(values.grad,2*v)
        matrix=torch.stack((values.detach(),2*values.detach()),1)
        torch.testing.assert_close(center_graph_values(matrix,ids),torch.stack((v.detach(),2*v.detach()),1))

    def test_float32_projection_with_large_common_offset(self):
        for device in ['cpu']+(['cuda'] if torch.cuda.is_available() else []):
            x=torch.linspace(-.3,.3,5001,device=device,dtype=torch.float32)+1000.
            graph_ids=torch.zeros(len(x),device=device,dtype=torch.long)
            centered=center_graph_values(x,graph_ids)
            self.assertLess(abs(float(centered.double().mean())),1e-8)
            torch.testing.assert_close(centered,(x.double()-x.double().mean()).float())

    def test_energy_projection_cpu_cuda_and_checkpoint(self):
        torch.set_num_threads(2)
        samples=graph_samples(graphs(),2)
        for sample in samples:
            m=ObservedGraphMoments.from_targets(sample['y'],std_floor=.001)
            sample['y']=m.transform(sample['y'])
        settings=dict(n_markers=2,interaction='pooled',activation='presence',size_dependent=False,
            center_response=False,pair_constraints='direct',interaction_radius=1,zero_mean_output=True)
        for pairs in [False,True]:
            for gamma in [0.,.8]:
                m=MeanCurvatureEnergy(**settings,pairs=pairs,fixed_strength=gamma).configure(samples)
                cpu=EnergyObjective(m,samples);loss,_=cpu(cpu.pack())
                for s in samples:
                    np.testing.assert_allclose(m.design(s).mean(0),0,atol=1e-14)
                    np.testing.assert_allclose(m.design(s),m.raw_design(s)@m.mapping(),atol=1e-14)
                    self.assertAlmostEqual(m.predict_sample(s).mean(),0,places=12)
                for device in ['cpu']+(['cuda'] if torch.cuda.is_available() else []):
                    other=copy.deepcopy(m);obj=TensorEnergyObjective(other,samples,device=device)
                    actual,_=obj(obj.pack());self.assertAlmostEqual(loss,actual,places=9)
                    for a,b in zip(m.predict_samples(samples),other.predict_samples(samples,device=device)):
                        np.testing.assert_allclose(a,b,atol=1e-9)
                # A common shift of every center coefficient must disappear after projection.
                original=m.predict_samples(samples);m.weights[:3]+=.3
                for a,b in zip(original,m.predict_samples(samples)):np.testing.assert_allclose(a,b,atol=1e-12)
                with tempfile.TemporaryDirectory() as d:
                    save_bundle(Path(d)/'fit',dict(model=m),splits={})
                    restored=load_bundle(Path(d)/'fit')['model'];self.assertTrue(restored.zero_mean_output)
                    np.testing.assert_allclose(restored.predict_sample(samples[0]),m.predict_sample(samples[0]),atol=1e-12)
        fit,info,_=fit_energy_fixed(samples,model_settings=settings,strength=.4,penalties=[dict(pair_ridge=.001)])
        self.assertTrue(info['optimizer']['success'])
        self.assertFalse(set(info['inner_train'])&set(info['inner_validation']))


if __name__=='__main__':unittest.main()
