"""Pooled exposure, centering, gradients, and saved-model numerical contracts."""
import copy
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from scipy.optimize._numdiff import approx_derivative
from src.artifacts.bundle import save_bundle,load_bundle
from src.models.curvature_energy import MeanCurvatureEnergy
from src.models.energy_ops import predict_energy_samples
from src.training.energy_fit import EnergyObjective,TensorEnergyObjective,fit_energy
from src.training.coupled_fit import graph_samples
from tests.test_coupled_fate import graphs


class PooledEnergyTests(unittest.TestCase):
    exposure_kind="fractions"
    def samples(self):
        samples=graph_samples(graphs(),2)
        for s in samples:
            ids=s['identity']
            s['y']=.03*(ids==0)-.02*(ids==1)+.01*np.sin(np.arange(len(ids)))+.004*np.log(s['N'])
        return samples

    def test_exposure_and_monotone_distance_sensitivity(self):
        m=MeanCurvatureEnergy(2,interaction='pooled',exposure_kind=self.exposure_kind)
        s=dict(identity=np.array([0]),counts=np.array([[[2.,1.,1.],[1.,2.,5.]]]))
        expected=(np.array([.5,.25,.25])+.5*np.array([.125,.25,.625]))/1.5
        np.testing.assert_allclose(m.exposure(s)[0],expected)
        # Duplicating all cells of a ring must not change its fate composition input.
        other=copy.deepcopy(s);other['counts'][:,1]*=3
        np.testing.assert_allclose(m.raw_response(s),m.raw_response(other))
        from src.models.coupled_fate import normalized_tanh
        p1,p2=.2,.3;step=1e-6
        f=lambda a,b:normalized_tanh(np.array([(a+.5*b)/1.5]),np.array([3.]))[0]
        d1=(f(p1+step,p2)-f(p1-step,p2))/(2*step)
        d2=(f(p1,p2+step)-f(p1,p2-step))/(2*step)
        self.assertGreater(d1,0);self.assertAlmostEqual(d2/d1,.5,places=7)
        s['counts'][:]=0
        np.testing.assert_array_equal(m.raw_response(s),0.)

    def test_gradients_centered_and_uncentered(self):
        samples=self.samples()
        for centered in (True,False):
            for device in ['cpu']+(['cuda'] if torch.cuda.is_available() else []):
                with self.subTest(centered=centered,device=device):
                    m=MeanCurvatureEnergy(2,interaction='pooled',exposure_kind=self.exposure_kind,center_response=centered,min_pair_organoids=1).configure(samples)
                    m.lambda_logits.copy_(torch.tensor([-.4,.15]));m.log_alpha.fill_(np.log(2.))
                    cpu=EnergyObjective(copy.deepcopy(m),samples)
                    tensor=TensorEnergyObjective(copy.deepcopy(m),samples,device=device)
                    p=cpu.pack();value,gradient=cpu(p);tv,tg=tensor(p)
                    self.assertAlmostEqual(value,tv,places=10)
                    np.testing.assert_allclose(gradient,tg,rtol=2e-5,atol=1e-9)
                    numerical=approx_derivative(lambda v:cpu(v)[0],p,method='3-point').ravel()
                    np.testing.assert_allclose(gradient,numerical,rtol=2e-4,atol=1e-7)
                    if not centered:np.testing.assert_array_equal(tensor.model.reference,0.)
                    self.assertEqual(cpu.select_signs()['passes'],0)

    def test_centering_is_exact_coordinate_change_at_fixed_N_coefficients(self):
        samples=self.samples()
        m=MeanCurvatureEnergy(2,interaction='pooled',exposure_kind=self.exposure_kind,min_pair_organoids=1).configure(samples)
        m.weights.copy_(torch.linspace(-.2,.3,m.hidden_dim));m.is_fitted.fill_(True)
        m.log_alpha.copy_(torch.linspace(-1.,3.,9).reshape(3,3));m.refresh_reference(samples)
        uncentered=copy.deepcopy(m);uncentered.center_response=False;uncentered.reference.zero_()
        ref=m.array(m.reference)[:,0];scale=float(m.target_scale)
        # At each N, removing mu requires a (generally N-dependent) center shift.
        for n in [80.,700.]:
            amplitude=m.coefficients([n])['amplitude'][0]
            uncentered.weights[:3].copy_(m.weights[:3]-torch.tensor((amplitude*ref).sum(1)/scale))
            for s in samples:
                s=dict(s,N=n)
                np.testing.assert_allclose(m.predict_sample(s),uncentered.predict_sample(s),atol=1e-12)
        # A single constant center shift generally cannot match both sizes.
        self.assertGreater(np.max(abs(m.predict_sample(dict(samples[0],N=80.))-uncentered.predict_sample(dict(samples[0],N=80.)))),1e-5)

    def test_fit_and_checkpoint_roundtrip(self):
        samples=self.samples()
        for centered in (True,False):
            model,info,trials=fit_energy(samples,model_settings=dict(n_markers=2,interaction='pooled',exposure_kind=self.exposure_kind,center_response=centered,min_pair_organoids=1),penalties=[dict(pair_ridge=.001)],starts=[(.2,3.,1)],blas_threads=1,device='cuda' if torch.cuda.is_available() else 'cpu')
            self.assertTrue(info['optimizer']['success'])
            self.assertTrue(all(v['passes']==0 for v in info['optimizer']['sign_rounds']))
            with tempfile.TemporaryDirectory() as directory:
                save_bundle(Path(directory)/'model',dict(model=model),splits={})
                restored=load_bundle(Path(directory)/'model')['model']
                self.assertEqual(restored.exposure_kind,self.exposure_kind);self.assertEqual(restored.interaction,'pooled');self.assertEqual(restored.center_response,centered)
                a=model.predict_samples(samples);b=predict_energy_samples(restored,samples,device='cuda' if torch.cuda.is_available() else 'cpu')
                for x,y in zip(a,b):np.testing.assert_allclose(x,y,atol=1e-10)


class CountEnergyTests(PooledEnergyTests):
    exposure_kind='counts'

    def test_exposure_and_monotone_distance_sensitivity(self):
        from src.models.coupled_fate import normalized_tanh
        m=MeanCurvatureEnergy(2,interaction='pooled',exposure_kind='counts')
        m.log_alpha.fill_(np.log(.1))
        s=dict(identity=np.array([0]),counts=np.array([[[2.,1.,1.],[1.,2.,5.]]]))
        expected=(s['counts'][:,0]+.5*s['counts'][:,1])/1.5
        np.testing.assert_allclose(m.exposure(s),expected)
        np.testing.assert_allclose(m.raw_response(s)[:,0],np.tanh(.1*expected)/np.tanh(.1))
        self.assertGreater(m.raw_response(s)[0,0,0],1.)
        other=copy.deepcopy(s);other['counts'][:,1]*=2
        self.assertTrue(np.all(m.exposure(other)>m.exposure(s)))
        self.assertTrue(np.all(m.raw_response(other)[:,0]>m.raw_response(s)[:,0]))
        # Two second-hop cells have the same exposure as one first-hop cell.
        a=copy.deepcopy(s);b=copy.deepcopy(s)
        a['counts'][0,0,0]+=1;b['counts'][0,1,0]+=2
        np.testing.assert_allclose(m.raw_response(a),m.raw_response(b))
        x=np.array([0.,1.,2.,10.])
        np.testing.assert_allclose(normalized_tanh(x,0.,input_max=None),x)
        with self.assertRaises(ValueError):normalized_tanh(x,.1)
        with self.assertRaises(ValueError):normalized_tanh([-1.],.1,input_max=None)
        with self.assertRaises(ValueError):MeanCurvatureEnergy(2,exposure_kind='counts')
        s['counts'][:]=0
        np.testing.assert_array_equal(m.raw_response(s),0.)


if __name__=='__main__':
    torch.set_num_threads(2)
    unittest.main()
