"""CPU/tensor/CUDA parity, solver certification, inference and full-fit checks."""
import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import torch
from scipy.optimize._numdiff import approx_derivative
from src.artifacts.bundle import save_bundle,load_bundle
from src.models.curvature_energy import MeanCurvatureEnergy
from src.models.energy_ops import EnergyBatch,AccommodationSolve,predict_energy_samples
from src.training.energy_fit import EnergyObjective,TensorEnergyObjective,fit_energy_partition,fit_energy
from src.training.coupled_fit import graph_samples
from tests.test_coupled_fate import graphs


class TensorEnergyTests(unittest.TestCase):
    device='cpu'

    def samples(self):
        samples=graph_samples(graphs(),2)
        # Include a disconnected organoid, empty shells and all-zero identities.
        g=copy.deepcopy(graphs()[0]);g.x.zero_();g.edge_index=torch.empty((2,0),dtype=torch.long)
        g.organoid_str='isolated';samples+=graph_samples([g],2)
        return samples

    def test_objective_gradient_and_sign_statistics(self):
        samples=self.samples()
        for kwargs in [dict(),dict(size_dependent=False),dict(activation='fixed'),dict(activation='linear'),dict(accommodation=False),dict(pairs=False),dict(pairs=False,accommodation=False)]:
            with self.subTest(kwargs=kwargs):
                m=MeanCurvatureEnergy(2,min_pair_organoids=1,**kwargs).configure(samples)
                m.lambda_logits[0]=-.4;m.signs[0,1]=-1
                m.log_alpha.copy_(torch.linspace(np.log(.02),np.log(100.),9).reshape(3,3))
                cpu=EnergyObjective(copy.deepcopy(m),samples)
                gpu=TensorEnergyObjective(copy.deepcopy(m),samples,device=self.device)
                p=cpu.pack();a,ga=cpu(p);b,gb=gpu(p)
                self.assertAlmostEqual(a,b,places=10)
                np.testing.assert_allclose(ga,gb,rtol=1e-7,atol=1e-10)
                np.testing.assert_allclose(cpu.model.weights,gpu.model.weights,rtol=1e-7,atol=1e-9)
                for v,w in zip(cpu.sign_statistics(),gpu.sign_statistics()):np.testing.assert_allclose(v,w,rtol=1e-8,atol=1e-10)
                si=cpu.select_signs();sj=gpu.select_signs();self.assertEqual(si,sj)
                np.testing.assert_array_equal(cpu.model.signs,gpu.model.signs)

    def test_tensor_gradient_finite_difference(self):
        s=self.samples();m=MeanCurvatureEnergy(2,min_pair_organoids=1).configure(s)
        o=TensorEnergyObjective(m,s,device=self.device);p=o.pack();_,gradient=o(p)
        numeric=approx_derivative(lambda x:o(x)[0],p,method='3-point').ravel()
        np.testing.assert_allclose(gradient,numeric,rtol=2e-4,atol=1e-7)

    def test_prediction_and_checkpoint_device_roundtrip(self):
        s=self.samples();m=MeanCurvatureEnergy(2,min_pair_organoids=1).configure(s)
        m.weights.copy_(torch.linspace(-.3,.5,m.hidden_dim));m.is_fitted.fill_(True);m.signs[1,0]=-1
        # Physical graph remains unchanged when only the supplied N changes.
        altered=[dict(v,N=80+200*i) for i,v in enumerate(s)]
        cpu=m.predict_samples(altered)
        for v in altered:v.pop('y')
        tensor=predict_energy_samples(m,altered,device=self.device)
        for a,b in zip(cpu,tensor):np.testing.assert_allclose(a,b,rtol=1e-9,atol=1e-10)
        m=m.to(self.device)
        graph=graphs()[0];data=SimpleNamespace(full_num_cells=torch.tensor([80.],device=self.device))
        (mu,_),_=m(graph.x.to(self.device),graph.edge_index.to(self.device),data=data)
        np.testing.assert_allclose(mu.cpu(),cpu[0],atol=1e-10)
        with tempfile.TemporaryDirectory() as directory:
            save_bundle(Path(directory)/'model',dict(model=m),splits={})
            restored=load_bundle(Path(directory)/'model')['model']
            np.testing.assert_allclose(restored.predict_sample(altered[0]),cpu[0],atol=1e-10)

    def test_solver_residual_and_failure(self):
        s=self.samples();m=MeanCurvatureEnergy(2,min_pair_organoids=1).configure(s)
        b=EnergyBatch(s,2,device=self.device);rhs=b.design(m)
        solve=AccommodationSolve(b,b.tensor([100.]*len(s)),rhs_chunk_size=2)
        value=solve(rhs)
        np.testing.assert_allclose(solve.apply(value).cpu(),rhs.cpu(),rtol=1e-9,atol=1e-10)
        self.assertLess(solve.last_relative_residual,5e-11)
        with self.assertRaisesRegex(RuntimeError,'CG failed'):
            AccommodationSolve(b,b.tensor([100.]*len(s)),max_iterations=1)(rhs)


class DeviceValidationTests(unittest.TestCase):
    def test_unavailable_cuda_fails_before_model_selection(self):
        with patch('torch.cuda.is_available',return_value=False):
            with self.assertRaisesRegex(RuntimeError,'CUDA requested but unavailable'):
                fit_energy([],model_settings={},penalties=[],device='cuda')
        with self.assertRaisesRegex(ValueError,'CPU or CUDA'):
            fit_energy([],model_settings={},penalties=[],device='mps')

    def test_out_of_memory_is_not_a_failed_optimizer_start(self):
        s=graph_samples(graphs(),2)
        with patch('src.training.energy_fit.fit_energy_partition',side_effect=torch.cuda.OutOfMemoryError('test')) as mocked:
            with self.assertRaises(torch.cuda.OutOfMemoryError):
                fit_energy(s,model_settings=dict(n_markers=2),penalties=[dict(pair_ridge=.001)])
            self.assertEqual(mocked.call_count,1)


@unittest.skipUnless(torch.cuda.is_available(),'CUDA unavailable')
class CudaEnergyTests(TensorEnergyTests):
    device='cuda'

    def test_matched_fit(self):
        s=self.samples();a=MeanCurvatureEnergy(2,min_pair_organoids=1).configure(s);b=copy.deepcopy(a)
        ai=fit_energy_partition(a,s,dict(pair_ridge=.001),device='cpu')
        bi=fit_energy_partition(b,s,dict(pair_ridge=.001),device='cuda')
        self.assertTrue(ai['success'] and bi['success'])
        self.assertAlmostEqual(ai['objective'],bi['objective'],places=7)
        np.testing.assert_array_equal(a.signs,b.signs)
        for left,right in zip(a.predict_samples(s),b.predict_samples(s,device='cuda')):
            np.testing.assert_allclose(left,right,rtol=2e-4,atol=2e-5)
        # Exercise inner selection, outer refit and GPU validation in the public API.
        model,metadata,trials=fit_energy(s,model_settings=dict(n_markers=2,min_pair_organoids=1000),penalties=[dict(pair_ridge=.001)],starts=[(.2,3.,1)],device='cuda',blas_threads=1)
        self.assertEqual(metadata['device'],'cuda');self.assertTrue(metadata['optimizer']['success'])
        np.testing.assert_allclose(model.alpha(),3.)


if __name__=='__main__':
    torch.set_num_threads(2)
    unittest.main()
