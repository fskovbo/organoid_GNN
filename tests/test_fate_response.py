"""Activation, analytic profiling, target separation and portable inference."""
import tempfile
import unittest
import numpy as np
import torch
from src.models.coupled_fate import CoupledFateResponse,normalized_tanh
from src.training.coupled_fit import graph_samples,ResponseObjective,fit_response_partition,response_mse
from src.artifacts.bundle import save_bundle,load_bundle
from tests.test_coupled_fate import graphs

class FateResponseTests(unittest.TestCase):
    def samples(self):
        samples=graph_samples(graphs(),2)
        for s in samples:s['y']=np.c_[s['y'],s['y']*.5+np.sin(np.arange(len(s['y'])))*.001]
        return samples

    def test_activation_and_distance_accounting(self):
        x=np.linspace(0,1,30)
        for alpha in [0.,.01,.5,3.,100.]:
            y=normalized_tanh(x,alpha);self.assertTrue(np.all((y>=0)&(y<=1)))
            self.assertEqual(y[0],0.);self.assertAlmostEqual(y[-1],1.)
        np.testing.assert_allclose(normalized_tanh(x,0.),x)
        m=CoupledFateResponse(2,radius=2,activation='linear')
        c=np.array([[[2.,1.,0.],[1.,0.,2.]]]);raw=m.raw_response(np.array([0]),c)
        np.testing.assert_allclose(raw,c/6)
        m.activation='fixed';m.log_alpha.fill_(np.log(3.))
        np.testing.assert_allclose(m.raw_response(np.array([0]),c).sum(axis=1),normalized_tanh(c.sum(axis=1)/6,3.))

    def test_profile_gradients_both_modes(self):
        for dependent in [False,True]:
            s=self.samples();m=CoupledFateResponse(2,n_splines=4,n_targets=2,activation='learned',min_pair_organoids=1,size_dependent=dependent).configure(s)
            objective=ResponseObjective(m,s);x=objective.pack()+np.random.default_rng(4).normal(0,1,len(objective.pack()));_,grad=objective(x)
            fd=np.array([(objective(x+np.eye(len(x))[i]*1e-5)[0]-objective(x-np.eye(len(x))[i]*1e-5)[0])/2e-5 for i in range(len(x))])
            np.testing.assert_allclose(grad,fd,atol=1e-7,rtol=1e-4)

    def test_constant_no_target_input_and_checkpoint(self):
        s=self.samples();m=CoupledFateResponse(2,n_targets=2,activation='learned',min_pair_organoids=1,size_dependent=False).configure(s)
        info=fit_response_partition(m,s,{},max_iterations=150,tolerance=1e-5)
        self.assertTrue(info['success']);a=m.predict_sample(s[0]);b=m.predict_sample(dict(s[0],y=np.zeros_like(s[0]['y'])),N=10000)
        np.testing.assert_allclose(a,b)
        np.testing.assert_allclose(m.contributions(s[0])['prediction'],a,atol=1e-12)
        with tempfile.TemporaryDirectory() as d:
            save_bundle(d+'/model',dict(model=m),splits={})
            restored=load_bundle(d+'/model')['model'];np.testing.assert_allclose(restored.predict_sample(s[0]),a)
        self.assertEqual(m.coefficients([100,200])['P'].shape,(2,2,2,3,3))

    def test_joint_linear_matches_separate(self):
        samples=self.samples();joint=CoupledFateResponse(2,n_targets=2,coupling=False,activation='linear',size_dependent=False).configure(samples)
        fit_response_partition(joint,samples,{})
        for target in range(2):
            scalar=[dict(s,y=s['y'][:,target:target+1]) for s in samples]
            single=CoupledFateResponse(2,n_targets=1,coupling=False,activation='linear',size_dependent=False).configure(scalar)
            fit_response_partition(single,scalar,{})
            np.testing.assert_allclose(joint.predict_sample(samples[0])[:,target],single.predict_sample(scalar[0])[:,0],atol=1e-10)

if __name__=='__main__':unittest.main()
