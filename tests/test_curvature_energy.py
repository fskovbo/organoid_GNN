import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import pandas as pd
import torch
from scipy.optimize._numdiff import approx_derivative
from src.models.curvature_energy import MeanCurvatureEnergy,graph_laplacian
from src.training.energy_fit import EnergyObjective,fit_energy_partition,fit_energy
from src.training.coupled_fit import graph_samples
from src.artifacts.bundle import save_bundle,load_bundle
from src.analysis.spatial.regions import distance_regions
from tests.test_coupled_fate import graphs

class EnergyTests(unittest.TestCase):
    def samples(self):return graph_samples(graphs(),2)
    def test_gradients_and_sign_scores(self):
        s=self.samples();m=MeanCurvatureEnergy(2,min_pair_organoids=1).configure(s)
        m.lambda_logits.copy_(torch.tensor([-.4,.2]));m.log_alpha.fill_(np.log(2.))
        m.signs[0,1]=-1
        for sample in s:
            np.testing.assert_allclose(m.design(sample),m.raw_design(sample)@m.mapping(),atol=1e-14)
        o=EnergyObjective(m,s);x=o.pack();v,g=o(x)
        numeric=approx_derivative(lambda p:o(p)[0],x,method='3-point').ravel()
        np.testing.assert_allclose(g,numeric,rtol=2e-4,atol=1e-7)
        o(x);before=o(x)[0];info=o.select_signs();after=o(x)[0]
        self.assertLessEqual(after,before+1e-9);self.assertTrue(info['coordinate_optimum'])
        signs=m.signs.clone()
        for a,b in np.argwhere(m.active_pairs.numpy()):
            m.signs[a,b]*=-1;candidate=o(x)[0];self.assertGreaterEqual(candidate,after-2e-9);m.signs.copy_(signs)
    def test_energy_ratio_and_restoration(self):
        s=self.samples();m=MeanCurvatureEnergy(2,min_pair_organoids=1).configure(s)
        m.weights.copy_(torch.linspace(-.2,.3,m.hidden_dim));m.signs[1,2]=-1;m.lambda_logits.copy_(torch.tensor([-.2,.1]));m.is_fitted.fill_(True)
        co=m.coefficients([80,300,700]);np.testing.assert_allclose(abs(co['hop2']),.5*abs(co['hop1']))
        np.testing.assert_allclose(co['center'][0],co['center'][-1]);np.testing.assert_allclose(co['hop1'][1],co['hop1'][0]+(co['hop1'][2]-co['hop1'][0])*np.log(300/80)/np.log(700/80))
        sample=s[0];prediction=m.predict_sample(sample);local=m.design(sample)@m.weights.numpy()*float(m.target_scale);lap=graph_laplacian(sample['transition'])
        np.testing.assert_allclose(prediction+m.strength([sample['N']])[0]*(lap@prediction),local,atol=1e-12)
        self.assertAlmostEqual(prediction.mean(),local.mean(),places=12)
        graph=graphs()[0]
        (forward,_),_=m(graph.x,graph.edge_index,data=SimpleNamespace(full_num_cells=torch.tensor([80.])))
        np.testing.assert_allclose(forward.numpy(),m.predict_sample(sample,N=80.))
        altered=dict(sample,y=np.full_like(sample['y'],999.));np.testing.assert_array_equal(prediction,m.predict_sample(altered))
        with tempfile.TemporaryDirectory() as d:
            save_bundle(Path(d)/'model',dict(model=m),splits={});restored=load_bundle(Path(d)/'model')['model']
            np.testing.assert_array_equal(prediction,restored.predict_sample(sample))
    def test_constant_center_and_no_propagation(self):
        s=self.samples();m=MeanCurvatureEnergy(2,pairs=False,accommodation=False).configure(s)
        info=fit_energy_partition(m,s,dict(ridge=.001));self.assertTrue(info['success'])
        np.testing.assert_array_equal(m.predict_sample(s[0],80),m.predict_sample(s[0],700))
    def test_unsupported_activation_is_fixed(self):
        m,_,_=fit_energy(self.samples(),model_settings=dict(n_markers=2,min_pair_organoids=1000),penalties=[dict(pair_ridge=.001)],starts=[(.2,10.,-1)],blas_threads=1)
        np.testing.assert_allclose(m.alpha(),3.)
        np.testing.assert_array_equal(m.signs.numpy(),1.)

    def test_regions(self):
        g=copy.deepcopy(graphs()[0]);n=len(g.x);g.x=torch.zeros((n,2));g.x[0,0]=1
        dist=np.full((2,n),4.);dist[0,:4]=[.2,.74,.75,1.1];dist[1,4:]=.3
        profile=['no_neck_support']*4+['local_minimum']*(n-4)
        base=pd.DataFrame(dict(region=['unqualified_crypt']*n,profile_class=profile,crypt_id=-1,crypt_distance=np.nan))
        with tempfile.TemporaryDirectory() as d:
            np.savez(Path(d)/f'{g.organoid_str}.npz',d_crypts_graph=dist)
            with patch('src.analysis.spatial.regions.graph_regions',return_value=base):
                r=distance_regions(g,d,marker_names=['LGR5','KI67'])
        self.assertEqual(r.region.iloc[0],'crypt_with_LGR5');self.assertEqual(r.region.iloc[1],'crypt_with_LGR5')
        self.assertEqual(r.region.iloc[2],'boundary_without_qualified_neck');self.assertEqual(r.region.iloc[3],'villus')
        self.assertTrue((r.region.iloc[4:]=='crypt_without_LGR5').all())

if __name__=='__main__':unittest.main()
