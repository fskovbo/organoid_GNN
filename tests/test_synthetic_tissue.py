"""Synthetic topology and constant-shift invariance of the projected energy."""
import unittest
import numpy as np
import torch
from src.data.synthetic import triangular_lattice_edges
from src.models.curvature_energy import MeanCurvatureEnergy
from src.training.coupled_fit import graph_samples
from torch_geometric.data import Data


class SyntheticTissueTests(unittest.TestCase):
    def test_six_neighbors_symmetry_and_gauge(self):
        side=9;edges=triangular_lattice_edges(side);n=side**2
        pairs=set(map(tuple,edges.T.tolist()))
        self.assertEqual(len(pairs),6*n)
        self.assertTrue(all((b,a) in pairs and a!=b for a,b in pairs))
        np.testing.assert_array_equal(torch.bincount(edges[0]).numpy(),6)
        x=torch.zeros(n,2);x[:,0]=1;x[n//2]=torch.tensor([0.,1.])
        graph=Data(x=x,edge_index=edges,y=torch.arange(n,dtype=torch.float64)/n,organoid_str='tissue')
        samples=graph_samples([graph],2)
        m=MeanCurvatureEnergy(2,interaction='pooled',activation='presence',center_response=False,
            pair_constraints='direct',fixed_strength=.5,size_dependent=False,zero_mean_output=True).configure(samples)
        m.weights.copy_(torch.arange(m.hidden_dim,dtype=torch.float64));m.is_fitted.fill_(True)
        samples[0].pop('y');before=m.predict_sample(samples[0]);m.weights[:3]+=7
        np.testing.assert_allclose(before,m.predict_sample(samples[0]),atol=1e-12)
        self.assertAlmostEqual(before.mean(),0,places=12)
        for invalid in [True,3,5.5]:
            with self.assertRaises(ValueError):triangular_lattice_edges(invalid)

if __name__=='__main__':unittest.main()
