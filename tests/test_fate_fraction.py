"""Nonlinear fraction models retain fate-only inputs and portable predictions."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from src.artifacts.bundle import save_bundle, load_bundle
from legacy.fate_interactions.models.fate_fraction import FateFractionSpline
from legacy.fate_interactions.training.spline_fit import graph_samples, fit_fraction_spline
from test_fate_spline import synthetic_graphs


class FateFractionTests(unittest.TestCase):
    def setUp(self):
        self.samples = graph_samples(synthetic_graphs(),2)
        rng = np.random.default_rng(78)
        for s in self.samples:
            # Nondegenerate simplex fractions, independent of actual ring sizes.
            s['counts'] = rng.integers(0,8,size=s['counts'].shape)
            s['y'] = rng.normal(0,.1,len(s['identity']))

    def test_fraction_invariance_decomposition_and_roundtrip(self):
        for degree,center,pairs in [(1,True,0),(2,False,0),(3,True,2)]:
            with self.subTest(degree=degree):
                model = FateFractionSpline(2,2,4,degree,center,pairs).configure(self.samples)
                model.weights.normal_(0,.03)
                model.is_fitted.fill_(True)
                s = self.samples[0]
                resized = dict(s,counts=s['counts']*np.array([3,17])[None,:,None])
                np.testing.assert_allclose(model.predict_sample(s),model.predict_sample(resized),atol=1e-12)
                for n in (20,30,1000):
                    np.testing.assert_allclose(model.predict_sample(s,n),model.contributions(s,n)['prediction'],atol=1e-12)
                with tempfile.TemporaryDirectory() as tmp:
                    save_bundle(Path(tmp)/'model',dict(model=model),splits={})
                    restored = load_bundle(Path(tmp)/'model')['model']
                    np.testing.assert_array_equal(model.predict_sample(s),restored.predict_sample(s))

    def test_new_features_are_orthogonal_to_lower_design(self):
        model = FateFractionSpline(2,2,4,3,True,2).configure(self.samples)
        abundance_cross = np.zeros((model.linear_dim,model.abundance_dim))
        pair_cross = np.zeros((model.linear_dim+model.abundance_dim,model.interaction_dim))
        for s in self.samples:
            design = model.local_design(s['identity'],s['counts'])
            cut = model.linear_dim+model.abundance_dim
            weight = 1/(len(self.samples)*len(design))
            abundance_cross += design[:,:model.linear_dim].T@design[:,model.linear_dim:cut]*weight
            pair_cross += design[:,:cut].T@design[:,cut:]*weight
        np.testing.assert_allclose(abundance_cross,0,atol=1e-10)
        np.testing.assert_allclose(pair_cross,0,atol=1e-10)

    def test_linear_nesting_and_training_only_selection(self):
        from legacy.fate_interactions.models.fate_spline import FateSplineCurvature
        original = FateSplineCurvature(2,2,'pairwise',4).configure(self.samples[:9])
        linear = FateFractionSpline(2,2,4,1).configure(self.samples[:9])
        s = self.samples[0]
        np.testing.assert_allclose(original.local_design(s['identity'],s['counts']),linear.local_design(s['identity'],s['counts']))
        model,meta,trials = fit_fraction_spline(self.samples[:9],n_markers=2,n_splines=4,
            fraction_degree=2,n_interactions=1,smoothness_grid=[0,.001],ridge_grid=[1e-5],extension_multipliers=[1.])
        self.assertEqual(len(trials),2)
        self.assertEqual(set(meta['inner_train'])|set(meta['inner_validation']),{str(i) for i in range(9)})
        self.assertFalse(set(meta['inner_train'])&set(meta['inner_validation']))
        self.assertAlmostEqual(np.exp(model.log_count_range[1].item()),self.samples[8]['N'])

    def test_recovery_of_nonlinear_fraction_response(self):
        for s in self.samples:
            p = s['counts'][:,0,0]/s['counts'][:,0].sum(axis=1)
            s['y'] = .2*p*(1-p)*(s['identity']==0)
        model,_,_ = fit_fraction_spline(self.samples,n_markers=2,n_splines=4,
            fraction_degree=2,smoothness_grid=[0],ridge_grid=[0],extension_multipliers=[1.])
        mse = np.mean([np.mean((model.predict_sample(s)-s['y'])**2) for s in self.samples])
        self.assertLess(mse,1e-10)


if __name__ == '__main__':
    unittest.main()
