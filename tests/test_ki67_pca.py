import unittest
import numpy as np
import torch
from src.models.gnn import SizeFiLMGINCurvature
from src.analysis.size_embedding_focus import head_prediction
from src.analysis.ki67_pca import fit_weighted_pca, fit_response_svd, exact_readout, bilinear_change


class KI67PCATests(unittest.TestCase):
    def test_response_projection_keeps_nonzero_mean_and_zero_origin(self):
        rng = np.random.default_rng(19)
        basis, _ = np.linalg.qr(rng.normal(size=(8, 2)))
        delta = (rng.normal(size=(20, 2)) + [8, 2]) @ basis.T
        projection = fit_response_svd([delta, 2*delta], ['a']*19+['b'])
        np.testing.assert_array_equal(projection.mean, 0)
        np.testing.assert_allclose(projection.reconstruct_displacement(delta, 2), delta, atol=1e-12)
        np.testing.assert_array_equal(projection.transform(np.zeros((1, 8))), 0)
        self.assertAlmostEqual(projection.variance_ratio[:2].sum(), 1.)

    def test_weighted_pca_preserves_geometry_in_retained_subspace(self):
        rng = np.random.default_rng(4)
        x = rng.normal(size=(20, 6))
        other = x + rng.normal(size=(20, 6))
        model = fit_weighted_pca([x, other], ['a']*19+['b'])
        expected = ((x[:19].mean(0)+x[-1])/2 + (other[:19].mean(0)+other[-1])/2)/2
        np.testing.assert_allclose(model.mean, expected)
        np.testing.assert_allclose(model.components @ model.components.T, np.eye(6), atol=1e-12)
        delta = other-x
        full = model.reconstruct_displacement(delta, 6)
        np.testing.assert_allclose(full, delta, atol=1e-12)
        retained = model.reconstruct_displacement(delta, 2)
        np.testing.assert_allclose(np.sum(retained**2,1), np.sum(model.displacement(delta,2)**2,1), atol=1e-12)
        self.assertTrue((np.sum(retained**2,1) <= np.sum(delta**2,1)+1e-12).all())

    def test_exact_finite_readout_across_relu_crossings(self):
        torch.manual_seed(2)
        model = SizeFiLMGINCurvature(3, hidden_dim=8, num_layers=2, global_dim=1, dropout=.2).eval()
        rng = np.random.default_rng(12)
        h = rng.normal(size=(30,8)).astype(np.float32)
        d = rng.normal(size=(30,8)).astype(np.float32)*3
        result = exact_readout(model,h,d,.3)
        exact = head_prediction(model,h+d,.3)-head_prediction(model,h,.3)
        np.testing.assert_allclose(result['effect'],exact,atol=2e-6,rtol=1e-5)
        np.testing.assert_allclose(result['positive']+result['negative'],result['effect'],atol=1e-10)
        zero = exact_readout(model,h,np.zeros_like(h),.3)
        np.testing.assert_array_equal(zero['effect'],0)

    def test_bilinear_change_is_exact_and_assigns_fixed_factors_zero(self):
        rng=np.random.default_rng(3)
        g1,g2,d1,d2=[rng.normal(size=(20,6)).astype(np.float32) for _ in range(4)]
        dr,dg=bilinear_change(g1,d1,g2,d2)
        expected=np.sum(g2.astype(float)*d2-g1.astype(float)*d1,axis=1)
        np.testing.assert_allclose(dr+dg,expected,atol=1e-12)
        dr,dg=bilinear_change(g1,d1,g1,d2)
        np.testing.assert_array_equal(dg,0)
        dr,dg=bilinear_change(g1,d1,g2,d1)
        np.testing.assert_array_equal(dr,0)


if __name__=='__main__':unittest.main()
