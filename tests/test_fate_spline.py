"""Scientific invariants for the explicit fate-interaction spline regression."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from torch_geometric.data import Data, Batch
from src.artifacts.bundle import save_bundle, load_bundle
from src.data.neighborhood_counts import fate_identities, exact_hop_counts
from src.data.target_transforms import IdentityTransform
from src.inference.predict import predict_targets
from src.models.fate_spline import FateSplineCurvature
from src.training.spline_fit import graph_samples, fit_spline, normal_equations, solve_spline


def synthetic_graphs():
    graphs = []
    for i in range(12):
        rng = np.random.default_rng(400+i)
        n = 12+3*i
        identities = rng.integers(0, 3, size=n)
        x = np.eye(3)[identities, :2]
        edges = np.vstack([np.arange(n), np.roll(np.arange(n), -1)])
        graphs.append(Data(x=torch.tensor(x, dtype=torch.float32), edge_index=torch.tensor(edges),
                           y=torch.zeros(n, dtype=torch.float64), full_num_cells=float(n), organoid_str=str(i)))
    return graphs


class FateSplineTests(unittest.TestCase):
    def setUp(self):
        self.graphs = synthetic_graphs()
        self.samples = graph_samples(self.graphs, 2)
        self.model = FateSplineCurvature(2, radius=2).configure(self.samples)
        rng = np.random.default_rng(71)
        self.model.weights.copy_(torch.tensor(rng.normal(0, .05, self.model.weights.shape)))
        self.model.is_fitted.fill_(True)
        for g, sample in zip(self.graphs, self.samples):
            sample['y'] = self.model.predict_sample(sample)
            g.y = torch.tensor(sample['y'])

    def test_unique_shortest_hop_counts_not_walks_and_unassigned(self):
        # Triangle plus tail, duplicate edge and self loop; disconnected last cell.
        x = torch.tensor([[1.,0.], [0.,1.], [0.,0.], [1.,0.], [0.,0.]])
        edge = torch.tensor([[0,1,2,2,0,0], [1,2,0,3,1,0]])
        before = x.clone()
        counts = exact_hop_counts(x, edge, 4)
        np.testing.assert_array_equal(counts[0], [[0,1,1],[1,0,0],[0,0,0],[0,0,0]])
        self.assertEqual(counts[4].sum(), 0)
        torch.testing.assert_close(x, before)
        np.testing.assert_array_equal(fate_identities(x), [0,1,2,0,2])
        with self.assertRaisesRegex(ValueError, 'Exclusive'):
            fate_identities(np.array([[1.,1.]]))

    def test_constraints_and_exact_decomposition_at_every_size(self):
        co = self.model.coefficients([12,20,40])
        cw, sw = self.model.center_reference.numpy(), self.model.source_reference.numpy()
        np.testing.assert_allclose(co['a'] @ cw, 0, atol=1e-15)
        np.testing.assert_allclose(np.einsum('nrt,rt->nr', co['c'], sw), 0, atol=1e-15)
        np.testing.assert_allclose(np.einsum('nrab,a->nrb', co['P'], cw), 0, atol=1e-15)
        np.testing.assert_allclose(np.einsum('nrab,rb->nra', co['P'], sw), 0, atol=1e-15)
        for sample in self.samples:
            for n in (12,25,40):
                np.testing.assert_allclose(self.model.contributions(sample,n)['prediction'],
                                           self.model.predict_sample(sample,n), atol=1e-14)
        self.assertFalse(np.allclose(co['P'], co['P'].transpose(0,1,3,2)))

    def test_empty_shell_zero_and_permutation_equivariance(self):
        sample = dict(self.samples[0], counts=np.zeros_like(self.samples[0]['counts']))
        terms = self.model.contributions(sample)
        self.assertEqual(terms['shared'].sum(), 0)
        self.assertEqual(terms['pairwise'].sum(), 0)
        g = self.graphs[0]
        perm = torch.randperm(len(g.x), generator=torch.Generator().manual_seed(7))
        inverse = torch.argsort(perm)
        pred = self.model(g.x, g.edge_index, data=g)[0][0]
        other = self.model(g.x[perm], inverse[g.edge_index], data=g)[0][0]
        torch.testing.assert_close(other, pred[perm], rtol=1e-13, atol=1e-13)

    def test_identity_swap_formula_and_fresh_features(self):
        g = self.graphs[0]
        center, source = 0, 1
        original = self.model(g.x, g.edge_index, data=g)[0][0].numpy()
        edited = g.x.clone()
        old = int(fate_identities(edited)[source])
        new = (old+1) % 3
        edited[source] = 0
        if new < 2:
            edited[source,new] = 1
        prediction = self.model(edited, g.edge_index, data=g)[0][0].numpy()
        co = self.model.coefficients([len(g.x)])
        a = self.samples[0]['identity'][center]
        m = self.samples[0]['counts'][center,0].sum()
        expected = (co['c'][0,0,new]-co['c'][0,0,old]+co['P'][0,0,a,new]-co['P'][0,0,a,old])/m
        self.assertAlmostEqual(prediction[center]-original[center], expected, places=12)

    def test_sufficient_statistics_equal_explicit_weighted_design(self):
        gram, rhs = normal_equations(self.model, self.samples)
        xx, yy = [], []
        for sample in self.samples:
            local = self.model.local_design(sample['identity'], sample['counts'])
            basis = self.model.basis([sample['N']])[0]
            scale = 1/np.sqrt(len(self.samples)*len(local))
            xx.append(np.einsum('nt,b->ntb',local,basis).reshape(len(local),-1)*scale)
            yy.append(sample['y']*scale)
        x, y = np.concatenate(xx), np.concatenate(yy)
        np.testing.assert_allclose(gram, x.T@x, atol=1e-15)
        np.testing.assert_allclose(rhs, x.T@y, atol=1e-15)
        model = FateSplineCurvature(2,radius=2).configure(self.samples)
        solve_spline(model, (gram,rhs), smoothness=0, ridge=0, pair_ridge=0)
        for sample in self.samples:
            np.testing.assert_allclose(model.predict_sample(sample), sample['y'], atol=2e-6)

    def test_tuning_train_only_and_portable_batched_predictions(self):
        model, meta, trials = fit_spline(self.samples[:9], n_markers=2, radius=2, variant='pairwise',
            smoothness_grid=[.001,.1], pair_ridge_grid=[.001], seed=2)
        self.assertEqual(len(trials),2)
        self.assertEqual(set(meta['inner_train']) | set(meta['inner_validation']), {str(i) for i in range(9)})
        self.assertFalse(set(meta['inner_train']) & set(meta['inner_validation']))
        self.assertAlmostEqual(np.exp(model.log_count_range[1].item()), self.samples[8]['N'])
        # Out-of-range counts clamp, and do not alter the fitted reference state.
        before = {k:v.clone() for k,v in model.state_dict().items()}
        np.testing.assert_allclose(model.coefficients([1000])['P'],model.coefficients([self.samples[8]['N']])['P'])
        for name, tensor in model.state_dict().items():
            torch.testing.assert_close(tensor,before[name])
        with tempfile.TemporaryDirectory() as d:
            save_bundle(Path(d)/'model',dict(model=model, transform=IdentityTransform().fit(self.graphs[:9])),splits={})
            loaded = load_bundle(Path(d)/'model')
            _, pred, _ = predict_targets(self.graphs[9:], loaded['model'], device='cpu', batch_size=2, target_transform=loaded['transform'])
            expected = np.concatenate([model.predict_sample(s) for s in self.samples[9:]])
            np.testing.assert_allclose(pred, expected, atol=1e-13)

    def test_nested_variants_have_no_hidden_pair_terms(self):
        for variant,radius in [('center',0),('shared',2)]:
            model,_,_ = fit_spline(self.samples,n_markers=2,radius=radius,variant=variant,
                smoothness_grid=[.01], pair_ridge_grid=[.001])
            self.assertEqual(model.coefficients([20])['P'].sum(),0.)
            if radius == 0:
                sample = self.samples[0]
                altered = dict(sample, counts=np.zeros_like(sample['counts']))
                np.testing.assert_array_equal(model.predict_sample(sample),model.predict_sample(altered))


if __name__ == '__main__':
    unittest.main()
