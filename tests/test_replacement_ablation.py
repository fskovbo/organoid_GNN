import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch_geometric.data import Data

from src.analysis.interventions.replacement import (
    MatchConfig, ReplacementMatcher, describe_graph, identity_codes, make_cases,
    replacement_weights, evaluate_replacements, contrast_table, load_comparison,
)
from src.analysis.interventions.size_sweeps import evaluate_size_ablation
from src.data.target_transforms import IdentityTransform


def graph(name='validation'):
    return Data(x=torch.tensor([[1., 0.], [0., 1.], [0., 0.], [1., 0.]]),
        edge_index=torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]]),
        y=torch.zeros(4, 1), organoid_str=name, full_num_cells=4.,
        global_feat=torch.tensor([[np.log(4)]], dtype=torch.float32))


class KnownModel(nn.Module):
    def forward(self, x, edge_index, data=None):
        signal = 2*x[:, 0] + 3*x[:, 1]
        neighbors = torch.zeros_like(signal)
        neighbors.index_add_(0, edge_index[1], signal[edge_index[0]])
        mu = signal + neighbors * data.global_feat[data.batch, 0]
        return (mu, torch.zeros_like(mu)), x


def donor_pool():
    rows = []
    # Small organs provide identity 0; large organs provide unassigned (2).
    for n, identity in [(100, 0), (400, 2)]:
        for org in range(3):
            for node in range(4):
                rows.append(dict(organoid_str=f'{n}_{org}', node=node, identity=identity,
                    n_cells=n, degree=2., fraction_0=.5, fraction_1=.5, fraction_2=0., reach_1=1, reach_2=1))
    return pd.DataFrame(rows)


def match_config(**kwargs):
    return MatchConfig(neighbors=24, min_cells=3, min_organoids=2,
        min_identity_cells=2, min_identity_organoids=2, max_size_ratio=1.5, **kwargs)


class ReplacementTests(unittest.TestCase):
    def test_descriptors_and_exclusive_validation(self):
        g = graph(); d = describe_graph(g)
        np.testing.assert_array_equal(d.identity, [0, 1, 2, 0])
        self.assertEqual(d.loc[1, 'fraction_1'], 0)  # own fate excluded
        self.assertEqual(d.loc[1, 'fraction_2'], .5)
        self.assertEqual(d.loc[0, 'reach_2'], 1 << 2)
        with self.assertRaises(ValueError):
            identity_codes([[1., 1.]])
        # A source two hops from recipient still uses all its full-graph neighbors.
        self.assertEqual(d.loc[2, 'degree'], 2)

    def test_fold_leakage_support_and_N_dependence(self):
        donors = donor_pool(); matcher = ReplacementMatcher(donors, 3, match_config())
        with self.assertRaises(ValueError):
            ReplacementMatcher(donors, 3, validation_organoids=['100_0'])
        context = donors.iloc[0].to_dict()
        kwargs = dict(center_identity=0, source_identity=1, hop=1)
        a, audit, _, _ = matcher.query(context, count=100, **kwargs)
        b, _, _, _ = matcher.query(context, count=400, **kwargs)
        np.testing.assert_array_equal(a, [1, 0, 0]); np.testing.assert_array_equal(b, [0, 0, 1])
        self.assertTrue(audit['supported'])
        empty, audit, _, _ = matcher.query(context, count=10000, **kwargs)
        self.assertTrue(np.isnan(empty).all()); self.assertFalse(audit['supported'])
        # Original source identity can never be a replacement.
        empty, audit, _, _ = matcher.query(context, center_identity=0, source_identity=0, hop=1, count=100)
        self.assertFalse(audit['supported'])

    def test_cases_include_unassigned_and_keep_sources_paired(self):
        g = graph()
        subs, cases, contexts = make_cases([g], ['A', 'B'], centers_per_identity=2)
        self.assertIn('Unassigned', cases.center_marker.tolist())
        self.assertIn('Unassigned', cases.source_marker_name.tolist())
        self.assertTrue((cases.orig_center != cases.orig_source_node).all())
        _, again, _ = make_cases([g], ['A', 'B'], centers_per_identity=2)
        pd.testing.assert_frame_equal(cases, again)
        for c, context in zip(cases.to_dict('records'), contexts):
            self.assertEqual(context['node'], c['orig_source_node'])

    def test_equal_organoid_weight_and_rare_identity_filter(self):
        donors = donor_pool()
        donors.n_cells = 100
        # One organoid contributes 12 cells, the other contributes one. Each
        # organoid still contributes half of the replacement probability.
        donors = donors.iloc[[*range(12), 12]].copy()
        donors.loc[donors.identity == 0, 'organoid_str'] = 'large_donor'
        cfg = MatchConfig(neighbors=24, min_cells=1, min_organoids=1,
                          min_identity_cells=1, min_identity_organoids=1)
        matcher = ReplacementMatcher(donors, 3, cfg)
        p, _, _, _ = matcher.query(donors.iloc[0].to_dict(), center_identity=0,
                                   source_identity=1, hop=1, count=100)
        np.testing.assert_allclose(p, [.5, 0, .5])
        cfg = MatchConfig(neighbors=24, min_cells=1, min_organoids=1,
                          min_identity_cells=2, min_identity_organoids=1)
        matcher = ReplacementMatcher(donors, 3, cfg)
        p, _, _, _ = matcher.query(donors.iloc[0].to_dict(), center_identity=0,
                                   source_identity=1, hop=1, count=100)
        np.testing.assert_array_equal(p, [1, 0, 0])

    def test_fixed_reference_uses_observed_size(self):
        donors = donor_pool()
        matcher = ReplacementMatcher(donors, 3, match_config())
        cases = pd.DataFrame([dict(case_id=0, observed_n=100, center_identity=0,
                                   source_identity=1, hop=1)])
        contexts = [donors.iloc[0].to_dict()]
        fixed, _ = replacement_weights(cases, contexts, matcher)
        dependent, _ = replacement_weights(cases, contexts, matcher, count=400)
        repeated_fixed, _ = replacement_weights(cases, contexts, matcher)
        np.testing.assert_array_equal(fixed, repeated_fixed)
        self.assertFalse(np.array_equal(fixed, dependent))

    def test_old_zero_equivalence_nonmutation_and_size_override(self):
        g = graph(); original_x = g.x.clone(); original_edges = g.edge_index.clone()
        subs, cases, _ = make_cases([g], ['A', 'B'])
        model = KnownModel(); transform = IdentityTransform()
        prediction = evaluate_replacements(subs, cases, model, transform, size_center=0,
                                          size_scale=1, count=100, batch_size=7)
        marked = cases[cases.source_identity != 2]
        old_cases = []
        for c in marked.to_dict('records'):
            c['source_marker'] = c['source_identity']
            old_cases.append(c)
        old = evaluate_size_ablation(subs, old_cases, model, transform,
            size_center=0, size_scale=1, count=100, device='cpu')
        delta_zero = prediction['mu'][:, 2] - prediction['base_mu']
        np.testing.assert_allclose(delta_zero[marked.index], old.delta_mu)
        np.testing.assert_array_equal(delta_zero[cases.source_identity == 2], 0)
        np.testing.assert_array_equal(g.x, original_x); np.testing.assert_array_equal(g.edge_index, original_edges)
        for c in cases.to_dict('records'):
            np.testing.assert_array_equal(subs[c['subgraph_index']].x, g.x[subs[c['subgraph_index']].orig_nodes])

    def test_weighting_physical_predictions_and_static_identity(self):
        cases = pd.DataFrame([dict(observed_n=100, source_identity=1)])
        p = np.array([[.25, 0, .75]])
        # Nonlinear inverse: weighted physical contrasts must not be inverse(weighted z).
        z = np.array([[0., 1., 2.]])
        pred = dict(z=z, mu=np.sinh(z), base_z=np.array([1.]), base_mu=np.sinh([1.]))
        result = contrast_table(cases, pred, p, p, count=None, fold=0, seed=42,
                                reference=dict(alpha=0., beta=0.))
        expected = .25*np.sinh(0) + .75*np.sinh(2) - np.sinh(1)
        self.assertAlmostEqual(result.replacement_fixed_delta_mu.item(), expected)
        self.assertNotAlmostEqual(expected, np.sinh(1.5)-np.sinh(1))
        self.assertEqual(result.replacement_fixed_delta_mu.item(), result.replacement_size_dependent_delta_mu.item())

    def test_constant_support_across_sweep(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp); (out/'tables').mkdir()
            (out/'settings.json').write_text(json.dumps(dict(counts=[100, 400], markers=['A', 'Unassigned'])))
            (out/'complete.json').write_text('{}')
            rows = []
            for n in [100, 400]:
                for case in [0, 1]:
                    row = dict(fold=0, seed=42, case_id=case, analysis='sweep', evaluated_n=n,
                        observed_n=100, size_fold_change=n/100, organoid_str=f'v{case}', center_marker='A',
                        source_marker_name='Unassigned', hop=1, fixed_supported=True,
                        dependent_supported=not (case == 1 and n == 400))
                    for method in ['zero', 'replacement_fixed', 'replacement_size_dependent']:
                        for metric in ['delta_mu', 'delta_relative', 'delta_z']:
                            row[f'{method}_{metric}'] = float(case)
                    rows.append(row)
            pd.DataFrame(rows).to_csv(out/'tables/test.csv.gz', index=False)
            result = load_comparison(out, bootstrap_samples=5)
            self.assertEqual(set(result['paired_cases'].case_id), {0})
            self.assertEqual(len(result['paired_cases']), 2)
            self.assertTrue((result['summary']['mean'] == 0).all())


if __name__ == '__main__':
    unittest.main()
