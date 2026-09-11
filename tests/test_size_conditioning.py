import copy


import json


from pathlib import Path


import numpy as np


import pandas as pd


import unittest


from unittest.mock import patch


from tempfile import TemporaryDirectory


import torch


from torch import nn


from torch_geometric.data import Batch, Data


from src.analysis.pseudotime import (
    evaluate_size_ablation,
    expand_center_markers,
    make_size_ablation_cases,
    summarize_by_organoid,
)


from src.data.target_transforms import StandardizeTransform


from src.models.gnn import GINCurvature, SizeFiLMGINCurvature


def graph(size=100.0, name="organoid_a"):
    return Data(
        x=torch.tensor([[1., 0.], [0., 1.], [1., 1.]]),
        y=torch.zeros(3, 1),
        edge_index=torch.tensor([[0, 1, 0, 2], [1, 0, 2, 0]]),
        global_feat=torch.tensor([[np.log(size / 100) / np.log(2)]], dtype=torch.float32),
        center_idx=0, orig_center=10, orig_nodes=torch.tensor([10, 11, 12]),
        organoid_str=name, full_num_cells=size, timepoint_label="day4",
    )


class _KnownSizeResponse(nn.Module):
    def forward(self, x, edge_index, data=None):
        neighborhood = torch.zeros(len(x), dtype=x.dtype)
        neighborhood.index_add_(0, edge_index[1], x[edge_index[0], 1])
        mu = neighborhood * (1 + data.global_feat[data.batch, 0])
        return (mu, torch.zeros_like(mu)), x



class SizeConditioningTests(unittest.TestCase):
    def patch_attribute(self, target, name, value):
        self.enterContext(patch.object(target, name, value))

    def test_identity_film_matches_normal_gin_and_has_trainable_modulation(self):
        torch.manual_seed(21)
        kwargs = dict(n_markers=2, hidden_dim=8, num_layers=2, global_dim=1, dropout=0)
        normal = GINCurvature(**kwargs).eval()
        film = SizeFiLMGINCurvature(**kwargs, film_hidden_dim=4).eval()
        film.load_state_dict(normal.state_dict(), strict=False)
        batch = Batch.from_data_list([graph(), graph(200, "organoid_b")])
        normal_out, normal_h = normal(batch.x, batch.edge_index, batch)
        film_out, film_h = film(batch.x, batch.edge_index, batch)
        torch.testing.assert_close(normal_h, film_h)
        for expected, actual in zip(normal_out, film_out):
            torch.testing.assert_close(expected, actual)
        (film_out[0].square().sum() + film_out[1].sum()).backward()
        assert any(layer[-1].weight.grad.abs().sum() > 0 for layer in film.film_layers)


    def test_film_conditions_hidden_features_and_respects_graph_batching(self):
        torch.manual_seed(4)
        model = SizeFiLMGINCurvature(
            2, hidden_dim=8, num_layers=2, global_dim=0, dropout=0, film_hidden_dim=4,
        ).eval()
        # Activate a deterministic nonconstant modulation, independent of the head.
        with torch.no_grad():
            for layer in model.film_layers:
                layer[0].weight.fill_(1)
                layer[0].bias.zero_()
                layer[-1].weight.fill_(0.1)
        a, b = graph(), graph(200, "organoid_b")
        batch = Batch.from_data_list([a, b])
        both_out, both_h = model(batch.x, batch.edge_index, batch)
        assert not torch.allclose(both_h[:3], both_h[3:])
        for idx, g in enumerate([a, b]):
            single = Batch.from_data_list([g])
            out, h = model(single.x, single.edge_index, single)
            torch.testing.assert_close(out[0], both_out[0][idx * 3:(idx + 1) * 3])
            torch.testing.assert_close(h, both_h[idx * 3:(idx + 1) * 3])
        bad = graph()
        bad.global_feat = torch.zeros(1, 3)
        batch = Batch.from_data_list([bad])
        with self.assertRaisesRegex(ValueError, "exactly 1"):
            model(batch.x, batch.edge_index, batch)


    def test_geometry_reaches_head_but_film_depends_only_on_size(self):
        torch.manual_seed(4)
        model = SizeFiLMGINCurvature(
            2, hidden_dim=8, num_layers=2, global_dim=4, dropout=0, film_hidden_dim=4,
        ).eval()
        normal = GINCurvature(2, hidden_dim=8, num_layers=2, global_dim=4, dropout=0).eval()
        model.load_state_dict(normal.state_dict(), strict=False)
        a, b, c = graph(), graph(), graph(200)
        a.global_feat = torch.tensor([[0., 2., 3., 4.]])
        b.global_feat = torch.tensor([[0., 5., 6., 7.]])
        c.global_feat = torch.tensor([[1., 2., 3., 4.]])
        batch = Batch.from_data_list([a, b, c])
        before, _ = model(batch.x, batch.edge_index, batch)
        expected, _ = normal(batch.x, batch.edge_index, batch)
        for actual, value in zip(before, expected):
            torch.testing.assert_close(actual, value)
        with torch.no_grad():
            for layer in model.film_layers:
                layer[0].weight.fill_(1)
                layer[0].bias.zero_()
                layer[-1].weight.fill_(0.1)
            # Make the head's geometry dependence deterministic.
            model.head.net[0].weight.zero_()
            model.head.net[0].bias.fill_(10)
            model.head.net[0].weight[0, 9] = 1  # embedding width 8, then N, then area
            model.head.net[-1].weight.zero_()
            model.head.net[-1].weight[0, 0] = 1
        output, h = model(batch.x, batch.edge_index, batch)
        torch.testing.assert_close(h[:3, :8], h[3:6, :8])
        assert not torch.allclose(h[:3, :8], h[6:, :8])
        torch.testing.assert_close(output[0][3:6] - output[0][:3], torch.full_like(output[0][:3], 3.))
        for i, g in enumerate([a, b, c]):
            single = Batch.from_data_list([g])
            single_output, single_h = model(single.x, single.edge_index, single)
            torch.testing.assert_close(single_h, h[i * 3:(i + 1) * 3])
            torch.testing.assert_close(single_output[0], output[0][i * 3:(i + 1) * 3])


    def test_sweep_preserves_geometry_for_intact_and_ablated_inputs(self):
        class GeometryResponse(_KnownSizeResponse):
            def forward(self, x, edge_index, data=None):
                expected = data.global_feat.new_tensor([[2., 3., 4.]]).expand(data.num_graphs, -1)
                torch.testing.assert_close(data.global_feat[:, 1:], expected)
                (mu, lv), h = super().forward(x, edge_index, data)
                return (mu * data.global_feat[data.batch, 1], lv), h

        g = graph()
        g.global_feat = torch.tensor([[0., 2., 3., 4.]])
        original = g.clone()
        cases = [c for c in make_size_ablation_cases([g], ['center', 'source'])
                 if c['source_marker_name'] == 'source']
        kwargs = dict(model=GeometryResponse(), target_transform=StandardizeTransform().fit_array(np.array([-1., 1.])),
                      size_center=np.log(100), size_scale=np.log(2), batch_size=1, device='cpu')
        observed = evaluate_size_ablation([g], cases, **kwargs)
        swept = evaluate_size_ablation([g], cases, count=200, **kwargs)
        np.testing.assert_allclose(observed.delta_mu, [-2])
        np.testing.assert_allclose(swept.delta_mu, [-4])
        torch.testing.assert_close(g.global_feat, original.global_feat)
        torch.testing.assert_close(g.x, original.x)


    def test_ablation_sweep_is_paired_and_does_not_mutate_graphs(self):
        graphs = [graph()]
        original = copy.deepcopy(graphs[0])
        cases = make_size_ablation_cases(graphs, ["center", "source"], seed=8)
        assert cases == make_size_ablation_cases(graphs, ["center", "source"], seed=8)
        source_cases = [c for c in cases if c['source_marker_name'] == 'source']
        assert len(source_cases) == 1
        transform = StandardizeTransform().fit_array(np.array([-1., 1.]))
        kwargs = dict(model=_KnownSizeResponse(), target_transform=transform,
                      size_center=np.log(100), size_scale=np.log(2), batch_size=1, device='cpu')
        observed = evaluate_size_ablation(graphs, source_cases, **kwargs)
        same = evaluate_size_ablation(graphs, source_cases, count=100, **kwargs)
        larger = evaluate_size_ablation(graphs, source_cases, count=200, **kwargs)
        np.testing.assert_allclose(observed.delta_mu, [-1])
        np.testing.assert_allclose(same.delta_mu, observed.delta_mu)
        np.testing.assert_allclose(larger.delta_mu, [-2])
        assert larger.orig_source_node.tolist() == observed.orig_source_node.tolist()
        assert observed.delta_mse.notna().all()
        assert larger.delta_mse.isna().all()
        assert observed.observed_n.tolist() == [100]
        assert larger.evaluated_n.tolist() == [200]
        torch.testing.assert_close(graphs[0].x, original.x)
        torch.testing.assert_close(graphs[0].global_feat, original.global_feat)
        torch.testing.assert_close(graphs[0].edge_index, original.edge_index)
        assert evaluate_size_ablation(graphs, [], **kwargs).empty
        with self.assertRaisesRegex(ValueError, 'positive'):
            evaluate_size_ablation(graphs, source_cases, count=0, **kwargs)


    def test_summaries_weight_organoids_not_cases_or_seeds(self):
        frame = pd.DataFrame(dict(model=['a'] * 11, organoid_str=['one'] * 10 + ['two'],
                                  delta_mu=[1.] * 10 + [5.], center_marker_names=[['a', 'b']] * 11))
        result = summarize_by_organoid(frame, ['model'], 'delta_mu', bootstrap_samples=20)
        assert result.iloc[0]['mean'] == 3
        assert result.iloc[0].n_organoids == 2
        assert len(expand_center_markers(frame)) == 22


    def test_notebook_synthetic_workflow_without_training(self):
        """Exercise all notebook cells with synthetic data and a no-optimization stub.

        No project data, experiment outputs, or trained checkpoints are used.
        The stub verifies the four-feature baseline and all four model input paths.
        """
        tmp_path = Path(self.enterContext(TemporaryDirectory()))
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import src.data.io as io
        import src.data.metadata as metadata
        import src.training.loop as training

        fixtures = []
        for i, n in enumerate([12, 16, 20, 24, 28, 32, 36, 40]):
            nodes = torch.arange(n - 1)
            x = torch.zeros(n, 2)
            x[:, 0] = 1
            x[::2, 1] = 1
            fixtures.append(Data(
                x=x, y=torch.linspace(-1, 1, n).reshape(-1, 1) + 0.1 * i,
                edge_index=torch.stack([torch.cat([nodes, nodes + 1]), torch.cat([nodes + 1, nodes])]),
                organoid_str=f'organoid_{i}',
                meta=dict(total_surface_area=10. + i, total_volume=20. + i,
                          num_nodes=n, dataset='synthetic', label_uid=i, timepoint=f'day{3 + i % 2}'),
            ))
        self.patch_attribute(io, 'load_graph_dataset_from_dir', lambda path: copy.deepcopy(fixtures))
        self.patch_attribute(metadata, 'load_aux_metadata_for_dir', lambda path: {})
        self.patch_attribute(metadata, 'load_marker_names_from_dir', lambda path: ['center', 'source'])
        calls = []

        def no_training(model, train_graphs, val_graphs, cfg):
            is_baseline = type(model).__name__ == 'GlobalFeatureMLP'
            expected_dim = 4 if is_baseline else model.global_dim
            assert all(g.global_feat.shape == (1, expected_dim) for g in train_graphs + val_graphs)
            if not is_baseline:
                assert {g.organoid_str for g in train_graphs}.isdisjoint(g.organoid_str for g in val_graphs)
                assert all(not hasattr(g, 'meta') for g in train_graphs + val_graphs)
            calls.append(type(model).__name__)
            return model.eval(), {'val_mae': 1.}, dict(train_loss=[1.], val_loss=[1.], train_mae=[1.], val_mae=[1.])

        self.patch_attribute(training, 'train', no_training)
        self.patch_attribute(plt, 'show', lambda: plt.close('all'))
        notebook = json.loads((Path(__file__).resolve().parents[1] / 'experiments/size_conditioned_ablation.ipynb').read_text())
        ns = {}
        for index, cell in enumerate(notebook['cells']):
            if cell['cell_type'] != 'code':
                continue
            if 'saved_results_plotting' in cell.get('metadata', {}).get('tags', []):
                continue  # This independent section reads the completed experiment.
            # Stored notebook outputs do not affect the synthetic workflow check.
            exec(compile(''.join(cell['source']), f'notebook_cell_{index}', 'exec'), ns)
            if index == 3:
                ns.update(RUN_TRAINING=True, RUN_ABLATIONS=True, N_FOLDS=2, MODEL_SEEDS=[42],
                          HIDDEN_DIM=8, FILM_HIDDEN_DIM=4, NUM_WORKERS=0, DEVICE='cpu',
                          FILTER_BLACKLISTED_ORGANOIDS=False, INTERPOLATE_TARGET_OUTLIERS=False,
                          BOOTSTRAP_SAMPLES=20, MIN_ORGANOIDS_PER_EFFECT=1,
                          ABLATION_CENTERS_PER_ORGANOID=2, SWEEP_CENTERS_PER_ORGANOID=1,
                          SWEEP_N_POINTS=3, N_SIZE_BINS=2, SAVE_DIR=tmp_path / 'synthetic')
        assert calls.count('GlobalFeatureMLP') == 2
        assert calls.count('GINCurvature') == calls.count('SizeFiLMGINCurvature') == 4
        assert len(ns['mse_df']) == 32
        assert set(ns['mse_df'].model) == set(ns['MODEL_NAMES'])
        assert set(ns['paired_overall'].comparison) == set(ns['comparisons'])
        assert ns['sweep_cases_df'].delta_mse.isna().all()
        assert ns['observed_cases'].delta_mse.notna().all()
        assert (tmp_path / 'synthetic/tables/paired_size_endpoint_contrast.csv').is_file()



if __name__ == "__main__":
    unittest.main()
