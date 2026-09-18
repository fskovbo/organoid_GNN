"""Scientific invariants for the paired embedding experiment."""
import unittest
from dataclasses import replace

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data, Batch

from src.analysis.embeddings.size_responses import (EmbeddingConfig, infer_states, fit_shared_atlas,
    project_atlas, representation_change, paired_bootstrap_curves, assign_observed_windows,
    _transitions)
from src.data.subgraphs import build_ego_subgraphs_for_graph
from src.models.gnn import SizeFiLMGINCurvature


class SizeEmbeddingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_local_embedding_routes_observed_sizes_and_nonmutation(self):
        torch.manual_seed(17)
        model = SizeFiLMGINCurvature(3, hidden_dim=8, num_layers=2, global_dim=1,
                                    dropout=0, norm="batch").eval()
        for film in model.film_layers:
            with torch.no_grad():
                film[-1].weight.fill_(.15)
        graph = Data(x=torch.eye(3), y=torch.zeros(3),
            edge_index=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]]),
            center_idx=0, full_num_cells=80)
        other = graph.clone()
        other.full_num_cells = 700
        pre = dict(size_center=5., size_scale=.7)
        req = [(0, ()), (0, ((1, 1),)), (1, ())]
        args = dict(model=model, subgraphs=[graph, other], requests=req, pre=pre, batch_size=2)
        ref = infer_states(**args, count=361)
        head = infer_states(**args, count=700, route="head_only")
        full = infer_states(**args, count=700)
        film = infer_states(**args, count=700, route="film_only")
        self.assertEqual(full["h"].shape, (3, 8))  # Explicit size excluded.
        np.testing.assert_allclose(head["h"], ref["h"], atol=1e-6)
        np.testing.assert_allclose(film["h"], full["h"], atol=1e-6)
        self.assertFalse(np.allclose(full["h"], ref["h"]))
        observed = infer_states(**args)
        at80 = infer_states(**args, count=80)
        np.testing.assert_allclose(observed["h"][:2], at80["h"][:2], atol=1e-6)
        np.testing.assert_allclose(observed["h"][2], full["h"][2], atol=1e-6)
        torch.testing.assert_close(graph.x, torch.eye(3))
        again = infer_states(**args, count=700)
        np.testing.assert_array_equal(full["h"], again["h"])
        with self.assertRaisesRegex(ValueError, "Center identity"):
            infer_states(model, [graph], [(0, ((0, 0),))], pre, count=80)

    def test_ego_center_matches_full_graph_at_depth_two(self):
        torch.manual_seed(3)
        model = SizeFiLMGINCurvature(3, hidden_dim=8, num_layers=2, global_dim=1,
                                    dropout=0, norm="batch").eval()
        graph = Data(x=torch.eye(3)[torch.arange(7) % 3], y=torch.zeros(7),
            edge_index=torch.tensor([[0,1,1,2,2,3,3,4,4,5,5,6],[1,0,2,1,3,2,4,3,5,4,6,5]]),
            global_feat=torch.tensor([[np.log(80.)]], dtype=torch.float32))
        sub = build_ego_subgraphs_for_graph(graph, centers=[2], num_hops=2)[0]
        sub.full_num_cells = 7
        predicted = infer_states(model, [sub], [(0, ())], dict(size_center=0., size_scale=1.), count=80)
        batch = Batch.from_data_list([graph])
        with torch.no_grad():
            (mu, _), h = model(batch.x, batch.edge_index, batch)
        np.testing.assert_allclose(predicted["h"][0], h[2, :8], atol=1e-6)
        np.testing.assert_allclose(predicted["z"][0], mu[2], atol=1e-6)

    def test_shared_projection_and_common_motion_control(self):
        rng = np.random.default_rng(2)
        ref = rng.normal(size=(40, 8))
        shifted = 2.5 * ref + np.arange(8)
        stats = representation_change(shifted, ref)
        self.assertAlmostEqual(stats["common_positive_gain"], 2.5)
        self.assertLess(stats["relative_shape_mismatch"], 1e-12)
        changed = shifted.copy()
        changed[:10, 0] += 12
        self.assertGreater(representation_change(changed, ref)["relative_shape_mismatch"], .1)
        config = replace(EmbeddingConfig(), pca_dim=3, n_clusters=2, sensitivity_k=(2, 3), gmm_n_init=1)
        atlas, sensitivity = fit_shared_atlas([ref, shifted], config)
        before = atlas["scaler"].mean_.copy()
        _, labels, probability = project_atlas(atlas, ref)
        np.testing.assert_allclose(probability.sum(1), 1)
        project_atlas(atlas, changed)
        np.testing.assert_array_equal(atlas["scaler"].mean_, before)
        np.testing.assert_array_equal(project_atlas(atlas, ref)[1], labels)
        self.assertEqual(len(sensitivity), 2)

    def test_bootstrap_pairs_n_and_weights_organs_not_nodes(self):
        frame = pd.DataFrame([dict(marker="a", organoid_str=org, n=n, value=v*scale)
            for org, copies, v in [("large", 20, 1.), ("small", 1, 5.)]
            for _ in range(copies) for n, scale in [(80, 1), (700, 2)]])
        result = paired_bootstrap_curves(frame, ["marker"], ["value"], draws=100, min_organoids=2).set_index("n")
        self.assertEqual(result.loc[80, "mean"], 3)
        self.assertEqual(result.loc[700, "mean"], 6)
        self.assertEqual(result.loc[700, "low"], 2 * result.loc[80, "low"])
        self.assertEqual(result.loc[700, "high"], 2 * result.loc[80, "high"])
        self.assertEqual(result.loc[80, "n_organoids"], 2)

    def test_transition_pairing_and_observed_nonoverlap(self):
        config = replace(EmbeddingConfig(), anchors=(80, 700), n_clusters=2)
        rows = [dict(node_id=i, state=f"N{n}", cluster=c)
                for n, labels in [(80, [0, 0, 1]), (700, [1, 0, 0])] for i, c in enumerate(labels)]
        nodes = pd.DataFrame(dict(node_id=[0, 1, 2], organoid_str=["a", "a", "b"]))
        result = _transitions(pd.DataFrame(rows), nodes, config)
        self.assertAlmostEqual(result.joint_fraction.sum(), 1)
        self.assertAlmostEqual(result.loc[(result.source == 1) & (result.target == 0), "joint_fraction"].item(), .5)
        labels = assign_observed_windows(np.array([80, 300, 550, 620, 700, 2000]), [80, 300, 550, 700], 1.25)
        self.assertEqual(labels[-1], -1)
        self.assertIn(labels[3], [550, 700])
        np.testing.assert_array_equal(labels[:3], [80, 300, 550])


if __name__ == "__main__":
    unittest.main()
