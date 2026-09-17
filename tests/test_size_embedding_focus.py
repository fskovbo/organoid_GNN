import unittest
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from src.analysis.size_embedding_focus import (curvature_order, remap_table,
    shapley_changes, head_prediction, decompose_interval)


class FocusedEmbeddingTests(unittest.TestCase):
    def test_curvature_order_and_empty_reference_cluster(self):
        order = curvature_order([0, 0, 1, 1, 2], [5., 7., -3., -1., 2.], 4)
        self.assertEqual(order.raw_cluster.tolist(), [1, 2, 0, 3])
        self.assertTrue(order.iloc[-1].empty_at_reference)
        np.testing.assert_allclose(order.median_predicted_residual[:3], [-2, 2, 6])

    def test_mapping_keeps_probabilities_labels_and_transitions_consistent(self):
        order = curvature_order([0, 1, 2], [5., -2., 1.], 3)
        frame = pd.DataFrame(dict(cluster=[0, 1], reference_cluster=[2, 0], source=[1, 0], target=[2, 1],
            prob_C0=[.8, .1], prob_C1=[.1, .7], prob_C2=[.1, .2], prediction_z=[4., 6.]))
        mapped = remap_table(frame, order)
        np.testing.assert_array_equal(mapped.cluster, [2, 0])
        np.testing.assert_array_equal(mapped.reference_cluster, [1, 2])
        np.testing.assert_array_equal(mapped.source, [0, 2])
        np.testing.assert_array_equal(mapped.target, [1, 0])
        np.testing.assert_array_equal(mapped[[f"prob_C{k}" for k in range(3)]].to_numpy().argmax(1), mapped.cluster)
        np.testing.assert_array_equal(mapped.prediction_z, frame.prediction_z)
        np.testing.assert_array_equal(frame.cluster, [0, 1])

    def test_exact_endpoint_decomposition_with_interactions(self):
        values = np.array([1 + 2*(m & 1) + 3*bool(m & 2) + 8*bool(m & 1)*bool(m & 2) for m in range(8)])
        parts = shapley_changes(values)
        np.testing.assert_allclose(parts, [6, 7, 0])
        self.assertEqual(parts.sum(), values[-1] - values[0])

    def test_linear_head_distinguishes_response_from_background(self):
        model = torch.nn.Module()
        model.head = torch.nn.Linear(3, 2, bias=False)
        with torch.no_grad():
            model.head.weight[:] = torch.tensor([[2., -1., 4.], [0., 0., 0.]])
        pre = dict(size_center=0., size_scale=1., residual_transform=SimpleNamespace(inverse=lambda x: x))
        states = []
        for h, n in [(np.array([[1., 1.], [2., 1.]], dtype=np.float32), 80),
                     (np.array([[4., -2.], [6., -1.]], dtype=np.float32), 700)]:
            states.append(dict(h=h, z=head_prediction(model, h, np.log(n))))
        cases = pd.DataFrame(dict(base_request=[0], edit_request=[1]))
        parts, error = decompose_interval(model, *states, cases, pre,
            SimpleNamespace(alpha=np.log(4*np.pi), beta=1.), 80, 700)
        np.testing.assert_allclose(parts["delta_z"].ravel(), [1, 0, 0, 0], atol=1e-5)
        np.testing.assert_allclose(parts["delta_relative"].sum(), 1940, atol=.01)
        self.assertLess(error, 3e-6)


if __name__ == "__main__":
    unittest.main()
