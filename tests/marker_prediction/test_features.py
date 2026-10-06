import unittest
import numpy as np
import pandas as pd
from src.marker_prediction.features import ring_features, select_features
from src.marker_prediction.classifiers import precision_threshold, organoid_splits


class MarkerPredictionTests(unittest.TestCase):
    def test_exact_rings_and_empty_shells(self):
        # Triangle plus a tail: path multiplicity must not duplicate a neighbor.
        x = np.eye(4, dtype=np.float32)
        edges = np.array([[0, 1, 2, 2], [1, 2, 0, 3]])
        f = ring_features(x, edges, 4)
        np.testing.assert_array_equal(f[:, :4], x)
        np.testing.assert_allclose(f[0, 4:8], [0, .5, .5, 0])
        np.testing.assert_allclose(f[0, 8:12], [0, 0, 0, 1])
        np.testing.assert_array_equal(f[:, 12:], 0)
        np.testing.assert_array_equal(x, np.eye(4))

    def test_curvature_is_optional_center_only(self):
        f = np.arange(60, dtype=np.float32).reshape(4, 15)
        k = np.arange(4, dtype=np.float32)[:, None]
        np.testing.assert_array_equal(select_features(f, k, 3, 1, False), f[:, :6])
        selected = select_features(f, k, 3, 1, True)
        self.assertEqual(selected.shape, (4, 7))
        np.testing.assert_array_equal(selected[:, -1], k[:, 0])

    def test_threshold_support_and_abstention(self):
        y = np.array([1, 0, 1, 0])
        p = np.array([.9, .8, .7, .1])
        groups = np.arange(4)
        self.assertTrue(np.isinf(precision_threshold(y, p, groups, .95, min_calls=2, min_organoids=2)))
        self.assertEqual(precision_threshold(y, p, groups, .95, min_calls=1, min_organoids=1), .9)

    def test_splits_exclude_whole_organoids(self):
        table = pd.DataFrame(dict(graph_id=np.arange(100), measured_A=True, measured_B=True,
            positive_A=np.tile([0, 1, 0, 1], 25), positive_B=np.tile([0, 0, 1, 1], 25)))
        folds = list(organoid_splits(table, ['A', 'B'], n_folds=5))
        all_test = []
        for fold in folds:
            sets = [set(fold[r]) for r in ('train', 'calibration', 'test')]
            self.assertEqual(len(set.union(*sets)), 100)
            self.assertEqual(sum(map(len, sets)), 100)
            all_test += fold['test']
        self.assertEqual(sorted(all_test), list(range(100)))
        table.loc[0, 'measured_A'] = False
        with self.assertRaises(ValueError):list(organoid_splits(table, ['A', 'B']))


if __name__ == '__main__':unittest.main()
