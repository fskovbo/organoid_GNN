"""Resuming a saved metrics table must not duplicate baseline columns."""
import unittest
import pandas as pd
from src.analysis.metrics.evaluation import compare_baseline_mse


class ResumedMetricsTests(unittest.TestCase):
    def test_repeated_pairing_and_new_fold(self):
        baseline = pd.DataFrame([
            dict(fold=0, organoid_str='a', mse=4.),
            dict(fold=1, organoid_str='b', mse=3.),
        ])
        scores = pd.DataFrame([dict(fold=0, organoid_str='a', mse=2., mae=1.)])
        first = compare_baseline_mse(scores, baseline)
        pd.testing.assert_frame_equal(first, compare_baseline_mse(first, baseline))
        resumed = pd.concat([first, pd.DataFrame([dict(fold=1, organoid_str='b', mse=1., mae=.5)])])
        paired = compare_baseline_mse(resumed, baseline)
        self.assertEqual(paired.baseline_mse.tolist(), [4., 3.])
        self.assertEqual(paired.mse_minus_baseline.tolist(), [-2., -2.])
        self.assertEqual(paired.mae.tolist(), [1., .5])


if __name__ == '__main__':
    unittest.main()
