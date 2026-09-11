import unittest
import numpy as np
import pandas as pd
from src.analysis.geometric_normalization import fit_area_references, normalize_cases


class GeometricNormalizationTests(unittest.TestCase):
    def test_power_law_fit_uses_only_training_organoids(self):
        n = np.array([10., 20., 40., 80.])
        cohort = pd.DataFrame(dict(organoid_str=list('abcd'), n_cells=n, surface_area=3*n**0.8))
        membership = pd.DataFrame(dict(fold=[0]*4, role=['train']*3+['val'], organoid_str=list('abcd')))
        first, _ = fit_area_references(cohort, membership)
        np.testing.assert_allclose(first[['alpha', 'beta']], [[np.log(3), 0.8]])
        cohort.loc[3, 'surface_area'] *= 1000
        second, diagnostics = fit_area_references(cohort, membership)
        np.testing.assert_allclose(first[['alpha', 'beta']], second[['alpha', 'beta']])
        self.assertGreater(second.val_rmse_log_area.iloc[0], 6)
        self.assertEqual(len(diagnostics), 4)

    def test_normalization_uses_supplied_count_and_preserves_raw_effects(self):
        frame = pd.DataFrame(dict(fold=[0, 0, 1], evaluated_n=[100., 400., 100.],
                                  observed_n=[100.]*3, delta_mu=[-0.2]*3))
        refs = pd.DataFrame(dict(fold=[0, 1], alpha=[np.log(4*np.pi), np.log(8*np.pi)], beta=[1., 1.]))
        result = normalize_cases(frame, refs)
        np.testing.assert_allclose(result.delta_relative, [-20., -80., -40.])
        np.testing.assert_allclose(result.delta_mu, frame.delta_mu)
        self.assertNotIn('delta_relative', frame)

    def test_overlap_and_missing_reference_are_rejected(self):
        cohort = pd.DataFrame(dict(organoid_str=list('abc'), n_cells=[1,2,3], surface_area=[2,3,4]))
        membership = pd.DataFrame(dict(fold=[0]*4, role=['train']*3+['val'], organoid_str=list('abca')))
        with self.assertRaisesRegex(ValueError, 'overlap'):
            fit_area_references(cohort, membership)
        refs = pd.DataFrame(dict(fold=[0], alpha=[0.], beta=[1.]))
        with self.assertRaisesRegex(ValueError, 'Missing'):
            normalize_cases(pd.DataFrame(dict(fold=[1], evaluated_n=[10.], delta_mu=[1.])), refs)
