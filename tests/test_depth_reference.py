"""Physical reference reconstruction and stable support for relative effects."""
import copy
import unittest
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from src.analysis.normalization.depth_reference import (
    predict_depth_reference, calibrate_depth_reference, add_depth_reference, validate_depth_reference,
)
from src.analysis.interventions.fate_edits import supported_effects


class CenterModel(torch.nn.Module):
    num_layers = 0
    def forward(self, x, edge_index, data):
        mu = x[:, 0] + data.global_feat[data.batch, 0]
        return (mu, torch.zeros_like(mu)), x


class Transform:
    def inverse(self, z):
        return np.sinh(z)


def reference():
    graphs = [Data(x=torch.tensor([[1.,0.],[0.,1.]]),
        edge_index=torch.tensor([[0,1],[1,0]]),organoid_str=org,
        global_feat=torch.tensor([[(np.log(2)-1)/2]],dtype=torch.float32)) for org in ['train','val']]
    return dict(model=CenterModel(),marker_names=['A','B'],global_features=['log_num_cells'],
        all_global_features=['unused','log_num_cells'],global_center=[99.,1.],global_scale=[99.,2.],
        groups=dict(train=graphs[:1],val=graphs[1:]),transform=Transform(),
        baseline_offsets={'train':np.array([2.,2.]),'val':np.array([-3.,-3.])})


class DepthReferenceTests(unittest.TestCase):
    def test_reconstructs_after_inverse_and_holds_baseline_fixed(self):
        ref = reference()
        requests = pd.DataFrame(dict(organoid_str=['val']*3,orig_center=[0,0,1],evaluated_n=[2,10,2]))
        result = predict_depth_reference(ref,requests)
        expected_z = np.array([1.,1.,0.]) + (np.log([2,10,2])-1)/2
        np.testing.assert_allclose(result.depth0_curvature,np.sinh(expected_z)-3,rtol=1e-6)
        np.testing.assert_array_equal(result.depth0_baseline,[-3.]*3)
        # A saved baseline model may exist without target residualization.
        ref['baseline_offsets']['val'] = np.zeros(2)
        ref['baseline_predictions'] = {'val':np.ones(2)*100}
        zero = predict_depth_reference(ref,requests)
        np.testing.assert_allclose(zero.depth0_curvature,np.sinh(expected_z),rtol=1e-6)

    def test_absolute_denominator_and_near_zero_exclusion(self):
        ref = reference()
        frame = pd.DataFrame(dict(organoid_str=['val']*2,orig_center=[0,1],evaluated_n=[2,2],delta_mu=[.5,-.5]))
        result = add_depth_reference(frame,ref,calibration={'threshold':.1})
        self.assertTrue((result.depth0_curvature<0).all())
        np.testing.assert_allclose(result.delta_depth0_relative,.5*np.array([1,-1])/result.depth0_curvature.abs())
        excluded = add_depth_reference(frame,ref,calibration={'threshold':10})
        self.assertTrue(excluded.delta_depth0_relative.isna().all())
        self.assertTrue(excluded.depth0_exclusion.eq('near_zero').all())

    def test_calibration_uses_training_only_and_validation_matches_folds(self):
        ref = reference()
        first = calibrate_depth_reference(ref)
        ref['baseline_offsets']['val'] += 10000
        self.assertEqual(first,calibrate_depth_reference(ref))
        validate_depth_reference(ref,ref)
        other = copy.deepcopy(ref)
        other['groups']['val'][0].x[0,0] = 0
        with self.assertRaisesRegex(ValueError,'fates or node ordering'):
            validate_depth_reference(ref,other)
        other = copy.deepcopy(ref)
        other['groups']['val'][0].organoid_str = 'wrong'
        with self.assertRaisesRegex(ValueError,'match the ablation fold'):
            validate_depth_reference(ref,other)

    def test_near_zero_sweep_exclusion_applies_at_every_n(self):
        frame = pd.DataFrame(dict(model_key=['m']*3,case_id=['c']*3,analysis=['observed','sweep','sweep'],
            supported=[True]*3,delta_mu=[1.]*3,delta_depth0_relative=[1.,1.,np.nan]))
        self.assertEqual(len(supported_effects(frame)),3)
        kept = supported_effects(frame,metric='delta_depth0_relative')
        self.assertEqual(kept.analysis.tolist(),['observed'])


if __name__ == '__main__':
    unittest.main()
