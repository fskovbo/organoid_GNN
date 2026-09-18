import unittest
import numpy as np
import pandas as pd
from src.analysis.spatial.neighborhoods import adjacency_rings, ring_average, neck_regions, within_organoid_contrast


class ObservedKI67Tests(unittest.TestCase):
    def test_exact_rings_exclude_center_first_ring_and_duplicate_edges(self):
        a,b=adjacency_rings([[0,1],[1,0],[1,2],[2,3],[1,1]],5)
        np.testing.assert_array_equal(a.toarray()[1],[1,0,1,0,0])
        np.testing.assert_array_equal(b.toarray()[1],[0,0,0,1,0])
        avg=ring_average(a,np.arange(5,dtype=float))
        self.assertEqual(avg[1],1)
        self.assertTrue(np.isnan(avg[4]))

    def test_missing_crypt_is_distinct_from_outside(self):
        np.testing.assert_array_equal(neck_regions(np.array([np.nan,.2,.8,1,1.2,1.3])),
            ['no_detected_crypt','crypt_body','neck_band','neck_band','neck_band','beyond_neck'])

    def test_contrasts_require_both_groups_and_keep_sign(self):
        frame=pd.DataFrame({'has_ki67_neighbor':[True]*3+[False]*3,'region':['crypt_body']*6})
        for col in ['curvature','curvature_norm','curvature_rank','negative','axis','ring1_fraction_AldoB','ring1_fraction_LGR5']:
            frame[col]=[1]*3+[4]*3
        result=within_organoid_contrast(frame,3)
        self.assertEqual(len(result),2)
        self.assertEqual(result[0]['curvature_norm'],-3)
        self.assertEqual(within_organoid_contrast(frame.iloc[:5],3),[])


if __name__=='__main__':unittest.main()
