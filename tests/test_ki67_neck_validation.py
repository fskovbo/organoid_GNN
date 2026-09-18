import unittest
import numpy as np
from src.analysis.spatial.necks import classify_profile, qualified_assignment


class NeckValidationTests(unittest.TestCase):
    def setUp(self):self.s=np.linspace(.01,2,150)

    def test_trough_requires_two_sides(self):
        self.assertEqual(classify_profile(self.s,1+2*(self.s-1)**2)['profile_class'],'local_minimum')
        self.assertEqual(classify_profile(self.s,.2+self.s)['profile_class'],'no_neck_support')

    def test_plateau_and_not_flat_dome(self):
        c=np.where(self.s<.8,1+.5*(self.s-.8),np.where(self.s>1.2,1+2*(self.s-1.2),1.))
        self.assertEqual(classify_profile(self.s,c)['profile_class'],'flat_section')
        self.assertEqual(classify_profile(self.s,1-.3*(self.s-1)**2)['profile_class'],'no_neck_support')

    def test_invalid_profile_is_unresolved(self):
        c=np.ones_like(self.s);c[70]=np.nan
        self.assertEqual(classify_profile(self.s,c)['profile_class'],'unresolved')

    def test_nonqualifying_nearest_not_reassigned(self):
        ids,axis=qualified_assignment(np.array([[.9,1.4],[1.1,.9]]),[False,True])
        np.testing.assert_array_equal(ids,[0,1])
        self.assertTrue(np.isnan(axis[0]));self.assertEqual(axis[1],.9)


if __name__=='__main__':unittest.main()
