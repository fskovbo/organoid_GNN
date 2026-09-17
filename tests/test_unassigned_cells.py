import unittest
import numpy as np
import pandas as pd
from src.analysis.unassigned_cells import majority_context, neighbor_contrasts, UnassignedConfig
from src.analysis.ki67_observed import MARKERS


class UnassignedTests(unittest.TestCase):
    def test_context_requires_strict_majority_and_retains_zero_ring(self):
        x=np.zeros((4,len(MARKERS)));x[0,0]=.6;x[0,1]=.4;x[1,:2]=.5;x[2,-1]=1;x[3]=np.nan
        f=pd.DataFrame(x,columns=[f'ring1_fraction_{m}' for m in MARKERS])
        np.testing.assert_array_equal(majority_context(f),['Agr2','Mixed','Unmarked','Isolated'])

    def test_paired_contrast_degree_matching_and_eligibility(self):
        rows=[]
        for degree,count,effect in [(3,3,-2),(4,6,1)]:
            for exposed in [False,True]:
                for _ in range(count):
                    r=dict(organoid_str='a',marker='LGR5',size_bin=2,stratum_time='day4',y=effect if exposed else 0)
                    for hop in [1,2]:r[f'ring{hop}_degree']=degree;r[f'ring{hop}_fraction_Unmarked']=1/degree if exposed else 0
                    rows.append(r)
        f=pd.DataFrame(rows)
        result=neighbor_contrasts(f,['y'],UnassignedConfig(),True)
        self.assertEqual(len(result),2)
        np.testing.assert_allclose(result.y,0)
        # Groups below the required count cannot contribute a contrast.
        f=f[f.ring1_degree==3].iloc[:5]
        self.assertTrue(neighbor_contrasts(f,['y'],UnassignedConfig(),True).empty)


if __name__=='__main__':unittest.main()
