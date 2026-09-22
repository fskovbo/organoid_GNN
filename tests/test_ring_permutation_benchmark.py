"""Paired ring-shuffle inference on the actual notebook workflow."""
import copy
import unittest
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from notebook_workflows import workflow
from src.models.gnn import GINCurvature
from src.data.target_transforms import AsinhStandardizeTransform
from src.analysis.metrics.evaluation import prediction_table
from src.graph.neighborhood import compute_hop_rings


class RingPermutationBenchmarkTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.evaluate = workflow('benchmarks/graph_signal_controls.ipynb', 'ring_permutation_node_errors')
        edges = torch.tensor([[0,0,1,1,2,2,3,4,5,6,7],[1,2,3,4,5,6,7,7,8,8,8]])
        edges = torch.cat([edges,edges.flip(0)],dim=1)
        self.g = Data(x=torch.eye(3).repeat(3,1), y=torch.linspace(-.8,.9,9), edge_index=edges,
                      organoid_str='fixture', global_feat=torch.tensor([[.3]]))
        self.transform = AsinhStandardizeTransform().fit([self.g])
        self.transform.transform_graphs([self.g],in_place=True)
        self.annotations = {'fixture':pd.DataFrame(dict(node=np.arange(9),
            region=['crypt']*3+['neck']*3+['villus']*3))}

    def selection(self, depth):
        torch.manual_seed(101)
        return dict(model=GINCurvature(3,hidden_dim=8,num_layers=depth,norm='batch',global_dim=1,dropout=.2),
                    groups={'val':[self.g]}, transform=self.transform,
                    baseline_offsets={'fixture':np.linspace(.1,.3,9)},
                    baseline_predictions={'fixture':np.full(9,.2)})

    def test_invariance_and_physical_errors_on_matched_centers(self):
        original=self.g.clone()
        for depth in (0,1,2):
            selected=self.selection(depth)
            scores,checks=self.evaluate(selected,self.annotations,radius=2,repeats=2,batch_size=3)
            self.assertEqual(len(scores),27)
            self.assertEqual(set(scores.region),{'crypt','neck','villus'})
            self.assertEqual(checks.centers.tolist(),[9])
            self.assertLess(checks.max_abs_intact_difference.max(),2e-6)
            reference=prediction_table(selected).set_index('node')
            intact=scores[scores.condition=='intact'].set_index('node')
            np.testing.assert_allclose(intact.y_true, reference.y_true, rtol=0,atol=0)
            np.testing.assert_allclose(intact.squared_error, reference.squared_error,rtol=1e-5,atol=2e-6)
            np.testing.assert_allclose(scores.squared_error,(scores.y_pred-scores.y_true)**2)
            for repeat in (0,1):
                edited=scores[(scores.condition=='ring_permuted') & (scores.repeat==repeat)].set_index('node')
                if depth < 2:
                    np.testing.assert_allclose(edited.y_pred,intact.y_pred,rtol=1e-5,atol=2e-6)
            for attr in ['x','y','edge_index','global_feat']:
                torch.testing.assert_close(getattr(self.g,attr),getattr(original,attr),rtol=0,atol=0)

    def test_permutation_content_and_repeatability_across_batch_sizes(self):
        selected=self.selection(2)
        captured=[]
        original_permute=self.evaluate.__globals__['permute_ego_neighborhood_within_rings']
        def checked(subs,radius,**kwargs):
            edited=original_permute(subs,radius,**kwargs)
            for before,after in zip(subs,edited):
                c=int(before.center_idx)
                torch.testing.assert_close(before.x[c],after.x[c],rtol=0,atol=0)
                for nodes in compute_hop_rings(before.edge_index,c,radius):
                    self.assertEqual(sorted(map(tuple,before.x[nodes].tolist())),sorted(map(tuple,after.x[nodes].tolist())))
                for attr in ['edge_index','y','global_feat']:
                    torch.testing.assert_close(getattr(before,attr),getattr(after,attr),rtol=0,atol=0)
                captured.append((int(before.orig_center),after.x.clone()))
            return edited
        self.evaluate.__globals__['permute_ego_neighborhood_within_rings']=checked
        a,_=self.evaluate(selected,self.annotations,radius=2,centers_per_organoid=5,repeats=2,batch_size=2)
        first=captured.copy();captured.clear()
        b,_=self.evaluate(selected,self.annotations,radius=2,centers_per_organoid=5,repeats=2,batch_size=4)
        order=['condition','repeat','node']
        pd.testing.assert_frame_equal(a.sort_values(order).reset_index(drop=True),
            b.sort_values(order).reset_index(drop=True),check_exact=False,rtol=1e-5,atol=2e-6)
        for center in set(c for c,_ in first):
            before=[x for c,x in first if c==center]
            after=[x for c,x in captured if c==center]
            for x,y in zip(before,after):torch.testing.assert_close(x,y,rtol=0,atol=0)


if __name__=='__main__':unittest.main()
