import unittest
import numpy as np
import torch
from torch_geometric.data import Data, Batch
from src.models.gnn import SizeFiLMGINCurvature
from src.analysis.lgr5_film import infer_requests, mixed_difference, build_requests, hidden_comparison
from src.analysis.lgr5_film_summary import scale_statistics, cluster_summary
import pandas as pd


class Lgr5FiLMTests(unittest.TestCase):
    def test_common_scale_null_and_changed_marker_preference(self):
        curves=np.array([[1.,2.],[2.,4.],[4.,8.]])
        gain,deviation,mismatch=scale_statistics(curves,1)
        np.testing.assert_allclose(gain,[.5,1,2])
        np.testing.assert_allclose(deviation,0)
        np.testing.assert_allclose(mismatch,0)
        curves[2]=[8.,4.]
        gain,deviation,mismatch=scale_statistics(curves,1)
        self.assertGreater(mismatch[2],.5)
        gain,deviation,mismatch=scale_statistics(np.zeros((3,2)),1)
        self.assertTrue(np.isnan(mismatch).all())

    def test_bootstrap_summary_weights_organoids_equally(self):
        frame=pd.DataFrame(dict(group=['x']*11,organoid_str=['a']*10+['b'],value=[1.]*10+[5.]))
        summary=cluster_summary(frame,['group'],['value'],draws=30)
        self.assertEqual(summary.iloc[0]['mean'],3)
        self.assertEqual(summary.iloc[0].n_organoids,2)

    def test_instrumentation_matches_forward_and_separates_head(self):
        torch.manual_seed(1)
        model = SizeFiLMGINCurvature(3, hidden_dim=8, num_layers=2, global_dim=1, norm='batch', dropout=0).eval()
        for layer in model.film_layers:
            with torch.no_grad(): layer[-1].weight.fill_(0.1)
        g = Data(x=torch.tensor([[1.,1.,0.],[1.,0.,1.],[0.,0.,1.]]), y=torch.zeros(3),
                 edge_index=torch.tensor([[0,1,0,2],[1,0,2,0]]), center_idx=0, global_feat=torch.tensor([[0.3]]))
        req=[(0,()),(0,((1,0),))]
        result = infer_requests(model,[g],req,head_size=.3,film_sizes=[.3,.3],device='cpu')
        batch=Batch.from_data_list([g])
        with torch.no_grad(): output,_=model(batch.x,batch.edge_index,batch)
        np.testing.assert_allclose(result['z'][0],output[0][0].item(),atol=1e-6)
        different_head=infer_requests(model,[g],req,head_size=2.,film_sizes=[.3,.3],device='cpu')
        np.testing.assert_allclose(result['h1'],different_head['h1'],atol=1e-7)
        np.testing.assert_allclose(result['h2'],different_head['h2'],atol=1e-7)
        different_film=infer_requests(model,[g],req,head_size=.3,film_sizes=[1.,1.],device='cpu')
        self.assertFalse(np.allclose(result['h2'],different_film['h2']))
        self.assertEqual(g.x[1,0],1)
        with torch.no_grad(): after,_=model(batch.x,batch.edge_index,batch)
        torch.testing.assert_close(output[0],after[0])

    def test_factorial_contrast_and_zero_direction(self):
        indices=np.array([[0,1,2,3]])
        np.testing.assert_allclose(mixed_difference(np.array([10.,12.,13.,15.]),indices),0)
        np.testing.assert_allclose(mixed_difference(np.array([10.,12.,13.,19.]),indices),4)
        norm,ratio,cosine=hidden_comparison(np.array([[2.,0.],[0.,0.]]),np.array([[1.,0.],[0.,0.]]))
        self.assertEqual(ratio[0],2);self.assertEqual(cosine[0],1);self.assertTrue(np.isnan(cosine[1]))

    def test_neighbor_pairs_exclude_same_cell(self):
        g=Data(x=torch.ones(3,3),center_idx=0,orig_center=100)
        cases=[dict(subgraph_index=0,source_node=node,source_marker=m,source_marker_name=str(m),hop=1,
                    organoid_str='a',orig_center=100,orig_source_node=100+node,center_marker_names=['LGR5'])
               for m,node in [(0,1),(1,1),(2,2)]]
        requests,singles,n_single,meta,indices,skipped=build_requests([g],cases,1)
        self.assertEqual(skipped,1)
        self.assertEqual((meta.kind=='neighbor_neighbor').sum(),2)
        self.assertEqual((meta.kind=='center_neighbor').sum(),3)
        self.assertTrue(all(requests[i][1]==() for i in singles[:,0]))
