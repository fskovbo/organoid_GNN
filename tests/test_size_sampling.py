"""Coverage across original sizes, unique interventions and balanced summaries."""
import copy
import unittest
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from src.analysis.interventions.fate_edits import sample_fate_contexts, summarize_fate_effects
from src.analysis.interventions.size_sweeps import summarize_by_organoid


def cohort():
    graphs = []
    for i,n in enumerate([12,13,14,15,36,37,38,39]):
        nodes = torch.arange(n)
        edge = torch.stack([nodes, (nodes+1)%n])
        graphs.append(Data(x=torch.eye(3)[nodes%3,:2],y=torch.zeros(n),
            edge_index=torch.cat([edge,edge.flip(0)],dim=1),organoid_str=f'org_{i}',
            full_num_cells=n,global_feat=torch.tensor([[float(np.log(n))]])))
    return graphs


class SizeSamplingTests(unittest.TestCase):
    def test_pair_bin_quotas_unique_centers_organoid_cap_and_seed(self):
        graphs=cohort()
        before=copy.deepcopy(graphs)
        settings=dict(scheme='size_stratified',size_bins=[0,20,np.inf],
            cases_per_pair_bin=4,max_cases_per_organoid=1,seed=7,return_info=True)
        subs,cases,info=sample_fate_contexts(graphs,['A','B'],2,**settings)
        self.assertFalse(cases.duplicated(['organoid_str','orig_center','source_marker_name','hop']).any())
        centers=[(g.organoid_str,int(g.orig_center)) for g in subs]
        self.assertEqual(len(centers),len(set(centers)))
        self.assertEqual(set(cases.subgraph_index),set(range(len(subs))))
        expanded=info['case_non_overlap']
        selected=expanded[expanded.selected]
        self.assertEqual(set(selected.subgraph_index.astype(int)),set(cases.subgraph_index))
        keys=['size_bin','hop','center_marker','source_marker_name']
        self.assertTrue((selected.groupby(keys).size()<=4).all())
        self.assertTrue((selected.groupby(keys+['organoid_str']).size()<=1).all())
        ab=info['size_pair_coverage'].query("center_marker == 'A' and source_marker_name == 'B'")
        self.assertEqual(len(ab),4)
        self.assertTrue((ab.selected_count==4).all())
        self.assertTrue((ab.selected_organoids==4).all())
        # Sampling ignores graph order and target values.
        for g in graphs:g.y.fill_(999.)
        _,again,_=sample_fate_contexts(graphs[::-1],['A','B'],2,**settings)
        pd.testing.assert_frame_equal(cases,again)
        for a,b in zip(graphs,before):
            torch.testing.assert_close(a.x,b.x)
            torch.testing.assert_close(a.edge_index,b.edge_index)
        for row in cases.itertuples():
            self.assertEqual(subs[row.subgraph_index].organoid_str,row.organoid_str)
            self.assertEqual(int(subs[row.subgraph_index].orig_nodes[row.source_node]),row.orig_source_node)

    def test_unavailable_pairs_are_not_filled_from_other_sizes(self):
        graphs=cohort()
        for g in graphs[:4]:g.x[:,1]=0
        _,cases,info=sample_fate_contexts(graphs,['A','B'],2,scheme='size_stratified',
            size_bins=[0,20,np.inf],cases_per_pair_bin=10,max_cases_per_organoid=1,return_info=True)
        audit=info['size_pair_coverage']
        missing=audit.query("size_bin == 0 and source_marker_name == 'B'")
        self.assertTrue((missing.available_cases==0).all())
        self.assertTrue((missing.selected_count==0).all())
        self.assertTrue((missing.shortfall==10).all())
        self.assertFalse(((cases.observed_n<20)&cases.source_marker_name.eq('B')).any())
        with self.assertRaisesRegex(ValueError,'cover every'):
            sample_fate_contexts(graphs,['A','B'],2,scheme='size_stratified',size_bins=[20,np.inf])

    def test_equal_size_bins_are_not_weighted_by_organoid_abundance(self):
        frame=pd.DataFrame(dict(pair=['A']*10,organoid_str=[f'o{i}' for i in range(10)],
            size_bin=[0]*8+[1]*2,effect=[0.]*8+[10.]*2))
        ordinary=summarize_by_organoid(frame,['pair'],'effect',bootstrap_samples=20)
        balanced=summarize_by_organoid(frame,['pair'],'effect',bootstrap_samples=20,strata_column='size_bin')
        self.assertEqual(ordinary['mean'].item(),2.)
        self.assertEqual(balanced['mean'].item(),5.)
        self.assertEqual(balanced.ci_low.item(),5.)
        self.assertEqual(balanced.ci_high.item(),5.)
        # Multiple cases or fitted seeds for the same organoid do not add weight.
        repeated=pd.concat([frame,frame.iloc[:8]],ignore_index=True)
        check=summarize_by_organoid(repeated,['pair'],'effect',bootstrap_samples=20,strata_column='size_bin')
        self.assertEqual(check['mean'].item(),5.)

    def test_sweep_bin_support_and_observed_means_are_explicit(self):
        rows=[]
        for org in range(10):
            for mode,n in [('observed',10 if org<8 else 30),('sweep',80),('sweep',700)]:
                rows.append(dict(organoid_str=f'o{org}',observed_n=10 if org<8 else 30,
                    analysis=mode,evaluated_n=n,delta_mu=0. if org<8 else 10.,supported=True,
                    center_marker='A',source_marker_name='B',hop=1,orig_center=0,
                    orig_source_node=1,case_id=f'o{org}',model_key='model'))
        cases=pd.DataFrame(rows)
        summary=summarize_fate_effects(cases,view='size',size_bins=[0,20,40,np.inf],draws=20,
            sweep_weighting='equal_size_bins',sweep_min_organoids_per_bin=2)
        sweep=summary.query("analysis == 'sweep'")
        self.assertTrue((sweep['mean']==5.).all())
        self.assertTrue((sweep.n_cases==10).all())
        self.assertTrue((sweep.n_size_bins==2).all())
        self.assertEqual(set(sweep.size_bins_used),{'[0, 1]'})
        self.assertEqual(summary.query("analysis == 'observed'")['mean'].tolist(),[0.,10.])
        sparse=summarize_fate_effects(cases,view='size',size_bins=[0,20,40,np.inf],draws=20,
            sweep_weighting='equal_size_bins',sweep_min_organoids_per_bin=3).query("analysis == 'sweep'")
        self.assertTrue((sparse['mean']==0.).all())
        self.assertTrue((sparse.n_cases==8).all())
        self.assertEqual(set(sparse.size_bins_used),{'[0]'})
        no_sweep=summarize_fate_effects(cases,view='size',size_bins=[0,20,40,np.inf],draws=20,
            sweep_weighting='equal_size_bins',sweep_min_organoids_per_bin=20)
        self.assertEqual(set(no_sweep.analysis),{'observed'})


if __name__=='__main__':unittest.main()
