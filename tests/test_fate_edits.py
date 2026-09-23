"""Intervention semantics, all-depth coverage and common-support safeguards."""
import copy
import unittest
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch_geometric.data import Data
from src.data.fate_masking import encode_fates
from src.data.target_transforms import IdentityTransform
from src.analysis.interventions.fate_edits import (
    fate_graphs,sample_fate_contexts,evaluate_fate_edit,align_method_cases,supported_effects,
    build_replacement_reference,replacement_distribution,add_geometric_scale,
    expand_recipients,summarize_fate_effects,
)
from src.analysis.interventions.replacement import describe_graph,MatchConfig
from src.graph.neighborhood import compute_hop_rings
from src.data.subgraph_sampling import sample_subgraphs_coverage, flag_non_overlapping_source_cases
from src.data.subgraphs import build_ego_subgraphs_for_graph
from src.plotting.influence_maps import plot_pair_heatmaps, plot_pair_size_curves

class AdditiveFlagModel(nn.Module):
    def forward(self,x,edge_index,data=None):
        signal=x[:,0]+3*x[:,1]+5*x[:,2]
        output=signal.clone();output.index_add_(0,edge_index[1],signal[edge_index[0]])
        return (output,torch.zeros_like(output)),x


def selection():
    x=torch.tensor([[1.,0.],[0.,1.],[0.,0.]]*4)
    a=torch.arange(len(x)-1)
    edges=torch.stack([torch.cat([a,a+1]),torch.cat([a+1,a])])
    def graph(name,n=12):
        return Data(x=encode_fates(x[:n]),edge_index=edges[:,(edges<n).all(0)],y=torch.zeros(n),
                    organoid_str=name,global_feat=torch.tensor([[17.,np.log(n)]],dtype=torch.float32))
    groups={'val':[graph('validation')],'train':[graph(f'train_{i}',n) for i,n in enumerate([8,10,12])]}
    raw={g.organoid_str:Data(meta={'total_surface_area':float(len(g.x)**2)}) for gs in groups.values() for g in gs}
    return dict(model=AdditiveFlagModel(),marker_names=['A','B'],groups=groups,
        record={'rate':.05},transform=IdentityTransform(),global_features=['log_volume','log_num_cells'],
        all_global_features=['log_volume','log_num_cells'],global_center=np.zeros(2),global_scale=np.ones(2),raw_graphs=raw)


class FateEditTests(unittest.TestCase):
    def test_total_heatmaps_keep_method_values_and_exclude_sweep_rows(self):
        pack=selection()
        subs,cases=sample_fate_contexts(fate_graphs(pack),['A','B'],2,centers=99)
        tables={}
        for method in ['marker_zeroing','masking']:
            observed=evaluate_fate_edit(pack,subs,cases,method).assign(fold=0,seed=1,model_key=method)
            sweep=observed.assign(analysis='sweep',evaluated_n=300,delta_mu=999.)
            tables[method]=pd.concat([observed,sweep],ignore_index=True)
        aligned=align_method_cases(tables)
        arrays=[]
        for method,frame in aligned.items():
            total=summarize_fate_effects(frame,view='total',draws=5)
            # A at the center, B as direct neighbor: zeroing=-3, masking=+2.
            expected=-3. if method=='marker_zeroing' else 2.
            mean=total.query("hop == 1 and center_marker == 'A' and source_marker_name == 'B'")['mean'].item()
            self.assertEqual(mean,expected)
            fig=plot_pair_heatmaps(total,hops=[1],marker_names=['A','B'],limit=5.)
            values=np.asarray(fig.axes[0].collections[0].get_array()).reshape(2,2)
            self.assertEqual(values[0,1],expected)
            arrays.append(values)
            plt.close(fig)
        self.assertFalse(np.array_equal(*arrays))

    def test_historical_coverage_selection_and_non_overlap(self):
        pack = selection()
        graphs = fate_graphs(pack)
        original = graphs[0].x.clone()
        pool = build_ego_subgraphs_for_graph(graphs[0], num_hops=3, graph_idx=0)
        proxies = []
        for sub in pool:
            proxy = copy.copy(sub)
            proxy.x = torch.cat([sub.x, (sub.x.sum(1)==0).float()[:, None]], 1)
            proxies.append(proxy)
        _, old_info = sample_subgraphs_coverage(proxies, ['A','B','Unassigned'], 3, 8,
            min_center_count=2, min_pair_count=2, seed=12)
        subs, raw, info = sample_fate_contexts(graphs, ['A','B'], 3, scheme='coverage',
            max_subgraphs=8, min_center_count=2, min_pair_count=2, seed=12, return_info=True)
        self.assertEqual([int(g.orig_center) for g in subs],
                         [int(pool[i].orig_center) for i in old_info['selected_indices']])
        np.testing.assert_array_equal(info['pair_coverage'].coverage_selected_count.to_numpy(),
                                      old_info['pair_covered'].reshape(-1))
        # Single-source choice matches the old first-positive-in-ring convention.
        for row in raw.itertuples():
            sub = subs[row.subgraph_index]
            ring = compute_hop_rings(sub.edge_index, int(sub.center_idx), 3)[row.hop]
            candidates = [node for node in ring if (sub.x[node, row.source_marker] > .5
                if row.source_marker < 2 else sub.x[node].sum()==0)]
            self.assertEqual(row.source_node, candidates[0])
        _, filtered, audit = sample_fate_contexts(graphs, ['A','B'], 3, scheme='coverage',
            max_subgraphs=8, min_center_count=2, min_pair_count=2, seed=12,
            apply_non_overlap=True, return_info=True)
        legacy = flag_non_overlapping_source_cases(expand_recipients(raw), graphs=graphs, seed=12)
        expected = legacy.loc[legacy.passes_non_overlap, ['case_id','center_marker']]
        actual = expand_recipients(filtered)[['case_id','center_marker']]
        self.assertEqual(set(map(tuple, actual.values)), set(map(tuple, expected.values)))
        self.assertGreater(audit['pair_coverage'].non_overlap_excluded_count.sum(), 0)
        torch.testing.assert_close(graphs[0].x, original)
        self.assertTrue(all(g.x.shape[1] == 2 for g in subs))
        # Large budgets still report target shortfalls, including unavailable pairs.
        _, _, census = sample_fate_contexts(graphs, ['A','B'], 3, scheme='coverage',
            max_subgraphs=None, min_pair_count=100, return_info=True)
        self.assertFalse(census['pair_coverage'].target_met.any())
        self.assertTrue(census['pair_coverage'].global_available.notna().all())

    def test_case_counts_do_not_multiply_with_model_seeds_and_heatmap_hatches(self):
        pack=selection()
        subs,cases=sample_fate_contexts(fate_graphs(pack), ['A','B'], 2, centers=99)
        one=evaluate_fate_edit(pack,subs,cases,'marker_zeroing').assign(fold=0,seed=1,model_key='a')
        repeated=pd.concat([one,one.assign(seed=2,model_key='b')],ignore_index=True)
        a=summarize_fate_effects(one,draws=5)
        b=summarize_fate_effects(repeated,draws=5)
        np.testing.assert_array_equal(a.n_cases,b.n_cases)
        np.testing.assert_array_equal(2*a.n_rows,b.n_rows)
        # Test below, exactly at, and above the threshold; absent pairs are hatched.
        table=pd.DataFrame(dict(hop=[1,1,1],center_marker=['A','A','B'],
            source_marker_name=['A','B','A'],mean=[.1,.2,.3],n_cases=[19,20,21],n_organoids=[3,3,3]))
        fig=plot_pair_heatmaps(table,min_cases={1:20},hops=[1,2],marker_names=['A','B'])
        axes=[ax for ax in fig.axes if ax.get_title().startswith('Hop')]
        self.assertEqual(len(axes[0].patches),2)
        self.assertEqual(len(axes[1].patches),4)
        self.assertTrue(all(p.get_hatch()=='////' for ax in axes for p in ax.patches))
        plt.close(fig)
        curve=pd.DataFrame(dict(hop=[1]*3,center_marker=['A']*3,source_marker_name=['B']*3,
            analysis=['sweep']*3,N=[80,300,700],mean=[1.,2.,3.],ci_low=[0.,1.,2.],
            ci_high=[2.,3.,4.],n_cases=[20,19,20],n_organoids=[3]*3))
        fig=plot_pair_size_curves(curve,hop=1,min_cases=20)
        self.assertTrue(np.isnan(fig.axes[0].lines[0].get_ydata()[1]))
        plt.close(fig)

    def test_exact_hops_flags_alternatives_and_nonmutation(self):
        pack=selection();original=copy.deepcopy(pack['groups'])
        subs,cases=sample_fate_contexts(fate_graphs(pack),['A','B'],4,centers=99)
        self.assertEqual(set(cases.hop),{1,2,3,4})
        self.assertTrue((cases.orig_center!=cases.orig_source_node).all())
        weights=np.ones((len(cases),3))*.5
        weights[np.arange(len(cases)),cases.source_marker.to_numpy(int)]=0
        strengths=np.array([1.,3.,0.])[cases.source_marker]
        active=(cases.hop==1).to_numpy()
        expected={'marker_zeroing':-strengths,'masking':5-strengths,
                  'replacement':weights@np.array([1.,3.,0.])-strengths}
        for method in expected:
            frame=evaluate_fate_edit(pack,subs,cases,method,weights=weights,device='cpu')
            np.testing.assert_allclose(frame.delta_mu,expected[method]*active,atol=1e-6)
            swept=evaluate_fate_edit(pack,subs,cases,method,count=40,weights=weights,device='cpu')
            np.testing.assert_allclose(swept.delta_mu,frame.delta_mu,atol=1e-6)
        for role in original:
            for a,b in zip(original[role],pack['groups'][role]):
                torch.testing.assert_close(a.x,b.x);torch.testing.assert_close(a.global_feat,b.global_feat)
        pack['record']['rate']=0.
        with self.assertRaisesRegex(ValueError,'positive'):
            evaluate_fate_edit(pack,subs,cases,'masking')
        with self.assertRaisesRegex(ValueError,'depth'):
            sample_fate_contexts(fate_graphs(pack),['A','B'],0)

    def test_higher_hop_replacement_context_and_training_only_donors(self):
        pack=selection();graph=fate_graphs(pack)[0]
        desc=describe_graph(graph,max_hops=4)
        labels=np.where(graph.x.sum(1).numpy()==0,2,graph.x.argmax(1).numpy())
        for node in range(len(graph.x)):
            rings=compute_hop_rings(graph.edge_index,node,4)
            for hop in range(1,5):
                expected=sum(1<<int(label) for label in set(labels[list(rings[hop])]))
                self.assertEqual(desc.iloc[node][f'reach_{hop}'],expected)
        subs,cases=sample_fate_contexts([graph],['A','B'],4,centers=99)
        config=MatchConfig(min_cells=1,min_organoids=1,min_identity_cells=1,min_identity_organoids=1,
                           max_composition_l1=2.,neighbors=128)
        matcher,contexts=build_replacement_reference(pack,cases,4,config=config)
        self.assertNotIn('validation',set(matcher.donors.organoid_str))
        weights,audit=replacement_distribution(cases,contexts,matcher,['A','B'])
        self.assertTrue(audit.supported.any())
        np.testing.assert_allclose(weights[audit.supported].sum(1),1)
        normalized,reference=add_geometric_scale(pd.DataFrame({'evaluated_n':[20.],'delta_mu':[2.]}),pack)
        np.testing.assert_allclose(normalized.delta_relative,2*20**2/(4*np.pi))

    def test_common_support_stays_constant_along_sweep(self):
        pack=selection();subs,cases=sample_fate_contexts(fate_graphs(pack),['A','B'],2,centers=2)
        parts=[evaluate_fate_edit(pack,subs,cases,'marker_zeroing',count=n).assign(fold=0,seed=1,model_key='m') for n in [None,12,24]]
        a=pd.concat(parts,ignore_index=True);b=a.copy()
        selected=b.case_id.iloc[0]
        b.loc[(b.case_id==selected)&(b.evaluated_n==24),'supported']=False
        aligned=align_method_cases({'zero':a,'replacement':b})
        for table in aligned.values():
            self.assertFalse(((table.case_id==selected)&(table.analysis=='sweep')).any())
            self.assertTrue(((table.case_id==selected)&(table.analysis=='observed')).any())
        altered=a.copy();altered['graph_signature']='different_fate_encoding'
        with self.assertRaisesRegex(ValueError,'common'):
            align_method_cases({'original':a,'altered':altered})

if __name__=='__main__':unittest.main()
