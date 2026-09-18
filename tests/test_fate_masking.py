import contextlib
import io
import json
from pathlib import Path
import pickle
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch_geometric.data import Data, Batch

from src.analysis.interventions.masking import (
    MaskTrainingConfig, encode_fates, random_mask, make_mask_model, ObservedFateAdapter,
    inner_split, train_mask_model, evaluate_single_mask, score_arrays,
    load_mask_benchmarks,
    make_quality_cases, _forward_with_mask,
)
from src.analysis.size_conditioning.cohort_inputs import make_model
from src.analysis.interventions.replacement import make_cases, MatchConfig
from src.analysis.interventions.masking_comparison import reconstruct_cases, load_masking_comparison
from src.data.target_transforms import IdentityTransform
from src.artifacts.runs import AnalysisRun

from notebook_workflows import workflow
train_masking_experiment = workflow('training/fate_masking_training.ipynb', 'train_masking_experiment')
benchmark_masking_experiment = workflow('benchmarks/masking_quality_and_robustness.ipynb', 'benchmark_masking_experiment')
run_replacement_analysis = workflow('ablation/fate_interventions.ipynb', 'run_replacement_analysis')
run_masking_ablation = workflow('ablation/fate_interventions.ipynb', 'run_masking_ablation')


SETTINGS=dict(HIDDEN_DIM=8, NUM_LAYERS=2, DROPOUT=0., RESIDUAL=True, NORM='batch',
    FILM_HIDDEN_DIM=4, LR=.003, WEIGHT_DECAY=0., EDGE_LOSS_WEIGHT=.1,
    EDGE_LOSS_PARAMS={'weighted':False}, FEATURE_ENCODING='exclusive_ordered',
    MODEL_GLOBAL_FEATURES={'gin_film_size':['log_num_cells']}, MODEL_SEEDS=[42],
    DATASET_NAME='synthetic', INTERPOLATE_TARGET_OUTLIERS=False)


def graph(name='organoid_0'):
    x=torch.tensor([[1.,0.],[0.,1.],[0.,0.],[1.,0.],[0.,1.],[0.,0.]])
    edge=torch.tensor([[0,1,1,2,2,3,3,4,4,5,5,0],[1,0,2,1,3,2,4,3,5,4,0,5]])
    return Data(x=x,edge_index=edge,y=torch.arange(6,dtype=torch.float32)/10,
                organoid_str=name,global_feat=torch.tensor([[1.]]),full_num_cells=6.)


class FlagModel(nn.Module):
    def forward(self,x,edge_index,data=None):
        signal=x[:,0]+3*x[:,1]+5*x[:,2]
        out=signal.clone();out.index_add_(0,edge_index[1],signal[edge_index[0]])
        return (out,torch.zeros_like(out)),x


class DummyBaseline:
    def __init__(self): self.prediction_cache={}
    def transform_graphs(self,graphs,in_place=True): return graphs


class FateMaskingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(2)

    def test_encoding_hides_identity_but_distinguishes_unassigned(self):
        g=graph();original=g.x.clone()
        missing=torch.tensor([False,True,True,False,False,False])
        x=encode_fates(g.x,missing)
        np.testing.assert_array_equal(x[1],[0,0,1])
        np.testing.assert_array_equal(x[2],[0,0,1])
        np.testing.assert_array_equal(x[5],[0,0,0])
        torch.testing.assert_close(original,g.x)
        a=torch.Generator().manual_seed(99);b=torch.Generator().manual_seed(99)
        low=random_mask(10000,.01,a);high=random_mask(10000,.05,b)
        self.assertTrue((~low|high).all());self.assertGreater(int(high.sum()),int(low.sum()))

    def test_paired_initialization_matches_seven_channel_network(self):
        original=make_model(SETTINGS,['A','B'],seed=42).eval()
        new=make_mask_model(SETTINGS,['A','B'],seed=42).eval()
        batch=Batch.from_data_list([graph()])
        (old,_),h_old=original(batch.x,batch.edge_index,data=batch)
        (actual,_),h_new=new(encode_fates(batch.x),batch.edge_index,data=batch)
        torch.testing.assert_close(old,actual);torch.testing.assert_close(h_old,h_new)
        self.assertEqual(float(new.convs[0].nn[0].weight[:,-1].detach().abs().sum()),0.)
        self.assertEqual(float(new.input_proj.weight[:,-1].detach().abs().sum()),0.)

    def test_no_identity_leak_through_batch_and_fate_blind_quality_sampling(self):
        class Inspect(nn.Module):
            def forward(self,x,edge_index,data=None):
                torch.testing.assert_close(x,data.x)
                torch.testing.assert_close(x[1],torch.tensor([0.,0.,1.]))
                return (x[:,0],x[:,0]),x
        g=graph();batch=Batch.from_data_list([g]);original=batch.x.clone()
        mask=torch.zeros(len(g.x),dtype=torch.bool);mask[1]=True
        _forward_with_mask(Inspect(),batch,mask)
        torch.testing.assert_close(batch.x,original)
        _,a=make_quality_cases([g],['A','B'],centers_per_organoid=3,seed=3)
        changed=g.clone();changed.x=torch.roll(g.x,1,dims=0)
        _,b=make_quality_cases([changed],['A','B'],centers_per_organoid=3,seed=3)
        pd.testing.assert_frame_equal(a[['orig_center','orig_source_node','hop']],
                                      b[['orig_center','orig_source_node','hop']])

    def test_initialization_and_receptive_fields_across_depths(self):
        g = graph()
        for depth in (0, 1, 3, 4):
            for width in (2, 3, 8):
                with self.subTest(depth=depth, width=width):
                    settings = dict(SETTINGS, NUM_LAYERS=depth, HIDDEN_DIM=width, NORM='layer')
                    base = make_model(settings, ['A', 'B'], seed=42).eval()
                    masked = make_mask_model(settings, ['A', 'B'], seed=42).eval()
                    batch = Batch.from_data_list([g])
                    (expected, _), _ = base(batch.x, batch.edge_index, data=batch)
                    (actual, _), _ = masked(encode_fates(batch.x), batch.edge_index, data=batch)
                    torch.testing.assert_close(actual, expected)
                    subs, cases, _ = make_cases([g], ['A', 'B'], receptive_hops=max(2, depth))
                    frame = evaluate_single_mask(subs, cases, masked, IdentityTransform(),
                        size_center=0, size_scale=1, device='cpu')
                    np.testing.assert_allclose(frame.intact_z,
                        actual.detach().numpy()[frame.orig_center.to_numpy(int)], atol=1e-6)
                    if depth == 0:
                        np.testing.assert_allclose(frame.mask_delta_z, 0, atol=1e-7)

    def test_single_neighbor_masks_keep_center_and_graph_observed(self):
        g=graph();subs,cases,_=make_cases([g],['A','B'],centers_per_identity=1)
        original=[s.x.clone() for s in subs]
        frame=evaluate_single_mask(subs,cases,FlagModel(),IdentityTransform(),size_center=0,size_scale=1,
                                   batch_size=7,device='cpu')
        unassigned=frame[(frame.source_marker_name=='Unassigned')&(frame.hop==1)]
        self.assertGreater(len(unassigned),0)
        np.testing.assert_allclose(unassigned.zero_delta_mu,0)
        np.testing.assert_allclose(unassigned.mask_delta_mu,5)
        for g,x in zip(subs,original):torch.testing.assert_close(g.x,x)
        swept=evaluate_single_mask(subs,cases,FlagModel(),IdentityTransform(),size_center=0,size_scale=1,count=30)
        self.assertNotIn('truth_z',swept)
        self.assertNotIn('delta_mse_z',swept)

    def test_masks_train_and_validation_sets_are_disjoint(self):
        fit,stop=inner_split(10,.2,42)
        self.assertFalse(set(fit)&set(stop));self.assertEqual(set(fit)|set(stop),set(range(10)))
        graphs=[graph(f'organoid_{i}') for i in range(4)]
        x=[g.x.clone() for g in graphs];y=[g.y.clone() for g in graphs]
        config=MaskTrainingConfig(rates=(0.,.5),max_epochs=2,patience=2,batch_size=2)
        model=make_mask_model(SETTINGS,['A','B'],seed=42)
        with contextlib.redirect_stdout(io.StringIO()):
            model,history=train_mask_model(model,graphs[:3],graphs[3:],SETTINGS,config,rate=.5,seed=42,device='cpu')
        self.assertEqual(len(history),2)
        self.assertGreater(float(model.convs[0].nn[0].weight[:,-1].detach().abs().sum()),0)
        for g,xx,yy in zip(graphs,x,y):torch.testing.assert_close(g.x,xx);torch.testing.assert_close(g.y,yy)

    def test_probability_scores_and_reconstruction(self):
        scores=score_arrays(np.array([0,2]),np.zeros(2),np.zeros(2),IdentityTransform())
        np.testing.assert_array_equal(scores['coverage_95'],[1,0])
        np.testing.assert_array_equal(scores['mse_z'],[0,4])
        gs=[graph('organoid_a'),graph('organoid_b')]
        subs,cases,_=make_cases(gs,['A','B'])
        new,manifest=reconstruct_cases(list(reversed(gs)),cases)
        for c in manifest.to_dict('records'):
            sub=new[c['subgraph_index']]
            self.assertEqual(int(sub.orig_nodes[c['source_node']]),c['orig_source_node'])
            self.assertEqual(int(sub.orig_center),c['orig_center'])

    def test_end_to_end_training_benchmarks_and_replacement_extension(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);data=root/'training_data/synthetic';data.mkdir(parents=True)
            reference=root/'reference'
            for folder in ['tables','checkpoints','geometric_normalization']:(reference/folder).mkdir(parents=True)
            settings=dict(SETTINGS)
            (reference/'settings.json').write_text(json.dumps(settings))
            rows=[]
            for i in range(8):
                org=f'organoid_{i}';n=12
                labels=(np.arange(n)+i)%3;x=np.eye(3,dtype=np.float32)[labels,:2]
                edges=np.column_stack([np.arange(n),np.roll(np.arange(n),-1)])
                np.savez(data/f'{org}.npz',x=x,y=x[:,0]*.2-x[:,1]*.1,edges=edges)
                (data/f'{org}_markers.json').write_text(json.dumps(['A','B']))
                (data/f'{org}_aux.json').write_text(json.dumps(dict(total_surface_area=100+i,total_volume=80+i)))
                rows.append(dict(organoid_str=org,n_cells=n))
            pd.DataFrame(rows).to_csv(reference/'tables/cohort.csv',index=False)
            split=[dict(fold=0,train_indices=list(range(6)),val_indices=[6,7])]
            (reference/'splits.json').write_text(json.dumps(split));(reference/'sweep_grid.json').write_text('[12,24]')
            pd.DataFrame([dict(fold=0,alpha=0.,beta=1.)]).to_csv(reference/'geometric_normalization/references.csv',index=False)
            pre=dict(baseline=DummyBaseline(),residual_transform=IdentityTransform().fit_array(np.zeros((12,1))),
                     global_center=np.zeros(4),global_scale=np.ones(4),size_center=0.,size_scale=1.)
            with (reference/'checkpoints/fold_0_preprocessing.pkl').open('wb') as f:pickle.dump(pre,f)
            torch.save(make_model(settings,['A','B'],seed=42).state_dict(),reference/'checkpoints/fold_0_seed_42_gin_film_size.pt')
            out=root/'masking';old=root/'replacement';aug=root/'mask_ablation'
            config=MaskTrainingConfig(rates=(0.,.1),max_epochs=2,patience=2,batch_size=2)
            with contextlib.redirect_stdout(io.StringIO()):
                train_masking_experiment(root,reference,out,config=config,device='cpu')
                # Resume must reuse completed checkpoints without retraining.
                before=(out/'checkpoints/fold_0_seed_42_p0.1.pt').stat().st_mtime_ns
                train_masking_experiment(root,reference,out,config=config,device='cpu')
                self.assertEqual(before,(out/'checkpoints/fold_0_seed_42_p0.1.pt').stat().st_mtime_ns)
                benchmark_masking_experiment(root,out,centers_per_identity=1,batch_size=16,device='cpu')
                benchmark=load_mask_benchmarks(out)
                self.assertEqual(set(benchmark['intact'].mask_rate),{0.,.1})
                self.assertEqual(set(benchmark['masked'].mask_rate),{.1})
                run_replacement_analysis(root,reference,old,centers_per_identity=1,batch_size=16,device='cpu',
                    matcher_config=MatchConfig(neighbors=64,min_cells=1,min_organoids=1,min_identity_cells=1,
                        min_identity_organoids=1,max_composition_l1=2))
                run_masking_ablation(root,old,out,aug,rate=.1,batch_size=16,device='cpu')
                result=load_masking_comparison(aug,bootstrap_samples=5)
                # Override the reference architecture, then load it through the
                # same saved-run interface used by downstream analysis notebooks.
                custom = root / 'masking_depth3'
                overrides = dict(depth=3, hidden_dim=12, film_hidden_dim=6,
                                 dropout=.2, norm='layer', residual=False, lr=.001,
                                 weight_decay=.01, edge_loss_weight=0.)
                train_masking_experiment(root, reference, custom, config=config,
                    device='cpu', model_overrides=overrides)
                run = AnalysisRun(custom, root=root)
                selected = run.select(run.records.iloc[0].key)
                self.assertEqual(selected['model'].num_layers, 3)
                self.assertEqual(selected['model'].hidden_dim, 12)
                self.assertEqual(selected['model'].dropout, .2)
                self.assertEqual(selected['model'].norm, 'layer')
                self.assertFalse(selected['model'].residual)
                saved = json.loads((custom/'settings.json').read_text())
                self.assertEqual(saved['model_overrides'], overrides)
                self.assertEqual(json.loads((reference/'settings.json').read_text())['NUM_LAYERS'], 2)
                benchmark_masking_experiment(root, custom, centers_per_identity=1,
                                             batch_size=16, device='cpu')
                run_masking_ablation(root, old, custom, root/'mask_ablation_depth3',
                                    rate=.1, batch_size=16, device='cpu')
            self.assertIn('masking',result['summary'].method.unique())
            self.assertEqual(len(result['methods']),4)
            frame=result['cases'];u=frame[frame.source_marker_name=='Unassigned']
            np.testing.assert_allclose(u.zero_delta_mu,0,atol=1e-7)
            self.assertTrue((frame.fixed_supported==frame.dependent_supported).all())
            membership=json.loads((out/'fold_0_membership.json').read_text())
            self.assertFalse(set(membership['benchmark']) & set(membership['early_stopping']))
            # Exercise plots and summaries as well as all inference entry points.
            import matplotlib
            matplotlib.use('Agg')
            benchmark_tables = workflow('benchmarks/masking_quality_and_robustness.ipynb', 'masking_benchmark_tables')
            plot_intact_quality = workflow('benchmarks/masking_quality_and_robustness.ipynb', 'masking_plot_intact_quality')
            plot_masked_quality = workflow('benchmarks/masking_quality_and_robustness.ipynb', 'masking_plot_masked_quality')
            plot_training_robustness = workflow('benchmarks/masking_quality_and_robustness.ipynb', 'masking_plot_training_robustness')
            plot_effect_robustness = workflow('benchmarks/masking_quality_and_robustness.ipynb', 'masking_plot_effect_robustness')
            plot_pair_grid = workflow('ablation/fate_interventions.ipynb', 'replacement_plot_pair_grid')
            import matplotlib.pyplot as plt
            tables=benchmark_tables(benchmark,bootstrap_samples=5)
            figures=[plot_intact_quality(tables),plot_masked_quality(tables),
                     plot_training_robustness(benchmark,tables),plot_effect_robustness(benchmark,tables,min_organoids=1),
                     plot_pair_grid(result,min_organoids=1)]
            for i,fig in enumerate(figures):fig.savefig(root/f'figure_{i}.png');plt.close(fig)


if __name__=='__main__':unittest.main()
