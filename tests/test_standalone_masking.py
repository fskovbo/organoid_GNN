"""Standalone masking grid, model choices, saved data and downstream analysis."""
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch_geometric.data import Data, Batch
from src.models.gnn import GINCurvature, SizeFiLMGINCurvature
from src.models.fate_masking import with_missing_fate_input
from src.data.fate_masking import encode_fates
from src.artifacts.bundle import save_bundle, load_bundle
from src.artifacts.runs import AnalysisRun
from src.analysis.interventions.masking import load_mask_benchmarks
from notebook_workflows import workflow

ROOT=Path(__file__).resolve().parents[1]


class StandaloneMaskingTests(unittest.TestCase):
    def test_mask_input_preserves_predictions_and_roundtrips_narrow_widths(self):
        torch.set_num_threads(2)
        g=Data(x=torch.eye(2).repeat(3,1),edge_index=torch.tensor([[0,1,1,2,2,3],[1,0,2,1,3,2]]),
               global_feat=torch.tensor([[.1,.2]]))
        intact=Batch.from_data_list([g]);encoded=Batch.from_data_list([g.clone()]);encoded.x=encode_fates(encoded.x)
        with tempfile.TemporaryDirectory() as tmp:
            for cls in [GINCurvature,SizeFiLMGINCurvature]:
                for depth in [0,2]:
                    for width in [2,3,8]:
                        with self.subTest(model=cls.__name__,depth=depth,width=width):
                            torch.manual_seed(42)
                            base=cls(n_markers=2,hidden_dim=width,num_layers=depth,global_dim=2,norm='layer',dropout=.1).eval()
                            new=with_missing_fate_input(base).eval()
                            a,_=base(intact.x,intact.edge_index,data=intact)
                            b,_=new(encoded.x,encoded.edge_index,data=encoded)
                            torch.testing.assert_close(a[0],b[0],rtol=1e-5,atol=1e-6)
                            dest=Path(tmp)/f'{cls.__name__}_{depth}_{width}'
                            save_bundle(dest,{'model':new},splits={})
                            restored=load_bundle(dest)['model']
                            c,_=restored(encoded.x,encoded.edge_index,data=encoded)
                            torch.testing.assert_close(b[0],c[0],rtol=0,atol=0)

    def test_notebook_trains_both_families_depths_and_loads_for_analysis(self):
        torch.set_num_threads(2)
        graphs=[]
        for i in range(8):
            n=12+i; edge=torch.arange(n-1)
            graphs.append(Data(x=torch.eye(3)[torch.arange(n)%3,:2],y=torch.linspace(-.5,.5,n)+i*.03,
                edge_index=torch.stack([torch.cat([edge,edge+1]),torch.cat([edge+1,edge])]),
                organoid_str=f'org_{i}',meta=dict(timepoint='day4',total_surface_area=10.+i,total_volume=20.+i)))
        notebook=json.loads((ROOT/'experiments/training/fate_masking_training.ipynb').read_text())
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'masking'
            with contextlib.redirect_stdout(io.StringIO()), \
                 patch('src.data.io.load_graph_dataset_from_dir',return_value=copy.deepcopy(graphs)), \
                 patch('src.data.metadata.load_aux_metadata_for_dir',return_value={}), \
                 patch('src.data.metadata.load_marker_names_from_dir',return_value=['LGR5','KI67']), \
                 patch('src.artifacts.paths.resolve_training_run',side_effect=AssertionError('Loaded a reference run')), \
                 patch('src.artifacts.size_reference.size_reference',side_effect=AssertionError('Converted a reference run')), \
                 patch.object(plt,'show',side_effect=lambda:plt.close('all')):
                namespace={}
                for cell in notebook['cells']:
                    if cell['cell_type']!='code':continue
                    source=''.join(cell['source']);exec(compile(source,'standalone masking','exec'),namespace)
                    if 'SETTINGS = dict(' in source:
                        namespace.update(RUN_TRAINING=True,RUN_DIR=out,DEVICE='cpu',SHOW_PROGRESS=False)
                        namespace['SETTINGS'].update(tag='fixture',blacklist=False,interpolate_outliers=False,sphericity_max=None,
                            model_types=['gin','film'],depths=[0,2],hidden_dims=[3],seeds=[42],n_folds=2,
                            global_features=['log_num_cells','log_surface_area'],norm='layer',dropout=0.,
                            max_epochs=1,patience=1,batch_size=4,num_workers=0,rates=[0.,.1],edge_loss_weight=0.,
                            baseline_max_epochs=1,baseline_patience=1,residualize=True)
            run=AnalysisRun(out)
            self.assertEqual(len(run.records),16)
            self.assertFalse((out/'reference').exists())
            self.assertEqual(run.settings['model_types'],['gin','film'])
            self.assertEqual(set(run.records.depth),{0,2})
            self.assertEqual(set(run.records.rate),{0.,.1})
            for split in json.loads((out/'inner_splits.json').read_text()):
                self.assertFalse(set(split['fit']) & set(split['early_stopping']))
                self.assertFalse((set(split['fit']) | set(split['early_stopping'])) & set(split['validation']))
            rows=pd.read_csv(out/'training_progress.csv')
            self.assertEqual((rows.kind=='baseline').sum(),2)
            self.assertEqual((rows.kind=='model').sum(),16)
            for record in run.records.to_dict('records'):
                selected=run.select(record['key'])
                self.assertIs(type(selected['model']),GINCurvature if record['name']=='gin' else SizeFiLMGINCurvature)
                self.assertIsNotNone(selected['baseline'])
                self.assertEqual(selected['groups']['val'][0].x.shape[1],3)
                self.assertTrue((selected['groups']['val'][0].x[:,-1]==0).all())
                self.assertEqual(selected['model'].num_layers,record['depth'])
            # Independent benchmarks and all three interventions can consume the new bundles.
            benchmark=workflow('benchmarks/masking_quality_and_robustness.ipynb','benchmark_masking_experiment')
            report=workflow('benchmarks/masking_quality_and_robustness.ipynb','masking_report_directory')
            with contextlib.redirect_stdout(io.StringIO()):
                for name in ['gin','film']:
                    benchmark(ROOT,out,model_name=name,depth=2,hidden_dim=3,centers_per_identity=1,
                              quality_centers_per_organoid=2,batch_size=16,device='cpu')
                    results=load_mask_benchmarks(report(run,name,2,3))
                    self.assertEqual(set(results['intact'].mask_rate),{0.,.1})
                    self.assertEqual(set(results['masked'].mask_rate),{.1})
            from src.analysis.interventions.fate_edits import fate_graphs,sample_fate_contexts,evaluate_fate_edit
            selected=run.select(run.records.query("name == 'gin' and depth == 2 and rate > 0").iloc[0].key)
            subs,cases=sample_fate_contexts(fate_graphs(selected),selected['marker_names'],2,centers=1)
            masked=evaluate_fate_edit(selected,subs,cases,'masking',device='cpu')
            self.assertTrue(np.isfinite(masked.delta_mu).all())


if __name__=='__main__':unittest.main()
