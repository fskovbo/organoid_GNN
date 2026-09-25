"""Execute all three notebook workflows on a tiny saved reference, never real data."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from src.artifacts.bundle import save_bundle, load_bundle, graph_membership
from src.artifacts.runs import AnalysisRun
from src.data.target_transforms import IdentityTransform
from src.models.gnn import GINCurvature
from src.training.preparation import prepare_fold
from tests.test_fate_spline import synthetic_graphs

ROOT = Path(__file__).resolve().parents[1]


def cells(path):
    return [''.join(c['source']) for c in json.loads((ROOT/path).read_text())['cells'] if c['cell_type'] == 'code']


class SplineNotebookTests(unittest.TestCase):
    def test_reference_training_comparison_and_coefficient_figures(self):
        torch.set_num_threads(2)
        directory = Path(self.enterContext(tempfile.TemporaryDirectory()))
        source = directory/'reference'
        source.mkdir()
        graphs = synthetic_graphs()
        for g in graphs:
            n = len(g.x)
            g.meta = dict(timepoint='day4', total_surface_area=float(n*2), total_volume=float(n**1.5))
            g.y = .01*g.x[:,0] - .02*g.x[:,1] + np.log(n)*.005
        markers = ['LGR5','KI67']
        settings = dict(dataset='synthetic_missing_annotations', target_indices=[0], seeds=[42],
                        exclusive_markers=True, loss_name='gaussian', timepoints=['day4'],
                        sphericity_max=.95, blacklist=True, complexity_min=None,
                        interpolate_outliers=False, outlier_quantiles=[.005,.995],
                        n_folds=2, val_fraction=.5, split_seed=815,
                        baseline_features=['log_num_cells','log_surface_area','log_volume','log_volume_over_area'],
                        baseline_hidden_dim=4, baseline_max_epochs=1, baseline_patience=1,
                        batch_size=128, num_workers=0)
        (source/'settings.json').write_text(json.dumps(settings))
        membership = [dict(fold=f, **graph_membership(train=[graphs[i] for i in range(12) if i%2 != f],
                                                     validation=[graphs[i] for i in range(12) if i%2 == f])) for f in range(2)]
        (source/'splits.json').write_text(json.dumps(membership))
        save_bundle(source/'cohort',dict(raw_graphs={g.organoid_str:g for g in graphs},settings=settings,
                    all_marker_names=markers),splits=membership)
        records = []
        for split in membership:
            fold = split['fold']
            prepared = prepare_fold(graphs, dict(train_indices=[i for i in range(12) if i%2 != fold],
                val_indices=[i for i in range(12) if i%2 == fold]), global_features=['log_num_cells'],
                residualize=False, baseline_kwargs=dict(max_epochs=1,patience=1,hidden_dim=4,num_workers=0,
                                                       train_kwargs=dict(device='cpu')))
            inputs = f'inputs/fold_{fold}'
            save_bundle(source/inputs,dict(**prepared,marker_names=markers),splits=split)
            for depth in range(5):
                key = f'gin_d{depth}_f{fold}'
                record = dict(key=key,name='gin',depth=depth,fold=fold,seed=42,hidden_dim=8,
                              subset='all',signal='intact',bundle=f'models/{key}',input_bundle=inputs)
                model = GINCurvature(2,hidden_dim=8,num_layers=depth,global_dim=1,norm='layer',dropout=0.)
                save_bundle(source/record['bundle'],dict(model=model,record=record),splits=split)
                records.append(record)
        pd.DataFrame(records).to_json(source/'models.json',orient='records')
        # Dataset reuse is independent of all predictor checkpoints/architecture choices.
        reader = AnalysisRun(source)
        with patch('src.artifacts.bundle.load_bundle', wraps=load_bundle) as loader:
            restored_inputs = reader.fold_inputs(0)
        self.assertEqual({Path(call.args[0]) for call in loader.call_args_list},
                         {source/'cohort', source/'inputs/fold_0'})
        self.assertNotIn('model',restored_inputs)
        self.assertEqual(restored_inputs['source_input_bundle'],'inputs/fold_0')
        restored_inputs['groups']['train'][0].x.zero_()
        self.assertTrue(reader.fold_inputs(0)['groups']['train'][0].x.any())
        conflicting = dict(records[0], input_bundle='inputs/different_preprocessing')
        reader.records = pd.concat([reader.records,pd.DataFrame([conflicting])],ignore_index=True)
        with self.assertRaisesRegex(ValueError,'unambiguous'):
            reader.fold_inputs(0)
        code = cells('experiments/training/fate_spline_training.ipynb')
        ns = {}
        exec(code[0],ns); exec(code[1],ns)
        ns.update(REFERENCE_RUN=str(source),RUN_TRAINING=True,
                  RUN_DIR=directory/'trained',SHOW_PROGRESS=False,DEVICE='cpu',EVALUATE_REGIONS=True,
                  display=lambda *args:None)
        ns['SETTINGS'].update(radii=[1,2,3,4],n_splines=4,smoothness_grid=[.01],pair_ridge_grid=[.01],
                              inner_fraction=.3,blas_threads=1)
        original_select = AnalysisRun.select
        def forbid_reference_predictor(run, *args, **kwargs):
            if run.directory == source:
                raise AssertionError('Spline training must not select a reference predictor.')
            return original_select(run, *args, **kwargs)
        with patch('matplotlib.pyplot.show'), patch.object(AnalysisRun,'select',forbid_reference_predictor):
            for source_code in code[2:]:
                exec(source_code,ns)
        run = AnalysisRun(directory/'trained')
        self.assertEqual(len(run.records),18)
        self.assertEqual(run.settings['n_folds'],2)
        self.assertEqual(run.settings['reference_inputs'],{'0':'inputs/fold_0','1':'inputs/fold_1'})
        self.assertNotIn('reference_models',run.settings)
        for name in ('dataset','target_indices','timepoints','blacklist','sphericity_max','complexity_min',
                     'interpolate_outliers','outlier_quantiles','n_folds','val_fraction','split_seed',
                     'baseline_features','baseline_hidden_dim','baseline_max_epochs','baseline_patience',
                     'batch_size','num_workers'):
            self.assertEqual(run.settings[name],settings[name])
        self.assertTrue((directory/'trained'/'reference_artifacts.json').exists())
        saved = run.select(run.records.iloc[0].key)
        self.assertIsInstance(saved['transform'],IdentityTransform)
        self.assertTrue(saved['residualized'])
        self.assertEqual(json.loads((directory/'trained'/'splits.json').read_text()),membership)
        original = AnalysisRun(directory/'reference').select('gin_d0_f0')
        for g in saved['groups']['val']:
            old = next(x for x in original['groups']['val'] if x.organoid_str == g.organoid_str)
            physical = original['transform'].inverse(old.y.numpy())
            np.testing.assert_allclose(g.y.numpy()+saved['baseline_offsets'][g.organoid_str],physical,atol=1e-12)
        for k,v in original['baseline'].model.state_dict().items():
            torch.testing.assert_close(saved['baseline'].model.state_dict()[k],v,rtol=0,atol=0)
        code = cells('experiments/benchmarks/fate_spline_comparison.ipynb')
        compare = {}
        exec(code[0],compare); exec(code[1],compare)
        compare.update(SPLINE_RUN=str(directory/'trained'),REFERENCE_WIDTH=8,REFERENCE_NAME='gin',
                       REFERENCE_RATE=None,REFERENCE_DEPTHS=[2],RADII=[0,1,2,3,4],DEVICE='cpu',display=lambda *args:None)
        with patch('matplotlib.pyplot.show'):
            for source_code in code[2:]:
                exec(source_code,compare)
        self.assertTrue((directory/'trained'/'analysis'/'radius_comparison'/'mse_comparison.png').exists())
        self.assertEqual(set(compare['score_table'].source), {'Spline','Reference'})
        self.assertEqual(set(compare['score_table'].query("source == 'Reference'").depth), {2})
        self.assertEqual(set(compare['score_table'].query("source == 'Spline'").depth), {0,1,2,3,4})
        code = cells('experiments/neighborhoods/fate_spline_coefficients.ipynb')
        analysis = {}
        exec(code[0],analysis); exec(code[1],analysis)
        analysis.update(TRAINING_RUN=str(directory/'trained'),CENTER_MARKERS=['LGR5','KI67'],
                        N_POINTS=12,N_SUPPORT_BINS=3,display=lambda *args:None)
        with patch('matplotlib.pyplot.show'):
            for source_code in code[2:]:
                exec(source_code,analysis)
        self.assertEqual(len(list(analysis['OUTPUT_DIR'].glob('*.png'))),6)
        self.assertAlmostEqual(sum(analysis['values'].values()),analysis['prediction'])
        plt.close('all')

    def test_training_requires_reference_and_has_no_fresh_data_controls(self):
        code = cells('experiments/training/fate_spline_training.ipynb')
        namespace = {}
        exec(code[0],namespace)
        exec(code[1],namespace)
        forbidden = {'dataset','target_indices','timepoints','blacklist','sphericity_max','complexity_min',
                     'interpolate_outliers','outlier_quantiles','n_folds','val_fraction','split_seed',
                     'baseline_features','baseline_hidden_dim','baseline_max_epochs','baseline_patience',
                     'batch_size','num_workers'}
        self.assertFalse(forbidden & namespace['SETTINGS'].keys())
        self.assertNotIn('prepare_fold','\n'.join(code))
        self.assertNotIn('load_graph_dataset_from_dir','\n'.join(code))
        namespace['REFERENCE_RUN'] = None
        with patch.object(AnalysisRun,'__init__',side_effect=AssertionError('Must require a reference first')):
            with self.assertRaisesRegex(ValueError,'REFERENCE_RUN is required'):
                exec(code[2],namespace)

    def test_identity_preparation_preserves_physical_targets_and_baseline(self):
        torch.set_num_threads(2)
        graphs = synthetic_graphs()[:8]
        for graph in graphs:
            n=len(graph.x)
            graph.meta=dict(total_surface_area=float(n*2), total_volume=float(n**1.5))
            graph.y = torch.arange(n,dtype=torch.float32)*.001
        for residualize in (False,True):
            prepared=prepare_fold(graphs,dict(train_indices=list(range(6)),val_indices=[6,7]),
                global_features=[],residualize=residualize,target_scaling='identity',
                baseline_kwargs=dict(max_epochs=1,patience=1,hidden_dim=4,num_workers=0,train_kwargs=dict(device='cpu')))
            self.assertIsInstance(prepared['transform'],IdentityTransform)
            self.assertIsNotNone(prepared['baseline'].model)
            for graph in prepared['groups']['val']:
                original = next(g for g in graphs if g.organoid_str == graph.organoid_str)
                np.testing.assert_allclose(graph.y.numpy()+prepared['baseline_offsets'][graph.organoid_str],original.y,atol=1e-8)


if __name__ == '__main__':
    unittest.main()
