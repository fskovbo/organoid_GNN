"""Small end-to-end checks of the actual workhorse and independent analysis cells."""
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

from src.artifacts.runs import AnalysisRun
from src.analysis.metrics.evaluation import prediction_table
from src.training.preparation import prepare_fold

ROOT = Path(__file__).resolve().parents[1]


def notebook_sources(relative):
    notebook = json.loads((ROOT/'experiments'/relative).read_text())
    return [''.join(c['source']) for c in notebook['cells'] if c['cell_type']=='code']


class ConsolidatedWorkflowTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.out = Path(self.temp.name)/'run'
        self.graphs = []
        for i in range(8):
            n=12+i
            a=torch.arange(n-1)
            x=torch.zeros(n,2);x[::2,0]=1;x[1::2,1]=1
            self.graphs.append(Data(x=x,y=torch.linspace(-1.,1.,n)+i*.1,
                edge_index=torch.stack([torch.cat([a,a+1]),torch.cat([a+1,a])]),
                organoid_str=f'org_{i}',meta=dict(total_surface_area=10.+i,total_volume=20.+i,
                    timepoint='day4',dataset='fixture',label_uid=str(i))))

    def check_reference_controls(self, max_folds=None):
        """Run the actual control notebook, forbidding preprocessing/data refits."""
        from src.training.loop import train as actual_train
        controls_out = self.out.parent/f'controls_{max_folds}'
        expected_folds = [0, 1] if max_folds is None else [0]
        reference_count = 2 * len(expected_folds)
        namespace = {}
        with patch('src.training.preparation.prepare_fold', side_effect=AssertionError('Refitted preprocessing')), \
             patch('src.data.io.load_graph_dataset_from_dir', side_effect=AssertionError('Reloaded raw data')), \
             patch('src.models.ring_mlp._compute_ring_fraction_features_and_sizes',
                   side_effect=AssertionError('Rebuilt ring features during forward pass')), \
             patch('src.training.loop.train', wraps=actual_train) as training:
            for source in notebook_sources('training/graph_controls_training.ipynb'):
                source = source.replace(
                    "resolve_training_run(ROOT, CONTROL_SETTINGS['reference_notebook'], CONTROL_SETTINGS['reference_run'])",
                    f'Path({str(self.out)!r})')
                exec(compile(source,'controls','exec'),namespace)
                if 'CONTROL_SETTINGS = dict(' in source:
                    namespace.update(RUN_TRAINING=True, RUN_DIR=controls_out, DEVICE='cpu')
                    namespace['CONTROL_SETTINGS'].update(reference_family='film', max_folds=max_folds)
            self.assertEqual(training.call_count, reference_count * 8)
        control = AnalysisRun(controls_out)
        reference = AnalysisRun(self.out)
        progress_rows = pd.read_csv(controls_out/'training_progress.csv')
        self.assertEqual((progress_rows.kind == 'copied').sum(), reference_count)
        self.assertEqual((progress_rows.kind == 'model').sum(), reference_count * 8)
        self.assertTrue(progress_rows.status.eq('completed').all())
        self.assertEqual(len(control.records), reference_count * 9)
        source_splits = json.loads((self.out/'splits.json').read_text())
        self.assertEqual(json.loads((controls_out/'splits.json').read_text()),
                         [s for s in source_splits if s['fold'] in expected_folds])
        self.assertEqual(set(control.records.fold), set(expected_folds))
        self.assertEqual(control.settings['selected_folds'], expected_folds)
        self.assertEqual(control.settings['n_folds'], len(expected_folds))
        self.assertEqual(control.settings['reference_n_folds'], 2)
        from src.artifacts.runs import select_depth_records
        self.assertEqual(set(select_depth_records(control,2,model_name='film').fold), set(expected_folds))
        for setting in ['depths','hidden_dims','seeds','timepoints','residualize','global_features','max_epochs','lr']:
            self.assertEqual(control.settings[setting],reference.settings[setting])
        for row in control.records.to_dict('records'):
            restored = control.select(row['key'])
            original = reference.select(row['reference_key'])
            if row['reused_reference']:
                source_record = reference.records.set_index('key').loc[row['reference_key']]
                for source_file in (self.out/source_record.bundle).iterdir():
                    self.assertEqual(source_file.read_bytes(),
                                     (controls_out/row['bundle']/source_file.name).read_bytes())
                for name,weights in original['model'].state_dict().items():
                    torch.testing.assert_close(restored['model'].state_dict()[name],weights,rtol=0,atol=0)
            for name, weights in original['baseline'].model.state_dict().items():
                torch.testing.assert_close(restored['baseline'].model.state_dict()[name],weights,rtol=0,atol=0)
            for role in ['train','val']:
                self.assertEqual([g.organoid_str for g in restored['groups'][role]],
                                 [g.organoid_str for g in original['groups'][role]])
                for a,b in zip(restored['groups'][role],original['groups'][role]):
                    for attr in ['y','edge_index','global_feat']:
                        torch.testing.assert_close(getattr(a,attr),getattr(b,attr),rtol=0,atol=0)
                    if row['signal']=='intact':torch.testing.assert_close(a.x,b.x,rtol=0,atol=0)
                    elif row['signal']=='constant':torch.testing.assert_close(a.x,torch.ones_like(b.x))
                    else:self.assertEqual(sorted(map(tuple,a.x.tolist())),sorted(map(tuple,b.x.tolist())))
                    np.testing.assert_array_equal(restored['transform'].inverse(a.y),original['transform'].inverse(b.y))
                    np.testing.assert_array_equal(restored['baseline_offsets'][a.organoid_str],original['baseline_offsets'][a.organoid_str])
        self.assertTrue((controls_out/'reference.json').exists())
        self.assertTrue((controls_out/'regional_evaluation/validation_mse.csv').exists())
        # All model/data restoration works with the original reference unavailable.
        reference_location = self.out.with_name('hidden_reference')
        self.out.rename(reference_location)
        try:
            restored = AnalysisRun(controls_out).select(control.records.iloc[0].key)
            self.assertTrue(restored['groups']['val'])
        finally:
            reference_location.rename(self.out)

    def test_train_save_and_analyze_without_refitting(self):
        import src.data.io as data_io
        import src.data.metadata as metadata
        for graph in self.graphs[-2:]:
            graph.meta['timepoint'] = 'day5'
        with contextlib.redirect_stdout(io.StringIO()), \
             patch.object(data_io,'load_graph_dataset_from_dir',side_effect=lambda _:copy.deepcopy(self.graphs)), \
             patch.object(metadata,'load_aux_metadata_for_dir',return_value={}), \
             patch.object(metadata,'load_marker_names_from_dir',return_value=['LGR5','KI67']), \
             patch.object(plt,'show',side_effect=lambda:plt.close('all')):
            ns={}
            for source in notebook_sources('training/gin_depth_training.ipynb'):
                exec(compile(source,'workhorse','exec'),ns)
                if 'SETTINGS = dict(' in source:
                    ns['RUN_TRAINING']=True;ns['RUN_DIR']=self.out;ns['DEVICE']='cpu'
                    ns['SETTINGS'].update(timepoints=['day4'],blacklist=False,interpolate_outliers=False,exclusive_markers=True,
                        sphericity_max=None, complexity_min=None,
                        model_types=['gin','film'],global_features=['log_num_cells'],depths=[0,2],hidden_dims=[8],n_folds=2,seeds=[42],max_epochs=1,
                        norm='layer',dropout=0.,edge_loss_weight=0.,batch_size=8,num_workers=0,
                        residualize=True,baseline_max_epochs=1,baseline_patience=1)
            run=AnalysisRun(self.out)
            self.assertEqual(len(run.records),8)
            self.assertTrue(run.settings['residualize'])
            self.assertEqual(run.settings['timepoints'], ['day4'])
            self.assertTrue((self.out/'cohort/inputs.pkl').exists())
            self.assertEqual(len(list((self.out/'inputs').glob('*/inputs.pkl'))),2)
            self.check_reference_controls()
            self.check_reference_controls(max_folds=1)
            restored=run.select(run.records.iloc[0].key)
            self.assertEqual(set(restored['raw_graphs']), {g.organoid_str for g in self.graphs[:6]})
            self.assertTrue(all(g.meta['timepoint'] == 'day4' for g in restored['raw_graphs'].values()))
            self.assertEqual({g.organoid_str for group in restored['groups'].values() for g in group},
                             set(restored['raw_graphs']))
            table=prediction_table(restored)
            for org,frame in table.groupby('organoid_str'):
                np.testing.assert_allclose(frame.y_true,restored['raw_graphs'][org].y.numpy(),atol=2e-6)
            self.assertFalse(restored['model'].training)
            # New multi-depth FiLM runs remain usable by existing size/masking workflows.
            from src.artifacts.size_reference import size_reference
            from src.analysis.size_conditioning.cohort_inputs import load_cohort, fold_graphs
            from torch_geometric.data import Batch
            film_key = run.records.query("name == 'film' and depth == 2").iloc[0].key
            with self.assertRaisesRegex(ValueError, 'REFERENCE_MODEL_KEY'):
                size_reference(self.out)
            reference = size_reference(self.out, film_key)
            source = run.select(film_key)
            compatibility = AnalysisRun(reference)
            adapted = compatibility.select('gin_film_size_f0_s42')
            with torch.no_grad():
                for original, loaded in zip(source['groups']['val'], adapted['groups']['val']):
                    a = Batch.from_data_list([original]); b = Batch.from_data_list([loaded])
                    expected, hidden_a = source['model'](a.x, a.edge_index, data=a)
                    actual, hidden_b = adapted['model'](b.x, b.edge_index, data=b)
                    torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
                    torch.testing.assert_close(hidden_a, hidden_b, rtol=0, atol=0)
                    torch.testing.assert_close(original.y, loaded.y, rtol=0, atol=0)
            self.assertEqual(size_reference(self.out, film_key), reference)
            with patch.object(data_io, 'load_graph_dataset_from_dir', side_effect=AssertionError('Reloaded source data')):
                raw = load_cohort(Path('/missing/dataset'), compatibility.settings, reference)
                pre = compatibility.legacy.preprocessing(0)
                exact = fold_graphs(raw, compatibility.legacy.splits[0], pre, compatibility.settings)
                torch.testing.assert_close(exact['val'][0].y, source['groups']['val'][0].y, rtol=0, atol=0)
                from notebook_workflows import workflow
                from src.training.masking import MaskTrainingConfig
                train_masking = workflow('training/fate_masking_training.ipynb', 'train_masking_experiment')
                masking_out = self.out.parent/'masking'
                train_masking(ROOT, reference, masking_out, folds=[0], seeds=[42], device='cpu', run_tag='fixture',
                    config=MaskTrainingConfig(rates=(0., .02), max_epochs=1, patience=1, batch_size=4))
                masking = AnalysisRun(masking_out)
                masked = masking.select('gin_film_size_f0_s42_p0.02')
                self.assertEqual(masked['groups']['val'][0].x.shape[1], len(source['marker_names'])+1)
                torch.testing.assert_close(masked['groups']['val'][0].y, source['groups']['val'][0].y)
                self.assertEqual(json.loads((masking_out/'settings.json').read_text())['tag'], 'fixture')
                benchmark = workflow('benchmarks/masking_quality_and_robustness.ipynb', 'benchmark_masking_experiment')
                benchmark(ROOT, masking_out, centers_per_identity=1, quality_centers_per_organoid=2,
                          batch_size=8, device='cpu')
                from src.analysis.metrics.evaluation import organoid_mse_summary
                region_ns = dict(ROOT=ROOT, MASKING_RUN=masking_out, DEVICE='cpu', REGION_SETTINGS={},
                                 AnalysisRun=AnalysisRun, json=json, pd=pd, plt=plt,
                                 organoid_mse_summary=organoid_mse_summary, display=lambda *args: None)
                exec(notebook_sources('training/fate_masking_training.ipynb')[-1], region_ns)
                mask_regions = pd.read_csv(masking_out/'regional_evaluation/validation_mse.csv')
                self.assertEqual(set(mask_regions.rate), {0., .02})
                self.assertTrue(mask_regions.baseline_mse.notna().all())
                from src.analysis.interventions.replacement import MatchConfig
                from src.analysis.interventions.fate_edits import (
                    fate_graphs, sample_fate_contexts, build_replacement_reference,
                    replacement_distribution, evaluate_fate_edit)
                subs,cases = sample_fate_contexts(fate_graphs(source),source['marker_names'],2,centers=1)
                matcher,contexts = build_replacement_reference(source,cases,2,
                    config=MatchConfig(neighbors=16,min_cells=1,min_organoids=1,min_identity_cells=1,min_identity_organoids=1))
                weights,_ = replacement_distribution(cases,contexts,matcher,source['marker_names'])
                effects = evaluate_fate_edit(source,subs,cases,'replacement',count=14,weights=weights,device='cpu')
                self.assertEqual(len(effects),len(cases))
            # Reopening starts from saved data; no fitted state comes from the training namespace.
            del ns
            with patch('src.training.loop.train',side_effect=AssertionError('Analysis retrained')):
                for path in ['benchmarks/model_comparison.ipynb','ablation/total_analysis.ipynb',
                             'ablation/size_dependent_ablation.ipynb',
                             'ablation/sampling_diagnostics.ipynb','data_quality/cohort_review.ipynb','embeddings/embedding_responses.ipynb',
                             'embeddings/clustering.ipynb', 'embeddings/patch_composition.ipynb']:
                    ns={}
                    for source in notebook_sources(path):
                        source=source.replace("resolve_training_run(ROOT, TRAINING_NOTEBOOK, TRAINING_RUN)",f'Path({str(self.out)!r})')
                        source=source.replace("resolve_training_run(ROOT, TRAINING_SUBFOLDER, TRAINING_RUN)",f'Path({str(self.out)!r})')
                        source=source.replace("resolve_training_run(ROOT,TRAINING_SUBFOLDER,TRAINING_RUN)",f'Path({str(self.out)!r})')
                        exec(compile(source,path,'exec'),ns)
                        if 'MODEL_DEPTH =' in source and 'MODEL_FOLD =' in source:
                            ns.update(MODEL_DEPTH=2, MODEL_FOLD=0, MODEL_NAME='gin', MODEL_KEY=None,
                                      MODEL_SEED=None, HIDDEN_DIM=None, DEVICE='cpu')
                        if "ABLATION_TYPE = 'marker_zeroing'" in source:
                            ns.update(RUN_INFERENCE=True, DEVICE='cpu', SWEEP_COUNTS=[16,32],
                                      BOOTSTRAP_SAMPLES=5, MIN_PLOT_ORGANOIDS=1, MODEL_NAME='gin')
                        if 'SAMPLING_SCHEMES =' in source:
                            ns.update(MODEL_NAME='gin',SAMPLING_REPEATS=[1701])
                        if 'MODEL_KEY = run.records.iloc[0].key' in source:
                            ns['DEVICE']='cpu'
                    if path == 'data_quality/cohort_review.ipynb':
                        self.assertEqual(ns['selected_record'].depth,2)
                        self.assertEqual(ns['selected_record'].fold,0)
                        self.assertTrue(ns['ranked'].mse.is_monotonic_decreasing)
                        np.testing.assert_array_equal(ns['ranked'].mse_rank,np.arange(1,len(ns['ranked'])+1))
                        self.assertTrue(ns['ranked'].sphericity.notna().all())
                        for org,row in ns['ranked'].iterrows():
                            meta=ns['selection']['raw_graphs'][org].meta
                            self.assertAlmostEqual(row.sphericity,36*np.pi*meta['total_volume']**2/meta['total_surface_area']**3)
                        ns['candidate_browser'].close()
                        for control in ns['candidate_controls'].values():control.close()
                    if path == 'ablation/total_analysis.ipynb':
                        self.assertEqual(set(ns['cases'].fold), {0,1})
                        for fold, rows in ns['cases'].groupby('fold'):
                            record = run.records.query("name == 'gin' and depth == 2 and fold == @fold").iloc[0]
                            pack = run.select(record.key)
                            self.assertEqual(set(rows.organoid_str), {g.organoid_str for g in pack['groups']['val']})
                    if path == 'ablation/size_dependent_ablation.ipynb':
                        self.assertEqual(set(ns['cases'].fold), {0,1})
                        self.assertEqual(set(ns['cases'].analysis), {'observed','sweep'})
                        self.assertEqual(set(ns['cases'].query("analysis == 'sweep'").evaluated_n), {16,32})
            cluster_key = run.records.query("name == 'gin' and depth == 2 and fold == 0").iloc[0].key
            cluster_dir = self.out/'analysis/clustering'/cluster_key
            self.assertTrue((cluster_dir/'atlas.pkl').exists())
            for figure in ['tsne_clusters_full.png','tsne_clusters_filtered.png',
                           'marker_composition_full.png','marker_composition_filtered.png',
                           'cluster_residual_curvature.png','crypt_distance_categories.png']:
                self.assertTrue((cluster_dir/figure).is_file(),figure)
            # Marker-population denominators include all positive cells in the
            # displayed subset, even when some clusters have no assigned cells.
            for subset in ['full','filtered']:
                counts = pd.read_csv(cluster_dir/f'marker_counts_{subset}.csv',index_col=0)
                distribution = pd.read_csv(cluster_dir/f'marker_population_{subset}.csv',index_col=0)
                present = counts.sum()>0
                np.testing.assert_allclose(distribution.loc[:,present].sum(),1.)
                self.assertTrue(distribution.loc[:,~present].isna().all().all())
                distances = pd.read_csv(cluster_dir/f'dcrypt_category_fractions_{subset}.csv',index_col=0)
                np.testing.assert_allclose(distances.dropna().sum(axis=1),1.)
            cells = pd.read_csv(cluster_dir/'cells.csv.gz')
            np.testing.assert_allclose(cells.y_true_residual+cells.baseline_prediction,cells.y_true,atol=1e-10)
            np.testing.assert_allclose(cells.y_pred_residual+cells.baseline_prediction,cells.y_pred,atol=1e-10)

    def test_clustering_selects_one_explicit_depth_fold_and_reselects(self):
        from types import SimpleNamespace
        source = next(source for source in notebook_sources('embeddings/clustering.ipynb')
                      if '# Saved model selection: one depth' in source)
        source = source[source.index('# Saved model selection: one depth'):source.index('# Inference resources')]
        rows = [dict(key=f'{name}_d{depth}_f{fold}',name=name,depth=depth,fold=fold,seed=42,hidden_dim=8)
                for name in ['gin','film'] for depth in [0,2] for fold in [0,1]]
        ns = dict(run=SimpleNamespace(records=pd.DataFrame(rows),legacy=None),MODEL_KEY=None,
                  MODEL_DEPTH=2,MODEL_FOLD=0,MODEL_NAME='gin',MODEL_SEED=None,HIDDEN_DIM=None)
        with contextlib.redirect_stdout(io.StringIO()):
            exec(source,ns)
            self.assertEqual(ns['SELECTED_MODEL_KEY'],'gin_d2_f0')
            self.assertIsNone(ns['MODEL_KEY'])
            ns.update(MODEL_DEPTH=0,MODEL_FOLD=1)
            exec(source,ns)
            self.assertEqual(ns['SELECTED_MODEL_KEY'],'gin_d0_f1')
            ns.update(MODEL_DEPTH=2,MODEL_NAME='film')
            exec(source,ns)
            self.assertEqual(ns['SELECTED_MODEL_KEY'],'film_d2_f1')
            ns['MODEL_NAME']=None
            with self.assertRaisesRegex(ValueError,'exactly one'):
                exec(source,ns)
            ns['MODEL_DEPTH']=99
            with self.assertRaisesRegex(ValueError,'exactly one'):
                exec(source,ns)
            ns['MODEL_KEY']='gin_d2_f0'
            exec(source,ns)
            self.assertEqual(ns['SELECTED_MODEL_KEY'],'gin_d2_f0')

    def test_restored_clustering_distance_and_residual_semantics(self):
        import ast
        function = next(node for source in notebook_sources('embeddings/clustering.ipynb')
                        for node in ast.parse(source).body if isinstance(node,ast.FunctionDef) and node.name=='min_node_dcrypt')
        namespace = dict(np=np,torch=torch)
        exec(compile(ast.Module(body=[function],type_ignores=[]),'clustering helper','exec'),namespace)
        nearest = namespace['min_node_dcrypt']
        distances = np.array([[.1,.9,1.4,np.nan],[.4,1.,2.,np.nan]])
        for array in [distances,distances.T]:
            np.testing.assert_allclose(nearest(array,4),[.1,.9,1.4,np.nan],equal_nan=True)
        self.assertTrue(np.isnan(nearest(None,4)).all())
        self.assertTrue(np.isnan(nearest(np.empty((0,4)),4)).all())
        with self.assertRaisesRegex(ValueError,'unambiguously'):
            nearest(np.zeros((4,4)),4)
        prepare = next(source for source in notebook_sources('embeddings/clustering.ipynb')
                       if "if 'baseline_prediction' not in predictions:" in source)
        table = pd.DataFrame(dict(y_true=[2.,3.,4.,5.],y_pred=[2.1,3.2,4.3,5.4],
            baseline_prediction=[1.,1.,2.,2.],squared_error=[.01,.04,.09,.16],node=range(4),cluster=[0,0,1,1]))
        ns=dict(predictions=table.copy(),selection={'residualized':False,'baseline_offsets':{'org':np.zeros(4)}},
                labels=np.array([0,0,1,1]),cluster_order=np.arange(2),np=np,pd=pd,
                ERROR_FILTER_DROP_FRACTION=.5,ERROR_FILTER_MIN_RETAINED_PER_CLUSTER=1,
                SHOW_ACCURACY_FILTERED=True,display=lambda *args:None)
        exec(prepare,ns)
        np.testing.assert_allclose(ns['predictions'].y_true_residual,[1.,2.,2.,3.])
        np.testing.assert_allclose(ns['predictions'].y_pred_residual,[1.1,2.2,2.3,3.4])
        np.testing.assert_array_equal(ns['retained_mask'],[True,False,True,False])

    def test_baseline_is_saved_without_residualization_or_gnn_globals(self):
        split=dict(train_indices=list(range(6)),val_indices=[6,7])
        prepared=prepare_fold(self.graphs,split,global_features=[],residualize=False,
            baseline_kwargs=dict(max_epochs=1, patience=1, num_workers=0, train_kwargs=dict(device='cpu')))
        self.assertIsNotNone(prepared['baseline'].model)
        self.assertFalse(prepared['residualized'])
        self.assertTrue(all('global_feat' not in g for group in prepared['groups'].values() for g in group))
        for g in prepared['groups']['val']:
            original=next(x for x in self.graphs if x.organoid_str==g.organoid_str)
            np.testing.assert_allclose(prepared['transform'].inverse(g.y),original.y,atol=2e-6)
            np.testing.assert_array_equal(prepared['baseline_offsets'][g.organoid_str], 0.)
            baseline_prediction = prepared['baseline_predictions'][g.organoid_str]
            expected_mse = np.mean((original.y.numpy()-baseline_prediction)**2)
            score = next(row['mse'] for row in prepared['baseline_validation_mse'] if row['organoid_str']==g.organoid_str)
            self.assertAlmostEqual(score, expected_mse)
        from src.artifacts.bundle import save_bundle, load_bundle
        save_bundle(self.out, prepared, splits=split)
        restored = load_bundle(self.out)
        self.assertIsNotNone(restored['baseline'].model)
        for name, value in prepared['baseline'].model.state_dict().items():
            torch.testing.assert_close(value, restored['baseline'].model.state_dict()[name], rtol=0, atol=0)

    def test_shared_timepoint_filter_rejects_invalid_selection(self):
        from src.data.filters import filter_graphs_by_timepoints
        for graph in self.graphs[-2:]:
            graph.meta['timepoint'] = 'day5'
        selected = filter_graphs_by_timepoints(self.graphs, ['day4'])
        self.assertEqual([g.organoid_str for g in selected], [g.organoid_str for g in self.graphs[:6]])
        self.assertEqual(len(filter_graphs_by_timepoints(self.graphs)), 8)
        for invalid in ([], 'day4', ['day4', 'day_typo']):
            with self.subTest(selection=invalid), self.assertRaises(ValueError):
                filter_graphs_by_timepoints(self.graphs, invalid)

    def test_marker_combinations_have_matching_saved_inputs_and_shared_splits(self):
        import src.data.io as data_io
        import src.data.metadata as metadata
        with contextlib.redirect_stdout(io.StringIO()), \
             patch.object(data_io, 'load_graph_dataset_from_dir', side_effect=lambda _: copy.deepcopy(self.graphs)), \
             patch.object(metadata, 'load_aux_metadata_for_dir', return_value={}), \
             patch.object(metadata, 'load_marker_names_from_dir', return_value=['LGR5', 'KI67']), \
             patch.object(plt, 'show', side_effect=lambda: plt.close('all')):
            ns = {}
            for source in notebook_sources('training/lineage_removal_training.ipynb'):
                exec(compile(source, 'lineage', 'exec'), ns)
                if 'SETTINGS = dict(' in source:
                    ns.update(RUN_TRAINING=True, RUN_DIR=self.out, DEVICE='cpu')
                    ns['SETTINGS'].update(blacklist=False, interpolate_outliers=False, depths=[1],
                        hidden_dims=[8], n_folds=2, seeds=[42], max_epochs=1, num_workers=0,
                        baseline_max_epochs=1, baseline_patience=1,
                        marker_subsets={'both':['LGR5','KI67'], 'stem':['LGR5']})
            run = AnalysisRun(self.out)
            self.assertEqual(len(run.records), 4)
            validation = pd.read_csv(self.out/'validation_mse.csv')
            baseline_scores = pd.read_csv(self.out/'baseline_validation_mse.csv')
            self.assertEqual(len(baseline_scores), len(self.graphs))
            self.assertTrue(validation.baseline_mse.notna().all())
            np.testing.assert_allclose(validation.mse_minus_baseline, validation.mse-validation.baseline_mse)
            memberships = {}
            for record in run.records.itertuples():
                saved = run.select(record.key)
                self.assertIsNotNone(saved['baseline'].model)
                expected = ['LGR5','KI67'] if record.subset == 'both' else ['LGR5']
                self.assertEqual(saved['marker_names'], expected)
                self.assertTrue(all(g.x.shape[1] == len(expected)
                                    for group in saved['groups'].values() for g in group))
                ids = {role: [g.organoid_str for g in graphs] for role, graphs in saved['groups'].items()}
                if record.fold in memberships:self.assertEqual(memberships[record.fold], ids)
                memberships[record.fold] = ids

    def test_training_notebooks_split_only_the_shared_cohort(self):
        import src.data.io as data_io
        import src.data.metadata as metadata
        for graph in self.graphs[-2:]:
            graph.meta['timepoint'] = 'day5'
        notebooks = ['gin_depth_training', 'lineage_removal_training']
        with contextlib.redirect_stdout(io.StringIO()), \
             patch.object(data_io, 'load_graph_dataset_from_dir', side_effect=lambda _: copy.deepcopy(self.graphs)), \
             patch.object(metadata, 'load_aux_metadata_for_dir', return_value={}), \
             patch.object(metadata, 'load_marker_names_from_dir', return_value=['LGR5', 'KI67']), \
             patch.object(plt, 'show', side_effect=lambda: plt.close('all')):
            for name in notebooks:
                for timepoints in (None, ['day4']):
                    with self.subTest(notebook=name, timepoints=timepoints):
                        ns = {}
                        for source in notebook_sources(f'training/{name}.ipynb'):
                            exec(compile(source, name, 'exec'), ns)
                            if 'SETTINGS = dict(' in source:
                                ns['SETTINGS'].update(timepoints=timepoints, blacklist=False,
                                    interpolate_outliers=False, n_folds=2, sphericity_max=None, complexity_min=None)
                            if 'splits' in ns:
                                break
                        expected = {g.organoid_str for g in self.graphs
                                    if timepoints is None or g.meta['timepoint'] in timepoints}
                        self.assertEqual({g.organoid_str for g in ns['graphs']}, expected)
                        for split in ns['splits']:
                            train = set(split['train_indices'])
                            val = set(split['val_indices'])
                            self.assertTrue(train and val)
                            self.assertFalse(train & val)
                            self.assertEqual(train | val, set(range(len(expected))))
                        key = 'timepoints'
                        self.assertEqual(ns['SETTINGS'][key], timepoints)


if __name__=='__main__':
    unittest.main()
