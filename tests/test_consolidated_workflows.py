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
                from src.analysis.interventions.replacement import MatchConfig
                replacement = workflow('ablation/fate_interventions.ipynb', 'run_replacement_analysis')
                replacement(ROOT, reference, self.out.parent/'replacement', counts=[14,16], folds=[0], seeds=[42],
                    centers_per_identity=1, sweep_centers_per_identity=1, batch_size=8, device='cpu',
                    matcher_config=MatchConfig(neighbors=16, min_cells=1, min_organoids=1,
                        min_identity_cells=1, min_identity_organoids=1))
            # Reopening starts from saved data; no fitted state comes from the training namespace.
            del ns
            with patch('src.training.loop.train',side_effect=AssertionError('Analysis retrained')):
                for path in ['benchmarks/model_comparison.ipynb','ablation/marker_zeroing.ipynb',
                             'ablation/sampling_diagnostics.ipynb','embeddings/embedding_responses.ipynb',
                             'embeddings/clustering.ipynb', 'embeddings/patch_composition.ipynb']:
                    ns={}
                    for source in notebook_sources(path):
                        source=source.replace("resolve_training_run(ROOT, TRAINING_NOTEBOOK, TRAINING_RUN)",f'Path({str(self.out)!r})')
                        exec(compile(source,path,'exec'),ns)
                        if 'MODEL_KEY = run.records.iloc[0].key' in source:
                            ns['DEVICE']='cpu'
            self.assertTrue((self.out/'analysis/clustering'/run.records.iloc[0].key/'atlas.pkl').exists())

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
        notebooks = ['gin_depth_training', 'graph_controls_training',
                     'lineage_removal_training']
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
