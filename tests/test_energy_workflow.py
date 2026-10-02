"""Exercise the fresh-cohort notebook on synthetic data; never touch project results."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nbformat
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from src.artifacts.bundle import load_bundle

ROOT=Path(__file__).resolve().parents[1]
MARKERS=['Agr2','AldoB','Chroma','KI67','LGR5','Lysozyme','Serotonin']


def synthetic_graphs():
    graphs=[];rng=np.random.default_rng(7)
    for i in range(18):
        n=12+i%3;identity=rng.integers(0,8,n);x=np.eye(8)[identity,:7]
        edge=np.array([(j,(j+1)%n) for j in range(n)]+[((j+1)%n,j) for j in range(n)]).T
        g=Data(x=torch.tensor(x,dtype=torch.float32),edge_index=torch.tensor(edge),y=torch.tensor(.02*identity+.001*np.arange(n),dtype=torch.float64))
        q=.97 if i>=14 else .8;area=50.+i;volume=np.sqrt(q*area**3/(36*np.pi))
        g.organoid_str=f'synthetic_{i}';g.meta=dict(timepoint='day4',total_surface_area=area,total_volume=volume)
        graphs.append(g)
    graphs[-1].meta['total_volume']=None  # Invalid metadata must not enter the spherical set.
    graphs[-2].meta['timepoint']='day3p5'  # Earlier timepoints must not enter either cohort.
    return graphs


def fake_regions(g,*args,**kwargs):
    labels=np.resize(['crypt_with_LGR5','crypt_without_LGR5','neck','villus','boundary_without_qualified_neck','no_detected_crypt'],len(g.x))
    return pd.DataFrame(dict(organoid_str=g.organoid_str,node=np.arange(len(g.x)),region=labels))


class EnergyWorkflowTests(unittest.TestCase):
    def test_fresh_holdout_and_folded_roundtrip(self):
        torch.set_num_threads(2)
        for folds in [1,2]:
            with self.subTest(folds=folds),tempfile.TemporaryDirectory() as d:
                run=Path(d)/'run'
                cells=[c.source for c in nbformat.read(ROOT/'experiments/training/energy_model_training.ipynb',4).cells if c.cell_type=='code']
                ns={};exec(cells[0],ns);exec(cells[1],ns)
                ns['SETTINGS'].update(activation='presence',n_folds=folds,gin_enabled=folds==1,source_markers=['Agr2','Chroma','Lysozyme','Serotonin'],recipient_markers=['AldoB','KI67','LGR5','Unassigned'],timepoints=['day4'],blacklist=False,interpolate_outliers=True,baseline_features=['log_num_cells'],
                    gin_hidden_dim=8,gin_max_epochs=2,gin_patience=1,gin_batch_size=4,baseline_hidden_dim=4,baseline_max_epochs=1,baseline_patience=1,baseline_num_workers=0,baseline_batch_size=4,gamma_grid=[0.,.2],pair_ridge_grid=[.001],blas_threads=2,min_pair_organoids=1)
                ns.update(DEVICE='cpu',SHOW_PROGRESS=False,RUN_TRAINING=True,RUN_DIR=run,PREPARED_INPUT_RUN=None,
                    load_graph_dataset_from_dir=lambda *a,**k:synthetic_graphs(),load_aux_metadata_for_dir=lambda *a:{},attach_metadata_to_graphs=lambda *a,**k:None,
                    load_marker_names_from_dir=lambda *a:MARKERS,distance_regions=fake_regions)
                with patch('matplotlib.pyplot.show',side_effect=lambda:plt.close('all')):
                    for cell in cells[2:]:exec(cell,ns)
                splits=json.loads((run/'splits.json').read_text());records=json.loads((run/'models.json').read_text())
                self.assertEqual(len(records),folds*(3 if folds==1 else 2))
                self.assertEqual(set(splits[0]['spherical_validation']),{'synthetic_14','synthetic_15'})
                for split in splits:
                    self.assertFalse(set(split['train'])&set(split['spherical_validation']))
                    self.assertFalse(set(split['validation'])&set(split['spherical_validation']))
                    prepared=load_bundle(run/f"inputs/fold_{split['fold']}_mean")
                    self.assertEqual([g.organoid_str for g in prepared['groups']['val']],split['validation'])
                    self.assertEqual({r['organoid_str'] for r in prepared['baseline_validation_mse']},set(split['validation']))
                    self.assertEqual({r['organoid_str'] for r in prepared['baseline_spherical_mse']},set(split['spherical_validation']))
                    for record in [r for r in records if r['fold']==split['fold']]:
                        payload=load_bundle(run/record['bundle'])
                        self.assertEqual(set(payload['metrics']['fit_organoids']),set(split['train']))
                        self.assertFalse(set(payload['metrics']['inner_validation'])&set(split['spherical_validation']))
                        if record['family']=='energy':self.assertEqual(payload['model'].hidden_dim,24)
                        else:
                            self.assertEqual(payload['model'].global_dim,0)
                            self.assertEqual(payload['model'].num_layers,1)
                            self.assertFalse(payload['metrics']['film'])
                ev=[c.source for c in nbformat.read(ROOT/'experiments/benchmarks/energy_model_evaluation.ipynb',4).cells if c.cell_type=='code']
                scope={};exec(ev[0],scope);exec(ev[1],scope);scope.update(TRAINING_RUN=str(run),DEVICE='cpu',STABILITY_REFERENCE_RUN=None)
                with patch('matplotlib.pyplot.show',side_effect=lambda:plt.close('all')):
                    for cell in ev[2:]:exec(cell,scope)
                out=scope['OUTPUT_DIR'];self.assertTrue((out/'complete.json').exists())
                scores=pd.read_csv(out/'validation_mse.csv')
                self.assertNotIn('boundary_without_qualified_neck',set(scores.region));self.assertNotIn('excluded_boundary',set(scores.region))
                self.assertEqual(set(scores.model),{'energy','gin'} if folds==1 else {'energy'})
                self.assertEqual(set(scores[scores.region=='total'].cohort),{'val'})
                self.assertEqual(set(scores[scores.region=='spherical_or_no_detected_crypt'].cohort),{'val','spherical'})
                self.assertEqual(set(scores[scores.region=='spherical_validation_only'].organoid_str),{'synthetic_14','synthetic_15'})
                composition=pd.read_csv(out/'cohort_composition.csv')
                self.assertEqual(composition.organoid_str.nunique(),16)
                np.testing.assert_allclose(composition.groupby('organoid_str').fraction.sum(),1)
                neighborhoods=pd.read_csv(out/'source_neighborhood_composition.csv')
                np.testing.assert_allclose(neighborhoods.groupby(['organoid_str','source']).fraction.sum(),1)
                stability=pd.read_csv(out/'interaction_stability.csv')
                self.assertTrue((stability.folds==folds).all())
                self.assertEqual(len(stability),32)
                if folds==2:
                    values=pd.DataFrame(dict(fold=[0,1],gamma=[0.,0.],identity=['A','A'],value=[.003,-.003]))
                    result=scope['summarize_stability'](values,['identity']).iloc[0]
                    self.assertEqual(result.signed_consensus,0)
                    self.assertEqual(result.positive_folds,1)
                    self.assertEqual(result.negative_folds,1)
                    self.assertFalse(result.unanimous_nonzero_sign)
                    values['value']=[.0001,-.0001]
                    self.assertTrue(scope['summarize_stability'](values,['identity']).iloc[0].all_near_zero)
                for name in ['cohort_composition.png','source_neighborhood_composition.png','center_fold_stability.png','interaction_fold_stability.png']:
                    self.assertTrue((out/name).exists())
                if folds==2:
                    matched=Path(d)/'all_pairs'
                    ns['SETTINGS'].update(source_markers=MARKERS+['Unassigned'],recipient_markers=MARKERS+['Unassigned'])
                    ns.update(RUN_DIR=matched,PREPARED_INPUT_RUN=str(run))
                    with patch('matplotlib.pyplot.show',side_effect=lambda:plt.close('all')):
                        for cell in cells[2:]:exec(cell,ns)
                    for split in splits:
                        relative=f"inputs/fold_{split['fold']}_mean/manifest.json"
                        self.assertEqual((run/relative).read_bytes(),(matched/relative).read_bytes())
                    rec=json.loads((matched/'models.json').read_text())[0]
                    self.assertEqual(load_bundle(matched/rec['bundle'])['model'].hidden_dim,72)
                    scope={};exec(ev[0],scope);exec(ev[1],scope)
                    scope.update(TRAINING_RUN=str(matched),DEVICE='cpu',STABILITY_REFERENCE_RUN=str(run))
                    with patch('matplotlib.pyplot.show',side_effect=lambda:plt.close('all')):
                        for cell in ev[2:]:exec(cell,scope)
                    stability=pd.read_csv(scope['OUTPUT_DIR']/'interaction_stability.csv')
                    self.assertEqual(len(stability),128)
                    shared=pd.read_csv(scope['OUTPUT_DIR']/'common_interaction_stability.csv')
                    self.assertEqual(len(shared),32)
                    self.assertTrue((scope['OUTPUT_DIR']/'reference_mse_comparison.png').exists())
                    linear=Path(d)/'linear_pairs'
                    ns['SETTINGS']['activation']='linear'
                    ns.update(RUN_DIR=linear,PREPARED_INPUT_RUN=str(run))
                    with patch('matplotlib.pyplot.show',side_effect=lambda:plt.close('all')):
                        for cell in cells[2:]:exec(cell,ns)
                    scope={};exec(ev[0],scope);exec(ev[1],scope)
                    scope.update(TRAINING_RUN=str(linear),DEVICE='cpu',STABILITY_REFERENCE_RUN=str(matched))
                    with patch('matplotlib.pyplot.show',side_effect=lambda:plt.close('all')):
                        for cell in ev[2:]:exec(cell,scope)
                    variance=pd.read_csv(scope['OUTPUT_DIR']/'parameter_variance_summary.csv')
                    self.assertEqual(len(variance),8)
                    diagnostics=pd.read_csv(scope['OUTPUT_DIR']/'unpenalized_design_diagnostics.csv')
                    self.assertTrue((diagnostics[diagnostics.activation=='linear'].nullity>=8).all())
                    self.assertLess(diagnostics[diagnostics.activation=='linear'].row_shift_max_local_change.max(),1e-12)
                    self.assertEqual(len(pd.read_csv(scope['OUTPUT_DIR']/'common_context_variance_summary.csv')),8)
                    # Reference-relative effects must be invariant to a row-wide linear coefficient shift.
                    original=scope['reference_relative'](scope['centers'],scope['pairs'])
                    shifted_c=scope['centers'].copy();shifted_p=scope['pairs'].copy()
                    shifted_c['value']-=.1;shifted_p['value']+=.1
                    shifted=scope['reference_relative'](shifted_c,shifted_p)
                    for a,b in zip(original,shifted):
                        np.testing.assert_allclose(a['mean'],b['mean'],atol=1e-12)
                        np.testing.assert_allclose(a.sd,b.sd,atol=1e-12)


if __name__=='__main__':unittest.main()
