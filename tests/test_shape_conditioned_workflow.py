"""Run the new notebooks on a tiny matched cohort, never touching project results."""
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
from src.artifacts.bundle import save_bundle, load_bundle
from src.data.target_transforms import IdentityTransform
from src.training.coupled_fit import graph_samples
from src.training.energy_fit import fit_energy_fixed
from tests.test_energy_workflow import synthetic_graphs, fake_regions, MARKERS

ROOT=Path(__file__).resolve().parents[1]


class ConditionalWorkflowTests(unittest.TestCase):
    def test_training_restoration_and_evaluation(self):
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as tmp:
            ref=Path(tmp)/'reference';run=Path(tmp)/'conditional';ref.mkdir()
            cells=[c.source for c in nbformat.read(ROOT/'experiments/training/shape_conditioned_energy_training.ipynb',4).cells if c.cell_type=='code']
            ns={};exec(cells[0],ns);exec(cells[1],ns)
            ns['SETTINGS'].update(n_folds=1,gin_hidden_dim=8,gin_batch_size=4,gin_max_epochs=2,gin_patience=1,gin_num_workers=0,
                gamma_grid=[0.,.8],pair_ridge_grid=[.001],blas_threads=2,min_pair_organoids=1)
            ns.update(DEVICE='cpu',SHOW_PROGRESS=False,REFERENCE_RUN=str(ref),RUN_DIR=run,RESUME_RUN=None)
            gs=synthetic_graphs()[:12];groups=dict(train=gs[:8],val=gs[8:10],spherical=gs[10:])
            split=dict(fold=0,train=[g.organoid_str for g in groups['train']],validation=[g.organoid_str for g in groups['val']],spherical_validation=[g.organoid_str for g in groups['spherical']])
            settings={k:ns['SETTINGS'][k] for k in ['dataset','target_indices','timepoints','sphericity_max','exclusive_markers','n_folds','split_seed','interpolate_outliers']}
            (ref/'settings.json').write_text(json.dumps(settings));(ref/'splits.json').write_text(json.dumps([split]));(ref/'exclusivity_rules.json').write_text('{}');(ref/'complete.json').write_text('{}')
            pd.DataFrame([dict(organoid_str=g.organoid_str,cohort='ordinary' if g in gs[:10] else 'spherical',timepoint='day4',N=len(g.x),sphericity=.8) for g in gs]).to_csv(ref/'cohort.csv',index=False)
            pd.concat([fake_regions(g).replace({'boundary_without_qualified_neck':'excluded_boundary','no_detected_crypt':'spherical_or_no_detected_crypt'}) for g in gs]).to_csv(ref/'regions.csv.gz',index=False)
            zeros={g.organoid_str:np.zeros(len(g.x)) for g in gs}
            data=dict(groups=groups,marker_names=MARKERS,transform=IdentityTransform().fit(groups['train']),baseline_offsets=zeros,baseline_predictions=zeros,baseline=None)
            save_bundle(ref/'inputs/fold_0_mean',data,splits=split)
            model,metrics,_=fit_energy_fixed(graph_samples(groups['train'],2),model_settings=dict(n_markers=7,interaction='pooled',activation='presence',center_response=False,pair_constraints='direct',interaction_radius=1),strength=0.,penalties=[dict(pair_ridge=.001)])
            rec=dict(key='prior',fold=0,family='energy',gamma=0.,bundle='models/prior')
            save_bundle(ref/'models/prior',dict(model=model,metrics=metrics),splits=split);(ref/'models.json').write_text(json.dumps([rec]))
            dest=ref/'analysis/accommodation_profiles/predictions';dest.mkdir(parents=True)
            for role in ['val','spherical']:
                group=groups[role];ss=graph_samples(group,2)
                np.savez_compressed(dest/f'prior_{role}.npz',organoid_ids=np.array([g.organoid_str for g in group]),offsets=np.r_[0,np.cumsum([len(g.x) for g in group])],truth=np.concatenate([g.y.numpy() for g in group]),prediction=np.concatenate(model.predict_samples(ss)))
            with patch('matplotlib.pyplot.show',side_effect=lambda:plt.close('all')):
                for cell in cells[2:]:exec(cell,ns)
            self.assertTrue((run/'complete.json').exists());self.assertEqual(len(ns['records']),6)
            prepared=load_bundle(run/'inputs/fold_0_mean')
            self.assertEqual(prepared['inner_train'],metrics['inner_train'])
            for role,group in prepared['physical_groups'].items():
                for g in group:
                    m=prepared['moments'][g.organoid_str]
                    np.testing.assert_allclose(m.inverse(m.transform(g.y.numpy())),g.y.numpy(),atol=1e-12)
            ev=[c.source for c in nbformat.read(ROOT/'experiments/benchmarks/shape_conditioned_energy_evaluation.ipynb',4).cells if c.cell_type=='code']
            scope={};exec(ev[0],scope);exec(ev[1],scope);scope.update(TRAINING_RUN=str(run),DEVICE='cpu',MIN_BIN_ORGANOIDS=1)
            with patch('matplotlib.pyplot.show',side_effect=lambda:plt.close('all')):
                for cell in ev[2:]:exec(cell,scope)
            self.assertTrue((scope['OUTPUT_DIR']/'complete.json').exists())
            self.assertTrue((run/'REPORT.md').exists())
            self.assertLess(max(x['max_difference'] for x in scope['verification']),1e-6)
            selected=pd.read_csv(run/'selected_models.csv')
            self.assertEqual(len(selected),2)


if __name__=='__main__':unittest.main()
