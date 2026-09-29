"""Execute new training/evaluation cells on small saved multi-target folds."""
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
from src.models.gnn import GINCurvature
from src.training.preparation import prepare_fold,combine_target_folds
from src.artifacts.bundle import save_bundle,graph_membership
from tests.test_coupled_fate import graphs
ROOT=Path(__file__).resolve().parents[1]

class CoupledNotebookTests(unittest.TestCase):
    def test_saved_multitarget_training_and_evaluation(self):
        torch.set_num_threads(2);torch.manual_seed(42)
        tmp=Path(self.enterContext(tempfile.TemporaryDirectory()));ref=tmp/'reference';ref.mkdir();out=tmp/'new'
        cohort=graphs();markers=['LGR5','KI67']
        for g in cohort:
            g.meta=dict(total_surface_area=2.*len(g.x),total_volume=float(len(g.x)**1.5))
            g.y=(.02*g.x[:,0]-.01*g.x[:,1]+.003*np.log(len(g.x))).double()
        settings=dict(dataset='synthetic_missing',exclusive_markers=True,target_indices=[0],seeds=[42],n_folds=2)
        splits=[dict(fold=f,**graph_membership(train=cohort[f::2],validation=cohort[1-f::2])) for f in range(2)]
        (ref/'settings.json').write_text(json.dumps(settings));(ref/'splits.json').write_text(json.dumps(splits))
        data=dict(raw_graphs={g.organoid_str:g for g in cohort},all_marker_names=markers,settings=settings)
        save_bundle(ref/'cohort',data,splits=splits);recs=[];out.mkdir();save_bundle(out/'cohort',data,splits=splits)
        (out/'splits.json').write_text(json.dumps(splits))
        for split in splits:
            f=split['fold'];folds={}
            for target in [0,1]:
                gs=copy.deepcopy(cohort)
                if target:
                    for g in gs:g.y=g.y*.3+.01
                prepared=prepare_fold(gs,dict(train_indices=list(range(f,12,2)),val_indices=list(range(1-f,12,2))),global_features=[],residualize=True,target_scaling='identity',
                    baseline_kwargs=dict(max_epochs=1,patience=1,hidden_dim=4,num_workers=0,train_kwargs=dict(device='cpu')))
                prepared['marker_names']=markers;folds[target]=prepared
            save_bundle(ref/f'inputs/fold_{f}',folds[0],splits=split)
            model=GINCurvature(2,hidden_dim=8,num_layers=2,global_dim=0,norm='layer',dropout=0.)
            rec=dict(key=f'gin_f{f}',name='gin',depth=2,hidden_dim=8,fold=f,seed=42,rate=0.,subset='all',signal='intact',bundle=f'models/gin_f{f}',input_bundle=f'inputs/fold_{f}')
            save_bundle(ref/rec['bundle'],dict(model=model,record=rec),splits=split);recs.append(rec)
            for name,indices in dict(gaussian=[0],mean=[1],joint=[0,1]).items():save_bundle(out/f'inputs/fold_{f}_{name}',combine_target_folds(folds,indices),splits=split)
        pd.DataFrame(recs).to_json(ref/'models.json',orient='records')
        cells=[c.source for c in nbformat.read(ROOT/'experiments/training/coupled_fate_training.ipynb',4).cells if c.cell_type=='code']
        ns={};exec(cells[0],ns);exec(cells[1],ns)
        ns.update(REFERENCE_RUN=str(ref),SHOW_PROGRESS=False,RESUME_RUN=None,CONCURRENT_FITS=2)
        ns['SETTINGS'].update(n_splines=4,pair_radius=1,min_pair_organoids=1,pair_ridge_grid=[.001],starts=[(.2,3.)],max_iterations=180,tolerance=1e-4)
        ns['training_run_path']=lambda *a,**k:out
        with patch('matplotlib.pyplot.show',side_effect=lambda:plt.close('all')):
            exec(cells[2],ns)
            (out/'settings.json').write_text(json.dumps(ns['SETTINGS']))
            ns['provenance']={}
            for c in cells[4:]:exec(c,ns)
        self.assertEqual(len(json.loads((out/'models.json').read_text())),48)
        cells=[c.source for c in nbformat.read(ROOT/'experiments/benchmarks/coupled_fate_evaluation.ipynb',4).cells if c.cell_type=='code']
        ns={};exec(cells[0],ns);exec(cells[1],ns)
        ns.update(TRAINING_RUN=str(out),REFERENCE_MODEL='gin',REFERENCE_WIDTH=8,N_POINTS=5,BOOTSTRAP_DRAWS=20)
        with patch('matplotlib.pyplot.show',side_effect=lambda:plt.close('all')):
            for c in cells[2:]:exec(c,ns)
        self.assertTrue((ns['OUTPUT_DIR']/'complete.json').exists())
        self.assertEqual(set(ns['scores'].target),{0,1})
        self.assertEqual(len(ns['audits']),48)

if __name__=='__main__':unittest.main()
