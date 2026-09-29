"""Execute complete training and analysis workflows on small saved folds."""
import copy,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nbformat,numpy as np,pandas as pd,torch
from src.artifacts.bundle import save_bundle,graph_membership
from src.models.gnn import SizeFiLMGINCurvature
from src.training.preparation import prepare_fold
from tests.test_coupled_fate import graphs
ROOT=Path(__file__).resolve().parents[1]
class MeanEnergyNotebookTests(unittest.TestCase):
    def test_round_trip(self):
        torch.set_num_threads(2)
        root=Path(self.enterContext(tempfile.TemporaryDirectory()));ref=root/'reference';source=root/'source';out=root/'new'
        ref.mkdir();source.mkdir();gs=graphs();markers=['LGR5','KI67']
        for g in gs:
            g.meta=dict(total_surface_area=2.*len(g.x),total_volume=float(len(g.x)**1.5))
            g.y=(.02*g.x[:,0]-.01*g.x[:,1]+.003*np.log(len(g.x))).double()
        membership=[dict(fold=f,**graph_membership(train=gs[f::2],validation=gs[1-f::2])) for f in range(2)]
        settings=dict(dataset='synthetic_missing',exclusive_markers=True,baseline_features=['log_num_cells'],masked_loss_weight=.5,inner_val_fraction=.2,inner_split_seed=8123,max_epochs=2,patience=1,batch_size=3,grad_clip=2.,lr=.001,weight_decay=.0001,edge_loss_weight=.2,edge_loss_params=dict(weighted=False,alpha=2.,normalize_by='graph_std',clip_weight=4.))
        (ref/'settings.json').write_text(json.dumps(settings));(source/'settings.json').write_text(json.dumps(dict(reference_run=str(ref),dataset=settings['dataset'])))
        for r in [ref,source]:(r/'splits.json').write_text(json.dumps(membership))
        cohort=dict(raw_graphs={g.organoid_str:g for g in gs},all_marker_names=markers)
        save_bundle(source/'cohort',cohort,splits=membership);recs=[]
        for split in membership:
            f=split['fold'];data=prepare_fold(gs,dict(train_indices=list(range(f,12,2)),val_indices=list(range(1-f,12,2))),global_features=['log_num_cells'],residualize=True,target_scaling='identity',baseline_features=['log_num_cells'],baseline_kwargs=dict(max_epochs=1,patience=1,hidden_dim=4,num_workers=0,train_kwargs=dict(device='cpu')))
            data['marker_names']=markers
            save_bundle(ref/f'inputs/fold_{f}_all_observed',data,splits=split)
            save_bundle(source/f'inputs/fold_{f}_mean',data,splits=split)
            model=SizeFiLMGINCurvature(3,hidden_dim=8,num_layers=2,global_dim=1,norm='layer',dropout=0.)
            rec=dict(key=f'film_f{f}',name='film',depth=2,hidden_dim=8,fold=f,seed=42,rate=0.,bundle=f'models/film_f{f}',input_bundle=f'inputs/fold_{f}_all_observed')
            save_bundle(ref/rec['bundle'],dict(model=model,record=rec),splits=split);recs.append(rec)
        pd.DataFrame(recs).to_json(ref/'models.json',orient='records');(source/'models.json').write_text('[]')
        cells=[c.source for c in nbformat.read(ROOT/'experiments/training/mean_curvature_energy_training.ipynb',4).cells if c.cell_type=='code'];ns={}
        exec(cells[0],ns);exec(cells[1],ns);ns.update(SOURCE_RUN=str(source),RESUME_RUN=str(out),SHOW_PROGRESS=False,DEVICE='cpu')
        ns['SETTINGS'].update(variants=['center_local','learned_accommodated'],film_width=8,pair_ridge_grid=[.001],starts=[[.2,3.,1]],min_pair_organoids=1,blas_threads=2,max_iterations=180,tolerance=1e-4)
        with patch('matplotlib.pyplot.show',side_effect=lambda:plt.close('all')):
            for cell in cells[2:]:exec(cell,ns)
        self.assertEqual(len(json.loads((out/'models.json').read_text())),6)
        cells=[c.source for c in nbformat.read(ROOT/'experiments/benchmarks/mean_curvature_energy_evaluation.ipynb',4).cells if c.cell_type=='code'];ns={}
        exec(cells[0],ns);exec(cells[1],ns);ns.update(TRAINING_RUN=str(out),COMPARE_PREVIOUS=False,N_POINTS=5,BOOTSTRAP_DRAWS=20,DEVICE='cpu')
        with patch('matplotlib.pyplot.show',side_effect=lambda:plt.close('all')):
            for cell in cells[2:]:exec(cell,ns)
        self.assertTrue((out/'REPORT.md').exists());self.assertTrue((ns['OUTPUT_DIR']/'complete.json').exists())
        self.assertTrue((ns['OUTPUT_DIR']/'pair_amplitudes.png').exists())
if __name__=='__main__':unittest.main()
