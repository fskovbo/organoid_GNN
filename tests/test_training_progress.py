"""Progress reflects real early stopping without changing training or RNG state."""
import contextlib
import copy
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from src.training.progress import TrainingProgress, report_epoch, log_epochs
from src.training.loop import TrainConfig, train
from src.training.masking import MaskTrainingConfig, train_mask_model
from src.models.gnn import GINCurvature
from src.models.fate_masking import make_mask_model


class TrainingProgressTests(unittest.TestCase):
    def test_counts_table_copy_and_single_display(self):
        with tempfile.TemporaryDirectory() as tmp:
            with TrainingProgress(1,total_copies=1,total_baselines=1,log_path=Path(tmp)/'progress.csv') as progress:
                with progress.task(kind='baseline',model='baseline',fold=0):
                    report_epoch(dict(epoch=1,max_epochs=5,metric='Validation MAE',value=.4,
                        best_value=.4,best_epoch=1,bad_epochs=0,patience=2))
                self.assertEqual(progress.completed['model'],0)
                with progress.task(kind='copied',model='gin',depth=2,fold=0):pass
                with progress.task(model='ring',depth=2,fold=0,signal='intact',seed=42):
                    self.assertFalse(log_epochs())
                    report_epoch(dict(epoch=3,max_epochs=10,metric='Validation MAE',value=.3,
                        best_value=.2,best_epoch=2,bad_epochs=1,patience=4))
                    self.assertEqual(progress.widgets['epoch'].value,3)
                    self.assertIn('Patience used: 1/4',progress.widgets['detail'].value)
                self.assertEqual(progress.widgets['overall'].value,1)
                self.assertEqual(progress.widgets['table'].value.count('<table'),1)
            self.assertEqual(progress.state,'Complete')
            table=pd.read_csv(Path(tmp)/'progress.csv')
            self.assertEqual(table.kind.tolist(),['baseline','copied','model'])
            self.assertEqual(table.epochs.tolist(),[1,0,3])
            self.assertEqual(table.best_epoch.iloc[-1],2)
            progress.close()
        self.assertTrue(log_epochs())  # No stale progress context outside tasks.

    def test_interruption_and_failure_do_not_complete_task(self):
        for failure in [KeyboardInterrupt(), RuntimeError('checkpoint write failed')]:
            progress=TrainingProgress(1,show=False)
            with self.assertRaises(type(failure)):
                with progress:
                    with progress.task(model='gin',fold=2):raise failure
            self.assertEqual(progress.completed['model'],0)
            self.assertEqual(progress.rows[0]['status'],'interrupted' if isinstance(failure,KeyboardInterrupt) else 'failed')
            self.assertIn('1 remaining',progress.status_html)
            self.assertTrue(log_epochs())

    def test_standard_loop_reports_actual_patience_and_quiet_logging(self):
        model=GINCurvature(n_markers=2,hidden_dim=4,num_layers=0,dropout=0.,norm='none')
        graph=Data(x=torch.eye(2),y=torch.zeros(2),edge_index=torch.tensor([[0,1],[1,0]]))
        config=TrainConfig(device='cpu',max_epochs=10,patience=2)
        events=[]
        # Alternating training/validation passes: improvement at epoch 2, stop at epoch 4.
        passes=[(1.,1.),(1.,.8),(1.,1.),(1.,.5),(1.,1.),(1.,.7),(1.,1.),(1.,.6)]
        output=io.StringIO()
        with patch('src.training.loop.epoch_pass',side_effect=passes),contextlib.redirect_stdout(output):
            with TrainingProgress(1,show=False) as progress:
                with progress.task(model='gin'):
                    _,_,history=train(model,[graph],[graph],config,epoch_callback=events.append)
        self.assertEqual(len(history['val_mae']),4)
        self.assertEqual([e['bad_epochs'] for e in events],[0,0,1,2])
        self.assertEqual(events[-1]['best_epoch'],2)
        self.assertEqual(output.getvalue(),'')
        self.assertEqual(progress.rows[0]['epochs'],4)

    def test_mask_loop_reports_inner_mse(self):
        from src.models.size_models import seed_all
        seed_all(3)
        settings=dict(HIDDEN_DIM=4,NUM_LAYERS=0,FILM_HIDDEN_DIM=4,DROPOUT=0.,NORM='none',RESIDUAL=True,
            MODEL_GLOBAL_FEATURES={'gin_film_size':['log_num_cells']},LR=.001,WEIGHT_DECAY=0.,
            EDGE_LOSS_WEIGHT=0.,EDGE_LOSS_PARAMS={})
        model=make_mask_model(settings,['A','B'])
        graph=Data(x=torch.eye(2),y=torch.zeros(2),edge_index=torch.tensor([[0,1],[1,0]]),global_feat=torch.zeros(1,1))
        config=MaskTrainingConfig(rates=(0.,.1),max_epochs=8,patience=2,batch_size=1)
        events=[]
        with patch('src.training.masking.intact_validation_mse',side_effect=[.4,.2,.3,.3]):
            with TrainingProgress(1,show=False) as progress:
                with progress.task(model='masking',rate=.1):
                    _,history=train_mask_model(model,[graph],[graph],settings,config,rate=.1,seed=3,
                                               device='cpu',epoch_callback=events.append)
        self.assertEqual(len(history),4)
        self.assertEqual(events[-1]['metric'],'Inner intact MSE (transformed)')
        self.assertEqual(events[-1]['best_epoch'],2)
        self.assertEqual(events[-1]['bad_epochs'],2)

    def test_reporting_preserves_actual_training_results(self):
        base=GINCurvature(n_markers=2,hidden_dim=4,num_layers=1,dropout=.2,norm='none')
        graph=Data(x=torch.eye(2),y=torch.tensor([.1,-.2]),edge_index=torch.tensor([[0,1],[1,0]]))
        config=TrainConfig(device='cpu',max_epochs=2,patience=3,batch_size=1)
        torch.manual_seed(81)
        ordinary,metrics_a,history_a=train(copy.deepcopy(base),[graph],[graph],config,verbose=False)
        torch.manual_seed(81)
        with TrainingProgress(1,show=False) as progress:
            with progress.task(model='gin'):
                monitored,metrics_b,history_b=train(copy.deepcopy(base),[graph],[graph],config)
        self.assertEqual(metrics_a,metrics_b)
        self.assertEqual(history_a,history_b)
        for key,weight in ordinary.state_dict().items():
            torch.testing.assert_close(weight,monitored.state_dict()[key],rtol=0,atol=0)


if __name__=='__main__':unittest.main()
