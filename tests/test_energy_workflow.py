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


def run_training(notebook,run,**overrides):
    cells=[c.source for c in nbformat.read(ROOT/'experiments/training'/notebook,4).cells if c.cell_type=='code']
    ns={};exec(cells[0],ns);exec(cells[1],ns)
    ns['SETTINGS'].update(n_folds=1,timepoints=['day4'],blacklist=False,interpolate_outliers=True,
        pair_ridge_grid=[.001],blas_threads=2,min_pair_organoids=1,**overrides)
    ns.update(DEVICE='cpu',SHOW_PROGRESS=False,RUN_TRAINING=True,RUN_DIR=run,
        load_graph_dataset_from_dir=lambda *a,**k:synthetic_graphs(),load_aux_metadata_for_dir=lambda *a:{},
        attach_metadata_to_graphs=lambda *a,**k:None,load_marker_names_from_dir=lambda *a:MARKERS,distance_regions=fake_regions)
    with patch('matplotlib.pyplot.show',side_effect=lambda:plt.close('all')),patch('matplotlib.figure.Figure.savefig'):
        for cell in cells[2:]:exec(cell,ns)
    return ns


def run_evaluation(notebook,run,**overrides):
    cells=[c.source for c in nbformat.read(ROOT/'experiments/benchmarks'/notebook,4).cells if c.cell_type=='code']
    ns={};exec(cells[0],ns);exec(cells[1],ns);ns.update(TRAINING_RUN=str(run),**overrides)
    with patch('matplotlib.pyplot.show',side_effect=lambda:plt.close('all')),patch('matplotlib.figure.Figure.savefig'):
        for cell in cells[2:]:exec(cell,ns)
    return ns


class EnergyWorkflowTests(unittest.TestCase):
    def test_conditional_scan_and_regional_averaging(self):
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as d:
            run=Path(d)/'scan'
            ns=run_training('energy_model_training.ipynb',run,gamma_grid=[0.,.5])
            self.assertEqual(len(ns['records']),4)
            split=ns['membership'][0]
            self.assertEqual(set(split['spherical_validation']),{'synthetic_14','synthetic_15'})
            prepared=load_bundle(run/'inputs/fold_0_mean')
            self.assertFalse(set(prepared['inner_train'])&set(split['validation']))
            self.assertFalse(set(prepared['inner_validation'])&set(split['spherical_validation']))
            ev=run_evaluation('energy_model_evaluation.ipynb',run)
            self.assertLess(max(row['max_difference'] for row in ev['verification']),1e-7)
            scores=ev['metrics']
            self.assertEqual(set(scores[scores.region=='total'].cohort),{'val'})
            self.assertEqual(set(scores[scores.region=='spherical_or_no_detected_crypt'].cohort),{'val','spherical'})
            self.assertNotIn('excluded_boundary',set(scores.region))
            self.assertEqual(len(ev['selected']),4)
            for mode in ['mean_only','standardized']:
                graphs=ns['target_groups'](prepared,mode)['val']
                for graph in graphs:
                    mom=prepared['moments'][graph.organoid_str]
                    self.assertAlmostEqual(float(graph.y.mean()),0,places=12)
                    if mode=='standardized':self.assertAlmostEqual(float(graph.y.std(correction=0)),1,places=12)


if __name__=='__main__':unittest.main()
