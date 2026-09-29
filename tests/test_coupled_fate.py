"""Coupled solves, fitted gradients, customizable pairs and portable artifacts."""
import copy
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from scipy.special import logit
from torch_geometric.data import Data, Batch
from src.models.coupled_fate import CoupledFateCurvature,neighbor_average_matrix,pair_response
from src.training.coupled_fit import graph_samples,ProfileObjective,fit_coupled_fate
from src.artifacts.bundle import save_bundle,load_bundle
from src.artifacts.checkpoints import build_model
from src.training.preparation import physical_fate_fold
from src.data.target_transforms import IdentityTransform


def graphs():
    result=[]
    for k in range(12):
        n=12+k*3;ids=np.random.default_rng(80+k).integers(0,3,n)
        result.append(Data(x=torch.tensor(np.eye(3)[ids,:2],dtype=torch.float32),
            edge_index=torch.tensor(np.array([np.arange(n),np.roll(np.arange(n),1)])),
            y=torch.zeros(n,dtype=torch.float64),organoid_str=str(k),full_num_cells=float(n)))
    return result


class CoupledTests(unittest.TestCase):
    def setUp(self):
        self.graphs=graphs();self.samples=graph_samples(self.graphs,2)
        self.model=CoupledFateCurvature(2,radius=2,n_splines=4).configure(self.samples)
        self.model.weights.copy_(torch.tensor(np.random.default_rng(0).normal(0,.1,self.model.weights.shape)))
        self.model.gamma_logits.fill_(logit(.4/.95));self.model.is_fitted.fill_(True)
        for s in self.samples:s['y']=self.model.predict_sample(s)

    def test_self_consistency_components_constant_and_isolates(self):
        s=self.samples[0];terms=self.model.contributions(s);g=self.model.gamma([s['N']])[0]
        np.testing.assert_allclose(terms['prediction'],(1-g)*terms['local']+g*s['transition']@terms['prediction'],atol=1e-14)
        np.testing.assert_allclose(terms['prediction'],terms['propagated_center']+terms['propagated_pairs'].sum(axis=(1,2)),atol=1e-14)
        np.testing.assert_allclose(self.model.propagate(np.full(len(s['y']),.3),s['transition'],s['N']),.3,atol=1e-14)
        w=neighbor_average_matrix(3,np.array([[0,0,1,1],[1,1,0,1]]))
        np.testing.assert_array_equal(w.toarray(),[[0,1,0],[1,0,0],[0,0,1]])
        self.assertAlmostEqual(self.model.propagate(np.array([1.,2.,3.]),w,20)[2],3.)

    def test_no_target_input_and_batched_permutation(self):
        s=self.samples[0];edited=dict(s,y=np.full_like(s['y'],np.nan))
        np.testing.assert_array_equal(self.model.predict_sample(s),self.model.predict_sample(edited))
        g=self.graphs[0];p=torch.randperm(len(g.x));inverse=torch.argsort(p)
        np.testing.assert_allclose(self.model(g.x,g.edge_index,data=g)[0][0][p],self.model(g.x[p],inverse[g.edge_index],data=g)[0][0],atol=1e-14)
        batch=Batch.from_data_list(self.graphs[:2]);pred=self.model(batch.x,batch.edge_index,batch)[0][0]
        np.testing.assert_allclose(pred,np.concatenate([self.model.predict_sample(s) for s in self.samples[:2]]),atol=1e-14)

    def test_pair_constraints_and_custom_responses(self):
        co=self.model.coefficients([15,35]);c=self.model.center_reference.numpy();s=self.model.source_reference.numpy()
        np.testing.assert_allclose(np.einsum('nrab,a->nrb',co['P'],c),0,atol=1e-14)
        np.testing.assert_allclose(np.einsum('nrab,rb->nra',co['P'],s),0,atol=1e-14)
        for spec in [dict(response='linear'),dict(response='presence'),dict(response='hill',half=.1),dict(response='threshold',threshold=.2,width=.03)]:
            out=pair_response([0,.1,.5,1],spec);self.assertEqual(out[0],0);self.assertAlmostEqual(out[-1],1);self.assertTrue((np.diff(out)>=0).all())
        custom=CoupledFateCurvature(2,2,4,pair_responses=[dict(center=0,source=1,hop=1,response='presence')]).configure(self.samples)
        ids=np.array([0,1,0]);counts=np.zeros((3,2,3));counts[:,0,:]=[1,1,8]
        raw=custom.response_features(ids,counts,False)
        np.testing.assert_allclose(raw[:,0,1],[1,.1,1])
        np.testing.assert_allclose(custom.local_design(ids,counts),custom.local_design(ids,counts*np.array([2,9])[None,:,None]))
        with self.assertRaises(ValueError):pair_response([.2],dict(response='unrecognized'))

    def test_profile_gradient_and_uncoupled_limit(self):
        m=CoupledFateCurvature(2,0,4).configure(self.samples)
        objective=ProfileObjective(m,self.samples,ridge=1e-4,pair_ridge=1e-3,smoothness=.001,gamma_smoothness=.01,gamma_ridge=.0001)
        eta=np.array([-.5,-.3,-.1,.2]);_,grad=objective(eta);eps=1e-5
        numeric=np.array([(objective(eta+eps*np.eye(4)[k])[0]-objective(eta-eps*np.eye(4)[k])[0])/(2*eps) for k in range(4)])
        np.testing.assert_allclose(grad,numeric,rtol=1e-4,atol=1e-8)
        m=CoupledFateCurvature(2,0,4,coupling=False).configure(self.samples);m.weights.fill_(.25);m.is_fitted.fill_(True)
        np.testing.assert_allclose(m.predict_sample(self.samples[0]),.25)
        self.assertEqual(m.coefficients([20])['P'].shape,(1,0,3,3))

    def test_optimizer_recovers_coupling_at_physical_target_scale(self):
        from src.training.coupled_fit import fit_partition
        samples=graph_samples(self.graphs,0)
        truth=CoupledFateCurvature(2,0,4).configure(samples)
        truth.weights[:]=truth.weights.new_tensor([[.02]*4,[-.03]*4,[.01]*4])
        truth.gamma_logits.fill_(logit(.7/.95));truth.is_fitted.fill_(True)
        for sample in samples:sample['y']=truth.predict_sample(sample)
        fitted=CoupledFateCurvature(2,0,4).configure(samples)
        fit_partition(fitted,samples,ridge=1e-10,pair_ridge=1e-10,smoothness=1e-10,
            gamma_smoothness=0,gamma_ridge=0,gamma_start=.15,max_iterations=200,tolerance=1e-10)
        np.testing.assert_allclose(fitted.gamma([12,25,45]),.7,atol=.005)

    def test_measured_neighbors_gradient_and_no_self_target(self):
        m=CoupledFateCurvature(2,1,4,neighbor_curvature='measured').configure(self.samples)
        obj=ProfileObjective(m,self.samples,ridge=1e-4,pair_ridge=1e-3,smoothness=.001,gamma_smoothness=.01,gamma_ridge=.0001)
        eta=np.array([-.5,-.3,-.1,.2]);_,grad=obj(eta);eps=1e-5
        numeric=np.array([(obj(eta+eps*np.eye(4)[k])[0]-obj(eta-eps*np.eye(4)[k])[0])/(2*eps) for k in range(4)])
        np.testing.assert_allclose(grad,numeric,rtol=1e-4,atol=1e-8)
        obj(eta);sample=self.samples[0];pred=m.predict_sample(sample)
        edited=dict(sample,y=sample['y'].copy());edited['y'][0]+=100
        self.assertAlmostEqual(m.predict_sample(edited)[0],pred[0])
        self.assertGreater(np.max(np.abs(m.predict_sample(edited)[1:]-pred[1:])),1)
        terms=m.contributions(sample)
        np.testing.assert_allclose(pred,terms['propagated_center']+terms['propagated_pairs'].sum(axis=(1,2))+terms['coupling_term'])
        alone=dict(N=20,identity=np.array([0]),counts=np.zeros((1,2,3)),transition=neighbor_average_matrix(1,np.zeros((2,0),int)),y=np.array([1000.]))
        self.assertAlmostEqual(m.predict_sample(alone)[0],m.coefficients([20])['a'][0,0])
        graph=self.graphs[0].clone();graph.y=torch.tensor(sample['y'])
        np.testing.assert_allclose(m(graph.x,graph.edge_index,graph)[0][0],pred)
        with tempfile.TemporaryDirectory() as directory:
            save_bundle(Path(directory)/'model',dict(model=m),splits={})
            loaded=load_bundle(Path(directory)/'model')['model']
            self.assertEqual(loaded.neighbor_curvature,'measured')
            np.testing.assert_array_equal(loaded.predict_sample(sample),pred)

    def test_historical_baseline_roundoff_preserves_physical_targets(self):
        graph=self.graphs[0].clone();graph.y=torch.linspace(-.1,.1,len(graph.x))
        n=len(graph.x);oid=graph.organoid_str
        baseline=np.float32(.00126313)
        offset=(graph.y.numpy()-(graph.y.numpy()-baseline)).astype(np.float32)
        source=dict(marker_names=['LGR5','KI67'],baseline=object(),residualized=True,
            baseline_predictions={oid:np.full(n,baseline)},baseline_offsets={oid:offset},
            transform=IdentityTransform(),groups={'train':[graph]})
        restored=physical_fate_fold(source)
        np.testing.assert_allclose(restored['groups']['train'][0].y.numpy()+restored['baseline_offsets'][oid],
            graph.y.numpy().astype(float)+offset,atol=1e-16)
        self.assertEqual(np.ptp(restored['baseline_offsets'][oid]),0)
        np.testing.assert_array_equal(source['baseline_offsets'][oid],offset)
        source['baseline_offsets'][oid]=offset.copy();source['baseline_offsets'][oid][0]+=.001
        with self.assertRaises(ValueError):physical_fate_fold(source)

    def test_fit_membership_and_checkpoint_roundtrip(self):
        model,meta,trials=fit_coupled_fate(self.samples[:9],model_settings=dict(n_markers=2,radius=0,n_splines=4,coupling=True),
            penalties=[dict(ridge=1e-5,pair_ridge=1e-4,smoothness=.001)],gamma_starts=[.2],max_iterations=100)
        self.assertEqual(set(meta['fit_organoids']),{str(k) for k in range(9)})
        self.assertFalse(set(meta['inner_train'])&set(meta['inner_validation']))
        self.assertTrue(meta['optimizer']['success'])
        self.assertAlmostEqual(np.exp(model.log_count_range[1].item()),self.samples[8]['N'])
        with tempfile.TemporaryDirectory() as directory:
            save_bundle(Path(directory)/'model',dict(model=model),splits={})
            restored=load_bundle(Path(directory)/'model')['model']
            np.testing.assert_array_equal(restored.predict_sample(self.samples[-1]),model.predict_sample(self.samples[-1]))
        old=build_model(dict(module='src.models.fate_spline',**{'class':'FateSplineCurvature'},kwargs=dict(n_markers=2,radius=0,variant='center')))
        self.assertIn('legacy.fate_interactions',type(old).__module__)


if __name__=='__main__':unittest.main()
