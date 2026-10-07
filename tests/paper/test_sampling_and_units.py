import numpy as np
import torch
from torch_geometric.data import Data
from src.analysis.interventions.fate_edits import sample_random_topup_contexts, evaluate_fate_edit
from src.data.target_transforms import IdentityTransform


def test_random_topup_preserves_random_pool_and_reports_shortfalls():
    x=torch.tensor([[1.,0.],[0.,1.],[0.,0.],[1.,0.],[0.,1.]])
    edges=torch.tensor([[0,1,1,2,2,3,3,4],[1,0,2,1,3,2,4,3]])
    graph=Data(x=x.clone(),y=torch.zeros(5),edge_index=edges,organoid_str='toy')
    subs,cases,support=sample_random_topup_contexts([graph],['A','B'],2,random_centers=1,minimum_count=3,seed=3)
    assert len({int(s.orig_center) for s in subs})==len(subs)
    assert cases[cases.sample_origin=='random'].orig_center.nunique()==1
    assert not cases.duplicated(['organoid_str','orig_center','hop','source_marker']).any()
    np.testing.assert_array_equal(support.selected_count,np.minimum(support.available,3).where(support.available<3,support.selected_count))
    assert (support.selected_count >= np.minimum(support.available,3)).all()
    assert support.loc[(support.hop==0)&(support.source_marker_name=='Unassigned'),'shortfall'].iloc[0]==2
    for c in cases.itertuples():
        assert int(subs[c.subgraph_index].orig_nodes[c.source_node])==c.orig_source_node
        if c.hop==0:assert c.orig_center==c.orig_source_node
    assert torch.equal(x,graph.x)


class ToyMaskModel(torch.nn.Module):
    def forward(self,x,edge_index,data=None):
        mu=x[:,0]+2*x[:,-1]
        return (mu,torch.zeros_like(mu)),x


def test_center_mask_errors_restore_baseline_and_leave_inputs_unchanged():
    graph=Data(x=torch.tensor([[1.,0.],[0.,1.]]),y=torch.tensor([.5,1.]),
        edge_index=torch.tensor([[0,1],[1,0]]),organoid_str='toy')
    subs,cases,_=sample_random_topup_contexts([graph],['A','B'],1,random_centers=2,minimum_count=1,hops=(0,),seed=0)
    observed=graph.clone();observed.x=torch.cat([graph.x,torch.zeros(2,1)],1)
    selection=dict(model=ToyMaskModel(),marker_names=['A','B'],groups={'val':[observed]},record={'rate':.02},
        transform=IdentityTransform(),baseline_offsets={'toy':np.array([10.,20.])})
    result=evaluate_fate_edit(selection,subs,cases,'masking',allow_center=True)
    np.testing.assert_allclose(result.delta_mse,result.edited_mse-result.intact_mse)
    np.testing.assert_allclose(result.delta_mae,result.edited_mae-result.intact_mae)
    row=result[result.orig_center==0].iloc[0]
    assert row.truth_mu==10.5 and row.intact_physical_mu==11 and row.edited_physical_mu==12
    assert row.delta_mse==2 and row.delta_mae==1
    assert torch.equal(graph.x,torch.tensor([[1.,0.],[0.,1.]]))
    assert (result.hop==0).all()
