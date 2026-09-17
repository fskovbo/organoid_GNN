"""Checks for the scientific estimand and graph context of niche diagnostics."""
import numpy as np
import torch
from torch_geometric.data import Data
from src.analysis.niche_hypotheses import crypt_membership, source_state, rings, predict
from src.data.subgraphs import build_ego_subgraphs_for_graph
from src.models.gnn import SizeFiLMGINCurvature


def test_full_graph_matches_complete_recipient_receptive_field_and_no_mutation():
    torch.manual_seed(16)
    x=torch.ones(7,7)
    edges=torch.tensor([[i,i+1] for i in range(6)]+[[i+1,i] for i in range(6)]).T
    graph=Data(x=x,edge_index=edges,y=torch.zeros(7,1))
    model=SizeFiLMGINCurvature(7,hidden_dim=16,num_layers=2,global_dim=1,film_hidden_dim=4,
        dropout=0,norm='batch',residual=True).eval()
    before=graph.x.clone()
    full=predict(model,[graph],[(0,(),7),(0,((0,5),),7)],[0.,0.],'cpu')
    sub=build_ego_subgraphs_for_graph(graph,num_hops=4,centers=[0])[0]
    local=int(torch.nonzero(sub.orig_nodes==2).item())
    extracted=predict(model,[sub],[(0,(),7),(0,((0,5),),7)],[0.,0.],'cpu')
    np.testing.assert_allclose([v[2] for v in full],[v[local] for v in extracted],atol=1e-6)
    assert torch.equal(before,graph.x)
    assert graph.x[2,4]==1 # recipient LGR5 identity remains intact
    np.testing.assert_allclose(full[0][3:],full[1][3:],atol=1e-6) # no path in two layers


def test_source_states_keep_double_positives_separate():
    x=np.zeros((4,7)); x[0,0]=1; x[1,[0,5]]=1; x[2,5]=1
    assert source_state(x,'Agr2','Lysozyme').tolist()==['precursor_only','double_positive','mature_only','neither']


def test_crypt_membership_uses_projected_final_labels_and_handles_none():
    projection=np.array([12,4,99,12])
    assert crypt_membership(projection,[[4],[12,15]]).tolist()==[1,0,-1,1]
    assert crypt_membership(projection,[]).tolist()==[-1]*4


def test_hops_exclude_center_and_do_not_double_count():
    adjacency=[{1,2},{0,2},{0,1,3},{2}]
    assert rings(adjacency,0)==[[1,2],[3]]
