import numpy as np
import torch
from torch_geometric.data import Data
from src.data.subgraphs import build_ego_subgraphs_for_graph
from src.data.subgraph_sampling import sample_subgraphs_coverage, sample_coverage_indices, _subgraph_marker_coverage


def test_compact_coverage_matches_legacy_selection():
    rng = np.random.default_rng(71)
    ids = rng.integers(0, 4, 120)
    x = torch.tensor(np.eye(4)[ids], dtype=torch.float32)
    edges = [(i, (i+j)%120) for i in range(120) for j in (-2,-1,1,2)]
    graph = Data(x=x, edge_index=torch.tensor(edges).T, y=torch.zeros(120))
    subs = build_ego_subgraphs_for_graph(graph, num_hops=2)
    center, rings = _subgraph_marker_coverage(subs, 2)
    for seed in range(5):
        for budget in (12, 60, None):
            _, old = sample_subgraphs_coverage(subs, list('ABCD'), 2, budget,
                min_center_count=5, min_pair_count=3, seed=seed)
            indices, new = sample_coverage_indices(center, rings, list('ABCD'), budget,
                min_center_count=5, min_pair_count=3, seed=seed)
            np.testing.assert_array_equal(old['selected_indices'], indices)
            for key in ('center_covered', 'pair_covered', 'center_target', 'pair_target'):
                np.testing.assert_array_equal(old[key], new[key])


def test_fate_coverage_builds_only_selected_egos():
    from src.analysis.interventions.fate_edits import sample_fate_contexts
    from unittest.mock import patch
    import src.analysis.interventions.fate_edits as edits
    n=90
    x=torch.tensor(np.eye(3)[np.arange(n)%3],dtype=torch.float32)
    x[::7]=0
    edges=torch.tensor([(i,(i+j)%n) for i in range(n) for j in (-1,1)]).T
    graph=Data(x=x,y=torch.zeros(n),edge_index=edges,organoid_str='toy')
    augmented=graph.clone();augmented.x=torch.cat([x,(x.sum(1)==0).float()[:,None]],1)
    old_subs=build_ego_subgraphs_for_graph(augmented,num_hops=2)
    _,old=sample_subgraphs_coverage(old_subs,['A','B','C','Unassigned'],2,20,min_center_count=2,min_pair_count=2,seed=17)
    with patch.object(edits,'build_ego_subgraphs_for_graph',wraps=build_ego_subgraphs_for_graph) as build:
        subs,cases,info=sample_fate_contexts([graph],['A','B','C'],2,scheme='coverage',max_subgraphs=20,min_center_count=2,min_pair_count=2,seed=17,return_info=True)
        assert sum(len(c.kwargs['centers']) for c in build.call_args_list)==20
    np.testing.assert_array_equal([int(s.orig_center) for s in subs],old['selected_indices'])
    assert not cases.empty and len(subs)==20
