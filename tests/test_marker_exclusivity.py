import itertools
import numpy as np
from src.data.marker_exclusivity import exclusive_markers, EXCLUSIVITY_RULES


def test_every_binary_pattern_is_exclusive_and_preserves_unmarked():
    for names in [['Agr2','AldoB','Chroma','KI67','LGR5','Lysozyme','Serotonin'],list(EXCLUSIVITY_RULES)]:
        original=np.array(list(itertools.product([0.,1.],repeat=len(names))),dtype=np.float32)
        before=original.copy();result=exclusive_markers(original,names)
        assert np.all(result.sum(1)<=1)
        np.testing.assert_array_equal(result.sum(1)>0,original.sum(1)>0)
        np.testing.assert_array_equal(original,before)
        assert result.shape==original.shape and result.dtype==original.dtype
        assert np.all(result<=original)
        np.testing.assert_array_equal(exclusive_markers(result,names),result)


def test_order_uses_updated_state_and_ignores_absent_markers():
    names=['Chroma','Mucin 2']
    # Chroma clears first; Mucin 2 then sees Chroma absent and survives.
    np.testing.assert_array_equal(exclusive_markers([[1,1]],names),[[0,1]])
    np.testing.assert_array_equal(exclusive_markers([[1,1,1]],['LGR5','Agr2','Lysozyme']),[[0,0,1]])
    np.testing.assert_array_equal(exclusive_markers([[1,1]],['KI67','LGR5']),[[0,1]])


def test_full_fate_control_changes_only_selected_source_and_preserves_input():
    import torch
    from torch_geometric.data import Data
    from unittest.mock import patch
    from src.analysis.pseudotime import evaluate_size_ablation
    graph=Data(x=torch.tensor([[1.,0.],[1.,1.],[0.,1.]]),global_feat=torch.tensor([[0.]]))
    original=graph.x.clone();captured=[]
    def predict(graphs,*args,**kwargs):
        captured.extend(g.x.clone() for g in graphs)
        return np.zeros(len(graphs)),np.zeros(len(graphs))
    class Identity:
        def inverse_distribution(self,y,z,log_var=None):return y,z,log_var
    case=dict(subgraph_index=0,source_node=1,source_marker=0,observed_n=100.)
    with patch('src.analysis.pseudotime.predict_subgraph_center_distribution',predict):
        evaluate_size_ablation([graph],[case],None,Identity(),size_center=0,size_scale=1,count=100)
        evaluate_size_ablation([graph],[dict(case,source_markers_to_zero=[0,1])],None,Identity(),size_center=0,size_scale=1,count=100)
    assert torch.equal(captured[1][1],torch.tensor([0.,1.]))
    assert torch.equal(captured[3][1],torch.tensor([0.,0.]))
    assert torch.equal(captured[3][[0,2]],original[[0,2]])
    assert torch.equal(graph.x,original)
