import numpy as np
import torch
from torch import nn
from torch_geometric.data import Data

from src.analysis.perturbation import compute_perturbation_influence_maps


class _DummyModel(nn.Module):
    def forward(self, x, edge_index, data=None):
        mu = x[:, 0] + 0.25 * x[:, 1]
        logvar = torch.zeros_like(mu)
        return (mu, logvar), x


class _VarianceModel(nn.Module):
    def forward(self, x, edge_index, data=None):
        mu = torch.zeros(x.shape[0], dtype=x.dtype, device=x.device)
        logvar = x[:, 0]
        return (mu, logvar), x


def test_single_ablation_records_exact_physical_source_cell():
    subgraph = Data(
        x=torch.tensor(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [0.0, 1.0],
            ]
        ),
        y=torch.zeros((3, 1)),
        edge_index=torch.tensor(
            [[0, 1, 0, 2], [1, 0, 2, 0]],
            dtype=torch.long,
        ),
        center_idx=0,
        orig_center=100,
        orig_nodes=torch.tensor([100, 205, 309]),
        graph_idx=7,
        organoid_str="organoid_7",
    )

    result = compute_perturbation_influence_maps(
        [subgraph],
        _DummyModel(),
        ["center", "source"],
        k_hops=1,
        mode="single",
        max_subgraphs=None,
        batch_size=1,
        normalize_by="cases",
        return_case_effects=True,
        source_markers=["source"],
    )

    assert len(result["case_effects"]) == 1
    case = result["case_effects"][0]
    assert case["source_node"] == 1
    assert case["source_node_indices"] == [1]
    assert case["orig_source_node"] == 205
    assert case["orig_source_node_indices"] == [205]
    assert case["delta_variance"] == 0.0
    assert case["delta_mse"] is not None


def test_center_inclusive_ablation_records_hop_zero():
    subgraph = Data(
        x=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        y=torch.zeros((2, 1)),
        edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
        center_idx=0,
        orig_center=10,
        orig_nodes=torch.tensor([10, 11]),
        graph_idx=0,
    )

    result = compute_perturbation_influence_maps(
        [subgraph],
        _DummyModel(),
        ["center", "source"],
        k_hops=1,
        mode="single",
        max_subgraphs=None,
        batch_size=1,
        normalize_by="cases",
        return_case_effects=True,
        source_markers=["center"],
        include_center=True,
    )

    assert result["hops"] == [0, 1]
    assert [case["hop"] for case in result["case_effects"]] == [0]
    assert result["case_effects"][0]["source_node"] == 0
    assert result["case_effects"][0]["orig_source_node"] == 10
    assert result["case_effects"][0]["delta_mu"] == -1.0
    assert result["case_effects"][0]["delta_mse"] == -1.0


def test_variance_change_is_computed_from_exponentiated_log_variance():
    subgraph = Data(
        x=torch.tensor([[1.0], [0.0]]),
        y=torch.zeros((2, 1)),
        edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
        center_idx=0,
        orig_nodes=torch.tensor([0, 1]),
        graph_idx=0,
    )

    result = compute_perturbation_influence_maps(
        [subgraph],
        _VarianceModel(),
        ["marker"],
        k_hops=1,
        mode="single",
        max_subgraphs=None,
        batch_size=1,
        normalize_by="cases",
        return_case_effects=True,
        include_center=True,
    )

    case = result["case_effects"][0]
    assert np.isclose(case["delta_logvar"], -1.0)
    assert np.isclose(case["delta_variance"], 1.0 - np.e)
