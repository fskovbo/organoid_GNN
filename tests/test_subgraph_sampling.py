import numpy as np
import torch
from torch_geometric.data import Data

from src.data.subgraph_sampling import sample_subgraphs_population


def _make_subgraph(graph_idx, center_idx, *, hop1_target=False, hop2_target=False):
    x = torch.zeros((3, 2), dtype=torch.float32)
    x[center_idx, 0] = 1.0
    if hop1_target:
        x[1, 1] = 1.0
    if hop2_target:
        x[2, 1] = 1.0

    return Data(
        x=x,
        y=torch.zeros((3, 1), dtype=torch.float32),
        edge_index=torch.tensor(
            [[0, 1, 1, 2], [1, 0, 2, 1]],
            dtype=torch.long,
        ),
        center_idx=center_idx,
        orig_center=graph_idx * 100 + center_idx,
        graph_idx=graph_idx,
        organoid_str=f"organoid_{graph_idx}",
    )


def _population_fixture():
    subgraphs = []
    for graph_idx, n_cells in enumerate([2, 4, 6]):
        for cell_idx in range(n_cells):
            subgraphs.append(
                _make_subgraph(
                    graph_idx,
                    0,
                    hop1_target=(cell_idx % 2 == 0),
                    hop2_target=(cell_idx % 3 == 0),
                )
            )
    return subgraphs


def test_cell_population_sample_has_uniform_weights():
    subgraphs = _population_fixture()
    sampled, info = sample_subgraphs_population(
        subgraphs,
        max_subgraphs=6,
        weighting="cell",
        seed=7,
    )

    assert len(sampled) == 6
    assert info["n_population"] == 6
    np.testing.assert_allclose(info["population_weights"], np.full(6, 1 / 6))
    np.testing.assert_allclose(info["sample_weights"].sum(), 1.0)


def test_cell_population_sample_does_not_require_organoid_metadata():
    subgraphs = [
        Data(
            x=torch.zeros((1, 2), dtype=torch.float32),
            y=torch.zeros((1, 1), dtype=torch.float32),
            edge_index=torch.empty((2, 0), dtype=torch.long),
            center_idx=0,
        )
        for _ in range(4)
    ]
    sampled, info = sample_subgraphs_population(
        subgraphs,
        max_subgraphs=2,
        weighting="cell",
        seed=5,
    )

    assert len(sampled) == 2
    np.testing.assert_allclose(info["population_weights"], [0.5, 0.5])


def test_organoid_population_sample_gives_each_organoid_equal_total_weight():
    subgraphs = _population_fixture()
    sampled, info = sample_subgraphs_population(
        subgraphs,
        max_subgraphs=9,
        weighting="organoid",
        seed=3,
    )

    weights_by_graph = {}
    for union_idx in info["population_indices"]:
        graph_idx = int(sampled[int(union_idx)].graph_idx)
        weights_by_graph[graph_idx] = (
            weights_by_graph.get(graph_idx, 0.0)
            + info["sample_weights"][int(union_idx)]
        )

    assert set(weights_by_graph) == {0, 1, 2}
    np.testing.assert_allclose(list(weights_by_graph.values()), np.full(3, 1 / 3))


def test_marker_topups_are_separate_from_population_draw():
    subgraphs = _population_fixture()
    sampled, info = sample_subgraphs_population(
        subgraphs,
        max_subgraphs=4,
        weighting="organoid",
        seed=11,
        marker_names=["center", "target"],
        k_hops=2,
        min_marker_count_per_hop=5,
    )

    assert info["n_population"] == 4
    assert len(sampled) == 4
    assert len(info["marker_sample_source_indices_by_hop"]["target"][1]) == 5
    assert len(info["marker_sample_source_indices_by_hop"]["target"][2]) == 5
    np.testing.assert_allclose(info["population_weights"].sum(), 1.0)
    np.testing.assert_allclose(
        info["marker_sample_weights_by_hop"]["target"][1].sum(),
        1.0,
    )
    np.testing.assert_allclose(
        info["marker_sample_weights_by_hop"]["target"][2].sum(),
        1.0,
    )

    population_sources = set(info["population_source_indices"].tolist())
    for hop in (1, 2):
        topups = set(
            info["marker_topup_source_indices_by_hop"]["target"][hop].tolist()
        )
        assert topups.isdisjoint(population_sources)
