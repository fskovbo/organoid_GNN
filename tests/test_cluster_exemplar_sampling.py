import numpy as np
import torch
from torch_geometric.data import Data

from src.analysis.cluster_analysis import (
    _marker_presence_within_hops,
    build_cluster_exemplar_subgraphs,
    build_marker_enriched_cluster_exemplar_subgraphs,
)
from src.analysis.motif_clustering import (
    ClusteringResult,
    EmbeddingExtractionResult,
)


def _single_node_graph(markers, organoid_id):
    return Data(
        x=torch.tensor([markers], dtype=torch.float32),
        y=torch.zeros((1, 1), dtype=torch.float32),
        edge_index=torch.empty((2, 0), dtype=torch.long),
        organoid_str=organoid_id,
    )


def _extraction(x_markers, graph_index, local_node_index):
    n_nodes = len(graph_index)
    return EmbeddingExtractionResult(
        embeddings_full=np.zeros((n_nodes, 2), dtype=float),
        embeddings_local=np.zeros((n_nodes, 2), dtype=float),
        y_true=np.arange(n_nodes, dtype=float),
        y_pred=np.arange(n_nodes, dtype=float),
        log_var=np.zeros(n_nodes, dtype=float),
        x_markers=np.asarray(x_markers, dtype=float),
        global_features=None,
        graph_index=np.asarray(graph_index, dtype=int),
        local_node_index=np.asarray(local_node_index, dtype=int),
        center_only=False,
        organoid_ids=[f"organoid_{i}" for i in graph_index],
    )


def _clustering(labels, probabilities):
    probabilities = np.asarray(probabilities, dtype=float)
    return ClusteringResult(
        labels=np.asarray(labels, dtype=int),
        probabilities=probabilities,
        model=None,
        scaler=None,
        pca=None,
        embeddings_used=np.zeros((len(labels), 2), dtype=float),
        cluster_centers=None,
    )


def test_marker_enriched_selector_covers_sparse_cluster_marker_before_filling():
    graphs = [
        _single_node_graph([1.0, 0.0], "organoid_0"),
        _single_node_graph([1.0, 0.0], "organoid_1"),
        _single_node_graph([1.0, 1.0], "organoid_2"),
        _single_node_graph([1.0, 0.0], "organoid_3"),
    ]
    extraction = _extraction(
        x_markers=[[1.0, 0.0], [1.0, 0.0], [1.0, 1.0], [1.0, 0.0]],
        graph_index=[0, 1, 2, 3],
        local_node_index=[0, 0, 0, 0],
    )
    clustering = _clustering(
        labels=[0, 0, 0, 1],
        probabilities=[
            [0.99, 0.01],
            [0.95, 0.05],
            [0.60, 0.40],
            [0.10, 0.90],
        ],
    )

    exemplars, diagnostics = build_marker_enriched_cluster_exemplar_subgraphs(
        graphs,
        extraction,
        clustering,
        marker_names=["Common", "Serotonin"],
        top_k_per_cluster=2,
        num_hops=0,
        min_global_positive_nodes=1,
        min_cluster_positive_nodes=1,
        min_enrichment_ratio=1.2,
    )

    assert [item["row_index"] for item in exemplars[0]] == [2, 0]
    assert exemplars[0][0]["selection_reason"] == "marker_coverage"
    assert exemplars[0][0]["target_marker"] == "Serotonin"
    assert exemplars[0][1]["selection_reason"] == "confidence_fill"
    assert len({item["graph_index"] for item in exemplars[0]}) == 2

    marker_diagnostics = diagnostics["marker_diagnostics"]
    serotonin = marker_diagnostics[
        (marker_diagnostics["cluster"] == 0)
        & (marker_diagnostics["marker"] == "Serotonin")
    ].iloc[0]
    assert serotonin["qualifies"]
    assert serotonin["covered"]


def test_marker_coverage_includes_neighboring_cells_within_radius():
    graph = Data(
        x=torch.tensor(
            [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]],
            dtype=torch.float32,
        ),
        y=torch.zeros((3, 1), dtype=torch.float32),
        edge_index=torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
        organoid_str="organoid_0",
    )
    extraction = _extraction(
        x_markers=graph.x.numpy(),
        graph_index=[0, 0, 0],
        local_node_index=[0, 1, 2],
    )

    radius_zero = _marker_presence_within_hops(
        [graph],
        extraction,
        num_hops=0,
        n_markers=2,
    )
    radius_one = _marker_presence_within_hops(
        [graph],
        extraction,
        num_hops=1,
        n_markers=2,
    )

    assert not radius_zero[0, 1]
    assert radius_one[0, 1]
    assert radius_one[2, 1]


def test_existing_confidence_selector_still_ranks_probability_with_unique_graphs():
    graphs = [
        _single_node_graph([1.0, 0.0], "organoid_0"),
        _single_node_graph([1.0, 0.0], "organoid_1"),
        _single_node_graph([1.0, 0.0], "organoid_2"),
    ]
    extraction = _extraction(
        x_markers=[[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]],
        graph_index=[0, 1, 2],
        local_node_index=[0, 0, 0],
    )
    clustering = _clustering(
        labels=[0, 0, 1],
        probabilities=[[0.8, 0.2], [0.9, 0.1], [0.1, 0.9]],
    )

    exemplars = build_cluster_exemplar_subgraphs(
        graphs,
        extraction,
        clustering,
        top_k_per_cluster=2,
        num_hops=0,
    )

    assert [item["row_index"] for item in exemplars[0]] == [1, 0]
    assert [item["row_index"] for item in exemplars[1]] == [2]
