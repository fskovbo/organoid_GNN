from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph
import numpy as np


def build_ego_subgraphs_for_graph(
    g: Data,
    num_hops: int = 2,
    max_centers: int | None = None,
    rng: np.random.Generator | None = None,
    centers: list[int] | None = None,
    graph_idx: int | None = None,
) -> list[Data]:
    """
    Build ego-subgraphs for selected center nodes from a single full graph.

    Each returned subgraph stores:
    - node features and targets restricted to the ego-neighborhood
    - relabeled edge_index
    - mapping back to the original graph via orig_center and orig_nodes
    """
        
    if rng is None:
        rng = np.random.default_rng()

    N = g.x.size(0)

    if centers is not None:
        centers = [int(c) for c in centers]
    else:
        all_centers = list(range(N))
        if max_centers is not None and max_centers < N:
            chosen = rng.choice(all_centers, size=max_centers, replace=False)
            centers = [int(c) for c in chosen]
        else:
            centers = all_centers

    subs: list[Data] = []
    for c in centers:
        nodes, edge_index_sub, mapping, mask = k_hop_subgraph(
            c,
            num_hops,
            g.edge_index,
            relabel_nodes=True,
            num_nodes=N,
        )

        x_sub = g.x[nodes]
        y_sub = g.y[nodes]

        sub = Data(
            x=x_sub,
            y=y_sub,
            edge_index=edge_index_sub,
        )

        if hasattr(g, "organoid_str"):
            sub.organoid_str = g.organoid_str

        if graph_idx is not None:
            sub.graph_idx = int(graph_idx)

        sub.center_idx = int(mapping.item())   # index in subgraph
        sub.orig_center = int(c)               # index in original graph
        sub.orig_nodes = nodes                 # tensor of original indices

        subs.append(sub)

    return subs


def build_ego_subgraphs_for_dataset(
    graphs: list[Data],
    num_hops: int = 2,
    max_centers_per_graph: int | None = None,
    seed: int = 0,
    centers_per_graph: list[list[int] | None] | None = None,
) -> list[Data]:
    """
    Build ego-subgraphs for all graphs in a dataset.

    Centers can be sampled randomly per graph or supplied explicitly via
    centers_per_graph. Returns one flat list of subgraphs.
    """
        
    rng = np.random.default_rng(seed)
    all_subs: list[Data] = []

    if centers_per_graph is not None and len(centers_per_graph) != len(graphs):
        raise ValueError("centers_per_graph must have same length as graphs")

    for gi, g in enumerate(graphs):
        centers = None if centers_per_graph is None else centers_per_graph[gi]

        subs = build_ego_subgraphs_for_graph(
            g,
            num_hops=num_hops,
            max_centers=max_centers_per_graph if centers is None else None,
            rng=rng,
            centers=centers,
            graph_idx=gi,
        )
        all_subs.extend(subs)

    return all_subs


def build_ego_subgraphs_for_center_specs(
    graphs: list[Data],
    center_specs: list[tuple[int, int]],
    num_hops: int,
) -> list[Data]:
    """
    Build ego-subgraphs for an explicit ordered list of (graph_idx, center_node) pairs.

    This preserves the exact order of center_specs in the returned subgraphs.
    """
    out: list[Data] = []

    for gi, c in center_specs:
        gi = int(gi)
        c = int(c)
        g = graphs[gi]

        subs = build_ego_subgraphs_for_graph(
            g,
            num_hops=num_hops,
            centers=[c],
            graph_idx=gi,
        )
        out.extend(subs)

    return out