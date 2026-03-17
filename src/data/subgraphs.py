from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph
import numpy as np


def build_ego_subgraphs_for_graph(
    g: Data,
    num_hops: int = 2,
    max_centers: int | None = None,
    rng: np.random.Generator | None = None,
) -> list[Data]:
    if rng is None:
        rng = np.random.default_rng()

    N = g.x.size(0)

    # Start with Python ints, not NumPy ints
    all_centers = list(range(N))

    if max_centers is not None and max_centers < N:
        # rng.choice returns NumPy scalars; convert to Python ints
        chosen = rng.choice(all_centers, size=max_centers, replace=False)
        centers = [int(c) for c in chosen]
    else:
        centers = all_centers

    subs: list[Data] = []
    for c in centers:
        # c is guaranteed to be a Python int here
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
) -> list[Data]:
    rng = np.random.default_rng(seed)
    all_subs: list[Data] = []
    for g in graphs:
        subs = build_ego_subgraphs_for_graph(
            g,
            num_hops=num_hops,
            max_centers=max_centers_per_graph,
            rng=rng,
        )
        all_subs.extend(subs)
    return all_subs
