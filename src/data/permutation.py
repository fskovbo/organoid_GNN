import copy
import torch

from src.data.subgraphs import build_ego_subgraphs_for_center_specs
from src.data.subgraph_sampling import center_specs_from_subgraphs
from src.graph.neighborhood import compute_hop_rings


def permute_graph_targets_within_organoid(graphs, inplace=False, generator=None):
    """
    Permute y within each graph independently.
    Preserves per-graph target distribution but destroys node-target alignment.
    """
    graphs_out = graphs if inplace else [copy.deepcopy(g) for g in graphs]

    for g in graphs_out:
        if not hasattr(g, "y"):
            raise ValueError("Graph is missing attribute 'y'")
        n = g.y.shape[0]
        perm = torch.randperm(n, generator=generator, device=g.y.device)
        g.y = g.y[perm]

    return graphs_out


def permute_graph_features_within_organoid(graphs, inplace=False, generator=None):
    """
    Permute rows of x within each graph independently.
    Preserves graph topology and per-graph marker prevalence, but destroys
    which cell carries which marker vector.
    """
    graphs_out = graphs if inplace else [copy.deepcopy(g) for g in graphs]

    for g in graphs_out:
        if not hasattr(g, "x"):
            raise ValueError("Graph is missing attribute 'x'")
        n = g.x.shape[0]
        perm = torch.randperm(n, generator=generator, device=g.x.device)
        g.x = g.x[perm]

    return graphs_out


def permute_ego_neighborhood_within_rings(subgraphs, k_hops, inplace=False, generator=None):
    """
    Permute non-center node features only within each exact hop ring.
    Preserves ring-wise composition but destroys finer arrangement.
    """
    subs_out = subgraphs if inplace else [copy.deepcopy(g) for g in subgraphs]

    for g in subs_out:
        c = int(g.center_idx)
        rings = compute_hop_rings(g.edge_index, c, k_hops)

        x_new = g.x.clone()

        for nodes in rings[1:]:   # exclude center ring
            if len(nodes) <= 1:
                continue
            idx = torch.tensor(nodes, device=g.x.device, dtype=torch.long)
            perm = torch.randperm(idx.numel(), generator=generator, device=g.x.device)
            x_new[idx] = g.x[idx[perm]]

        g.x = x_new

    return subs_out


def build_center_preserving_permuted_subgraphs(
    graphs,
    sampled_subgraphs,
    num_hops,
    permute_graphs_fn,
    seed=0,
    inplace=False,
):
    """
    Build a permuted set of ego-subgraphs for previously sampled centers.

    Workflow:
      1) extract sampled centers from original sampled_subgraphs
      2) permute full organoid graphs with permute_graphs_fn
      3) rebuild ego-subgraphs at the same centers
      4) restore the center node feature row from the original graph

    Parameters
    ----------
    graphs : list[Data]
        Original full graphs
    sampled_subgraphs : list[Data]
        Previously sampled original ego-subgraphs
    num_hops : int
        Ego radius to rebuild
    permute_graphs_fn : callable
        Function like permute_graph_features_within_organoid(graphs, inplace=False, generator=...)
    seed : int
    inplace : bool
        Passed to the permutation function if supported

    Returns
    -------
    perm_subs : list[Data]
        Permuted ego-subgraphs with center features restored
    """
    center_specs = center_specs_from_subgraphs(sampled_subgraphs)

    # build generator if the permutation function supports it
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))

    try:
        permuted_graphs = permute_graphs_fn(graphs, inplace=inplace, generator=gen)
    except TypeError:
        permuted_graphs = permute_graphs_fn(graphs, inplace=inplace)

    perm_subs = build_ego_subgraphs_for_center_specs(
        permuted_graphs,
        center_specs=center_specs,
        num_hops=num_hops,
    )

    # restore center features from the original full graph
    for sub in perm_subs:
        gi = int(sub.graph_idx)
        c_orig = int(sub.orig_center)
        c_sub = int(sub.center_idx)

        sub.x[c_sub] = graphs[gi].x[c_orig]

    return perm_subs