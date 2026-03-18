import copy
import torch


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
