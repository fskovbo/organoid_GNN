import copy
import torch


def subtract_organoid_mean_curvature(
    graphs,
    inplace=False,
    store_mean=True,
    mean_attr_name="organoid_mean_curvature",
):
    """
    For each graph, subtract mean(y) from all node targets.

    Parameters
    ----------
    graphs : list[torch_geometric.data.Data]
    inplace : bool
        If False, returns shallow copies.
    store_mean : bool
        Whether to store the removed mean on the graph.
    mean_attr_name : str
        Attribute name to store the mean.

    Returns
    -------
    graphs_out : list[Data]
    """

    graphs_out = graphs if inplace else [copy.copy(g) for g in graphs]

    for g in graphs_out:

        if not hasattr(g, "y"):
            raise ValueError("Graph missing target 'y'")

        y = g.y

        if y.numel() == 0:
            continue

        mean_val = torch.mean(y)

        g.y = y - mean_val

        if store_mean:
            setattr(g, mean_attr_name, mean_val)

    return graphs_out