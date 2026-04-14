import copy
import torch
import numpy as np
from torch_geometric.utils import degree


def weight_targets_by_patch_area(
    graphs,
    *,
    inplace=False,
    area_key="cell_patch_area",
    y_attr="y",
):
    """
    Multiply node targets y by corresponding cell_patch_area.

    Parameters
    ----------
    graphs : list[Data]
        PyG graphs with g.y and g.meta[area_key]
    inplace : bool
        If False, returns shallow copies of graphs
    area_key : str
        Metadata key containing per-node areas
    y_attr : str
        Attribute name of target (default: 'y')

    Returns
    -------
    graphs_out : list[Data]
    """

    graphs_out = graphs if inplace else [copy.copy(g) for g in graphs]

    for i, g in enumerate(graphs_out):
        if not hasattr(g, y_attr):
            raise ValueError(f"Graph {i} has no attribute '{y_attr}'")

        y = getattr(g, y_attr)
        md = getattr(g, "meta", None)

        if md is None:
            raise ValueError(f"Graph {i} has no metadata (g.meta)")

        if area_key not in md:
            raise KeyError(
                f"Graph {i} missing '{area_key}' in metadata"
            )

        area = np.asarray(md[area_key], dtype=np.float32).reshape(-1)

        # convert y safely
        y_np = y.detach().cpu().numpy() if hasattr(y, "detach") else np.asarray(y)

        if y_np.shape[0] != area.shape[0]:
            raise ValueError(
                f"Shape mismatch in graph {i}: "
                f"len(y)={y_np.shape[0]} vs len(area)={area.shape[0]}"
            )

        y_weighted = y_np * area

        # write back (preserve tensor type if needed)
        if hasattr(y, "new_tensor"):
            setattr(g, y_attr, y.new_tensor(y_weighted))
        else:
            setattr(g, y_attr, y_weighted)

    return graphs_out


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


def robust_zscore_organoid_targets(
    graphs,
    inplace=False,
    store_stats=True,
    median_attr="organoid_y_median",
    mad_attr="organoid_y_mad",
    scale_consistency=True,
    eps=1e-8,
):
    """
    Replace node targets y with robust z-score per organoid:
        (y - median) / MAD

    Parameters
    ----------
    graphs : list[Data]
    inplace : bool
        If False, returns shallow copies.
    store_stats : bool
        Store median and MAD on graph for later reconstruction.
    median_attr : str
    mad_attr : str
    scale_consistency : bool
        If True, multiply MAD by 1.4826 to match std for Gaussian data.
    eps : float
        Minimum MAD to avoid division by zero.

    Returns
    -------
    graphs_out : list[Data]
    """

    graphs_out = graphs if inplace else [copy.copy(g) for g in graphs]

    for g in graphs_out:

        if not hasattr(g, "y"):
            raise ValueError("Graph missing attribute 'y'")

        y = g.y

        if y.numel() == 0:
            continue

        med = torch.median(y)

        mad = torch.median(torch.abs(y - med))

        if scale_consistency:
            mad = mad * 1.4826

        mad = torch.clamp(mad, min=eps)

        g.y = (y - med) / mad

        if store_stats:
            setattr(g, median_attr, med)
            setattr(g, mad_attr, mad)

    return graphs_out