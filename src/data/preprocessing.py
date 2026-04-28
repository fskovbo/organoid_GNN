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

        y_weighted = y_np * (area[:, None] if getattr(y_np, "ndim", 1) == 2 else area)

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

        mean_val = torch.mean(y, dim=0) if y.ndim == 2 else torch.mean(y)

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

        med = torch.median(y, dim=0).values if y.ndim == 2 else torch.median(y)

        mad = torch.median(torch.abs(y - med), dim=0).values if y.ndim == 2 else torch.median(torch.abs(y - med))

        if scale_consistency:
            mad = mad * 1.4826

        mad = torch.clamp(mad, min=eps)

        g.y = (y - med) / mad

        if store_stats:
            setattr(g, median_attr, med)
            setattr(g, mad_attr, mad)

    return graphs_out



def interpolate_target_outliers_from_neighbors(
    graphs,
    *,
    target_indices=None,
    clip_quantiles=(0.001, 0.999),
    min_neighbors=1,
    fallback="global_median",
    inplace=False,
    report=True,
):
    """
    Replace extreme target outliers with neighbor-interpolated values.

    Parameters
    ----------
    graphs : list[torch_geometric.data.Data]
        Graphs with g.y and g.edge_index.
    target_indices : list[int] or None
        Which target columns to process. None = all targets.
    clip_quantiles : tuple[float, float]
        Outliers are values outside these global quantiles per target.
    min_neighbors : int
        Minimum finite non-outlier neighbors required for interpolation.
    fallback : {"global_median", "node_median", "keep"}
        What to do if an outlier node has too few usable neighbors.
    inplace : bool
        If False, returns copied graphs.
    report : bool
        Print outlier counts.

    Returns
    -------
    graphs_out : list[Data]
    info : dict
    """

    if fallback not in {"global_median", "node_median", "keep"}:
        raise ValueError("fallback must be one of: 'global_median', 'node_median', 'keep'")

    graphs_out = graphs if inplace else [copy.deepcopy(g) for g in graphs]

    # -------------------------
    # Collect all targets
    # -------------------------
    ys = []
    graph_slices = []
    offset = 0

    for g in graphs_out:
        y = g.y.detach().cpu()
        if y.ndim == 1:
            y = y[:, None]
        ys.append(y)

        n = y.shape[0]
        graph_slices.append(slice(offset, offset + n))
        offset += n

    Y = torch.cat(ys, dim=0).numpy().astype(float)  # (total_nodes, target_dim)
    n_total, target_dim = Y.shape

    if target_indices is None:
        target_indices = list(range(target_dim))
    else:
        target_indices = list(target_indices)

    # -------------------------
    # Global outlier thresholds
    # -------------------------
    q_lo, q_hi = clip_quantiles
    thresholds = {}

    for t in target_indices:
        vals = Y[:, t]
        vals = vals[np.isfinite(vals)]
        lo = np.quantile(vals, q_lo)
        hi = np.quantile(vals, q_hi)
        thresholds[t] = (lo, hi)

    # -------------------------
    # Replace outliers graphwise
    # -------------------------
    info = {
        "clip_quantiles": clip_quantiles,
        "thresholds": thresholds,
        "n_replaced": {t: 0 for t in target_indices},
        "n_outliers": {t: 0 for t in target_indices},
        "n_fallback": {t: 0 for t in target_indices},
    }

    for gi, g in enumerate(graphs_out):
        y = g.y.detach().clone()
        original_was_1d = y.ndim == 1

        if y.ndim == 1:
            y2 = y[:, None]
        else:
            y2 = y

        y_new = y2.clone()
        n_nodes = y2.shape[0]

        edge_index = g.edge_index.detach().cpu()
        src = edge_index[0].numpy()
        dst = edge_index[1].numpy()

        neighbors = [[] for _ in range(n_nodes)]
        for u, v in zip(src, dst):
            u = int(u)
            v = int(v)
            if u != v:
                neighbors[u].append(v)

        for t in target_indices:
            lo, hi = thresholds[t]

            vals = y2[:, t].detach().cpu().numpy().astype(float)
            finite = np.isfinite(vals)
            outlier = finite & ((vals < lo) | (vals > hi))

            info["n_outliers"][t] += int(outlier.sum())

            non_outlier = finite & (~outlier)
            global_median = float(np.median(Y[np.isfinite(Y[:, t]), t]))

            for i in np.where(outlier)[0]:
                neigh = np.asarray(neighbors[i], dtype=int)

                if neigh.size > 0:
                    neigh_vals = vals[neigh]
                    good = np.isfinite(neigh_vals)

                    # Prefer non-outlier neighbors
                    good = good & non_outlier[neigh]

                    if good.sum() >= min_neighbors:
                        replacement = float(np.mean(neigh_vals[good]))
                    else:
                        replacement = None
                else:
                    replacement = None

                if replacement is None:
                    info["n_fallback"][t] += 1

                    if fallback == "global_median":
                        replacement = global_median
                    elif fallback == "node_median":
                        good_vals = vals[non_outlier]
                        replacement = float(np.median(good_vals)) if good_vals.size else global_median
                    elif fallback == "keep":
                        continue

                y_new[i, t] = y_new.new_tensor(replacement)
                info["n_replaced"][t] += 1

        g.y = y_new[:, 0].contiguous() if original_was_1d else y_new.contiguous()

    if report:
        print("=== Target outlier interpolation report ===")
        print(f"Quantiles: {clip_quantiles}")
        for t in target_indices:
            lo, hi = thresholds[t]
            print(
                f"target {t}: "
                f"range=[{lo:.6g}, {hi:.6g}] | "
                f"outliers={info['n_outliers'][t]} | "
                f"replaced={info['n_replaced'][t]} | "
                f"fallback={info['n_fallback'][t]}"
            )

    return graphs_out, info