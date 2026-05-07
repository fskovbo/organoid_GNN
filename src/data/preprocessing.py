import copy
from collections import deque
from numbers import Integral

import torch
import numpy as np
from torch_geometric.utils import degree


def remove_marker_features(
    graphs,
    markers,
    *,
    marker_names=None,
    x_attr="x",
    inplace=False,
    return_marker_names=False,
):
    """
    Remove marker-feature columns from graph node feature matrices.

    Parameters
    ----------
    graphs : list[Data]
        PyG graphs with node feature matrix ``g.x`` by default.
    markers : int | str | sequence[int | str]
        Marker columns to remove. Integers are interpreted as feature indices.
        Strings require ``marker_names`` and are resolved by name.
    marker_names : list[str] or None
        Optional full marker-name list corresponding to the feature columns.
    x_attr : str
        Graph attribute holding the node feature matrix.
    inplace : bool
        If False, returns shallow copies of graphs.
    return_marker_names : bool
        If True, return ``(graphs_out, marker_names_out)``.

    Returns
    -------
    graphs_out : list[Data]
        Graphs with the selected marker columns removed from ``x_attr``.
    marker_names_out : list[str], optional
        Updated marker-name list, returned when ``return_marker_names=True``.
    """

    if isinstance(markers, (Integral, str)):
        markers = [markers]
    else:
        markers = list(markers)

    if len(markers) == 0:
        graphs_out = graphs if inplace else [copy.copy(g) for g in graphs]
        if return_marker_names:
            return graphs_out, None if marker_names is None else list(marker_names)
        return graphs_out

    marker_names_out = None
    name_to_idx = None
    if marker_names is not None:
        marker_names = list(marker_names)
        name_to_idx = {}
        for i, name in enumerate(marker_names):
            if name in name_to_idx:
                raise ValueError(f"Duplicate marker name {name!r} in marker_names")
            name_to_idx[name] = i

    remove_indices = []
    for marker in markers:
        if isinstance(marker, str):
            if name_to_idx is None:
                raise ValueError("String marker selection requires marker_names.")
            if marker not in name_to_idx:
                raise KeyError(f"Unknown marker name {marker!r}")
            idx = name_to_idx[marker]
        else:
            idx = int(marker)

        if idx < 0:
            if marker_names is None:
                raise IndexError(
                    "Negative marker indices require marker_names so they can be resolved safely."
                )
            idx += len(marker_names)

        remove_indices.append(idx)

    remove_indices = sorted(set(remove_indices))

    if marker_names is not None:
        n_markers = len(marker_names)
        bad = [idx for idx in remove_indices if idx < 0 or idx >= n_markers]
        if bad:
            raise IndexError(
                f"Marker indices out of range for marker_names length {n_markers}: {bad}"
            )
        remove_set = set(remove_indices)
        marker_names_out = [
            name for i, name in enumerate(marker_names) if i not in remove_set
        ]

    graphs_out = graphs if inplace else [copy.copy(g) for g in graphs]

    for gi, g in enumerate(graphs_out):
        if not hasattr(g, x_attr):
            raise ValueError(f"Graph {gi} has no attribute '{x_attr}'")

        x = getattr(g, x_attr)
        if x is None:
            raise ValueError(f"Graph {gi} has '{x_attr}=None'")
        if x.ndim != 2:
            raise ValueError(
                f"Graph {gi} attribute '{x_attr}' must be 2-D, got shape {tuple(x.shape)}"
            )

        n_features = int(x.shape[1])
        bad = [idx for idx in remove_indices if idx < 0 or idx >= n_features]
        if bad:
            raise IndexError(
                f"Marker indices out of range for graph {gi} with {n_features} features: {bad}"
            )
        if len(remove_indices) >= n_features:
            raise ValueError(
                f"Cannot remove all {n_features} feature columns from graph {gi}."
            )

        remove_set = set(remove_indices)
        keep_indices = [i for i in range(n_features) if i not in remove_set]

        if torch.is_tensor(x):
            keep = torch.as_tensor(keep_indices, dtype=torch.long, device=x.device)
            x_new = x.index_select(dim=1, index=keep).contiguous()
        else:
            x_arr = np.asarray(x)
            x_new = np.ascontiguousarray(x_arr[:, keep_indices])

        setattr(g, x_attr, x_new)

    if return_marker_names:
        return graphs_out, marker_names_out

    return graphs_out


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


def _build_undirected_adjacency(edge_index, n_nodes):
    adjacency = [set() for _ in range(n_nodes)]
    src = edge_index[0].detach().cpu().numpy()
    dst = edge_index[1].detach().cpu().numpy()

    for u, v in zip(src, dst):
        u = int(u)
        v = int(v)
        if u == v:
            continue
        adjacency[u].add(v)
        adjacency[v].add(u)

    return [sorted(neigh) for neigh in adjacency]


def _nodes_within_hops(adjacency, center, max_hops):
    distances = {}
    queue = deque([(int(center), 0)])
    seen = {int(center)}

    while queue:
        node, dist = queue.popleft()
        if dist == max_hops:
            continue
        for nbr in adjacency[node]:
            if nbr in seen:
                continue
            seen.add(nbr)
            next_dist = dist + 1
            distances[nbr] = next_dist
            queue.append((nbr, next_dist))

    return distances


def _outlier_shell_depths(outlier, adjacency):
    """Return outlier-cluster depths so boundary outliers are processed first."""
    depth = np.full(outlier.shape[0], np.inf, dtype=float)
    queue = deque()

    for i in np.where(outlier)[0]:
        if any(not outlier[j] for j in adjacency[int(i)]):
            depth[int(i)] = 0.0
            queue.append(int(i))

    while queue:
        node = queue.popleft()
        for nbr in adjacency[node]:
            if outlier[nbr] and np.isinf(depth[nbr]):
                depth[nbr] = depth[node] + 1.0
                queue.append(nbr)

    # Fully isolated outlier components have no non-outlier boundary. Process
    # them last and let the normal fallback policy handle under-supported fits.
    finite_depth = np.isfinite(depth)
    fill_depth = np.nanmax(depth[finite_depth]) + 1.0 if finite_depth.any() else 0.0
    depth[np.isinf(depth)] = fill_depth
    return depth


def _outlier_severity(value, lo, hi):
    if not np.isfinite(value):
        return np.inf
    if value < lo:
        return lo - value
    if value > hi:
        return value - hi
    return 0.0


def _quadratic_hop_interpolation(
    distances,
    values,
    *,
    ridge=1e-3,
):
    distances = np.asarray(distances, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    valid = np.isfinite(distances) & np.isfinite(values)
    distances = distances[valid]
    values = values[valid]

    if values.size == 0:
        return None
    if values.size == 1:
        return float(values[0])

    # Fit y(d) = a*d^2 + b*d + c and evaluate at d=0, so c is the
    # interpolated center value. Distances are only 1..2 here, so a small ridge
    # term stabilizes the otherwise underdetermined quadratic component.
    X = np.column_stack([distances ** 2, distances, np.ones_like(distances)])
    weights = 1.0 / np.maximum(distances, 1.0)
    Xw = X * np.sqrt(weights)[:, None]
    yw = values * np.sqrt(weights)

    penalty = np.diag([ridge, ridge, 0.0])
    lhs = Xw.T @ Xw + penalty
    rhs = Xw.T @ yw

    try:
        coef = np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        coef = np.linalg.lstsq(lhs, rhs, rcond=None)[0]

    return float(coef[2])



def interpolate_target_outliers_from_neighbors(
    graphs,
    *,
    target_indices=None,
    clip_quantiles=(0.001, 0.999),
    min_neighbors=1,
    max_hops=2,
    interpolation="quadratic",
    ridge=1e-3,
    reject_farther_outliers=True,
    fallback="neighbor_mean",
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
        Minimum finite non-outlier or already-interpolated neighbors required
        within max_hops for interpolation.
    max_hops : int
        Maximum graph distance used for interpolation. The upgraded quadratic
        interpolation is designed for max_hops=2.
    interpolation : {"quadratic", "mean"}
        Local interpolation method. "quadratic" fits target value as a function
        of hop distance and evaluates it at the outlier cell.
    ridge : float
        Small regularization used by the quadratic fit.
    reject_farther_outliers : bool
        If True, reject interpolated values that are further outside the global
        valid range than the original outlier value.
    fallback : {"neighbor_mean", "global_median", "node_median", "keep"}
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

    if fallback not in {"neighbor_mean", "global_median", "node_median", "keep"}:
        raise ValueError(
            "fallback must be one of: "
            "'neighbor_mean', 'global_median', 'node_median', 'keep'"
        )
    if interpolation not in {"quadratic", "mean"}:
        raise ValueError("interpolation must be one of: 'quadratic', 'mean'")
    if max_hops < 1:
        raise ValueError("max_hops must be >= 1")

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
        "max_hops": max_hops,
        "interpolation": interpolation,
        "reject_farther_outliers": reject_farther_outliers,
        "n_replaced": {t: 0 for t in target_indices},
        "n_outliers": {t: 0 for t in target_indices},
        "n_fallback": {t: 0 for t in target_indices},
        "n_rejected_interpolation": {t: 0 for t in target_indices},
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

        adjacency = _build_undirected_adjacency(g.edge_index, n_nodes)
        hop_neighborhoods = [
            _nodes_within_hops(adjacency, i, max_hops)
            for i in range(n_nodes)
        ]

        for t in target_indices:
            lo, hi = thresholds[t]

            vals = y2[:, t].detach().cpu().numpy().astype(float)
            finite = np.isfinite(vals)
            outlier = finite & ((vals < lo) | (vals > hi))

            info["n_outliers"][t] += int(outlier.sum())

            non_outlier = finite & (~outlier)
            global_median = float(np.median(Y[np.isfinite(Y[:, t]), t]))
            current_vals = vals.copy()
            valid_for_fit = non_outlier.copy()
            outlier_nodes = np.where(outlier)[0]
            shell_depth = _outlier_shell_depths(outlier, adjacency)
            processing_order = sorted(
                outlier_nodes,
                key=lambda idx: (shell_depth[int(idx)], int(idx)),
            )

            for i in processing_order:
                hop_distances = hop_neighborhoods[int(i)]
                candidate_nodes = np.asarray(list(hop_distances.keys()), dtype=int)
                neighbor_fallback_values = np.asarray([], dtype=float)

                if candidate_nodes.size > 0:
                    good = valid_for_fit[candidate_nodes] & np.isfinite(
                        current_vals[candidate_nodes]
                    )
                    neighbor_fallback_values = current_vals[candidate_nodes][good]

                    if int(good.sum()) >= min_neighbors:
                        fit_nodes = candidate_nodes[good]
                        fit_distances = np.asarray(
                            [hop_distances[int(node)] for node in fit_nodes],
                            dtype=np.float64,
                        )
                        fit_values = current_vals[fit_nodes]
                        if interpolation == "quadratic":
                            replacement = _quadratic_hop_interpolation(
                                fit_distances,
                                fit_values,
                                ridge=ridge,
                            )
                        else:
                            replacement = float(np.mean(fit_values))
                    else:
                        replacement = None
                else:
                    replacement = None

                if replacement is not None:
                    original_severity = _outlier_severity(vals[int(i)], lo, hi)
                    replacement_severity = _outlier_severity(replacement, lo, hi)
                    if (
                        reject_farther_outliers
                        and replacement_severity > original_severity
                    ):
                        replacement = None
                        info["n_rejected_interpolation"][t] += 1

                if replacement is None:
                    info["n_fallback"][t] += 1

                    if fallback == "global_median":
                        replacement = global_median
                    elif fallback == "neighbor_mean":
                        if neighbor_fallback_values.size == 0:
                            continue
                        replacement = float(np.mean(neighbor_fallback_values))
                    elif fallback == "node_median":
                        good_vals = current_vals[valid_for_fit]
                        replacement = float(np.median(good_vals)) if good_vals.size else global_median
                    elif fallback == "keep":
                        continue

                y_new[i, t] = y_new.new_tensor(replacement)
                current_vals[int(i)] = replacement
                valid_for_fit[int(i)] = True
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
                f"fallback={info['n_fallback'][t]} | "
                f"rejected_interp={info['n_rejected_interpolation'][t]}"
            )

    return graphs_out, info
