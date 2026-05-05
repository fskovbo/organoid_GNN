"""Marker-domain curvature profile analysis for PyTorch Geometric organoid graphs.

This module implements the analysis discussed in the chat:

For one specified marker X:
    1. Find connected X-positive components.
    2. Keep all components of size >= 1 and annotate size/compactness.
    3. Compute signed graph-hop distance from the component boundary.
    4. Aggregate predicted curvature as a function of signed distance.
    5. Stratify by user-supplied component-size bins.
    6. Optionally repeat after shrink-only marker perturbations.

Important convention
--------------------
Shrinking does NOT delete graph nodes or edges. It only sets marker X to zero on
selected boundary cells, then recomputes model predictions on the same graph.
The remaining X-positive domain is kept connected by a deterministic greedy
peeling rule.
"""

import copy
from collections import deque
from dataclasses import dataclass
import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader


# -----------------------------------------------------------------------------
# Prediction helpers
# -----------------------------------------------------------------------------


def _select_target_output(arr: torch.Tensor, target_index: int | None = 0, *, name: str) -> torch.Tensor:
    """Return a 1-D per-node tensor for one selected target dimension."""
    if arr.ndim == 1:
        if target_index not in (None, 0):
            raise IndexError(f"{name} is 1-D; only target_index=0 is valid, got {target_index!r}.")
        return arr.contiguous()

    if arr.ndim == 2:
        if arr.shape[1] == 1:
            if target_index not in (None, 0):
                raise IndexError(f"{name} has one target; only target_index=0 is valid, got {target_index!r}.")
            return arr[:, 0].contiguous()
        if target_index is None:
            raise ValueError(f"{name} has shape {tuple(arr.shape)}. Pass target_index to select one target.")
        target_index = int(target_index)
        if not (0 <= target_index < arr.shape[1]):
            raise IndexError(f"target_index={target_index} out of range for {name} with shape {tuple(arr.shape)}.")
        return arr[:, target_index].contiguous()

    raise ValueError(f"Expected {name} to be 1-D or 2-D after covariance conversion, got {tuple(arr.shape)}.")


def _select_target_numpy(arr: np.ndarray, target_index: int | None = 0) -> np.ndarray:
    """Return a 1-D numpy array for one selected target dimension."""
    arr = np.asarray(arr)
    if arr.ndim == 1:
        if target_index not in (None, 0):
            raise IndexError(f"Array is 1-D; only target_index=0 is valid, got {target_index!r}.")
        return arr.astype(np.float64, copy=False)
    if arr.ndim == 2:
        if arr.shape[1] == 1:
            if target_index not in (None, 0):
                raise IndexError(f"Array has one target; only target_index=0 is valid, got {target_index!r}.")
            return arr[:, 0].astype(np.float64, copy=False)
        if target_index is None:
            raise ValueError(f"Array has shape {arr.shape}. Pass target_index to select one target.")
        return arr[:, int(target_index)].astype(np.float64, copy=False)
    return arr.reshape(arr.shape[0], -1)[:, int(target_index or 0)].astype(np.float64, copy=False)


@torch.no_grad()
def predict_graph_node_distributions(
    graphs: list,
    model: torch.nn.Module,
    *,
    device: str | None = None,
    batch_size: int = 32,
    num_workers: int = 0,
    pin_memory: bool = True,
    target_index: int | None = 0,
) -> dict[str, list[np.ndarray]]:
    """Predict node-wise mean and log-variance for each graph.

    The model is called in the same style as your existing analysis utilities:
    first as ``model(batch.x, batch.edge_index, data=batch)`` and, if that fails,
    as ``model(batch.x, batch.edge_index)``.

    Returns
    -------
    dict
        Per-graph lists with keys ``mu``, ``log_var``, ``y_true`` and ``x``.
        ``log_var`` entries are arrays of NaNs if the model returns no logvar.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = model.to(device).eval()
    loader = DataLoader(
        list(graphs),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    mu_by_graph: list[np.ndarray] = []
    lv_by_graph: list[np.ndarray] = []
    y_by_graph: list[np.ndarray] = []
    x_by_graph: list[np.ndarray] = []

    for batch in loader:
        batch = batch.to(device, non_blocking=True)

        try:
            out, _embedding = model(batch.x, batch.edge_index, data=batch)
        except TypeError:
            out, _embedding = model(batch.x, batch.edge_index)

        if isinstance(out, (tuple, list)) and len(out) == 2:
            mu, log_var = out
        else:
            mu, log_var = out, None

        if log_var is not None and getattr(log_var, "ndim", 0) == 3:
            # Convert a Cholesky/covariance-like output to marginal log-variances.
            cov = log_var @ log_var.transpose(-1, -2)
            log_var = torch.log(torch.diagonal(cov, dim1=-2, dim2=-1).clamp_min(1e-12))

        mu = _select_target_output(mu, target_index=target_index, name="mu")
        if log_var is None:
            lv = torch.full_like(mu, float("nan"))
        else:
            lv = _select_target_output(log_var, target_index=target_index, name="log_var")

        ptr = batch.ptr.detach().cpu().numpy().astype(int)
        mu_np = mu.detach().cpu().numpy().astype(np.float64)
        lv_np = lv.detach().cpu().numpy().astype(np.float64)
        x_np = batch.x.detach().cpu().numpy()

        if hasattr(batch, "y") and batch.y is not None:
            y_np = _select_target_numpy(batch.y.detach().cpu().numpy(), target_index=target_index)
        else:
            y_np = np.full(batch.num_nodes, np.nan, dtype=np.float64)

        for start, end in zip(ptr[:-1], ptr[1:]):
            mu_by_graph.append(mu_np[start:end])
            lv_by_graph.append(lv_np[start:end])
            y_by_graph.append(y_np[start:end])
            x_by_graph.append(x_np[start:end])

    return {"mu": mu_by_graph, "log_var": lv_by_graph, "y_true": y_by_graph, "x": x_by_graph}


# -----------------------------------------------------------------------------
# Graph/component helpers
# -----------------------------------------------------------------------------


def marker_index_from_name(marker: int | str, marker_names: list[str] | None = None) -> int:
    """Resolve a marker index from either an int or a marker name."""
    if isinstance(marker, (int, np.integer)):
        return int(marker)
    if marker_names is None:
        raise ValueError("marker_names must be supplied when marker is a string.")
    try:
        return list(marker_names).index(str(marker))
    except ValueError as exc:
        raise ValueError(f"Marker {marker!r} not found in marker_names.") from exc


def build_undirected_adjacency(edge_index: torch.Tensor | np.ndarray, num_nodes: int) -> list[set[int]]:
    """Build adjacency sets from a PyG edge_index, ignoring self-loops."""
    if torch.is_tensor(edge_index):
        edges = edge_index.detach().cpu().numpy()
    else:
        edges = np.asarray(edge_index)
    if edges.shape[0] != 2:
        raise ValueError(f"edge_index must have shape (2, E), got {edges.shape}.")

    adj = [set() for _ in range(int(num_nodes))]
    for u, v in edges.T:
        u = int(u)
        v = int(v)
        if u == v:
            continue
        adj[u].add(v)
        adj[v].add(u)
    return adj


def connected_components_from_nodes(nodes: object, adj: list[set[int]]) -> list[list[int]]:
    """Connected components induced by ``nodes``."""
    node_set = set(int(n) for n in nodes)
    comps: list[list[int]] = []

    while node_set:
        start = node_set.pop()
        q = deque([start])
        comp = [start]
        while q:
            u = q.popleft()
            for v in adj[u]:
                if v in node_set:
                    node_set.remove(v)
                    comp.append(v)
                    q.append(v)
        comps.append(sorted(comp))

    comps.sort(key=lambda c: (-len(c), c[0] if c else -1))
    return comps


def is_connected_node_set(nodes: object, adj: list[set[int]]) -> bool:
    """Return True if the induced subgraph on nodes is connected."""
    node_set = set(int(n) for n in nodes)
    if len(node_set) <= 1:
        return True
    start = next(iter(node_set))
    seen = {start}
    q = deque([start])
    while q:
        u = q.popleft()
        for v in adj[u]:
            if v in node_set and v not in seen:
                seen.add(v)
                q.append(v)
    return len(seen) == len(node_set)


def component_boundary_nodes(component_nodes: object, adj: list[set[int]]) -> list[int]:
    """Nodes inside a component that touch at least one node outside the component.

    If every node is internal, all component nodes are returned so that signed
    distances remain defined even for whole-graph components.
    """
    comp = set(int(n) for n in component_nodes)
    if len(comp) <= 1:
        return sorted(comp)
    boundary = sorted(u for u in comp if any(v not in comp for v in adj[u]))
    return boundary if boundary else sorted(comp)


def _bfs_distances_from_sources(
    sources: object,
    adj: list[set[int]],
    *,
    allowed_nodes: set[int] | None = None,
    max_distance: int | None = None,
) -> dict[int, int]:
    """Shortest-path distances from sources, optionally restricted to allowed nodes."""
    src = [int(s) for s in sources]
    if allowed_nodes is not None:
        allowed_nodes = set(int(n) for n in allowed_nodes)
        src = [s for s in src if s in allowed_nodes]
    if len(src) == 0:
        return {}

    dist = {s: 0 for s in src}
    q = deque(src)
    while q:
        u = q.popleft()
        du = dist[u]
        if max_distance is not None and du >= int(max_distance):
            continue
        for v in adj[u]:
            if allowed_nodes is not None and v not in allowed_nodes:
                continue
            if v not in dist:
                dist[v] = du + 1
                q.append(v)
    return dist


def signed_boundary_distances(
    component_nodes: object,
    adj: list[set[int]],
    *,
    max_outside_distance: int | None = None,
    max_inside_distance: int | None = None,
) -> dict[int, int]:
    """Compute signed distance to the component boundary.

    Convention
    ----------
    ``0``: X-positive component boundary cells.
    ``-1, -2, ...``: deeper inside the component.
    ``+1, +2, ...``: outside the component, moving away from the boundary.

    The returned dictionary contains only nodes reached within the requested
    inside/outside distance caps.
    """
    comp = set(int(n) for n in component_nodes)
    boundary = component_boundary_nodes(comp, adj)

    # Inside distances are measured within the component only.
    inside = _bfs_distances_from_sources(
        boundary,
        adj,
        allowed_nodes=comp,
        max_distance=max_inside_distance,
    )

    # Outside distances are measured in the full graph from the inside boundary.
    outside = _bfs_distances_from_sources(
        boundary,
        adj,
        allowed_nodes=None,
        max_distance=max_outside_distance,
    )

    signed: dict[int, int] = {}
    for u, d in inside.items():
        signed[int(u)] = -int(d)
    for u, d in outside.items():
        if u in comp:
            continue
        signed[int(u)] = int(d)
    return signed


def component_stats(component_nodes: object, adj: list[set[int]]) -> dict[str, float | int]:
    """Basic size and compactness statistics for one marker-positive component."""
    comp = set(int(n) for n in component_nodes)
    n = len(comp)

    internal_edges = 0
    boundary_edges = 0
    internal_degrees = []
    boundary_exposures = []

    for u in comp:
        int_deg = sum(1 for v in adj[u] if v in comp)
        out_deg = sum(1 for v in adj[u] if v not in comp)
        internal_degrees.append(int_deg)
        boundary_exposures.append(out_deg)
        boundary_edges += out_deg
        internal_edges += int_deg

    # Each internal undirected edge was counted twice.
    internal_edges //= 2
    possible_edges = n * (n - 1) / 2
    edge_density = internal_edges / possible_edges if possible_edges > 0 else np.nan
    mean_internal_degree = float(np.mean(internal_degrees)) if n else np.nan
    mean_boundary_exposure = float(np.mean(boundary_exposures)) if n else np.nan
    boundary_nodes = component_boundary_nodes(comp, adj)

    # Tree-like connected components have internal_edges = n-1. Values >1 mean
    # cycles/redundant connections; values near 1 are chain/tree-like.
    edge_surplus_ratio = internal_edges / max(n - 1, 1) if n > 1 else np.nan

    return {
        "cluster_size": int(n),
        "internal_edges": int(internal_edges),
        "edge_density": float(edge_density) if np.isfinite(edge_density) else np.nan,
        "mean_internal_degree": mean_internal_degree,
        "boundary_edges": int(boundary_edges),
        "n_boundary_nodes": int(len(boundary_nodes)),
        "boundary_fraction": float(len(boundary_nodes) / n) if n else np.nan,
        "mean_boundary_exposure": mean_boundary_exposure,
        "edge_surplus_ratio": float(edge_surplus_ratio) if np.isfinite(edge_surplus_ratio) else np.nan,
    }


def _normalize_size_bins(size_bins=None) -> list[int] | None:
    """Return sorted integer exact-size bins, or None for exact observed-size labels."""
    if size_bins is None:
        return None
    if isinstance(size_bins, (int, np.integer)):
        if int(size_bins) < 1:
            raise ValueError("size_bins as an integer must be >= 1.")
        return list(range(1, int(size_bins) + 1))
    bins = sorted({int(s) for s in size_bins})
    if any(s < 1 for s in bins):
        raise ValueError("All size_bins must be positive integers.")
    return bins


def format_size_bin(
    size: int,
    *,
    size_bins=None,
    overflow_size_bin: bool = True,
    overflow_label: str | None = None,
    other_label: str | None = None,
) -> str:
    """Convert a component size into a configurable size-bin label.

    Parameters
    ----------
    size
        Component size to label.
    size_bins
        Exact component sizes to keep as named bins. Pass ``None`` to label
        every observed size exactly. Pass an integer ``N`` for bins ``1..N``.
        Pass an iterable of integers for custom exact bins.
    overflow_size_bin
        If True and ``size_bins`` is provided, sizes larger than the largest
        exact bin are labeled as ``"{max_bin}+"`` unless ``overflow_label`` is
        supplied.
    overflow_label
        Optional custom label for the overflow bin.
    other_label
        Optional label for positive sizes that are not in ``size_bins`` and are
        not part of the overflow bin. If None, such sizes keep their exact label.
    """
    size = int(size)
    bins = _normalize_size_bins(size_bins)
    if bins is None:
        return str(size)
    if size in bins:
        return str(size)
    if len(bins) > 0 and overflow_size_bin and size > max(bins):
        return overflow_label or f"{max(bins)}+"
    return other_label if other_label is not None else str(size)


def size_bin_order_from_spec(
    df: pd.DataFrame | None = None,
    *,
    size_bins=None,
    overflow_size_bin: bool = True,
    overflow_label: str | None = None,
    size_col: str = "size_bin",
) -> list[str]:
    """Return an ordered list of size-bin labels for summaries and plots."""
    bins = _normalize_size_bins(size_bins)
    labels: list[str] = []

    if bins is not None:
        labels = [str(s) for s in bins]
        if overflow_size_bin and len(bins) > 0:
            label = overflow_label or f"{max(bins)}+"
            if df is None or (size_col in df.columns and (df[size_col].astype(str) == label).any()):
                labels.append(label)
    elif df is not None and size_col in df.columns:
        vals = [str(v) for v in pd.Series(df[size_col]).dropna().unique()]

        def _key(label: str):
            if label.endswith("+"):
                try:
                    return (int(label[:-1]), 1)
                except ValueError:
                    return (10**12, label)
            try:
                return (int(label), 0)
            except ValueError:
                return (10**12, label)

        labels = sorted(vals, key=_key)

    return labels


@dataclass(frozen=True)
class MarkerComponent:
    graph_index: int
    component_id: int
    nodes: tuple[int, ...]
    stats: dict[str, float | int]


# -----------------------------------------------------------------------------
# Observed marker-domain profile analysis
# -----------------------------------------------------------------------------


def find_marker_components_in_graph(
    graph,
    marker: int | str,
    *,
    marker_names: list[str] | None = None,
    threshold: float = 0.5,
    graph_index: int = 0,
) -> list[MarkerComponent]:
    """Find connected X-positive components for one graph and marker."""
    marker_idx = marker_index_from_name(marker, marker_names)
    x = graph.x.detach().cpu().numpy() if torch.is_tensor(graph.x) else np.asarray(graph.x)
    pos_nodes = np.where(x[:, marker_idx] > threshold)[0].astype(int).tolist()
    adj = build_undirected_adjacency(graph.edge_index, num_nodes=x.shape[0])
    comps = connected_components_from_nodes(pos_nodes, adj)

    out: list[MarkerComponent] = []
    for cid, nodes in enumerate(comps):
        stats = component_stats(nodes, adj)
        out.append(MarkerComponent(
            graph_index=int(graph_index),
            component_id=int(cid),
            nodes=tuple(int(n) for n in nodes),
            stats=stats,
        ))
    return out


def find_marker_components(
    graphs: list,
    marker: int | str,
    *,
    marker_names: list[str] | None = None,
    threshold: float = 0.5,
) -> list[MarkerComponent]:
    """Find connected X-positive components across a list of graphs."""
    components: list[MarkerComponent] = []
    for gi, g in enumerate(graphs):
        components.extend(find_marker_components_in_graph(
            g,
            marker,
            marker_names=marker_names,
            threshold=threshold,
            graph_index=gi,
        ))
    return components


def build_domain_profile_rows_for_component(
    *,
    graph,
    graph_index: int,
    component_id: int,
    component_nodes: list[int],
    mu: np.ndarray,
    y_true: np.ndarray | None = None,
    log_var: np.ndarray | None = None,
    marker_name: str | None = None,
    marker_index: int | None = None,
    condition: str = "observed",
    max_outside_distance: int = 4,
    max_inside_distance: int | None = None,
    extra_columns: dict | None = None,
    size_bins=None,
    overflow_size_bin: bool = True,
    overflow_label: str | None = None,
) -> pd.DataFrame:
    """Build one per-node signed-distance profile table for a component."""
    adj = build_undirected_adjacency(graph.edge_index, num_nodes=int(graph.num_nodes))
    stats = component_stats(component_nodes, adj)
    signed_dist = signed_boundary_distances(
        component_nodes,
        adj,
        max_outside_distance=max_outside_distance,
        max_inside_distance=max_inside_distance,
    )

    comp = set(int(n) for n in component_nodes)
    boundary = set(component_boundary_nodes(comp, adj))
    y_true = np.full_like(mu, np.nan, dtype=np.float64) if y_true is None else np.asarray(y_true, dtype=np.float64)
    log_var = np.full_like(mu, np.nan, dtype=np.float64) if log_var is None else np.asarray(log_var, dtype=np.float64)
    extra_columns = dict(extra_columns or {})

    rows: list[dict] = []
    for node, dist in sorted(signed_dist.items(), key=lambda kv: (kv[1], kv[0])):
        row: dict = {
            "condition": condition,
            "graph_index": int(graph_index),
            "organoid_str": getattr(graph, "organoid_str", None),
            "component_id": int(component_id),
            "component_key": f"g{int(graph_index)}_c{int(component_id)}",
            "node_index": int(node),
            "marker_name": marker_name,
            "marker_index": marker_index,
            "signed_distance": int(dist),
            "is_inside_component": bool(node in comp),
            "is_boundary": bool(node in boundary),
            "mu": float(mu[node]),
            "y_true": float(y_true[node]) if np.isfinite(y_true[node]) else np.nan,
            "log_var": float(log_var[node]) if np.isfinite(log_var[node]) else np.nan,
            "cluster_size": int(stats["cluster_size"]),
            "size_bin": format_size_bin(
                int(stats["cluster_size"]),
                size_bins=size_bins,
                overflow_size_bin=overflow_size_bin,
                overflow_label=overflow_label,
            ),
        }
        row.update(stats)
        row.update(extra_columns)
        rows.append(row)

    return pd.DataFrame(rows)


def compute_observed_marker_domain_profiles(
    graphs: list,
    model: torch.nn.Module,
    marker: int | str,
    *,
    marker_names: list[str] | None = None,
    marker_threshold: float = 0.5,
    max_outside_distance: int = 4,
    max_inside_distance: int | None = None,
    size_bins=None,
    overflow_size_bin: bool = True,
    overflow_label: str | None = None,
    device: str | None = None,
    batch_size: int = 32,
    target_index: int | None = 0,
) -> tuple[pd.DataFrame, list[MarkerComponent], dict[str, list[np.ndarray]]]:
    """Compute observed marker-domain curvature profiles for one marker.

    Parameters
    ----------
    graphs
        List of PyTorch Geometric graphs. Each graph must have ``x``,
        ``edge_index``, and usually ``y``.
    model
        Trained GNN. The function first tries
        ``model(batch.x, batch.edge_index, data=batch)`` and falls back to
        ``model(batch.x, batch.edge_index)``.
    marker
        Marker to analyze, either as an integer column index in ``graph.x`` or
        as a string found in ``marker_names``.
    marker_names
        Optional list of marker names. Required if ``marker`` is a string.
    marker_threshold
        Values in ``graph.x[:, marker]`` larger than this threshold are treated
        as marker-positive.
    max_outside_distance
        Largest positive signed graph distance to include outside each marker
        domain.
    max_inside_distance
        Largest depth to include inside each domain. ``None`` includes all
        marker-positive cells reachable from the boundary.
    size_bins
        Component-size bins used to create the ``size_bin`` column. Pass
        ``None`` to label every observed size exactly, an integer ``N`` for
        bins ``1..N``, or an iterable of exact sizes such as
        ``range(1, 9)``.
    overflow_size_bin
        If True and ``size_bins`` is supplied, sizes larger than the largest
        exact bin are labeled as an overflow bin, for example ``"8+"``.
    overflow_label
        Optional custom label for the overflow bin.
    device
        Torch device. If ``None``, CUDA is used when available.
    batch_size
        Number of graphs per inference batch.
    target_index
        Target dimension to analyze for multi-target models.

    Returns
    -------
    profile_df, components, predictions
        ``profile_df`` is the long-form table used for plotting.
        ``components`` stores the detected marker-positive components.
        ``predictions`` contains per-graph ``mu``, ``log_var``, ``y_true``,
        and ``x`` arrays.
    """
    marker_idx = marker_index_from_name(marker, marker_names)
    marker_name = marker_names[marker_idx] if marker_names is not None else f"marker_{marker_idx}"

    preds = predict_graph_node_distributions(
        graphs,
        model,
        device=device,
        batch_size=batch_size,
        target_index=target_index,
    )

    components = find_marker_components(
        graphs,
        marker_idx,
        marker_names=None,
        threshold=marker_threshold,
    )

    rows = []
    for comp in components:
        gi = comp.graph_index
        rows.append(build_domain_profile_rows_for_component(
            graph=graphs[gi],
            graph_index=gi,
            component_id=comp.component_id,
            component_nodes=comp.nodes,
            mu=preds["mu"][gi],
            y_true=preds["y_true"][gi],
            log_var=preds["log_var"][gi],
            marker_name=marker_name,
            marker_index=marker_idx,
            condition="observed",
            max_outside_distance=max_outside_distance,
            max_inside_distance=max_inside_distance,
            size_bins=size_bins,
            overflow_size_bin=overflow_size_bin,
            overflow_label=overflow_label,
        ))

    profile_df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    return profile_df, components, preds


# -----------------------------------------------------------------------------
# Shrink-only perturbations
# -----------------------------------------------------------------------------


def _valid_single_node_removals(nodes: set[int], adj: list[set[int]]) -> list[int]:
    """Boundary nodes whose removal keeps the remaining component connected."""
    if len(nodes) <= 1:
        return []
    boundary = component_boundary_nodes(nodes, adj)
    valid = []
    for u in boundary:
        rem = set(nodes)
        rem.remove(int(u))
        if is_connected_node_set(rem, adj):
            valid.append(int(u))
    return valid


def _removal_priority(u: int, nodes: set[int], adj: list[set[int]]) -> tuple[int, int, int]:
    """Deterministic priority for peeling a boundary node.

    Higher boundary exposure is preferred, then lower internal degree, then lower
    node index. This tends to remove protruding/outer cells before core cells.
    """
    out_deg = sum(1 for v in adj[u] if v not in nodes)
    in_deg = sum(1 for v in adj[u] if v in nodes)
    return (-out_deg, in_deg, int(u))


def greedy_connected_shrink_sequence(
    component_nodes: list[int],
    adj: list[set[int]],
    *,
    target_sizes: object | None = None,
    min_size: int = 1,
) -> list[dict]:
    """Generate deterministic connected shrink variants for one component.

    Parameters
    ----------
    component_nodes
        Original X-positive component.
    adj
        Undirected graph adjacency.
    target_sizes
        Desired remaining sizes. If None, all sizes from ``n-1`` down to
        ``min_size`` are returned.
    min_size
        Smallest allowed remaining component size.

    Returns
    -------
    list of dict
        Each entry has ``remaining_nodes``, ``removed_nodes``, ``removed_count``,
        ``remaining_size`` and ``shrink_step``.
    """
    original = set(int(n) for n in component_nodes)
    n0 = len(original)
    if n0 <= int(min_size):
        return []

    if target_sizes is None:
        target_sizes_list = list(range(n0 - 1, int(min_size) - 1, -1))
    else:
        target_sizes_list = sorted({int(s) for s in target_sizes if int(min_size) <= int(s) < n0}, reverse=True)

    current = set(original)
    removed: list[int] = []
    variants: list[dict] = []
    target_sizes_set = set(target_sizes_list)

    while len(current) > int(min_size):
        valid = _valid_single_node_removals(current, adj)
        if not valid:
            break
        chosen = sorted(valid, key=lambda u: _removal_priority(u, current, adj))[0]
        current.remove(chosen)
        removed.append(chosen)

        if len(current) in target_sizes_set:
            variants.append({
                "shrink_step": int(len(removed)),
                "remaining_size": int(len(current)),
                "removed_count": int(len(removed)),
                "remaining_nodes": tuple(sorted(current)),
                "removed_nodes": tuple(sorted(removed)),
            })

    return variants


def _default_shrink_target_sizes(
    n: int,
    *,
    size_bins=None,
    include_all_intermediate: bool = False,
) -> list[int]:
    """Choose shrink target sizes without hard-coding a bin scheme.

    If ``include_all_intermediate`` is True, every connected remaining size
    from 1 to ``n-1`` is requested. Otherwise, supplied ``size_bins`` are used
    as target remaining sizes. If neither is supplied, only a one-cell shrink
    target, ``n-1``, is requested.
    """
    n = int(n)
    if n <= 1:
        return []
    if include_all_intermediate:
        return list(range(1, n))
    bins = _normalize_size_bins(size_bins)
    if bins is not None:
        return [s for s in bins if 1 <= int(s) < n]
    return [n - 1]


def make_shrunk_graph_for_component(
    graph,
    marker_index: int,
    removed_nodes: list[int],
):
    """Deep-copy a graph and set marker X to zero on removed nodes."""
    g = copy.deepcopy(graph)
    nodes = torch.as_tensor(list(removed_nodes), dtype=torch.long, device=g.x.device)
    if nodes.numel() > 0:
        g.x[nodes, int(marker_index)] = 0
    return g


def compute_shrink_marker_domain_profiles(
    graphs: list,
    model: torch.nn.Module,
    marker: int | str,
    *,
    marker_names: list[str] | None = None,
    components: list[MarkerComponent] | None = None,
    baseline_predictions: dict[str, list[np.ndarray]] | None = None,
    marker_threshold: float = 0.5,
    max_outside_distance: int = 4,
    max_inside_distance: int | None = None,
    target_sizes: object | None = None,
    size_bins=None,
    overflow_size_bin: bool = True,
    overflow_label: str | None = None,
    include_all_intermediate_sizes: bool = False,
    max_components: int | None = None,
    device: str | None = None,
    batch_size: int = 32,
    target_index: int | None = 0,
) -> pd.DataFrame:
    """Run shrink-only marker perturbations and build profile tables.

    Parameters
    ----------
    graphs
        List of PyTorch Geometric graphs. The graph topology is not changed;
        shrink perturbations only set marker X to zero on selected cells.
    model
        Trained GNN used to recompute predictions on the shrunk graphs.
    marker
        Marker to shrink, either as an integer column index in ``graph.x`` or
        as a string found in ``marker_names``.
    marker_names
        Optional list of marker names. Required if ``marker`` is a string.
    components
        Optional precomputed marker-positive components, usually returned by
        ``compute_observed_marker_domain_profiles``. If omitted, components are
        detected here.
    baseline_predictions
        Optional baseline output from ``predict_graph_node_distributions``. If
        supplied, it is used to compute paired ``delta_mu`` and avoids another
        baseline model pass.
    marker_threshold
        Threshold for identifying marker-positive cells when components are not
        supplied.
    max_outside_distance
        Largest positive signed graph distance to include outside each shrunk
        domain.
    max_inside_distance
        Largest depth to include inside each shrunk domain. ``None`` includes
        all cells inside the remaining connected component.
    target_sizes
        Explicit remaining component sizes to generate. This overrides
        ``size_bins`` for shrink generation.
    size_bins
        Size-bin specification used for labeling output rows and, when
        ``target_sizes`` is None, as the default set of remaining sizes to
        generate. Pass ``None`` to avoid a predefined bin scheme.
    overflow_size_bin
        If True and ``size_bins`` is supplied, sizes larger than the largest
        exact bin are labeled as an overflow bin.
    overflow_label
        Optional custom label for the overflow bin.
    include_all_intermediate_sizes
        If True, generate every connected remaining size from 1 to ``n-1`` for
        each component. This can require many model forward passes.
    max_components
        Optional cap on the number of components to shrink, useful for testing.
    device
        Torch device. If ``None``, CUDA is used when available.
    batch_size
        Number of perturbed graphs per inference batch.
    target_index
        Target dimension to analyze for multi-target models.

    Returns
    -------
    pandas.DataFrame
        Long-form profile table. Distances are recomputed relative to the
        shrunk component. Rows include ``base_mu`` and ``delta_mu = mu -
        base_mu`` for the same physical node.
    """
    marker_idx = marker_index_from_name(marker, marker_names)
    marker_name = marker_names[marker_idx] if marker_names is not None else f"marker_{marker_idx}"

    if baseline_predictions is None:
        baseline_predictions = predict_graph_node_distributions(
            graphs,
            model,
            device=device,
            batch_size=batch_size,
            target_index=target_index,
        )

    if components is None:
        components = find_marker_components(
            graphs,
            marker_idx,
            threshold=marker_threshold,
        )
    components = list(components)
    if max_components is not None:
        components = components[: int(max_components)]

    # Build all shrunk graphs first so model inference can be batched.
    shrunk_graphs: list = []
    meta: list[dict] = []

    for comp in components:
        gi = comp.graph_index
        graph = graphs[gi]
        adj = build_undirected_adjacency(graph.edge_index, num_nodes=int(graph.num_nodes))
        n0 = len(comp.nodes)
        ts = list(target_sizes) if target_sizes is not None else _default_shrink_target_sizes(
            n0,
            size_bins=size_bins,
            include_all_intermediate=include_all_intermediate_sizes,
        )
        variants = greedy_connected_shrink_sequence(comp.nodes, adj, target_sizes=ts, min_size=1)

        for vi, var in enumerate(variants):
            gp = make_shrunk_graph_for_component(graph, marker_idx, var["removed_nodes"])
            shrunk_graphs.append(gp)
            meta.append({
                "source_graph_index": int(gi),
                "source_component_id": int(comp.component_id),
                "source_component_key": f"g{int(gi)}_c{int(comp.component_id)}",
                "shrink_variant_index": int(vi),
                "original_size": int(n0),
                "original_size_bin": format_size_bin(
                    n0,
                    size_bins=size_bins,
                    overflow_size_bin=overflow_size_bin,
                    overflow_label=overflow_label,
                ),
                **var,
            })

    if len(shrunk_graphs) == 0:
        return pd.DataFrame()

    pert_preds = predict_graph_node_distributions(
        shrunk_graphs,
        model,
        device=device,
        batch_size=batch_size,
        target_index=target_index,
    )

    rows = []
    for pi, info in enumerate(meta):
        gi = int(info["source_graph_index"])
        remaining_nodes = tuple(int(n) for n in info["remaining_nodes"])
        if len(remaining_nodes) == 0:
            continue

        # Recompute profile relative to the shrunk component on the perturbed graph.
        df = build_domain_profile_rows_for_component(
            graph=shrunk_graphs[pi],
            graph_index=gi,
            component_id=int(info["source_component_id"]),
            component_nodes=remaining_nodes,
            mu=pert_preds["mu"][pi],
            y_true=baseline_predictions["y_true"][gi],
            log_var=pert_preds["log_var"][pi],
            marker_name=marker_name,
            marker_index=marker_idx,
            condition="shrink",
            max_outside_distance=max_outside_distance,
            max_inside_distance=max_inside_distance,
            size_bins=size_bins,
            overflow_size_bin=overflow_size_bin,
            overflow_label=overflow_label,
            extra_columns={
                "perturbation_id": f"g{gi}_c{int(info['source_component_id'])}_shrink{int(info['shrink_variant_index'])}",
                "source_component_key": info["source_component_key"],
                "shrink_variant_index": int(info["shrink_variant_index"]),
                "shrink_step": int(info["shrink_step"]),
                "original_size": int(info["original_size"]),
                "original_size_bin": info["original_size_bin"],
                "remaining_size": int(info["remaining_size"]),
                "remaining_size_bin": format_size_bin(
                    int(info["remaining_size"]),
                    size_bins=size_bins,
                    overflow_size_bin=overflow_size_bin,
                    overflow_label=overflow_label,
                ),
                "removed_count": int(info["removed_count"]),
            },
        )
        if not df.empty:
            node_idx = df["node_index"].to_numpy(dtype=int)
            df["base_mu"] = baseline_predictions["mu"][gi][node_idx]
            df["delta_mu"] = df["mu"].to_numpy(dtype=float) - df["base_mu"].to_numpy(dtype=float)
            df["base_log_var"] = baseline_predictions["log_var"][gi][node_idx]
            df["delta_log_var"] = df["log_var"].to_numpy(dtype=float) - df["base_log_var"].to_numpy(dtype=float)
        rows.append(df)

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


# -----------------------------------------------------------------------------
# Summaries and plotting
# -----------------------------------------------------------------------------


def summarize_domain_profiles(
    profile_df: pd.DataFrame,
    *,
    value_col: str = "mu",
    distance_col: str = "signed_distance",
    size_col: str = "size_bin",
    condition_col: str = "condition",
    component_col: str = "component_key",
    weight_components_equally: bool = True,
    min_cells: int = 1,
    size_bins=None,
    overflow_size_bin: bool = True,
    overflow_label: str | None = None,
) -> pd.DataFrame:
    """Aggregate profile rows into mean/SEM curves for plotting.

    Parameters
    ----------
    profile_df
        Raw long-form dataframe returned by the observed or shrink profile
        functions.
    value_col
        Numeric column to summarize, for example ``"mu"`` or ``"delta_mu"``.
    distance_col
        Column containing signed graph distance from the marker-domain
        boundary.
    size_col
        Column containing component-size labels.
    condition_col
        Column identifying observed versus perturbation conditions.
    component_col
        Column identifying individual domains/components.
    weight_components_equally
        If True, cells are first averaged within each component and distance,
        then component means are averaged. This prevents large domains from
        dominating the curve.
    min_cells
        Minimum total number of contributing cells required for a summarized
        point to be retained.
    size_bins
        Optional size-bin specification used only for ordering the output. Pass
        ``None`` to infer the order from the dataframe.
    overflow_size_bin
        Whether the size-bin order should include an overflow label when
        ``size_bins`` is supplied.
    overflow_label
        Optional custom overflow label.
    """
    if profile_df.empty:
        return pd.DataFrame()
    if value_col not in profile_df.columns:
        raise ValueError(f"{value_col!r} not found in profile_df columns.")

    df = profile_df.copy()
    df = df[np.isfinite(df[value_col].to_numpy(dtype=float))]
    if df.empty:
        return pd.DataFrame()

    group_cols = [condition_col, size_col, distance_col]

    if weight_components_equally:
        per_comp = (
            df.groupby(group_cols + [component_col], observed=True)
            .agg(value=(value_col, "mean"), n_cells=(value_col, "size"))
            .reset_index()
        )
        src = per_comp.rename(columns={"value": value_col})
        n_unit_name = "n_components"
        agg = (
            src.groupby(group_cols, observed=True)
            .agg(
                mean=(value_col, "mean"),
                std=(value_col, "std"),
                n=(value_col, "size"),
                n_cells=("n_cells", "sum"),
            )
            .reset_index()
        )
        agg[n_unit_name] = agg["n"]
    else:
        agg = (
            df.groupby(group_cols, observed=True)
            .agg(
                mean=(value_col, "mean"),
                std=(value_col, "std"),
                n=(value_col, "size"),
                n_cells=(value_col, "size"),
            )
            .reset_index()
        )
        agg["n_components"] = np.nan

    agg = agg[agg["n_cells"] >= int(min_cells)].copy()
    agg["sem"] = agg["std"] / np.sqrt(agg["n"].clip(lower=1))
    agg["ci95"] = 1.96 * agg["sem"]

    order = size_bin_order_from_spec(
        agg,
        size_bins=size_bins,
        overflow_size_bin=overflow_size_bin,
        overflow_label=overflow_label,
        size_col=size_col,
    )
    if order:
        agg[size_col] = pd.Categorical(agg[size_col].astype(str), categories=order, ordered=True)
    agg = agg.sort_values([condition_col, size_col, distance_col]).reset_index(drop=True)
    return agg


def plot_domain_profiles(
    profile_df_or_summary: pd.DataFrame,
    *,
    value_col: str = "mu",
    condition: str | None = "observed",
    size_bins=None,
    overflow_size_bin: bool = True,
    overflow_label: str | None = None,
    size_bin_order: list[str] | None = None,
    distance_col: str = "signed_distance",
    size_col: str = "size_bin",
    condition_col: str = "condition",
    summary_already: bool = False,
    weight_components_equally: bool = True,
    show_ci: bool = True,
    ax: object | None = None,
    title: str | None = None,
    ylabel: str | None = None,
    xlabel: str = "Signed graph distance from marker-domain boundary",
):
    """Plot mean profile curves stratified by component-size bin.

    Parameters
    ----------
    profile_df_or_summary
        Either a raw long-form profile dataframe or the output of
        ``summarize_domain_profiles``.
    value_col
        Numeric quantity to plot, for example ``"mu"`` or ``"delta_mu"``.
    condition
        Condition to plot. Use ``"observed"``, ``"shrink"``, or ``None`` to
        plot all conditions present in a pre-filtered dataframe.
    size_bins
        Optional exact-size bin specification used for summary ordering. Pass
        ``None`` to infer bins from the dataframe, an integer ``N`` for bins
        ``1..N``, or an iterable of exact integer sizes.
    overflow_size_bin
        Whether an overflow bin should be included when ``size_bins`` is
        supplied.
    overflow_label
        Optional custom label for the overflow bin.
    size_bin_order
        Optional explicit list of size-bin labels to draw. This is useful when
        you want full manual control of the plotting order.
    distance_col
        Column containing signed graph distance from the marker-domain
        boundary.
    size_col
        Column containing component-size labels.
    condition_col
        Column containing condition labels.
    summary_already
        Set True when ``profile_df_or_summary`` is already summarized.
    weight_components_equally
        Passed to ``summarize_domain_profiles`` when summarizing raw rows.
    show_ci
        If True, draw ±95% normal-approximation confidence bands when available.
    ax
        Optional matplotlib axes object. If None, a new axes is created.
    title, ylabel, xlabel
        Plot labels.
    """
    import matplotlib.pyplot as plt

    if summary_already:
        summary = profile_df_or_summary.copy()
    else:
        summary = summarize_domain_profiles(
            profile_df_or_summary,
            value_col=value_col,
            distance_col=distance_col,
            size_col=size_col,
            condition_col=condition_col,
            weight_components_equally=weight_components_equally,
            size_bins=size_bins,
            overflow_size_bin=overflow_size_bin,
            overflow_label=overflow_label,
        )

    if condition is not None and condition_col in summary.columns:
        summary = summary[summary[condition_col] == condition].copy()

    if ax is None:
        _, ax = plt.subplots(figsize=(7.5, 4.8))

    if size_bin_order is None:
        size_bin_order = size_bin_order_from_spec(
            summary,
            size_bins=size_bins,
            overflow_size_bin=overflow_size_bin,
            overflow_label=overflow_label,
            size_col=size_col,
        )

    for sb in size_bin_order:
        sub = summary[summary[size_col].astype(str) == str(sb)].sort_values(distance_col)
        if sub.empty:
            continue
        x = sub[distance_col].to_numpy(dtype=float)
        y = sub["mean"].to_numpy(dtype=float)
        ax.plot(x, y, marker="o", label=f"size {sb}")
        if show_ci and "ci95" in sub.columns:
            ci = sub["ci95"].to_numpy(dtype=float)
            ax.fill_between(x, y - ci, y + ci, alpha=0.18)

    ax.axvline(0, linestyle="--", linewidth=1)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel or value_col)
    if title is not None:
        ax.set_title(title)
    ax.legend(title="component size", bbox_to_anchor=(1.02, 1.0), loc="upper left")
    return ax


def plot_observed_vs_shrink_profiles(
    observed_df: pd.DataFrame,
    shrink_df: pd.DataFrame,
    *,
    value_col: str = "mu",
    size_bins=None,
    overflow_size_bin: bool = True,
    overflow_label: str | None = None,
    size_bin_order: list[str] | None = None,
    use_remaining_size_for_shrink: bool = True,
    weight_components_equally: bool = True,
    show_ci: bool = True,
    axes: list | tuple | None = None,
):
    """Plot observed profiles next to shrink-only perturbation profiles.

    Parameters
    ----------
    observed_df
        Raw observed profile dataframe.
    shrink_df
        Raw shrink profile dataframe.
    value_col
        Numeric quantity to plot, usually ``"mu"``.
    size_bins
        Optional exact-size bin specification used for ordering. Pass ``None``
        to infer bins from the dataframe.
    overflow_size_bin
        Whether to include an overflow bin when ``size_bins`` is supplied.
    overflow_label
        Optional custom overflow-bin label.
    size_bin_order
        Optional explicit label order for plotting.
    use_remaining_size_for_shrink
        If True, shrink profiles are grouped by remaining component size rather
        than original component size.
    weight_components_equally
        Passed to ``summarize_domain_profiles``.
    show_ci
        If True, draw ±95% normal-approximation confidence bands.
    axes
        Optional pair of matplotlib axes. If None, a new two-panel figure is
        created.
    """
    import matplotlib.pyplot as plt

    if axes is None:
        _, axes = plt.subplots(1, 2, figsize=(14, 4.8), sharey=True)
    ax0, ax1 = axes

    plot_domain_profiles(
        observed_df,
        value_col=value_col,
        condition="observed",
        size_bins=size_bins,
        overflow_size_bin=overflow_size_bin,
        overflow_label=overflow_label,
        size_bin_order=size_bin_order,
        weight_components_equally=weight_components_equally,
        show_ci=show_ci,
        ax=ax0,
        title="Observed marker domains",
        ylabel=value_col,
    )

    shrink_plot_df = shrink_df.copy()
    if use_remaining_size_for_shrink and "remaining_size_bin" in shrink_plot_df.columns:
        shrink_plot_df["size_bin"] = shrink_plot_df["remaining_size_bin"]

    plot_domain_profiles(
        shrink_plot_df,
        value_col=value_col,
        condition="shrink",
        size_bins=size_bins,
        overflow_size_bin=overflow_size_bin,
        overflow_label=overflow_label,
        size_bin_order=size_bin_order,
        weight_components_equally=weight_components_equally,
        show_ci=show_ci,
        ax=ax1,
        title="Shrink-only counterfactuals",
        ylabel=value_col,
    )
    return axes


def plot_shrink_delta_profiles(
    shrink_df: pd.DataFrame,
    *,
    value_col: str = "delta_mu",
    size_bins=None,
    overflow_size_bin: bool = True,
    overflow_label: str | None = None,
    size_bin_order: list[str] | None = None,
    use_remaining_size_for_shrink: bool = True,
    weight_components_equally: bool = True,
    show_ci: bool = True,
    ax: object | None = None,
    title: str = "Shrink-only change in prediction",
):
    """Plot paired prediction change after shrink perturbations.

    Parameters
    ----------
    shrink_df
        Raw shrink profile dataframe returned by
        ``compute_shrink_marker_domain_profiles``.
    value_col
        Paired-change column to plot, usually ``"delta_mu"`` or
        ``"delta_log_var"``.
    size_bins
        Optional exact-size bin specification used for ordering. Pass ``None``
        to infer bins from the dataframe.
    overflow_size_bin
        Whether to include an overflow bin when ``size_bins`` is supplied.
    overflow_label
        Optional custom overflow-bin label.
    size_bin_order
        Optional explicit label order for plotting.
    use_remaining_size_for_shrink
        If True, profiles are grouped by remaining component size rather than
        original component size.
    weight_components_equally
        Passed to ``summarize_domain_profiles``.
    show_ci
        If True, draw ±95% normal-approximation confidence bands.
    ax
        Optional matplotlib axes. If None, a new axes is created.
    title
        Plot title.
    """
    plot_df = shrink_df.copy()
    if use_remaining_size_for_shrink and "remaining_size_bin" in plot_df.columns:
        plot_df["size_bin"] = plot_df["remaining_size_bin"]
    ax = plot_domain_profiles(
        plot_df,
        value_col=value_col,
        condition="shrink",
        size_bins=size_bins,
        overflow_size_bin=overflow_size_bin,
        overflow_label=overflow_label,
        size_bin_order=size_bin_order,
        weight_components_equally=weight_components_equally,
        show_ci=show_ci,
        ax=ax,
        title=title,
        ylabel=value_col,
    )
    ax.axhline(0, linewidth=1)
    return ax


# -----------------------------------------------------------------------------
# One-call convenience wrapper
# -----------------------------------------------------------------------------


def run_marker_domain_profile_analysis(
    graphs: list,
    model: torch.nn.Module,
    marker: int | str,
    *,
    marker_names: list[str] | None = None,
    marker_threshold: float = 0.5,
    max_outside_distance: int = 4,
    max_inside_distance: int | None = None,
    size_bins=None,
    overflow_size_bin: bool = True,
    overflow_label: str | None = None,
    run_shrink: bool = True,
    shrink_target_sizes: object | None = None,
    include_all_intermediate_shrink_sizes: bool = False,
    max_components_for_shrink: int | None = None,
    device: str | None = None,
    batch_size: int = 32,
    target_index: int | None = 0,
) -> dict:
    """Run observed marker-domain profiles and optional shrink counterfactuals.

    Parameters
    ----------
    graphs
        List of PyTorch Geometric graphs. Each graph must provide ``x`` and
        ``edge_index``; ``y`` is used when available.
    model
        Trained GNN used for node-wise prediction. The wrapper uses the same
        inference helper for baseline and shrink predictions.
    marker
        Marker to analyze, either as an integer column index in ``graph.x`` or
        as a string found in ``marker_names``.
    marker_names
        Optional list of marker names. Required if ``marker`` is a string.
    marker_threshold
        Values larger than this threshold are treated as marker-positive.
    max_outside_distance
        Largest positive signed graph-hop distance to include outside each
        marker domain.
    max_inside_distance
        Largest inside-domain depth to include. ``None`` includes all cells
        inside the marker-positive component.
    size_bins
        Component-size bin specification used throughout the analysis. Pass
        ``None`` to label every observed size exactly, an integer ``N`` for bins
        ``1..N``, or an iterable of exact sizes such as ``range(1, 9)``. When
        shrink targets are not explicitly supplied, these bins also determine
        the default remaining sizes generated by the shrink analysis.
    overflow_size_bin
        If True and ``size_bins`` is supplied, sizes larger than the largest
        exact bin are labeled as an overflow bin.
    overflow_label
        Optional custom label for the overflow bin.
    run_shrink
        If True, also run shrink-only marker perturbations.
    shrink_target_sizes
        Explicit remaining component sizes to generate during shrink analysis.
        This overrides ``size_bins`` for shrink generation.
    include_all_intermediate_shrink_sizes
        If True, generate every connected remaining size from 1 to ``n-1`` for
        each component. This can be computationally expensive.
    max_components_for_shrink
        Optional cap on the number of components passed to the shrink analysis,
        useful for quick tests.
    device
        Torch device. If ``None``, CUDA is used when available.
    batch_size
        Number of graphs per inference batch.
    target_index
        Target dimension to analyze for multi-target models.

    Returns
    -------
    dict
        Dictionary with ``observed_df``, ``components``,
        ``baseline_predictions``, ``observed_summary``, and, if ``run_shrink``
        is True, ``shrink_df``, ``shrink_summary_mu``, and
        ``shrink_summary_delta_mu``.
    """
    observed_df, components, baseline_preds = compute_observed_marker_domain_profiles(
        graphs,
        model,
        marker,
        marker_names=marker_names,
        marker_threshold=marker_threshold,
        max_outside_distance=max_outside_distance,
        max_inside_distance=max_inside_distance,
        size_bins=size_bins,
        overflow_size_bin=overflow_size_bin,
        overflow_label=overflow_label,
        device=device,
        batch_size=batch_size,
        target_index=target_index,
    )

    out: dict = {
        "observed_df": observed_df,
        "components": components,
        "baseline_predictions": baseline_preds,
        "observed_summary": summarize_domain_profiles(
            observed_df,
            value_col="mu",
            size_bins=size_bins,
            overflow_size_bin=overflow_size_bin,
            overflow_label=overflow_label,
        ),
    }

    if run_shrink:
        shrink_df = compute_shrink_marker_domain_profiles(
            graphs,
            model,
            marker,
            marker_names=marker_names,
            components=components,
            baseline_predictions=baseline_preds,
            marker_threshold=marker_threshold,
            max_outside_distance=max_outside_distance,
            max_inside_distance=max_inside_distance,
            target_sizes=shrink_target_sizes,
            size_bins=size_bins,
            overflow_size_bin=overflow_size_bin,
            overflow_label=overflow_label,
            include_all_intermediate_sizes=include_all_intermediate_shrink_sizes,
            max_components=max_components_for_shrink,
            device=device,
            batch_size=batch_size,
            target_index=target_index,
        )
        out["shrink_df"] = shrink_df
        out["shrink_summary_mu"] = summarize_domain_profiles(
            shrink_df,
            value_col="mu",
            size_bins=size_bins,
            overflow_size_bin=overflow_size_bin,
            overflow_label=overflow_label,
        ) if not shrink_df.empty else pd.DataFrame()
        out["shrink_summary_delta_mu"] = summarize_domain_profiles(
            shrink_df,
            value_col="delta_mu",
            size_bins=size_bins,
            overflow_size_bin=overflow_size_bin,
            overflow_label=overflow_label,
        ) if not shrink_df.empty else pd.DataFrame()

    return out