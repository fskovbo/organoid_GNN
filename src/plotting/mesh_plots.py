import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
from plotly.subplots import make_subplots

from src.analysis.mesh_mapping import project_node_categories_to_mesh
from organograph.plotting.meshes import plot_organoid_mesh, plot_mesh_by_regions


"""Plotly utilities for mesh-projected quantities."""



def plot_projected_true_vs_pred(
    result,
    *,
    colorscale="RdBu_r",
    center_at_zero=True,
    fig_size=(1200, 550),
    view=None,
    title=None,
):
    """Plot ground-truth and predicted vertex values side-by-side on the same mesh."""
    mesh = result["mesh"]
    mesh_true = np.asarray(result["mesh_true"], dtype=float)
    mesh_pred = np.asarray(result["mesh_pred"], dtype=float)

    all_vals = np.concatenate([mesh_true.ravel(), mesh_pred.ravel()])
    finite = np.isfinite(all_vals)
    if not np.any(finite):
        vmin, vmax = -1.0, 1.0
    else:
        vv = all_vals[finite]
        if center_at_zero:
            m = float(np.max(np.abs(vv)))
            m = 1.0 if m == 0.0 else m
            vmin, vmax = -m, m
        else:
            vmin, vmax = float(np.min(vv)), float(np.max(vv))
            if vmin == vmax:
                vmin -= 1.0
                vmax += 1.0

    fig_true = plot_organoid_mesh(
        mesh,
        vertex_values=mesh_true,
        backend="plotly",
        colorscale=colorscale,
        center_at_zero=center_at_zero,
        vmin=vmin,
        vmax=vmax,
        show_colorbar=True,
        fig_size=fig_size,
        view=view,
    )
    fig_pred = plot_organoid_mesh(
        mesh,
        vertex_values=mesh_pred,
        backend="plotly",
        colorscale=colorscale,
        center_at_zero=center_at_zero,
        vmin=vmin,
        vmax=vmax,
        show_colorbar=False,
        fig_size=fig_size,
        view=view,
    )

    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "scene"}, {"type": "scene"}]],
        subplot_titles=("Ground truth", "Prediction"),
        horizontal_spacing=0.03,
    )
    for tr in fig_true.data:
        fig.add_trace(tr, row=1, col=1)
    for tr in fig_pred.data:
        fig.add_trace(tr, row=1, col=2)

    scene1 = fig_true.layout.scene.to_plotly_json() if fig_true.layout.scene else {}
    scene2 = fig_pred.layout.scene.to_plotly_json() if fig_pred.layout.scene else {}
    fig.update_layout(scene=scene1, scene2=scene2, width=int(fig_size[0]), height=int(fig_size[1]))

    organoid_str = result.get("organoid_str")
    if title is None:
        title = organoid_str if organoid_str is not None else "Projected truth vs prediction"
    fig.update_layout(title=title)
    return fig


# -----------------------------------------------------------------------------
# Categorical helpers for marker/cluster mesh plots
# -----------------------------------------------------------------------------


def marker_name_from_x_row(x_row, marker_names, threshold=0.5, priority_names=None):
    """Return the first positive marker name in priority order, else 'none'.

    Parameters
    ----------
    marker_names : sequence[str]
        Marker names in the exact column order of x_row / graph.x.
    priority_names : sequence[str] or None
        Marker priority order for resolving multi-positive cells. If None,
        marker_names order is used.
    """
    x_row = np.asarray(x_row)
    marker_to_col = {name: j for j, name in enumerate(marker_names)}
    ordered = marker_names if priority_names is None else list(priority_names)

    for name in ordered:
        j = marker_to_col.get(name)
        if j is None:
            continue
        if x_row[j] > threshold:
            return name
    return "none"



def marker_categories_for_graph(graph, marker_names, threshold=0.5, priority_names=None):
    """Map each node in a graph to one marker category using priority order.

    marker_names must match graph.x column order. priority_names controls the
    first-hit assignment when a node is positive for multiple markers.
    """
    X = graph.x.detach().cpu().numpy()
    if X.ndim != 2:
        raise ValueError(f"graph.x must be 2-D, got shape {X.shape}")
    if X.shape[1] < len(marker_names):
        raise ValueError(
            f"graph.x has only {X.shape[1]} columns, but marker_names has length {len(marker_names)}"
        )

    return np.array(
        [
            marker_name_from_x_row(
                row,
                marker_names,
                threshold=threshold,
                priority_names=priority_names,
            )
            for row in X
        ],
        dtype=object,
    )



def get_graph_slice_bounds(graphs, graph_index):
    """Return [start, end) bounds into concatenated node arrays for one graph."""
    sizes = [int(g.y.shape[0]) for g in graphs]
    start = sum(sizes[:graph_index])
    end = start + sizes[graph_index]
    return start, end



def cluster_color_map(K, cmap_name="tab10"):
    """Return cluster_id -> hex color using the same matplotlib cmap as the TSNE plot."""
    cmap = plt.get_cmap(cmap_name, K)
    return {k: to_hex(cmap(k)) for k in range(K)}



def _prepare_regions_from_projected_categories(mesh_categories, category_order):
    """Convert projected per-vertex categories into one vertex-region per category."""
    mesh_categories = np.asarray(mesh_categories, dtype=object)
    regions = []
    names = []

    for cat in category_order:
        idx = np.where(mesh_categories == cat)[0]
        regions.append(idx)
        names.append(cat)

    return regions, names



def _filter_empty_regions(regions, names, colors):
    """Filter regions/names/colors together so plot_mesh_by_regions sees matching lengths."""
    regions_f = []
    names_f = []
    colors_f = []

    for reg, name, color in zip(regions, names, colors):
        if reg is None:
            continue
        reg = np.asarray(reg)
        if reg.size == 0:
            continue
        regions_f.append(reg)
        names_f.append(name)
        colors_f.append(color)

    return regions_f, names_f, colors_f



def _prefix_plotly_region_legend(fig, prefix):
    """Prefix legend names so marker/cluster legends remain distinguishable after combining figs."""
    for tr in fig.data:
        name = getattr(tr, "name", None)
        if name:
            tr.name = f"{prefix}: {name}"



def project_marker_categories_for_graph(
    graph,
    *,
    marker_names,
    marker_colors,
    threshold=0.5,
    meta_lookup=None,
    normalize_mesh=True,
    missing_category="none",
):
    """Project graph marker categories to mesh vertices using graph-consistent priority.

    Marker assignment priority follows the order of marker_colors (excluding
    the special 'none' category), matching graph plots built from the same dict.
    """
    priority_names = [m for m in marker_colors.keys() if m != "none"]
    node_categories = marker_categories_for_graph(
        graph,
        marker_names,
        threshold=threshold,
        priority_names=priority_names,
    )
    return project_node_categories_to_mesh(
        graph,
        node_categories,
        meta_lookup=meta_lookup,
        normalize_mesh=normalize_mesh,
        missing_category=missing_category,
    )



def extract_cluster_labels_for_graph(graph_index, graphs, cluster_labels_all):
    """Extract one graph's per-node cluster labels from concatenated labels."""
    start, end = get_graph_slice_bounds(graphs, graph_index)
    return np.asarray(cluster_labels_all[start:end], dtype=int)



def cluster_categories_from_labels(cluster_labels):
    """Convert integer cluster labels to string categories like ['C0', 'C1', ...]."""
    cluster_labels = np.asarray(cluster_labels, dtype=int).reshape(-1)
    return np.array([f"C{k}" for k in cluster_labels], dtype=object)



def project_cluster_categories_for_graph(
    graph,
    cluster_labels,
    *,
    meta_lookup=None,
    normalize_mesh=True,
    missing_category=None,
):
    """Project one graph's per-node cluster labels to mesh vertices."""
    cluster_labels = np.asarray(cluster_labels, dtype=int).reshape(-1)
    cluster_categories = cluster_categories_from_labels(cluster_labels)
    proj = project_node_categories_to_mesh(
        graph,
        cluster_categories,
        meta_lookup=meta_lookup,
        normalize_mesh=normalize_mesh,
        missing_category=missing_category,
    )
    proj["cluster_labels"] = cluster_labels
    proj["cluster_categories"] = cluster_categories
    return proj



def plot_categorical_mesh(
    mesh,
    mesh_categories,
    *,
    category_order,
    category_colors,
    fig_size=(800, 700),
    view=None,
    baseline_color="lightgray",
    baseline_name="unassigned",
    add_legend=True,
    title=None,
):
    """Plot a single mesh from per-vertex categorical labels."""
    regions, names = _prepare_regions_from_projected_categories(mesh_categories, category_order)
    colors = [category_colors[name] for name in names]
    regions, names, colors = _filter_empty_regions(regions, names, colors)

    fig = plot_mesh_by_regions(
        mesh,
        regions,
        backend="plotly",
        region_kind="vertex",
        region_names=names,
        colors=colors,
        baseline_color=baseline_color,
        baseline_name=baseline_name,
        priority="last",
        alpha=1.0,
        view=view,
        fig_size=fig_size,
        add_legend=add_legend,
    )
    if title is not None:
        fig.update_layout(title=title)
    return fig



def plot_marker_mesh(
    graph,
    *,
    marker_names,
    marker_colors,
    meta_lookup=None,
    threshold=0.5,
    fig_size=(800, 700),
    view=None,
    normalize_mesh=True,
    add_legend=True,
):
    """Plot one graph's marker categories projected onto its mesh.

    Uses the same marker priority as graph plots: the order of marker_colors,
    with 'none' reserved as the fallback category.
    """
    proj = project_marker_categories_for_graph(
        graph,
        marker_names=marker_names,
        marker_colors=marker_colors,
        threshold=threshold,
        meta_lookup=meta_lookup,
        normalize_mesh=normalize_mesh,
        missing_category="none",
    )
    category_order = [m for m in marker_colors.keys() if m != "none" and m in marker_names] + ["none"]
    return plot_categorical_mesh(
        proj["mesh"],
        proj["mesh_categories"],
        category_order=category_order,
        category_colors=marker_colors,
        fig_size=fig_size,
        view=view,
        baseline_color=marker_colors.get("none", "lightgray"),
        baseline_name="none",
        add_legend=add_legend,
        title=getattr(graph, "organoid_str", None),
    )



def plot_cluster_mesh(
    graph,
    cluster_labels,
    *,
    meta_lookup=None,
    cluster_cmap_name="tab10",
    fig_size=(800, 700),
    view=None,
    normalize_mesh=True,
    add_legend=True,
    missing_category=None,
):
    """Plot one graph's cluster categories projected onto its mesh."""
    proj = project_cluster_categories_for_graph(
        graph,
        cluster_labels,
        meta_lookup=meta_lookup,
        normalize_mesh=normalize_mesh,
        missing_category=missing_category,
    )
    cluster_labels = proj["cluster_labels"]
    K = int(np.max(cluster_labels)) + 1 if cluster_labels.size > 0 else 0
    cluster_colors_raw = cluster_color_map(K, cmap_name=cluster_cmap_name) if K > 0 else {}
    category_order = [f"C{k}" for k in range(K)]
    category_colors = {f"C{k}": cluster_colors_raw[k] for k in range(K)}

    return plot_categorical_mesh(
        proj["mesh"],
        proj["mesh_categories"],
        category_order=category_order,
        category_colors=category_colors,
        fig_size=fig_size,
        view=view,
        baseline_color="lightgray",
        baseline_name="unassigned",
        add_legend=add_legend,
        title=getattr(graph, "organoid_str", None),
    )



def plot_marker_vs_cluster_mesh(
    graph_index,
    graphs,
    cluster_labels_all,
    *,
    marker_names,
    meta_lookup,
    marker_colors,
    cluster_cmap_name="tab10",
    fig_size=(1400, 600),
    view=None,
):
    """Plot one organoid as two Plotly 3D subplots: markers and clusters."""
    g = graphs[graph_index]

    marker_proj = project_marker_categories_for_graph(
        g,
        marker_names=marker_names,
        marker_colors=marker_colors,
        meta_lookup=meta_lookup,
        missing_category="none",
    )
    fig_marker = plot_categorical_mesh(
        marker_proj["mesh"],
        marker_proj["mesh_categories"],
        category_order=[m for m in marker_colors.keys() if m != "none" and m in marker_names] + ["none"],
        category_colors=marker_colors,
        fig_size=fig_size,
        view=view,
        baseline_color=marker_colors.get("none", "lightgray"),
        baseline_name="none",
        add_legend=True,
        title=None,
    )
    _prefix_plotly_region_legend(fig_marker, "marker")

    cluster_labels = extract_cluster_labels_for_graph(graph_index, graphs, cluster_labels_all)
    cluster_proj = project_cluster_categories_for_graph(
        g,
        cluster_labels,
        meta_lookup=meta_lookup,
        missing_category=None,
    )
    K = int(np.max(cluster_labels)) + 1 if cluster_labels.size > 0 else 0
    cluster_colors_raw = cluster_color_map(K, cmap_name=cluster_cmap_name) if K > 0 else {}
    cluster_color_dict = {f"C{k}": cluster_colors_raw[k] for k in range(K)}
    fig_cluster = plot_categorical_mesh(
        cluster_proj["mesh"],
        cluster_proj["mesh_categories"],
        category_order=[f"C{k}" for k in range(K)],
        category_colors=cluster_color_dict,
        fig_size=fig_size,
        view=view,
        baseline_color="lightgray",
        baseline_name="unassigned",
        add_legend=True,
        title=None,
    )
    _prefix_plotly_region_legend(fig_cluster, "cluster")

    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "scene"}, {"type": "scene"}]],
        subplot_titles=("Fate markers", "Cluster labels"),
        horizontal_spacing=0.03,
    )

    for tr in fig_marker.data:
        fig.add_trace(tr, row=1, col=1)
    for tr in fig_cluster.data:
        fig.add_trace(tr, row=1, col=2)

    scene1 = fig_marker.layout.scene.to_plotly_json() if fig_marker.layout.scene else {}
    scene2 = fig_cluster.layout.scene.to_plotly_json() if fig_cluster.layout.scene else {}

    fig.update_layout(
        scene=scene1,
        scene2=scene2,
        width=int(fig_size[0]),
        height=int(fig_size[1]),
        title=getattr(g, "organoid_str", None),
        legend=dict(
            x=1.02,
            y=1.0,
            xanchor="left",
            yanchor="top",
        ),
    )

    return fig