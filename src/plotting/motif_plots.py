import math
from collections import defaultdict

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.lines import Line2D
from plotly.subplots import make_subplots

from organograph.plotting.graphs import add_region_overlays, plot_graph_by_markers
from organograph.mesh.OrganoidMesh import OrganoidMesh

from src.data.metadata import get_graph_metadata


"""Plotting helpers for subgraph motifs and exemplar organoids."""


DEFAULT_MARKER_COLORS = {
    "Lysozyme": "#2C75D2",
    "Serotonin": "#F392E3",
    "Mucin 2": "#3EC1D9",
    "Chroma": "#D852CB",
    "Glucagon": "#983EE2",
    "LGR5": "#FFB431",
    "AldoB": "#F16C6A",
    "Agr2": "#359BD5",
    "KI67": "#808080",
    "none": "#EBEBEB",
}



def pyg_subgraph_to_nx(subgraph):
    """Convert a PyG subgraph to an undirected NetworkX graph."""
    G = nx.Graph()
    n = int(subgraph.x.shape[0])
    G.add_nodes_from(range(n))
    edge_index = subgraph.edge_index.detach().cpu().numpy()
    for u, v in edge_index.T:
        u = int(u)
        v = int(v)
        if u != v:
            G.add_edge(u, v)
    return G



def radial_layout_by_hops(G, center, angular_offset=0.0):
    """Place nodes on concentric circles based on hop distance from a center."""
    hop_dists = nx.single_source_shortest_path_length(G, center)
    rings = defaultdict(list)
    for node, h in hop_dists.items():
        rings[h].append(node)

    pos = {center: np.array([0.0, 0.0], dtype=float)}
    for h in sorted(rings.keys()):
        if h == 0:
            continue
        nodes = sorted(rings[h])
        n_ring = len(nodes)
        for j, node in enumerate(nodes):
            theta = angular_offset + 2.0 * np.pi * j / max(n_ring, 1)
            pos[node] = np.array([h * np.cos(theta), h * np.sin(theta)], dtype=float)
    return pos, hop_dists



def node_color_from_marker_priority(x_row, marker_names, marker_colors):
    """Assign one display color to a node based on the first positive marker."""
    x_row = np.asarray(x_row)
    for j, name in enumerate(marker_names):
        if x_row[j] > 0.5:
            return marker_colors.get(name, marker_colors["none"])
    return marker_colors["none"]



def plot_cluster_exemplar_subgraphs_radial(
    cluster_exemplars,
    cluster_id,
    marker_names,
    *,
    marker_colors=DEFAULT_MARKER_COLORS,
    max_cols=4,
    center_node_size=1100,
    other_node_size=500,
    edge_width=1.2,
):
    """Plot exemplar ego-subgraphs for one cluster using a radial hop layout."""
    items = cluster_exemplars[cluster_id]
    if len(items) == 0:
        print(f"Cluster {cluster_id}: no exemplars")
        return None

    n = len(items)
    ncols = min(max_cols, n)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.0 * ncols, 4.0 * nrows))

    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = np.array([axes])
    elif ncols == 1:
        axes = np.array([[ax] for ax in axes])

    axes_flat = axes.ravel()
    for ax, item in zip(axes_flat, items):
        sub = item["subgraph"]
        G = pyg_subgraph_to_nx(sub)
        center = int(sub.center_idx)
        pos, _ = radial_layout_by_hops(G, center)

        X = sub.x.detach().cpu().numpy()
        node_colors = [node_color_from_marker_priority(X[i], marker_names, marker_colors) for i in G.nodes()]
        node_sizes = [center_node_size if i == center else other_node_size for i in G.nodes()]
        node_edgecolors = ["black" if i == center else "gray" for i in G.nodes()]

        nx.draw_networkx_edges(G, pos, ax=ax, width=edge_width, edge_color="gray")
        nx.draw_networkx_nodes(
            G,
            pos,
            ax=ax,
            node_color=node_colors,
            node_size=node_sizes,
            edgecolors=node_edgecolors,
            linewidths=1.5,
        )

        organoid_str = getattr(sub, "organoid_str", None)
        title = (
            f"C{cluster_id} | p={item['probability']:.3f}\n"
            f"g={item['graph_index']} node={item['node_index']}\n"
            f"y={item['y_true']:.3f}, ŷ={item['y_pred']:.3f}"
        )
        if organoid_str is not None:
            title = f"{organoid_str}\n" + title
        ax.set_title(title, fontsize=9)
        ax.set_axis_off()
        ax.set_aspect("equal")

    for ax in axes_flat[n:]:
        ax.set_axis_off()

    plt.tight_layout()
    return fig, axes



def plot_marker_color_legend(marker_colors=DEFAULT_MARKER_COLORS, ncols=5):
    """Plot a standalone legend for marker colors."""
    handles = [
        Line2D(
            [0], [0], marker="o", color="none", markerfacecolor=color,
            markeredgecolor="black" if name == "none" else "gray",
            markersize=10, label=name,
        )
        for name, color in marker_colors.items()
    ]
    fig, ax = plt.subplots(figsize=(1.8 * ncols, 1.2))
    ax.legend(handles=handles, loc="center", ncol=ncols, frameon=False)
    ax.axis("off")
    plt.tight_layout()
    return fig, ax



def pyg_graph_to_nx_with_centroids(pyg_graph, *, marker_names, meta_lookup=None, normalize_mesh=True):
    """Convert one PyG graph to NetworkX and reconstruct centroids from mesh projection IDs."""
    md = get_graph_metadata(pyg_graph, meta_lookup=meta_lookup, strict=True)
    mesh_path = md.get("mesh_path")
    proj_vertex_ids = md.get("proj_vertex_ids")

    if mesh_path is None:
        raise ValueError(f"{getattr(pyg_graph, 'organoid_str', None)} missing meta['mesh_path']")
    if proj_vertex_ids is None:
        raise ValueError(f"{getattr(pyg_graph, 'organoid_str', None)} missing meta['proj_vertex_ids']")

    proj_vertex_ids = np.asarray(proj_vertex_ids, dtype=np.int64).reshape(-1)
    n_nodes = int(pyg_graph.x.shape[0])
    if proj_vertex_ids.shape[0] != n_nodes:
        raise ValueError(f"proj_vertex_ids length mismatch: {proj_vertex_ids.shape[0]} vs {n_nodes}")

    mesh = OrganoidMesh(mesh_path)
    if normalize_mesh:
        mesh.normalize_inplace()

    centroids = np.asarray(mesh.v[proj_vertex_ids], dtype=float)
    G = nx.Graph()
    G.graph["organoid_str"] = getattr(pyg_graph, "organoid_str", None)
    G.graph["marker_names"] = list(marker_names)

    X = pyg_graph.x.detach().cpu().numpy()
    for i in range(n_nodes):
        G.add_node(i, centroid=centroids[i, :3], x=X[i].copy(), marker_bin=X[i].copy(), markers_bin=X[i].copy())

    edge_index = pyg_graph.edge_index.detach().cpu().numpy()
    for u, v in edge_index.T:
        u = int(u)
        v = int(v)
        if u != v:
            G.add_edge(u, v)
    return G



def plot_cluster_exemplars_full_organoid(
    cluster_exemplars,
    graphs,
    *,
    marker_names,
    meta_lookup=None,
    marker_colors=DEFAULT_MARKER_COLORS,
    max_graphs_per_cluster=3,
    backend="plotly",
    node_size=4,
    edge_width=0.5,
    overlay_size=8,
    overlay_alpha=0.85,
    fig_size=(900, 700),
    view=None,
    show_center_as_separate_overlay=True,
):
    """Plot full organoids with exemplar regions overlaid for each cluster."""
    marker_map = [
        {"marker": name, "color": marker_colors[name], "name": name}
        for name in marker_names
        if name in marker_colors and name != "none"
    ]

    figs_by_cluster = {}
    for k, items in cluster_exemplars.items():
        figs_k = []
        for j, item in enumerate(items[:max_graphs_per_cluster]):
            gi = int(item["graph_index"])
            full_pyg = graphs[gi]
            sub = item["subgraph"]
            full_nx = pyg_graph_to_nx_with_centroids(full_pyg, marker_names=marker_names, meta_lookup=meta_lookup)

            region_nodes = set(np.asarray(sub.orig_nodes.detach().cpu().numpy(), dtype=int).tolist())
            center_node = int(sub.orig_center)

            fig = plot_graph_by_markers(
                full_nx,
                marker_map=marker_map,
                backend=backend,
                baseline_color=marker_colors["none"],
                node_size=node_size,
                edge_width=edge_width,
                priority="first",
                add_legend=True,
                legend_baseline_name="none",
                fig_size=fig_size,
                view=view,
            )

            regions = [region_nodes]
            colors = ["green"]
            if show_center_as_separate_overlay:
                regions.append({center_node})
                colors.append("red")

            add_region_overlays(
                fig,
                full_nx,
                regions=regions,
                backend=backend,
                colors=colors,
                size=overlay_size,
                alpha=overlay_alpha,
                name_prefix=f"cluster {k}",
            )

            organoid_str = getattr(full_pyg, "organoid_str", None)
            fig.update_layout(
                title=(
                    f"Cluster C{k} | exemplar {j+1}/{min(len(items), max_graphs_per_cluster)} | {organoid_str}<br>"
                    f"graph={gi}, node={item['node_index']}, p={item['probability']:.3f}, "
                    f"y={item['y_true']:.3f}, ŷ={item['y_pred']:.3f}"
                )
            )
            figs_k.append(fig)

        figs_by_cluster[k] = figs_k
    return figs_by_cluster



def combine_plotly_3d_figures_in_row(figs, subplot_titles=None, width_per_fig=420, height=420):
    """Combine multiple Plotly 3D figures into a single row."""
    n = len(figs)
    if n == 0:
        raise ValueError("No figures provided.")
    subplot_titles = [""] * n if subplot_titles is None else subplot_titles

    fig_row = make_subplots(
        rows=1,
        cols=n,
        specs=[[{"type": "scene"} for _ in range(n)]],
        subplot_titles=subplot_titles,
        horizontal_spacing=0.02,
    )

    for col, fig in enumerate(figs, start=1):
        for tr in fig.data:
            fig_row.add_trace(tr, row=1, col=col)

        scene_src = fig.layout.scene if "scene" in fig.layout else None
        scene_name = "scene" if col == 1 else f"scene{col}"
        if scene_src is not None:
            scene_update = {}
            for attr in ["camera", "xaxis", "yaxis", "zaxis", "aspectmode", "bgcolor"]:
                if hasattr(scene_src, attr) and getattr(scene_src, attr) is not None:
                    scene_update[attr] = getattr(scene_src, attr)
            fig_row.layout[scene_name].update(scene_update)

    fig_row.update_layout(width=width_per_fig * n, height=height, margin=dict(l=10, r=10, t=40, b=10), showlegend=(n == 1))
    return fig_row



def plot_cluster_exemplars_full_organoid_rows(
    cluster_exemplars,
    graphs,
    *,
    marker_names,
    meta_lookup=None,
    marker_colors=DEFAULT_MARKER_COLORS,
    max_graphs_per_cluster=3,
    backend="plotly",
    node_size=3,
    edge_width=0.5,
    overlay_size=6,
    overlay_alpha=0.85,
    fig_size_single=(700, 600),
    view=None,
    show_center_as_separate_overlay=True,
    width_per_fig=420,
    row_height=420,
):
    """Create one combined Plotly row per cluster for full-organoid exemplar plots."""
    figs_by_cluster = plot_cluster_exemplars_full_organoid(
        cluster_exemplars,
        graphs,
        marker_names=marker_names,
        meta_lookup=meta_lookup,
        marker_colors=marker_colors,
        max_graphs_per_cluster=max_graphs_per_cluster,
        backend=backend,
        node_size=node_size,
        edge_width=edge_width,
        overlay_size=overlay_size,
        overlay_alpha=overlay_alpha,
        fig_size=fig_size_single,
        view=view,
        show_center_as_separate_overlay=show_center_as_separate_overlay,
    )

    row_figs_by_cluster = {}
    for k, figs in figs_by_cluster.items():
        if len(figs) == 0:
            continue

        subplot_titles = []
        for item in cluster_exemplars[k][:max_graphs_per_cluster]:
            gi = item["graph_index"]
            organoid_str = getattr(graphs[gi], "organoid_str", None) or f"g{gi}"
            subplot_titles.append(
                f"{organoid_str}<br>p={item['probability']:.3f}, y={item['y_true']:.3f}, ŷ={item['y_pred']:.3f}"
            )

        row_fig = combine_plotly_3d_figures_in_row(figs, subplot_titles=subplot_titles, width_per_fig=width_per_fig, height=row_height)
        row_fig.update_layout(title=f"Cluster C{k}")
        row_figs_by_cluster[k] = row_fig

    return row_figs_by_cluster