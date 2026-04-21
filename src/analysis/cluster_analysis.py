import numpy as np
import pandas as pd

from src.data.subgraphs import build_ego_subgraphs_for_center_specs


# ---------------------------------------------------------------------------------
# Cluster-level summary utilities for marker, curvature, and metadata analyses.
# ---------------------------------------------------------------------------------

def cluster_marker_means(X, labels, *, center_mask=None):
    """Compute mean marker positivity per cluster."""
    X = np.asarray(X, dtype=float)
    labels = np.asarray(labels, dtype=int)
    K = int(labels.max()) + 1
    out = np.full((K, X.shape[1]), np.nan, dtype=float)

    for k in range(K):
        mask = labels == k
        if center_mask is not None:
            mask = mask & np.asarray(center_mask, dtype=bool)
        if np.any(mask):
            out[k] = X[mask].mean(axis=0)
    return out



def cluster_marker_distribution(X, labels, *, center_mask=None, threshold=0.5):
    """For each marker, compute how its positive nodes are distributed across clusters."""
    X = np.asarray(X, dtype=float)
    labels = np.asarray(labels, dtype=int)
    K = int(labels.max()) + 1
    out = np.full((K, X.shape[1]), np.nan, dtype=float)

    base_mask = np.ones(X.shape[0], dtype=bool) if center_mask is None else np.asarray(center_mask, dtype=bool)
    for m in range(X.shape[1]):
        pos_mask = (X[:, m] > threshold) & base_mask
        n_pos = int(pos_mask.sum())
        if n_pos == 0:
            continue
        for k in range(K):
            out[k, m] = float(np.sum(pos_mask & (labels == k)) / n_pos)
    return out



def cluster_order_from_values(labels, values, reducer=np.mean):
    """Order clusters by an aggregate of some node-wise value."""
    labels = np.asarray(labels, dtype=int)
    values = np.asarray(values)
    rows = []
    for k in range(int(labels.max()) + 1):
        mask = labels == k
        if np.any(mask):
            rows.append((k, float(reducer(values[mask]))))
    rows = sorted(rows, key=lambda x: x[1])
    return np.array([k for k, _ in rows], dtype=int)



def build_binned_cluster_fraction_table(
    labels,
    values,
    valid_mask,
    bin_func,
    *,
    bin_order,
    cluster_order_by="median_valid_value",
    include_all=True,
):
    """Build a table of per-cluster fractions across user-defined bins."""
    labels = np.asarray(labels)
    values = np.asarray(values)
    valid_mask = np.asarray(valid_mask, dtype=bool)

    df = pd.DataFrame({
        "cluster": labels,
        "value": values,
        "valid": valid_mask,
    })

    df["bin"] = [
        bin_func(v, ok)
        for v, ok in zip(df["value"].values, df["valid"].values)
    ]

    valid_df = df[df["valid"] & np.isfinite(df["value"])]

    if cluster_order_by == "median_valid_value":
        cluster_order = (
            valid_df.groupby("cluster")["value"]
            .median()
            .sort_values()
            .index
        )
    else:
        raise ValueError(f"Unknown cluster_order_by={cluster_order_by!r}")

    frac_clusters = pd.crosstab(
        df["cluster"],
        df["bin"],
        normalize="index",
    ).reindex(index=cluster_order, columns=bin_order, fill_value=0.0)

    if include_all:
        frac_all = (
            df["bin"]
            .value_counts(normalize=True)
            .reindex(bin_order, fill_value=0.0)
        )
        plot_table = pd.concat(
            [pd.DataFrame([frac_all.values], index=["All"], columns=bin_order),
             frac_clusters],
            axis=0,
        )
    else:
        plot_table = frac_clusters

    return plot_table, cluster_order, df


# -------------------------------------------------------------------------------------
# Helpers for selecting representative cluster exemplars and building ego-subgraphs.
# -------------------------------------------------------------------------------------

def get_top_cluster_exemplar_indices_unique_graphs(extraction, clustering_result, top_k_per_cluster=5, require_assigned_label=True):
    """Pick the most confident examples per cluster with at most one exemplar per graph."""
    labels = np.asarray(clustering_result.labels, dtype=int)
    probs = clustering_result.probabilities
    graph_index = np.asarray(extraction.graph_index, dtype=int)

    if probs is None:
        raise ValueError("This helper expects soft cluster probabilities, but probabilities is None.")

    K = probs.shape[1]
    out = {}
    for k in range(K):
        idx = np.where(labels == k)[0] if require_assigned_label else np.arange(len(labels))
        if len(idx) == 0:
            out[k] = np.array([], dtype=int)
            continue

        idx_sorted = idx[np.argsort(-probs[idx, k])]
        chosen = []
        used_graphs = set()
        for i in idx_sorted:
            gi = int(graph_index[i])
            if gi in used_graphs:
                continue
            chosen.append(int(i))
            used_graphs.add(gi)
            if len(chosen) >= top_k_per_cluster:
                break

        out[k] = np.asarray(chosen, dtype=int)

    return out



def build_cluster_exemplar_subgraphs(
    graphs,
    extraction,
    clustering_result,
    *,
    top_k_per_cluster=5,
    num_hops=2,
    require_assigned_label=True,
    copy_graph_level_attrs=True,
):
    """Build ego-subgraphs around the most confident examples in each cluster."""
    top_idx = get_top_cluster_exemplar_indices_unique_graphs(
        extraction,
        clustering_result,
        top_k_per_cluster=top_k_per_cluster,
        require_assigned_label=require_assigned_label,
    )

    probs = clustering_result.probabilities
    exemplars = {}

    for k, rows in top_idx.items():
        center_specs = [
            (int(extraction.graph_index[i]), int(extraction.local_node_index[i]))
            for i in rows
        ]
        subs = build_ego_subgraphs_for_center_specs(
            graphs=graphs,
            center_specs=center_specs,
            num_hops=num_hops,
            copy_graph_level_attrs=copy_graph_level_attrs,
        )

        items = []
        for i, sub in zip(rows, subs):
            items.append({
                "row_index": int(i),
                "probability": float(probs[i, k]),
                "graph_index": int(extraction.graph_index[i]),
                "node_index": int(extraction.local_node_index[i]),
                "y_true": float(extraction.y_true[i]),
                "y_pred": float(extraction.y_pred[i]),
                "subgraph": sub,
            })
        exemplars[k] = items

    return exemplars



def marker_label_from_x(x_row, marker_names, threshold=0.5):
    """Convert one binary marker row into a compact text label."""
    pos = [marker_names[j] for j, v in enumerate(np.asarray(x_row)) if v > threshold]
    return "None" if len(pos) == 0 else "|".join(pos)



def marker_labels_for_subgraph(subgraph, marker_names, threshold=0.5):
    """Convert all nodes in one subgraph to text marker labels."""
    X = subgraph.x.detach().cpu().numpy()
    return [marker_label_from_x(X[i], marker_names, threshold=threshold) for i in range(X.shape[0])]