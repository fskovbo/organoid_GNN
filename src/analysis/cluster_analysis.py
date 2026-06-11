import numpy as np
import pandas as pd

from src.data.subgraphs import build_ego_subgraphs_for_center_specs


def _select_exemplar_scalar(value, target_index=None, name="value"):
    """Return a scalar for exemplar display from scalar or multi-target values.

    If `target_index` is None and `value` is multi-dimensional, the first
    target is used for backwards-compatible display titles. The full arrays can
    still be stored separately by callers.
    """
    arr = np.asarray(value)
    if arr.ndim == 0 or arr.size == 1:
        return float(arr.reshape(-1)[0])
    if target_index is None:
        target_index = 0
    try:
        return float(arr.reshape(-1)[int(target_index)])
    except Exception as exc:
        raise ValueError(
            f"Could not select scalar {name} from shape {arr.shape} "
            f"with target_index={target_index!r}."
        ) from exc


def _as_serializable_target_value(value):
    """Convert scalar or vector target value to a Python float/list for storage."""
    arr = np.asarray(value)
    if arr.ndim == 0 or arr.size == 1:
        return float(arr.reshape(-1)[0])
    return arr.astype(float).tolist()


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

def get_top_cluster_exemplar_indices_unique_graphs(
    extraction,
    clustering_result,
    top_k_per_cluster=5,
    require_assigned_label=True,
    *,
    prefer_low_variance=False,
    variance_weight=1.0,
    variance_mode="rank",
):
    """Pick representative examples per cluster with at most one exemplar per graph.

    By default, examples are ranked only by cluster probability, preserving the
    previous behavior. If ``prefer_low_variance=True``, examples are additionally
    prioritized when the model predicts low variance for that node.

    Parameters
    ----------
    prefer_low_variance : bool
        If True, combine high cluster probability with low predicted variance.
        Requires ``extraction.log_var`` to be present.
    variance_weight : float
        Strength of the variance preference. Larger values prioritize low
        variance more strongly. With ``variance_mode='rank'``, 1.0 gives roughly
        equal influence to probability rank and variance rank.
    variance_mode : {"rank", "score"}
        ``"rank"`` is robust to the numerical scale of log-variance and is the
        recommended default. ``"score"`` uses ``probability - variance_weight *
        standardized_log_variance``.
    """
    labels = np.asarray(clustering_result.labels, dtype=int)
    probs = clustering_result.probabilities
    graph_index = np.asarray(extraction.graph_index, dtype=int)

    if probs is None:
        raise ValueError("This helper expects soft cluster probabilities, but probabilities is None.")

    if prefer_low_variance:
        if extraction.log_var is None:
            raise ValueError(
                "prefer_low_variance=True requires extraction.log_var, but it is None. "
                "Make sure extract_node_embeddings() was run with a model that returns log_var."
            )
        log_var = np.asarray(extraction.log_var, dtype=float).reshape(len(labels), -1)
        if log_var.shape[1] == 1:
            uncertainty = log_var[:, 0]
        else:
            # For multi-target outputs, use the mean log-variance across targets.
            uncertainty = np.nanmean(log_var, axis=1)
    else:
        uncertainty = None

    K = probs.shape[1]
    out = {}
    for k in range(K):
        idx = np.where(labels == k)[0] if require_assigned_label else np.arange(len(labels))
        if len(idx) == 0:
            out[k] = np.array([], dtype=int)
            continue

        if prefer_low_variance:
            if variance_mode == "rank":
                # Smaller rank_score is better. This avoids assumptions about the
                # absolute scale/calibration of log-variance.
                prob_order = np.argsort(-probs[idx, k])
                var_order = np.argsort(uncertainty[idx])

                prob_rank = np.empty(len(idx), dtype=float)
                var_rank = np.empty(len(idx), dtype=float)
                prob_rank[prob_order] = np.arange(len(idx), dtype=float)
                var_rank[var_order] = np.arange(len(idx), dtype=float)

                rank_score = prob_rank + float(variance_weight) * var_rank
                idx_sorted = idx[np.argsort(rank_score)]

            elif variance_mode == "score":
                u = uncertainty[idx]
                u_std = np.nanstd(u)
                if not np.isfinite(u_std) or u_std == 0.0:
                    u_z = np.zeros_like(u, dtype=float)
                else:
                    u_z = (u - np.nanmean(u)) / u_std
                score = probs[idx, k] - float(variance_weight) * u_z
                idx_sorted = idx[np.argsort(-score)]

            else:
                raise ValueError("variance_mode must be 'rank' or 'score'")
        else:
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


def _marker_presence_within_hops(
    graphs,
    extraction,
    *,
    num_hops,
    n_markers,
    marker_threshold=0.5,
):
    """Return whether each extracted node has each marker within its ego-neighborhood."""
    if int(num_hops) < 0:
        raise ValueError("num_hops must be non-negative.")
    graph_index = np.asarray(extraction.graph_index, dtype=int)
    local_node_index = np.asarray(extraction.local_node_index, dtype=int)
    if graph_index.shape != local_node_index.shape:
        raise ValueError("extraction graph and local-node indices must have equal shape.")

    presence = np.zeros((len(graph_index), int(n_markers)), dtype=bool)
    rows_by_graph = {}
    for row, graph_id in enumerate(graph_index):
        rows_by_graph.setdefault(int(graph_id), []).append(int(row))

    for graph_id, rows in rows_by_graph.items():
        graph = graphs[graph_id]
        x = graph.x.detach().cpu().numpy() if hasattr(graph.x, "detach") else np.asarray(graph.x)
        if x.ndim != 2 or x.shape[1] < n_markers:
            raise ValueError(
                f"Graph {graph_id} has marker shape {x.shape}; expected at least {n_markers} columns."
            )

        within_hops = np.asarray(x[:, :n_markers] > marker_threshold, dtype=bool)
        edge_index = (
            graph.edge_index.detach().cpu().numpy()
            if hasattr(graph.edge_index, "detach")
            else np.asarray(graph.edge_index)
        )
        if edge_index.size:
            source = edge_index[0].astype(int, copy=False)
            target = edge_index[1].astype(int, copy=False)
            for _ in range(int(num_hops)):
                previous = within_hops
                expanded = previous.copy()
                # Treat cell adjacency as undirected, matching the biological
                # ego-neighborhood even if only one edge direction is stored.
                np.logical_or.at(expanded, source, previous[target])
                np.logical_or.at(expanded, target, previous[source])
                within_hops = expanded

        rows_array = np.asarray(rows, dtype=int)
        nodes = local_node_index[rows_array]
        if np.any(nodes < 0) or np.any(nodes >= within_hops.shape[0]):
            raise IndexError(f"Extraction contains invalid node indices for graph {graph_id}.")
        presence[rows_array] = within_hops[nodes]

    return presence


def get_marker_enriched_cluster_exemplar_indices_unique_graphs(
    graphs,
    extraction,
    clustering_result,
    marker_names,
    *,
    top_k_per_cluster=5,
    num_hops=2,
    require_assigned_label=True,
    marker_threshold=0.5,
    min_global_positive_nodes=5,
    min_cluster_positive_nodes=1,
    min_enrichment_ratio=1.25,
):
    """Select cluster exemplars by enriched-marker coverage, then confidence.

    The enrichment score is ``P(marker positive | cluster) / P(marker positive)``.
    Qualifying markers are considered from highest to lowest enrichment. For each
    still-uncovered marker, the highest-membership candidate whose ego-subgraph
    contains that marker is selected, subject to at most one exemplar per graph.
    Remaining slots are filled by descending cluster-membership probability.

    Returns
    -------
    indices_by_cluster : dict[int, ndarray]
        Extraction row indices in selection order.
    diagnostics : dict
        Dataframe-friendly marker and selection records.
    """
    marker_names = list(marker_names)
    labels = np.asarray(clustering_result.labels, dtype=int)
    probabilities = clustering_result.probabilities
    graph_index = np.asarray(extraction.graph_index, dtype=int)
    x_markers = np.asarray(extraction.x_markers)

    if probabilities is None:
        raise ValueError("Marker-enriched exemplar selection requires soft cluster probabilities.")
    probabilities = np.asarray(probabilities, dtype=float)
    if x_markers.ndim != 2 or x_markers.shape[0] != len(labels):
        raise ValueError("extraction.x_markers must align with clustering labels.")
    if x_markers.shape[1] < len(marker_names):
        raise ValueError("marker_names is longer than extraction.x_markers columns.")
    if probabilities.shape[0] != len(labels):
        raise ValueError("Cluster probabilities must align with clustering labels.")
    if not 0 <= float(marker_threshold):
        raise ValueError("marker_threshold must be non-negative.")
    if int(top_k_per_cluster) < 1:
        raise ValueError("top_k_per_cluster must be at least 1.")
    if int(min_global_positive_nodes) < 1:
        raise ValueError("min_global_positive_nodes must be at least 1.")
    if int(min_cluster_positive_nodes) < 1:
        raise ValueError("min_cluster_positive_nodes must be at least 1.")
    if float(min_enrichment_ratio) <= 0:
        raise ValueError("min_enrichment_ratio must be positive.")

    positive = x_markers[:, :len(marker_names)] > float(marker_threshold)
    global_counts = positive.sum(axis=0).astype(int)
    global_prevalence = global_counts / max(len(labels), 1)
    neighborhood_presence = _marker_presence_within_hops(
        graphs,
        extraction,
        num_hops=num_hops,
        n_markers=len(marker_names),
        marker_threshold=marker_threshold,
    )

    indices_by_cluster = {}
    marker_rows = []
    selection_rows = []
    selection_details = {}

    for cluster in range(probabilities.shape[1]):
        assigned = labels == cluster
        candidate_indices = (
            np.flatnonzero(assigned)
            if require_assigned_label
            else np.arange(len(labels), dtype=int)
        )
        cluster_size = int(assigned.sum())
        cluster_counts = positive[assigned].sum(axis=0).astype(int)
        cluster_prevalence = cluster_counts / max(cluster_size, 1)
        enrichment = np.divide(
            cluster_prevalence,
            global_prevalence,
            out=np.full(len(marker_names), np.nan, dtype=float),
            where=global_prevalence > 0,
        )
        qualifies = (
            (global_counts >= int(min_global_positive_nodes))
            & (cluster_counts >= int(min_cluster_positive_nodes))
            & np.isfinite(enrichment)
            & (enrichment >= float(min_enrichment_ratio))
        )
        qualifying_markers = np.flatnonzero(qualifies)
        qualifying_markers = sorted(
            qualifying_markers.tolist(),
            key=lambda marker: (
                -enrichment[marker],
                global_counts[marker],
                marker_names[marker],
            ),
        )

        chosen = []
        chosen_set = set()
        used_graphs = set()
        covered_markers = set()

        for target_marker in qualifying_markers:
            if target_marker in covered_markers or len(chosen) >= top_k_per_cluster:
                continue
            eligible = [
                int(row)
                for row in candidate_indices
                if int(row) not in chosen_set
                and int(graph_index[row]) not in used_graphs
                and neighborhood_presence[row, target_marker]
            ]
            if not eligible:
                continue
            selected = max(
                eligible,
                key=lambda row: (
                    probabilities[row, cluster],
                    int(np.sum([
                        neighborhood_presence[row, marker]
                        for marker in qualifying_markers
                        if marker not in covered_markers
                    ])),
                    -row,
                ),
            )
            newly_covered = {
                marker
                for marker in qualifying_markers
                if marker not in covered_markers
                and neighborhood_presence[selected, marker]
            }
            chosen.append(selected)
            chosen_set.add(selected)
            used_graphs.add(int(graph_index[selected]))
            covered_markers.update(newly_covered)
            selection_details[(cluster, selected)] = {
                "selection_reason": "marker_coverage",
                "target_marker": marker_names[target_marker],
                "newly_covered_markers": [
                    marker_names[marker] for marker in qualifying_markers
                    if marker in newly_covered
                ],
            }

        confidence_order = candidate_indices[
            np.argsort(-probabilities[candidate_indices, cluster], kind="stable")
        ]
        for row in confidence_order:
            row = int(row)
            if len(chosen) >= top_k_per_cluster:
                break
            graph_id = int(graph_index[row])
            if row in chosen_set or graph_id in used_graphs:
                continue
            chosen.append(row)
            chosen_set.add(row)
            used_graphs.add(graph_id)
            selection_details[(cluster, row)] = {
                "selection_reason": "confidence_fill",
                "target_marker": None,
                "newly_covered_markers": [],
            }

        indices_by_cluster[cluster] = np.asarray(chosen, dtype=int)
        for rank, row in enumerate(chosen, start=1):
            details = selection_details[(cluster, row)]
            selection_rows.append({
                "cluster": cluster,
                "rank": rank,
                "row_index": row,
                "graph_index": int(graph_index[row]),
                "node_index": int(extraction.local_node_index[row]),
                "probability": float(probabilities[row, cluster]),
                **details,
            })

        for marker, marker_name in enumerate(marker_names):
            marker_rows.append({
                "cluster": cluster,
                "marker_index": marker,
                "marker": marker_name,
                "global_positive_nodes": int(global_counts[marker]),
                "cluster_positive_nodes": int(cluster_counts[marker]),
                "global_prevalence": float(global_prevalence[marker]),
                "cluster_prevalence": float(cluster_prevalence[marker]),
                "enrichment_ratio": float(enrichment[marker]),
                "qualifies": bool(qualifies[marker]),
                "candidate_subgraphs_with_marker": int(
                    neighborhood_presence[candidate_indices, marker].sum()
                ),
                "covered": bool(marker in covered_markers),
            })

    return indices_by_cluster, {
        "marker_diagnostics": pd.DataFrame(marker_rows),
        "selection_diagnostics": pd.DataFrame(selection_rows),
        "selection_details": selection_details,
        "neighborhood_marker_presence": neighborhood_presence,
    }


def _build_cluster_exemplar_items(
    graphs,
    extraction,
    clustering_result,
    indices_by_cluster,
    *,
    num_hops,
    copy_graph_level_attrs,
    target_index,
    store_all_targets,
    selection_details=None,
):
    """Build exemplar subgraphs and metadata from preselected extraction rows."""
    probabilities = clustering_result.probabilities
    exemplars = {}

    for cluster, rows in indices_by_cluster.items():
        center_specs = [
            (int(extraction.graph_index[row]), int(extraction.local_node_index[row]))
            for row in rows
        ]
        subgraphs = build_ego_subgraphs_for_center_specs(
            graphs=graphs,
            center_specs=center_specs,
            num_hops=num_hops,
            copy_graph_level_attrs=copy_graph_level_attrs,
        )

        items = []
        for row, subgraph in zip(rows, subgraphs):
            row = int(row)
            y_true = extraction.y_true[row]
            y_pred = extraction.y_pred[row]
            item = {
                "row_index": row,
                "probability": float(probabilities[row, cluster]),
                "graph_index": int(extraction.graph_index[row]),
                "node_index": int(extraction.local_node_index[row]),
                "y_true": _select_exemplar_scalar(y_true, target_index, name="y_true"),
                "y_pred": _select_exemplar_scalar(y_pred, target_index, name="y_pred"),
                "subgraph": subgraph,
            }
            if selection_details is not None:
                item.update(selection_details.get((cluster, row), {}))
            if extraction.log_var is not None:
                log_var = extraction.log_var[row]
                item["log_var"] = _select_exemplar_scalar(
                    log_var, target_index, name="log_var"
                )
                item["pred_var"] = float(np.exp(item["log_var"]))
            if store_all_targets:
                item["y_true_all"] = _as_serializable_target_value(y_true)
                item["y_pred_all"] = _as_serializable_target_value(y_pred)
                if extraction.log_var is not None:
                    item["log_var_all"] = _as_serializable_target_value(
                        extraction.log_var[row]
                    )
            items.append(item)
        exemplars[cluster] = items

    return exemplars


def build_marker_enriched_cluster_exemplar_subgraphs(
    graphs,
    extraction,
    clustering_result,
    marker_names,
    *,
    top_k_per_cluster=5,
    num_hops=2,
    require_assigned_label=True,
    copy_graph_level_attrs=True,
    target_index=None,
    store_all_targets=True,
    marker_threshold=0.5,
    min_global_positive_nodes=5,
    min_cluster_positive_nodes=1,
    min_enrichment_ratio=1.25,
):
    """Build exemplars with greedy coverage of cluster-enriched markers."""
    indices, diagnostics = get_marker_enriched_cluster_exemplar_indices_unique_graphs(
        graphs,
        extraction,
        clustering_result,
        marker_names,
        top_k_per_cluster=top_k_per_cluster,
        num_hops=num_hops,
        require_assigned_label=require_assigned_label,
        marker_threshold=marker_threshold,
        min_global_positive_nodes=min_global_positive_nodes,
        min_cluster_positive_nodes=min_cluster_positive_nodes,
        min_enrichment_ratio=min_enrichment_ratio,
    )
    exemplars = _build_cluster_exemplar_items(
        graphs,
        extraction,
        clustering_result,
        indices,
        num_hops=num_hops,
        copy_graph_level_attrs=copy_graph_level_attrs,
        target_index=target_index,
        store_all_targets=store_all_targets,
        selection_details=diagnostics["selection_details"],
    )
    return exemplars, diagnostics



def build_cluster_exemplar_subgraphs(
    graphs,
    extraction,
    clustering_result,
    *,
    top_k_per_cluster=5,
    num_hops=2,
    require_assigned_label=True,
    copy_graph_level_attrs=True,
    target_index=None,
    store_all_targets=True,
    prefer_low_variance=False,
    variance_weight=1.0,
    variance_mode="rank",
):
    """Build ego-subgraphs around representative examples in each cluster.

    By default, exemplars are the most cluster-confident nodes, matching the
    previous behavior. Set ``prefer_low_variance=True`` to prefer examples where
    the model also predicts small variance.

    Parameters
    ----------
    target_index : int or None
        Target used for scalar display fields ``y_true`` and ``y_pred`` when
        extraction targets are multi-dimensional. If None, target 0 is used for
        backwards-compatible plot titles.
    store_all_targets : bool
        If True, also store full vector values as ``y_true_all`` and
        ``y_pred_all`` for multi-target extractions.
    prefer_low_variance : bool
        If True, rank examples using both high cluster probability and low
        predicted log-variance. Requires ``extraction.log_var``.
    variance_weight : float
        Strength of the variance preference. With ``variance_mode='rank'``, 1.0
        gives roughly equal influence to probability rank and variance rank.
    variance_mode : {"rank", "score"}
        ``"rank"`` is robust and recommended. ``"score"`` combines probability
        with standardized log-variance directly.
    """
    top_idx = get_top_cluster_exemplar_indices_unique_graphs(
        extraction,
        clustering_result,
        top_k_per_cluster=top_k_per_cluster,
        require_assigned_label=require_assigned_label,
        prefer_low_variance=prefer_low_variance,
        variance_weight=variance_weight,
        variance_mode=variance_mode,
    )

    return _build_cluster_exemplar_items(
        graphs,
        extraction,
        clustering_result,
        top_idx,
        num_hops=num_hops,
        copy_graph_level_attrs=copy_graph_level_attrs,
        target_index=target_index,
        store_all_targets=store_all_targets,
    )



def marker_label_from_x(x_row, marker_names, threshold=0.5):
    """Convert one binary marker row into a compact text label."""
    pos = [marker_names[j] for j, v in enumerate(np.asarray(x_row)) if v > threshold]
    return "None" if len(pos) == 0 else "|".join(pos)



def marker_labels_for_subgraph(subgraph, marker_names, threshold=0.5):
    """Convert all nodes in one subgraph to text marker labels."""
    X = subgraph.x.detach().cpu().numpy()
    return [marker_label_from_x(X[i], marker_names, threshold=threshold) for i in range(X.shape[0])]
