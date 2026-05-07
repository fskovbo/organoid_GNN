import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.analysis.marker_stats import (
    compute_markerwise_nll,
    compute_markerwise_residuals,
    compute_nodewise_nll,
)
from src.data.metadata import get_graph_metadata
from src.inference.predict import predict_targets


"""Utilities for evaluating curvature predictions and building analysis tables."""


def _select_target(a, target_index=0):
    a = np.asarray(a, dtype=np.float64)
    if a.ndim == 2:
        return a[:, int(target_index)]
    return a.reshape(-1)


# -----------------------------------------------------------------------------
# Existing helpers
# -----------------------------------------------------------------------------


def get_graph_slice_bounds(graphs, graph_index):
    """Return ``(start, end)`` indices into concatenated node arrays for one graph."""
    sizes = [int(g.y.shape[0]) for g in graphs]
    start = int(np.sum(sizes[:graph_index]))
    end = start + sizes[graph_index]
    return start, end



def split_node_arrays_by_graph(graphs, *arrays):
    """Split concatenated node-wise arrays into per-graph chunks."""
    sizes = [int(g.y.shape[0]) for g in graphs]
    total = int(np.sum(sizes))

    split_arrays = []
    for arr in arrays:
        arr = np.asarray(arr)
        if len(arr) != total:
            raise ValueError(f"Array length mismatch: expected {total}, got {len(arr)}")
        parts = []
        offset = 0
        for n in sizes:
            parts.append(arr[offset:offset + n])
            offset += n
        split_arrays.append(parts)

    return split_arrays if len(split_arrays) > 1 else split_arrays[0]



def build_graph_prediction_dataframe(graphs, y_true, y_pred, *, meta_lookup=None, include_metadata=True):
    """Build a per-graph performance table from node-wise predictions."""
    node_abs_err = np.abs(np.asarray(y_pred) - np.asarray(y_true))
    y_true_split, y_pred_split, err_split = split_node_arrays_by_graph(graphs, y_true, y_pred, node_abs_err)

    rows = []
    for i, (g, yt, yp, err) in enumerate(zip(graphs, y_true_split, y_pred_split, err_split)):
        row = {
            "graph_index": i,
            "organoid_str": getattr(g, "organoid_str", None),
            "num_nodes": int(len(yt)),
            "graph_mae": float(np.mean(err)),
            "graph_mse": float(np.mean((yp - yt) ** 2)),
            "graph_rmse": float(np.sqrt(np.mean((yp - yt) ** 2))),
        }

        if include_metadata:
            row.update(get_graph_metadata(g, meta_lookup=meta_lookup, strict=False, default={}))

        rows.append(row)

    return pd.DataFrame(rows)


def make_zero_centered_bin_edges(values, n_bins=5, eps=1e-12):
    """Return symmetric bin edges centered on zero for a 1-D value array."""
    if int(n_bins) != n_bins or n_bins < 1:
        raise ValueError("n_bins must be a positive integer.")
    n_bins = int(n_bins)
    if n_bins % 2 == 0:
        raise ValueError("n_bins must be odd so one bin is centered on zero.")

    values = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError("Cannot compute bins from an array with no finite values.")

    max_abs = float(np.max(np.abs(finite)))
    if max_abs <= eps:
        max_abs = 1.0

    return np.linspace(-max_abs, max_abs, n_bins + 1, dtype=np.float64)


def assign_bins(values, edges):
    """Assign values to bins defined by edges, returning -1 for non-finite values."""
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    edges = np.asarray(edges, dtype=np.float64)
    if edges.ndim != 1 or edges.size < 2:
        raise ValueError("edges must be a 1-D array with at least two entries.")

    bins = np.full(values.shape[0], -1, dtype=int)
    finite = np.isfinite(values)
    if np.any(finite):
        bins[finite] = np.searchsorted(edges[1:-1], values[finite], side="right")
        bins[finite] = np.clip(bins[finite], 0, edges.size - 2)
    return bins


def _bin_confusion_dataframe(true_bins, pred_bins, n_bins):
    true_bins = np.asarray(true_bins, dtype=int)
    pred_bins = np.asarray(pred_bins, dtype=int)
    valid = (true_bins >= 0) & (pred_bins >= 0)
    mat = np.zeros((n_bins, n_bins), dtype=int)
    for tb, pb in zip(true_bins[valid], pred_bins[valid]):
        mat[int(tb), int(pb)] += 1
    labels = [f"bin_{i}" for i in range(n_bins)]
    return pd.DataFrame(mat, index=labels, columns=labels)


def _graph_node_rows(graphs):
    graph_index = []
    node_index = []
    organoid_str = []
    for gi, g in enumerate(graphs):
        n = int(g.y.shape[0])
        graph_index.extend([gi] * n)
        node_index.extend(range(n))
        organoid_str.extend([getattr(g, "organoid_str", None)] * n)
    return (
        np.asarray(graph_index, dtype=int),
        np.asarray(node_index, dtype=int),
        np.asarray(organoid_str, dtype=object),
    )


def _accuracy_summary_row(label, true_bins, pred_bins):
    true_bins = np.asarray(true_bins, dtype=int)
    pred_bins = np.asarray(pred_bins, dtype=int)
    valid = (true_bins >= 0) & (pred_bins >= 0)
    correct = valid & (true_bins == pred_bins)
    n_valid = int(valid.sum())
    n_correct = int(correct.sum())
    return {
        "scope": label,
        "n_nodes": int(true_bins.size),
        "n_valid": n_valid,
        "n_correct": n_correct,
        "bin_accuracy": float(n_correct / n_valid) if n_valid else np.nan,
    }


def _marker_bin_accuracy_dataframe(
    true_bins,
    pred_bins,
    X,
    marker_names=None,
    *,
    include_negative=True,
):
    true_bins = np.asarray(true_bins, dtype=int)
    pred_bins = np.asarray(pred_bins, dtype=int)
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError(f"X must be a 2-D marker matrix, got shape {X.shape}.")
    if X.shape[0] != true_bins.shape[0]:
        raise ValueError(
            f"X row count mismatch: expected {true_bins.shape[0]}, got {X.shape[0]}."
        )

    n_markers = X.shape[1]
    if marker_names is None:
        marker_names = [f"marker_{i}" for i in range(n_markers)]
    else:
        marker_names = list(marker_names)
        if len(marker_names) != n_markers:
            raise ValueError(
                f"marker_names length mismatch: expected {n_markers}, got {len(marker_names)}."
            )

    rows = []
    for mi, name in enumerate(marker_names):
        pos = X[:, mi] > 0.5
        row = _accuracy_summary_row(f"{name}+", true_bins[pos], pred_bins[pos])
        row.update(
            {
                "marker_index": mi,
                "marker_name": name,
                "subset": "positive",
            }
        )
        rows.append(row)

        if include_negative:
            neg = ~pos
            row = _accuracy_summary_row(f"{name}-", true_bins[neg], pred_bins[neg])
            row.update(
                {
                    "marker_index": mi,
                    "marker_name": name,
                    "subset": "negative",
                }
            )
            rows.append(row)

    return pd.DataFrame(rows)


def compute_zero_centered_bin_accuracy(
    graphs,
    y_true,
    y_pred,
    *,
    n_bins=5,
    scope="dataset",
    target_index=0,
    meta_lookup=None,
    include_metadata=True,
    X=None,
    marker_names=None,
    marker_resolved=False,
    include_marker_negative=True,
):
    """Evaluate prediction accuracy by matching zero-centered true/predicted bins.

    True and predicted values are binned separately. With ``scope="dataset"``,
    one true binning and one predicted binning are fit across all supplied nodes.
    With ``scope="organoid"``, separate true/predicted bin edges are fit within
    each graph/organoid.

    Returns
    -------
    node_df : pandas.DataFrame
        One row per node with true/predicted values, bin indices, and correctness.
    summary : dict
        Overall accuracy, per-graph accuracy, confusion matrix, and bin edges.
        If ``marker_resolved=True``, also contains ``marker`` with per-marker
        positive-node accuracy, plus marker-negative complements when requested.
    """
    if scope not in {"dataset", "organoid", "graph"}:
        raise ValueError("scope must be one of: 'dataset', 'organoid', 'graph'")
    if scope == "graph":
        scope = "organoid"

    y_true = _select_target(y_true, target_index=target_index)
    y_pred = _select_target(y_pred, target_index=target_index)
    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError(
            f"y_true and y_pred length mismatch: {y_true.shape[0]} vs {y_pred.shape[0]}"
        )

    graph_index, node_index, organoid_str = _graph_node_rows(graphs)
    if y_true.shape[0] != graph_index.shape[0]:
        raise ValueError(
            f"Prediction length mismatch: expected {graph_index.shape[0]} nodes from graphs, "
            f"got {y_true.shape[0]}."
        )

    true_bins = np.full(y_true.shape[0], -1, dtype=int)
    pred_bins = np.full(y_pred.shape[0], -1, dtype=int)
    edge_records = []

    if scope == "dataset":
        true_edges = make_zero_centered_bin_edges(y_true, n_bins=n_bins)
        pred_edges = make_zero_centered_bin_edges(y_pred, n_bins=n_bins)
        true_bins[:] = assign_bins(y_true, true_edges)
        pred_bins[:] = assign_bins(y_pred, pred_edges)
        edge_records.append(
            {
                "scope": "dataset",
                "graph_index": None,
                "organoid_str": None,
                "true_edges": true_edges,
                "pred_edges": pred_edges,
            }
        )
    else:
        for gi, g in enumerate(graphs):
            mask = graph_index == gi
            true_edges = make_zero_centered_bin_edges(y_true[mask], n_bins=n_bins)
            pred_edges = make_zero_centered_bin_edges(y_pred[mask], n_bins=n_bins)
            true_bins[mask] = assign_bins(y_true[mask], true_edges)
            pred_bins[mask] = assign_bins(y_pred[mask], pred_edges)
            edge_records.append(
                {
                    "scope": "organoid",
                    "graph_index": gi,
                    "organoid_str": getattr(g, "organoid_str", None),
                    "true_edges": true_edges,
                    "pred_edges": pred_edges,
                }
            )

    valid = (true_bins >= 0) & (pred_bins >= 0)
    bin_correct = valid & (true_bins == pred_bins)

    node_df = pd.DataFrame(
        {
            "graph_index": graph_index,
            "node_index": node_index,
            "organoid_str": organoid_str,
            "y_true": y_true,
            "y_pred": y_pred,
            "true_bin": true_bins,
            "pred_bin": pred_bins,
            "bin_correct": bin_correct,
            "valid_bin": valid,
        }
    )

    per_graph_rows = []
    for gi, g in enumerate(graphs):
        mask = graph_index == gi
        row = _accuracy_summary_row(f"graph_{gi}", true_bins[mask], pred_bins[mask])
        row["graph_index"] = gi
        row["organoid_str"] = getattr(g, "organoid_str", None)
        if include_metadata:
            row.update(get_graph_metadata(g, meta_lookup=meta_lookup, strict=False, default={}))
        per_graph_rows.append(row)

    per_graph = pd.DataFrame(per_graph_rows)
    overall = _accuracy_summary_row("overall", true_bins, pred_bins)
    overall.update(
        {
            "bin_accuracy_mean_graph": float(per_graph["bin_accuracy"].mean())
            if len(per_graph)
            else np.nan,
            "bin_accuracy_sem_graph": _safe_sem(per_graph["bin_accuracy"].dropna().to_numpy())
            if len(per_graph)
            else np.nan,
        }
    )

    summary = {
        "scope": scope,
        "n_bins": int(n_bins),
        "target_index": target_index,
        "overall": overall,
        "per_graph": per_graph,
        "confusion": _bin_confusion_dataframe(true_bins, pred_bins, int(n_bins)),
        "edges": edge_records,
    }

    if marker_resolved:
        if X is None:
            raise ValueError("marker_resolved=True requires X.")
        marker_df = _marker_bin_accuracy_dataframe(
            true_bins,
            pred_bins,
            X,
            marker_names=marker_names,
            include_negative=include_marker_negative,
        )
        summary["marker"] = marker_df

    return node_df, summary


# -----------------------------------------------------------------------------
# Existing markerwise evaluation code
# -----------------------------------------------------------------------------

def compute_markerwise_uncertainty_correlation(y, mu, log_var, X, target_index=0):
    y = _select_target(y, target_index=target_index)
    mu = _select_target(mu, target_index=target_index)
    var = np.exp(_select_target(log_var, target_index=target_index))
    X = np.asarray(X)
    M = X.shape[1]

    rho = np.full(M, np.nan)
    pval = np.full(M, np.nan)
    n_pos = np.zeros(M, dtype=int)

    for m in range(M):
        mask = X[:, m] > 0.5
        n = int(mask.sum())
        n_pos[m] = n
        if n < 10:
            continue
        r2 = (y[mask] - mu[mask]) ** 2
        v = var[mask]
        rho[m], pval[m] = spearmanr(v, r2)

    return rho, pval, n_pos



def compute_global_uncertainty_correlation(y, mu, log_var, target_index=0):
    return spearmanr(np.exp(_select_target(log_var, target_index)), (_select_target(y, target_index) - _select_target(mu, target_index)) ** 2)



def _safe_sem(x):
    x = np.asarray(x, dtype=np.float64)
    if x.size <= 1:
        return np.nan
    return float(np.std(x, ddof=1) / np.sqrt(x.size))



def _aggregate_subset_metrics(y, mu, log_var, X, mask, eps=1e-12, target_index=0):
    y = _select_target(y, target_index=target_index)
    mu = _select_target(mu, target_index=target_index)
    log_var = _select_target(log_var, target_index=target_index)
    mask = np.asarray(mask, dtype=bool)
    n = int(mask.sum())
    if n == 0:
        return {
            "n": 0,
            "mse_model": np.nan,
            "sem_mse_model": np.nan,
            "mse_base": np.nan,
            "sem_mse_base": np.nan,
            "var_model": np.nan,
            "sem_var_model": np.nan,
            "var_base": np.nan,
            "sem_var_base": np.nan,
            "nll_model": np.nan,
            "sem_nll_model": np.nan,
            "nll_base": np.nan,
            "sem_nll_base": np.nan,
            "rho": np.nan,
            "pval": np.nan,
        }

    y_s = np.asarray(y[mask], dtype=np.float64)
    mu_s = np.asarray(mu[mask], dtype=np.float64)
    log_var_s = np.asarray(log_var[mask], dtype=np.float64)
    var_s = np.maximum(np.exp(log_var_s), eps)

    mu_b = float(np.mean(y_s))
    var_b = float(np.var(y_s)) + eps

    res_model = y_s - mu_s
    res_base = y_s - mu_b

    nll_model_node = compute_nodewise_nll(y_s, mu_s, var_s)
    nll_base_node = compute_nodewise_nll(y_s, mu_b, var_b)

    rho, pval = (np.nan, np.nan)
    if n >= 10:
        rho, pval = spearmanr(var_s, res_model**2)

    return {
        "n": n,
        "mse_model": float(np.mean(res_model**2)),
        "sem_mse_model": _safe_sem(res_model**2),
        "mse_base": float(np.mean(res_base**2)),
        "sem_mse_base": _safe_sem(res_base**2),
        "var_model": float(np.mean(var_s)),
        "sem_var_model": _safe_sem(var_s),
        "var_base": var_b,
        "sem_var_base": _safe_sem(np.full_like(y_s, var_b, dtype=np.float64)),
        "nll_model": float(np.mean(nll_model_node)),
        "sem_nll_model": _safe_sem(nll_model_node),
        "nll_base": float(np.mean(nll_base_node)),
        "sem_nll_base": _safe_sem(nll_base_node),
        "rho": float(rho) if np.isfinite(rho) else np.nan,
        "pval": float(pval) if np.isfinite(pval) else np.nan,
    }



def eval_model_per_marker(model, g_val, device, target_transform=None, eps=1e-12, center_only=False, target_index=0):
    y, mu, log_var, X = predict_targets(
        g_val,
        model,
        device=device,
        return_log_var=True,
        center_only=center_only,
        target_transform=target_transform,
    )

    y_eval = _select_target(y, target_index=target_index)
    mu_eval = _select_target(mu, target_index=target_index)
    log_var_eval = _select_target(log_var, target_index=target_index)
    var = np.exp(log_var_eval)

    residuals_model, residuals_base, n_pos_resid, mu_pos_resid = compute_markerwise_residuals(y_eval, mu_eval, X)
    mse_model = np.array([np.mean(r**2) if r.size > 0 else np.nan for r in residuals_model])
    mse_base = np.array([np.mean(r**2) if r.size > 0 else np.nan for r in residuals_base])

    sem_mse_model = np.array([np.std(r**2, ddof=1) / np.sqrt(r.size) if r.size > 1 else np.nan for r in residuals_model])
    sem_mse_base = np.array([np.std(r**2, ddof=1) / np.sqrt(r.size) if r.size > 1 else np.nan for r in residuals_base])

    Xb = X > 0.5
    var_model = np.array([np.mean(var[Xb[:, m]]) if np.any(Xb[:, m]) else np.nan for m in range(X.shape[1])])
    sem_var_model = np.array([
        np.std(var[Xb[:, m]], ddof=1) / np.sqrt(np.sum(Xb[:, m])) if np.sum(Xb[:, m]) > 1 else np.nan
        for m in range(X.shape[1])
    ])

    var_base = mse_base
    sem_var_base = sem_mse_base

    nll_model_list, nll_base_list, delta_mean_list, delta_var_list, n_pos_nll = compute_markerwise_nll(
        y_true=y_eval,
        mu_model=mu_eval,
        log_var_model=log_var_eval,
        X=X,
        eps=eps,
    )

    nll_model = np.array([np.mean(v) if v.size > 0 else np.nan for v in nll_model_list])
    nll_base = np.array([np.mean(v) if v.size > 0 else np.nan for v in nll_base_list])
    sem_nll_model = np.array([np.std(v, ddof=1) / np.sqrt(v.size) if v.size > 1 else np.nan for v in nll_model_list])
    sem_nll_base = np.array([np.std(v, ddof=1) / np.sqrt(v.size) if v.size > 1 else np.nan for v in nll_base_list])

    delta_mean = np.array([np.mean(v) if v.size > 0 else np.nan for v in delta_mean_list])
    delta_var = np.array([np.mean(v) if v.size > 0 else np.nan for v in delta_var_list])

    rho_marker, pval_marker, n_pos_corr = compute_markerwise_uncertainty_correlation(y=y_eval, mu=mu_eval, log_var=log_var_eval, X=X)
    rho_global, pval_global = compute_global_uncertainty_correlation(y=y_eval, mu=mu_eval, log_var=log_var_eval)

    row_pos = (X > 0.5).sum(axis=1)
    aggregate = {
        "any_marker": _aggregate_subset_metrics(y_eval, mu_eval, log_var_eval, X, row_pos > 0, eps=eps),
        "no_marker": _aggregate_subset_metrics(y_eval, mu_eval, log_var_eval, X, row_pos == 0, eps=eps),
        "all_nodes": _aggregate_subset_metrics(y_eval, mu_eval, log_var_eval, X, np.ones_like(row_pos, dtype=bool), eps=eps),
    }

    return {
        "mse_model": mse_model,
        "sem_mse_model": sem_mse_model,
        "mse_base": mse_base,
        "sem_mse_base": sem_mse_base,
        "var_model": var_model,
        "sem_var_model": sem_var_model,
        "var_base": var_base,
        "sem_var_base": sem_var_base,
        "nll_model": nll_model,
        "sem_nll_model": sem_nll_model,
        "nll_base": nll_base,
        "sem_nll_base": sem_nll_base,
        "delta_mean": delta_mean,
        "delta_var": delta_var,
        "rho_marker": rho_marker,
        "pval_marker": pval_marker,
        "n_pos_corr": n_pos_corr,
        "rho_global": rho_global,
        "pval_global": pval_global,
        "aggregate": aggregate,
    }
