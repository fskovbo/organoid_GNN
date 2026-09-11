"""Paired marker ablations at observed or overridden organoid cell counts.

These are conditional predictor sensitivities. Sweeping the global count does
not change the graph, add cells, or simulate a developmental trajectory.
"""

import copy

import numpy as np
import pandas as pd
import torch

from src.analysis.perturbation import predict_subgraph_center_distribution
from src.graph.neighborhood import compute_hop_rings


def make_size_ablation_cases(subgraphs, marker_names, *, hops=(1, 2), seed=0):
    """Select one random positive source per center/marker/exact-hop ring.

    Build this manifest once, then reuse it for every model and size. A center
    with multiple positive markers contributes to each corresponding pair in
    ``expand_center_markers``. Missing source markers produce no case.
    """
    if not hops or any(int(h) != h or h < 0 for h in hops):
        raise ValueError("hops must be a nonempty sequence of nonnegative integers.")
    if len(set(hops)) != len(hops):
        raise ValueError("hops must be unique.")
    rng = np.random.default_rng(seed)
    cases = []
    for si, graph in enumerate(subgraphs):
        if graph.x.shape[1] != len(marker_names):
            raise ValueError("Marker names must match the fate-only feature matrix.")
        center = int(graph.center_idx)
        rings = compute_hop_rings(graph.edge_index, center, max(hops))
        center_names = [name for j, name in enumerate(marker_names) if graph.x[center, j] > 0.5]
        observed_n = float(graph.full_num_cells)
        if not np.isfinite(observed_n) or observed_n <= 0:
            raise ValueError("full_num_cells must be the positive original organoid count.")
        for hop in hops:
            nodes = np.asarray(rings[hop], dtype=int)
            for marker, name in enumerate(marker_names):
                positive = nodes[graph.x[nodes, marker].detach().cpu().numpy() > 0.5]
                if not len(positive):
                    continue
                source = int(rng.choice(positive))
                cases.append({
                    "case_id": len(cases), "subgraph_index": si,
                    "organoid_str": str(graph.organoid_str),
                    "orig_center": int(graph.orig_center),
                    "source_node": source,
                    "orig_source_node": int(graph.orig_nodes[source]),
                    "source_marker": marker, "source_marker_name": name,
                    "center_marker_names": center_names, "hop": int(hop),
                    "n_perturbed": 1, "observed_n": observed_n,
                    "timepoint": str(getattr(graph, "timepoint_label", "unknown")),
                })
    return cases


def _size_override(graph, count, size_center, size_scale, size_feature_index=0):
    result = copy.copy(graph)
    if count is not None:
        if not np.isfinite(count) or count <= 0:
            raise ValueError("Override cell count must be finite and positive.")
        features = graph.global_feat
        if features.ndim != 2 or features.shape[0] != 1:
            raise ValueError("global_feat must have shape (1, D).")
        if not 0 <= size_feature_index < features.shape[1]:
            raise ValueError("size_feature_index must select a global size feature.")
        result.global_feat = features.clone()
        result.global_feat[:, size_feature_index] = (np.log(count) - float(size_center)) / float(size_scale)
    return result


def evaluate_size_ablation(
    subgraphs, cases, model, target_transform, *, size_center, size_scale,
    count=None, batch_size=128, device=None, max_fold_change=2.0,
    size_feature_index=0,
):
    """Evaluate a fixed case manifest, without mutating its original graphs.

    ``target_transform`` must invert only the residual target transformation,
    e.g. a fitted AsinhStandardizeTransform, NOT the global baseline. The
    per-organoid baseline cancels from every paired curvature difference.
    Sweeps report no MSE: ground truth at the hypothetical count is unknown.
    Only ``size_feature_index`` changes during a sweep; other global features
    retain their observed values for both intact and perturbed predictions.
    """
    if not np.isfinite(size_scale) or size_scale <= 0 or batch_size < 1:
        raise ValueError("size_scale and batch_size must be positive.")
    if max_fold_change < 1:
        raise ValueError("max_fold_change must be >= 1.")
    if not cases:
        return pd.DataFrame()
    base_graphs = [_size_override(g, count, size_center, size_scale, size_feature_index) for g in subgraphs]
    base_z, base_lv = predict_subgraph_center_distribution(
        base_graphs, model, batch_size=batch_size, device=device,
    )
    _, base_mu, base_lv_physical = target_transform.inverse_distribution(
        None, base_z, log_var=base_lv,
    )
    base_mu = np.asarray(base_mu).reshape(-1)
    base_lv_physical = np.asarray(base_lv_physical).reshape(-1)
    rows = []
    # Bound memory even when thousands of cases share large neighborhoods.
    for start in range(0, len(cases), batch_size):
        chunk = cases[start:start + batch_size]
        perturbed = []
        for case in chunk:
            graph = copy.copy(base_graphs[case["subgraph_index"]])
            graph.x = graph.x.clone()
            graph.x[case["source_node"], case["source_marker"]] = 0
            perturbed.append(graph)
        pert_z, pert_lv = predict_subgraph_center_distribution(
            perturbed, model, batch_size=batch_size, device=device,
        )
        _, pert_mu, pert_lv_physical = target_transform.inverse_distribution(
            None, pert_z, log_var=pert_lv,
        )
        pert_mu = np.asarray(pert_mu).reshape(-1)
        pert_lv_physical = np.asarray(pert_lv_physical).reshape(-1)
        for j, case in enumerate(chunk):
            si = case["subgraph_index"]
            evaluated_n = case["observed_n"] if count is None else float(count)
            ratio = max(evaluated_n / case["observed_n"], case["observed_n"] / evaluated_n)
            row = dict(case)
            row.update({
                "analysis": "observed" if count is None else "sweep",
                "evaluated_n": evaluated_n, "size_fold_change": ratio,
                "within_local_size_window": ratio <= max_fold_change,
                "base_mu": float(base_mu[si]), "perturbed_mu": float(pert_mu[j]),
                "delta_mu": float(pert_mu[j] - base_mu[si]),
                "base_mu_transformed": float(base_z[si]),
                "perturbed_mu_transformed": float(pert_z[j]),
                "delta_mu_transformed": float(pert_z[j] - base_z[si]),
                "delta_variance": float(np.exp(pert_lv_physical[j]) - np.exp(base_lv_physical[si])),
            })
            if count is None:
                g = subgraphs[si]
                true_z = float(g.y[int(g.center_idx)].reshape(-1)[0])
                truth = float(np.asarray(target_transform.inverse(np.array([true_z]))).reshape(-1)[0])
                row["delta_mse"] = float((pert_mu[j] - truth) ** 2 - (base_mu[si] - truth) ** 2)
            else:
                row["delta_mse"] = np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def expand_center_markers(cases):
    """One row per positive center marker, keeping unmarked centers separate."""
    result = cases.copy()
    result["center_marker"] = result["center_marker_names"].map(
        lambda names: list(names) or ["unmarked"]
    )
    return result.explode("center_marker", ignore_index=True)


def summarize_by_organoid(frame, group_columns, value, *, bootstrap_samples=1000, seed=0):
    """Average cases and repeated model seeds within organoid, then bootstrap.

    Every organoid has equal weight. Intervals describe between-organoid
    variation for fitted models; they do not include retraining uncertainty.
    """
    columns = list(group_columns) + [
        "mean", "ci_low", "ci_high", "n_organoids", "n_rows",
    ]
    if frame.empty:
        return pd.DataFrame(columns=columns)
    if bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be positive.")
    rng = np.random.default_rng(seed)
    rows = []
    for key, group in frame.groupby(list(group_columns), observed=True, dropna=False, sort=True):
        key = key if isinstance(key, tuple) else (key,)
        group = group[np.isfinite(group[value])]
        if group.empty:
            continue
        values = group.groupby("organoid_str", observed=True)[value].mean().to_numpy()
        if len(values) > 1:
            draws = rng.choice(values, size=(bootstrap_samples, len(values)), replace=True).mean(axis=1)
            low, high = np.quantile(draws, [0.025, 0.975])
        else:
            low = high = np.nan
        rows.append(dict(zip(group_columns, key), mean=float(values.mean()),
                         ci_low=float(low), ci_high=float(high),
                         n_organoids=len(values), n_rows=len(group)))
    return pd.DataFrame(rows, columns=columns)
