import numpy as np
import torch
from torch_geometric.loader import DataLoader
import matplotlib.pyplot as plt

def gather_predictions(
    graphs,
    model=None,
    device=None,
    batch_size=32,
    num_workers=0,
    pin_memory=True,
    *,
    y_pred=None,
    use_baseline=False,
    mu_dict=None,
):
    """
    Return concatenated (y_true, y_pred) as 1D numpy arrays for a list of graphs.

    Two modes:
      A) Provide (model, graphs) → runs the model and returns predictions.
         If use_baseline=True, adds per-node baseline from mu_dict (residual workflow).
      B) Provide y_pred directly (numpy or tensor) → just concatenates y_true, aligns shape.

    If graphs were residualized (y is residual), and you want original y, attach with:
      - y_orig on each graph (as in your residual pipeline)

    Args
    ----
    graphs : list[Data]
    model  : torch.nn.Module or None
    device : str or None
    y_pred : optional precomputed predictions (np.ndarray or torch.Tensor), shape (sum N_i,)
    use_baseline : bool, if True adds baseline b_i = avg(mu_m) across positive markers
    mu_dict : dict from compute_marker_baseline_means (needed if use_baseline=True)

    Returns
    -------
    y_true : (K,) float64
    y_pred : (K,) float64
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 1) collect true y (prefer original if present)
    Ys = []
    for g in graphs:
        if hasattr(g, "y_orig"):
            Ys.append(g.y_orig.detach().cpu().numpy())
        else:
            Ys.append(g.y.detach().cpu().numpy())
    y_true = np.concatenate(Ys, axis=0).astype(np.float64)

    # 2) predictions
    if y_pred is not None:
        yp = y_pred
        if torch.is_tensor(yp):
            yp = yp.detach().cpu().numpy()
        yp = np.asarray(yp, dtype=np.float64).reshape(-1)
        if yp.shape[0] != y_true.shape[0]:
            raise ValueError("Provided y_pred length does not match total nodes.")
        return y_true, yp

    if model is None:
        raise ValueError("Provide either (model, graphs) or y_pred explicitly.")

    model = model.to(device).eval()
    loader = DataLoader(graphs, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=pin_memory)

    yhats = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device, non_blocking=True)
            yhat, _ = model(batch.x, batch.edge_index)  # residual OR full depending on your model
            if use_baseline:
                if mu_dict is None:
                    raise ValueError("use_baseline=True requires mu_dict.")
                # compute baseline b for this batch
                mu_pos = torch.as_tensor(mu_dict["mu_pos"], dtype=batch.x.dtype, device=batch.x.device)
                pos_counts = batch.x.sum(dim=1, keepdim=True).clamp_min(0.0)
                b = (batch.x @ mu_pos.view(-1, 1))
                b = torch.where(pos_counts > 0, b[:, 0] / pos_counts[:, 0], torch.tensor(mu_dict["mu_none"], device=batch.x.device, dtype=batch.x.dtype))
                yhat = yhat + b
            yhats.append(yhat.detach().cpu().numpy())

    y_pred = np.concatenate(yhats, axis=0).astype(np.float64)
    return y_true, y_pred


def binned_curvature_metrics(
    y_true, y_pred,
    bins=None, nbins=12, strategy="quantile",
    clip_percentiles=(1.0, 99.0)
):
    """
    Bin data by TRUE curvature and compute metrics per bin.

    Metrics per bin:
      - count
      - y_true_mean, y_true_std
      - y_pred_mean
      - bias = mean(y_pred - y_true)
      - MAE = mean(|y_pred - y_true|)
      - RMSE = sqrt(mean((y_pred - y_true)^2))
      - res_std = std(y_pred - y_true)

    Args
    ----
    y_true, y_pred : 1D arrays of same length
    bins           : explicit bin edges (array-like), or None to build from data
    nbins          : number of bins if bins=None
    strategy       : 'quantile' (equal counts) or 'uniform' (equal width)
    clip_percentiles: for bins=None, clip extremes when forming edges (robust to outliers)

    Returns
    -------
    dict with:
      'df'      : structured numpy record array with per-bin metrics
      'edges'   : bin edges (len = n_bins+1)
      'centers' : bin centers (y_true means or midpoints depending on strategy)
    """
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have same shape")

    # Build edges if not provided
    if bins is None:
        lo, hi = np.percentile(y_true, [clip_percentiles[0], clip_percentiles[1]])
        if strategy == "uniform":
            edges = np.linspace(lo, hi, nbins + 1)
        elif strategy == "quantile":
            qs = np.linspace(0, 1, nbins + 1)
            edges = np.quantile(y_true, qs)
            # ensure strictly increasing
            edges = np.unique(edges)
            if edges.size < nbins + 1:
                # fallback to uniform if too many duplicates
                edges = np.linspace(lo, hi, nbins + 1)
        else:
            raise ValueError("strategy must be 'quantile' or 'uniform'")
    else:
        edges = np.asarray(bins, dtype=np.float64)
        if edges.ndim != 1 or edges.size < 2:
            raise ValueError("bins must be a 1D array of length >= 2")

    # Assign bins
    idx = np.digitize(y_true, edges, right=False) - 1  # 0..B-1
    B = edges.size - 1

    # Aggregate
    recs = []
    centers = []
    for b in range(B):
        sel = (idx == b)
        n = int(sel.sum())
        if n == 0:
            recs.append((b, edges[b], edges[b+1], 0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan))
            centers.append(0.5 * (edges[b] + edges[b+1]))
            continue
        yt = y_true[sel]; yp = y_pred[sel]
        r = yp - yt
        recs.append((
            b, edges[b], edges[b+1], n,
            float(yt.mean()), float(yt.std()),
            float(yp.mean()),
            float(np.mean(np.abs(r))),
            float(np.sqrt(np.mean(r**2))),
            float(np.mean(r)),  # bias
            # you could add more here (e.g., median abs err, etc.)
        ))
        centers.append(yt.mean() if strategy == "quantile" else 0.5 * (edges[b] + edges[b+1]))

    dtype = [
        ("bin", int), ("edge_lo", float), ("edge_hi", float), ("count", int),
        ("y_true_mean", float), ("y_true_std", float), ("y_pred_mean", float),
        ("MAE", float), ("RMSE", float), ("bias", float)
    ]
    df = np.array(recs, dtype=dtype)
    centers = np.asarray(centers, dtype=float)
    return {"df": df, "edges": edges, "centers": centers}


def plot_binned_error_curves(binned, label="",
                             show_mae=True, show_rmse=True, show_bias=True,
                             title="Error vs. curvature (binned by true y)"):
    """
    Line plots of MAE / RMSE / Bias vs bin center.
    """
    df = binned["df"]; x = binned["centers"]
    plt.figure(figsize=(6.5, 4))
    if show_mae and np.isfinite(df["MAE"]).any():
        plt.plot(x, df["MAE"], marker='o', label=f"{label} MAE" if label else "MAE")
    if show_rmse and np.isfinite(df["RMSE"]).any():
        plt.plot(x, df["RMSE"], marker='s', label=f"{label} RMSE" if label else "RMSE")
    if show_bias and np.isfinite(df["bias"]).any():
        plt.plot(x, df["bias"], marker='^', label=f"{label} Bias" if label else "Bias")
    plt.axhline(0.0, color='k', linewidth=0.7, alpha=0.5)
    plt.xlabel("Curvature (true y), bin center")
    plt.ylabel("Error")
    plt.title(title)
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.show()

import math
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch_geometric.loader import DataLoader


def plot_binned_error_curves_per_marker(
    graphs,
    model=None,
    *,
    marker_names=None,
    device=None,
    batch_size=32,
    use_baseline=False,
    mu_dict=None,
    # binning options
    bins=None,
    nbins=8,
    strategy="quantile",          # 'quantile' or 'uniform'
    clip_percentiles=(1.0, 99.0),
    # marker selection / filtering
    positives_only_threshold=30,  # require at least this many positive cells for a marker
    max_markers=12,               # plot at most this many markers, pick the ones with most positives
    show_mae=True,
    show_rmse=True,
    show_bias=False,
    title_prefix="Per-marker error vs curvature (positives only)",
):
    """
    For each marker m:
      - select nodes with x[:, m] > 0.5 (positives),
      - bin by TRUE curvature,
      - compute MAE / RMSE / bias per bin,
      - plot curves in a small-multiples grid.

    Returns
    -------
    results : dict
        {
          'marker_indices': [m1, m2, ...],
          'counts': {m: n_pos, ...},
          'binned': {m: {'df': recarray, 'edges': edges, 'centers': centers}, ...}
        }
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 1) Gather flattened y_true, y_pred across graphs
    y_true, y_pred = gather_predictions(
        graphs, model=model, device=device, batch_size=batch_size,
        use_baseline=use_baseline, mu_dict=mu_dict
    )

    # 2) Concatenate marker matrix X across graphs (on CPU as numpy)
    Xs = []
    for g in graphs:
        Xs.append(g.x.detach().cpu().numpy())
    X = np.concatenate(Xs, axis=0).astype(np.float32)  # (K, M)
    K, M = X.shape
    if marker_names is None:
        marker_names = [f"m{j}" for j in range(M)]
    else:
        # truncate or pad if needed
        if len(marker_names) != M:
            # keep it robust but warn
            print(f"Note: marker_names length {len(marker_names)} != X.shape[1] {M}. Will truncate/pad.")
            marker_names = (list(marker_names) + [f"m{j}" for j in range(M)])[0:M]

    # 3) Count positives per marker and select which ones to plot
    pos_counts = (X > 0.5).sum(axis=0).astype(int)
    eligible = [m for m in range(M) if pos_counts[m] >= positives_only_threshold]
    if not eligible:
        print("No markers have enough positive samples; nothing to plot.")
        return {"marker_indices": [], "counts": {}, "binned": {}}

    # sort by count desc and cap to max_markers
    eligible.sort(key=lambda m: pos_counts[m], reverse=True)
    selected = eligible[:max_markers]

    # 4) Compute binned metrics per selected marker
    binned_by_marker = {}
    for m in selected:
        mask = X[:, m] > 0.5
        yt_m = y_true[mask]
        yp_m = y_pred[mask]
        if yt_m.size < max(positives_only_threshold, nbins * 3):
            # too few points after mask for meaningful binning
            continue
        binned = binned_curvature_metrics(
            yt_m, yp_m, bins=bins, nbins=nbins, strategy=strategy,
            clip_percentiles=clip_percentiles
        )
        binned_by_marker[m] = binned

    if not binned_by_marker:
        print("Selected markers did not have enough samples per bin; nothing to plot.")
        return {"marker_indices": [], "counts": {}, "binned": {}}

    # 5) Plot small-multiples grid
    n_plots = len(binned_by_marker)
    ncols = min(4, max(1, int(math.ceil(math.sqrt(n_plots)))))
    nrows = int(math.ceil(n_plots / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5*ncols, 3.8*nrows), squeeze=False)
    ax_iter = (ax for row in axes for ax in row)

    for m, ax in zip(binned_by_marker.keys(), ax_iter):
        b = binned_by_marker[m]
        df = b["df"]; x = b["centers"]
        has_any = False
        if show_mae and np.isfinite(df["MAE"]).any():
            ax.plot(x, df["MAE"], marker='o', label="MAE"); has_any = True
        if show_rmse and np.isfinite(df["RMSE"]).any():
            ax.plot(x, df["RMSE"], marker='s', label="RMSE"); has_any = True
        if show_bias and np.isfinite(df["bias"]).any():
            ax.plot(x, df["bias"], marker='^', label="Bias"); has_any = True
        ax.axhline(0.0, color='k', lw=0.7, alpha=0.4)
        ax.set_xlabel("True curvature (bin center)")
        ax.set_ylabel("Error")
        name = marker_names[m] if m < len(marker_names) else f"m{m}"
        ax.set_title(f"{name}  (n⁺={pos_counts[m]})")
        ax.grid(True, alpha=0.25)
        if has_any:
            ax.legend()

    # hide any unused axes
    for ax in ax_iter:
        ax.axis("off")

    fig.suptitle(title_prefix, y=1.02)
    fig.tight_layout()
    plt.show()

    return {
        "marker_indices": list(binned_by_marker.keys()),
        "counts": {m: int(pos_counts[m]) for m in binned_by_marker.keys()},
        "binned": binned_by_marker,
    }



# --- Uniform-in-curvature helpers -------------------------------------------
import numpy as np
import torch
from torch_geometric.loader import DataLoader

# You already have gather_predictions(); we reuse it.

def _build_uniform_edges_from_true(y_true, nbins=12, strategy="quantile"):
    y_true = np.asarray(y_true, float).ravel()
    if strategy == "quantile":
        qs = np.linspace(0, 1, nbins + 1)
        edges = np.unique(np.quantile(y_true, qs))
        if edges.size < 3:  # degenerate: fall back to uniform width
            lo, hi = np.min(y_true), np.max(y_true)
            edges = np.linspace(lo, hi, nbins + 1)
    elif strategy == "uniform":
        lo, hi = np.min(y_true), np.max(y_true)
        edges = np.linspace(lo, hi, nbins + 1)
    else:
        raise ValueError("strategy must be 'quantile' or 'uniform'")
    return edges


def uniform_weighted_mse(y_true, y_pred, edges):
    """
    Equal-weighted MSE across curvature bins defined by `edges` (on TRUE y).
    Returns (uw_mse, per_bin list of dicts).
    """
    y_true = np.asarray(y_true, float).ravel()
    y_pred = np.asarray(y_pred, float).ravel()
    idx = np.digitize(y_true, edges, right=False) - 1
    B = edges.size - 1
    mses, per_bin = [], []
    for b in range(B):
        sel = (idx == b)
        if not np.any(sel):
            per_bin.append({"bin": b, "count": 0, "mse": np.nan})
            continue
        r = y_pred[sel] - y_true[sel]
        mse = float(np.mean(r**2))
        mses.append(mse)
        per_bin.append({"bin": b, "count": int(sel.sum()), "mse": mse,
                        "edge_lo": float(edges[b]), "edge_hi": float(edges[b+1]),
                        "y_mean": float(y_true[sel].mean())})
    uw = float(np.nanmean(mses)) if len(mses) else np.nan
    return uw, per_bin


# --- Baseline (per-marker mean) utilities -----------------------------------

def _compute_mu_dict_from_graphs(graphs):
    """
    μ_m = mean(y | marker m positive) from graphs; μ_none for rows with no positives.
    Works whether graphs store original y (g.y) or residualized y with g.y_orig present.
    """
    ys = []
    Xs = []
    for g in graphs:
        y = g.y_orig if hasattr(g, "y_orig") else g.y
        ys.append(y.detach().cpu().numpy())
        Xs.append(g.x.detach().cpu().numpy())
    y_all = np.concatenate(ys, axis=0).astype(np.float64)
    X_all = np.concatenate(Xs, axis=0).astype(np.float32)

    M = X_all.shape[1]
    mu_pos = np.zeros(M, dtype=np.float32)
    for m in range(M):
        mask = X_all[:, m] > 0.5
        mu_pos[m] = float(np.mean(y_all[mask])) if np.any(mask) else float(np.mean(y_all))
    row_pos = (X_all > 0.5).sum(axis=1)
    mask_none = row_pos == 0
    mu_none = float(np.mean(y_all[mask_none])) if np.any(mask_none) else float(np.mean(y_all))
    return {"mu_pos": mu_pos, "mu_none": mu_none}


def baseline_predictions_for_graphs(graphs, mu_dict):
    """
    Build y_base for each node in `graphs` using μ_m and μ_none.
    Returns a single concatenated numpy array aligned with gather_predictions(...).
    """
    Ys_base = []
    mu_pos = mu_dict["mu_pos"]; mu_none = mu_dict["mu_none"]
    for g in graphs:
        X = g.x.detach().cpu().numpy()
        pos_counts = np.clip(np.sum(X > 0.5, axis=1, keepdims=True), 0, None)
        wsum = X @ mu_pos.reshape(-1, 1)
        b = np.where(pos_counts > 0, (wsum / pos_counts), mu_none).reshape(-1)
        Ys_base.append(b.astype(np.float64))
    return np.concatenate(Ys_base, axis=0)


# --- Uniform residual MSE per marker (compatible adapter) --------------------

def per_marker_uniform_residual_mse(
    graphs, model, marker_names,
    device=None, nbins=8, strategy="quantile",
    use_baseline=False, mu_dict=None,  # pass your residual-baseline if you want y_pred to include it
    min_positives=30,
):
    """
    For each marker (positives only):
      - build edges from TRUE y (train/val set separately)
      - compute uniform-weighted MSE of model predictions
      - compute uniform-weighted MSE of per-marker baseline
    Returns two dicts keyed by marker_name: (model_uw_mse, baseline_uw_mse)
    plus a small adapter to reuse your existing plot_marker_residual_mse().
    """
    # 1) Gather y_true / y_pred (optionally add baseline inside predictions)
    y_true, y_pred = gather_predictions(
        graphs, model, device=device, use_baseline=use_baseline, mu_dict=mu_dict
    )
    # 2) Build baseline vector (pure baseline, independent of model)
    mu_train = mu_dict if mu_dict is not None else _compute_mu_dict_from_graphs(graphs)
    y_base = baseline_predictions_for_graphs(graphs, mu_train)

    # 3) Stack X to filter positives per marker
    X = np.concatenate([g.x.detach().cpu().numpy() for g in graphs], axis=0).astype(np.float32)
    M = X.shape[1]
    if marker_names is None or len(marker_names) != M:
        marker_names = [f"m{j}" for j in range(M)]

    model_mse, base_mse = {}, {}
    for m in range(M):
        mask = X[:, m] > 0.5
        n = int(mask.sum())
        if n < min_positives:
            continue
        edges = _build_uniform_edges_from_true(y_true[mask], nbins=nbins, strategy=strategy)
        uw_model, _ = uniform_weighted_mse(y_true[mask], y_pred[mask], edges)
        uw_base,  _ = uniform_weighted_mse(y_true[mask], y_base[mask],  edges)
        model_mse[marker_names[m]] = uw_model
        base_mse[marker_names[m]]  = uw_base

    return model_mse, base_mse


def adapt_marker_mse_dict_for_existing_plot(mse_dict):
    """
    Your existing plot_marker_residual_mse(mse_train, mse_val, ...) likely expects
    a structure with parallel name/value arrays. This adapter converts a
    {name: value} dict to {'names': [...], 'mse': np.array([...])}.
    """
    names = list(mse_dict.keys())
    vals  = np.array([mse_dict[k] for k in names], dtype=float)
    return {"names": names, "mse": vals}



def _uniform_mse_and_sem_from_bins(y_true, y_pred, edges):
    """
    Compute uniform-weighted MSE and a simple SEM estimate by treating
    per-bin MSEs as i.i.d. samples (equal-weighted bins).

    Returns:
      uw_mse : float
      sem    : float  (nan-safe; 0 if <2 non-empty bins)
      mse_bins : list[float] per non-empty bin (for debugging)
    """
    y_true = np.asarray(y_true, float).ravel()
    y_pred = np.asarray(y_pred, float).ravel()
    idx = np.digitize(y_true, edges, right=False) - 1
    B = edges.size - 1

    mse_bins = []
    for b in range(B):
        sel = (idx == b)
        if not np.any(sel):
            continue
        r = y_pred[sel] - y_true[sel]
        mse_bins.append(float(np.mean(r**2)))

    if len(mse_bins) == 0:
        return np.nan, np.nan, []
    uw_mse = float(np.mean(mse_bins))
    if len(mse_bins) >= 2:
        sem = float(np.std(mse_bins, ddof=1) / np.sqrt(len(mse_bins)))
    else:
        sem = 0.0
    return uw_mse, sem, mse_bins


def per_marker_uniform_residual_mse_for_plot(
    graphs, model, marker_names,
    device=None,
    nbins=8, strategy="quantile",
    use_baseline=False, mu_dict=None,
    min_positives=30,
):
    """
    Build a result dict compatible with your plot_marker_residual_mse(...).

    For each marker m (positives only):
      - Build bin edges on TRUE curvature (within the masked subset).
      - Compute uniform-weighted MSE for MODEL and for BASELINE.
      - Estimate SEM from per-bin MSE variability.
      - Collect into arrays in the required schema.

    Returns
    -------
    res_dict = {
        "names":         [str],
        "mse_baseline":  np.ndarray shape (K,),
        "mse_model":     np.ndarray shape (K,),
        "sem_baseline":  np.ndarray shape (K,),
        "sem_model":     np.ndarray shape (K,),
        "delta":         np.ndarray shape (K,)   # baseline - model
    }
    """
    # 1) Predictions and true y (optionally add per-node baseline into y_pred)
    y_true, y_pred = gather_predictions(
        graphs, model, device=device, use_baseline=use_baseline, mu_dict=mu_dict
    )
    # 2) Pure baseline predictions (independent of model output)
    mu_train = mu_dict if mu_dict is not None else _compute_mu_dict_from_graphs(graphs)
    y_base = baseline_predictions_for_graphs(graphs, mu_train)

    # 3) Marker mask
    X = np.concatenate([g.x.detach().cpu().numpy() for g in graphs], axis=0).astype(np.float32)
    M = X.shape[1]
    if marker_names is None or len(marker_names) != M:
        marker_names = [f"m{j}" for j in range(M)]

    names = []
    mse_b, mse_m, sem_b, sem_m = [], [], [], []

    for m in range(M):
        mask = X[:, m] > 0.5
        npos = int(mask.sum())
        if npos < min_positives:
            continue

        yt = y_true[mask]; yp = y_pred[mask]; yb = y_base[mask]
        # Build edges on TRUE curvature (masked subset)
        edges = _build_uniform_edges_from_true(yt, nbins=nbins, strategy=strategy)

        uw_m, se_m, _ = _uniform_mse_and_sem_from_bins(yt, yp, edges)
        uw_b, se_b, _ = _uniform_mse_and_sem_from_bins(yt, yb, edges)

        names.append(marker_names[m])
        mse_m.append(uw_m); sem_m.append(se_m)
        mse_b.append(uw_b); sem_b.append(se_b)

    if not names:
        # no markers passed the threshold
        return {
            "names": [], "mse_baseline": np.array([]), "mse_model": np.array([]),
            "sem_baseline": np.array([]), "sem_model": np.array([]), "delta": np.array([])
        }

    mse_b = np.asarray(mse_b, dtype=float)
    mse_m = np.asarray(mse_m, dtype=float)
    sem_b = np.asarray(sem_b, dtype=float)
    sem_m = np.asarray(sem_m, dtype=float)
    delta = mse_b - mse_m

    return {
        "names": names,
        "mse_baseline": mse_b,
        "mse_model":    mse_m,
        "sem_baseline": sem_b,
        "sem_model":    sem_m,
        "delta":        delta,
    }