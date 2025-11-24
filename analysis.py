"""
Analysis helpers for organoid curvature model.

Each compute_* function consumes a SINGLE dataset (list of PyG Data graphs).
Plotting functions can combine two result dicts (e.g., train and val).

Provided:

- print_head_coefficients_with_names(model, marker_names, top_k=10)
- run_inference(model, graphs, device=None, batch_size=8, num_workers=0, pin_memory=True)

- compute_overall_mean_std(graphs, model, device=None)
    -> {'y': (mean,std), 'yhat': (mean,std)}

- compute_per_marker_mean_std(graphs, model, marker_names, device=None)
    -> {'names': [...], 'y_mean':(M,), 'y_std':(M,), 'yhat_mean':(M,), 'yhat_std':(M,), 'n_pos':(M,)}

- compute_marker_baseline_mse(graphs, model, marker_names, device=None)
    -> {'names': [...], 'mse_mean':(M,), 'mse_sem':(M,), 'n_pos':(M,), 'mu':(M,)}

Plotters (errorbar/points, safe for negative values):

- plot_overall_mean_std(res_train=None, res_val=None, title="")
- plot_per_marker_mean_std(res_train=None, res_val=None, top_k=None, rotate=60, title="")
- plot_marker_baseline_mse(res_train=None, res_val=None, top_k=None, rotate=60, title="")
"""

import numpy as np
import torch
from torch_geometric.loader import DataLoader
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# Coefficient inspection with names
# ---------------------------------------------------------------------

def _infer_cross_pairs(n_markers, cross_len, model=None):
    """
    Map cross-term index -> (i,j) pair. If model has explicit_cross_pairs, use it.
    Otherwise assume all unordered pairs (i<j) in lexicographic order.
    """
    if model is not None and hasattr(model, "explicit_cross_pairs"):
        pairs = list(getattr(model, "explicit_cross_pairs"))
        if len(pairs) == cross_len:
            return pairs
    pairs = []
    for i in range(n_markers):
        for j in range(i+1, n_markers):
            pairs.append((i, j))
    if len(pairs) != cross_len:
        pairs = pairs[:cross_len]
    return pairs


def print_head_coefficients_with_names(model, marker_names, top_k=10):
    """
    Print top-|w| explicit head coefficients with marker names: self, neighbor, cross.
    """
    coeffs = model.unpack_head_coefficients()
    w_self = np.asarray(coeffs["self"])
    w_nb   = np.asarray(coeffs["nb"])
    w_cross = np.asarray(coeffs["cross"])

    M = len(marker_names)
    if w_self.shape[0] != M or w_nb.shape[0] != M:
        raise ValueError(f"Marker names ({M}) != coeff dims (self={w_self.shape[0]}, nb={w_nb.shape[0]})")

    print("\nTop explicit SELF (|weight|):")
    idx = np.argsort(np.abs(w_self))[::-1][:top_k]
    for k in idx:
        print(f"  {marker_names[k]:>20s}: {w_self[k]:+.4f}")

    print("\nTop explicit NEIGHBOR (|weight|):")
    idx = np.argsort(np.abs(w_nb))[::-1][:top_k]
    for k in idx:
        print(f"  {marker_names[k]:>20s}: {w_nb[k]:+.4f}")

    pairs = _infer_cross_pairs(M, len(w_cross), model=model)
    print("\nTop explicit CROSS (|weight|):")
    idx = np.argsort(np.abs(w_cross))[::-1][:top_k]
    for k in idx:
        i, j = pairs[k] if k < len(pairs) else (-1, -1)
        name = f"{marker_names[i]} × {marker_names[j]}" if 0 <= i < M and 0 <= j < M else f"pair[{k}]"
        print(f"  {name:>20s}: {w_cross[k]:+.4f}")


# ---------------------------------------------------------------------
# Inference helper (single dataset)
# ---------------------------------------------------------------------

@torch.no_grad()
def run_inference(model, graphs, device=None, batch_size=8, num_workers=0, pin_memory=True):
    """
    Run model on a list of graphs. Returns concatenated arrays:
      - y_true: (N_total,)
      - y_pred: (N_total,)
      - X:      (N_total, M)
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device).eval()

    loader = DataLoader(graphs, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=pin_memory)

    Yt, Yp, Xs = [], [], []
    for batch in loader:
        batch = batch.to(device, non_blocking=True)
        yhat, _ = model(batch.x, batch.edge_index)
        Yt.append(batch.y.detach().cpu()); Yp.append(yhat.detach().cpu()); Xs.append(batch.x.detach().cpu())

    y_true = torch.cat(Yt, dim=0).numpy()
    y_pred = torch.cat(Yp, dim=0).numpy()
    X = torch.cat(Xs, dim=0).numpy()

    return {"y_true": y_true, "y_pred": y_pred, "X": X}


# ---------------------------------------------------------------------
# Overall mean/std (single dataset)
# ---------------------------------------------------------------------

def compute_overall_mean_std(graphs, model, device=None):
    """
    Compute overall mean/std for truth and prediction on one dataset.
    Returns: {'y': (mean,std), 'yhat': (mean,std)}
    """
    out = run_inference(model, graphs, device=device)
    y, yhat = out["y_true"], out["y_pred"]
    return {
        "y":    (float(np.mean(y)), float(np.std(y))),
        "yhat": (float(np.mean(yhat)), float(np.std(yhat))),
    }


# ---------------------------------------------------------------------
# Per-marker mean/std (single dataset)
# ---------------------------------------------------------------------

def compute_per_marker_mean_std(graphs, model, marker_names, device=None):
    """
    For each marker m, restrict to positive cells (X[:,m]==1) and compute:
      - mean/std of y
      - mean/std of yhat
      - n_pos
    Returns dict with arrays of length M and 'names'.
    """
    out = run_inference(model, graphs, device=device)
    y, yhat, X = out["y_true"], out["y_pred"], out["X"]
    M = X.shape[1]
    if len(marker_names) != M:
        raise ValueError("marker_names length must match feature dimension M")

    y_mean = np.zeros(M); y_std = np.zeros(M)
    yh_mean = np.zeros(M); yh_std = np.zeros(M)
    n_pos = np.zeros(M, dtype=np.int64)

    for m in range(M):
        mask = X[:, m] > 0.5
        vals_y  = y[mask]
        vals_yh = yhat[mask]
        if vals_y.size == 0:
            y_mean[m] = np.nan; y_std[m] = np.nan
            yh_mean[m] = np.nan; yh_std[m] = np.nan
            n_pos[m] = 0
        else:
            y_mean[m]  = float(np.mean(vals_y))
            y_std[m]   = float(np.std(vals_y))
            yh_mean[m] = float(np.mean(vals_yh))
            yh_std[m]  = float(np.std(vals_yh))
            n_pos[m]   = int(vals_y.size)

    return {
        "names": list(marker_names),
        "y_mean": y_mean, "y_std": y_std,
        "yhat_mean": yh_mean, "yhat_std": yh_std,
        "n_pos": n_pos
    }


# ---------------------------------------------------------------------
# Marker-baseline MSE (single dataset)
# ---------------------------------------------------------------------

def compute_marker_baseline_mse(graphs, model, marker_names, device=None):
    """
    For each marker m:
      1) Compute μ_m = mean(true y | X[:,m]==1) using THIS dataset.
      2) For all positive cells of marker m, compute (yhat - μ_m)^2.
      3) Report mean MSE per marker, with SEM = std / sqrt(n_pos).

    Returns dict:
      {'names', 'mse_mean', 'mse_sem', 'n_pos', 'mu'}
    """
    out = run_inference(model, graphs, device=device)
    y, yhat, X = out["y_true"], out["y_pred"], out["X"]
    M = X.shape[1]
    if len(marker_names) != M:
        raise ValueError("marker_names length must match feature dimension M")

    mu = np.zeros(M)
    mse_mean = np.zeros(M)
    mse_sem  = np.zeros(M)
    n_pos    = np.zeros(M, dtype=np.int64)

    for m in range(M):
        mask = X[:, m] > 0.5
        if not np.any(mask):
            mu[m] = np.nan; mse_mean[m] = np.nan; mse_sem[m] = np.nan; n_pos[m] = 0
            continue
        mu_m = float(np.mean(y[mask]))
        se = (yhat[mask] - mu_m) ** 2
        mu[m] = mu_m
        mse_mean[m] = float(np.mean(se))
        n_pos[m] = int(se.size)
        std = float(np.std(se))
        mse_sem[m] = std / np.sqrt(max(n_pos[m], 1))

    return {"names": list(marker_names), "mse_mean": mse_mean, "mse_sem": mse_sem, "n_pos": n_pos, "mu": mu}


def compute_marker_residuals(graphs, model, marker_names, device=None):
    """
    For each marker m (positives only):
      - μ_m = mean(true y | X[:,m]==1) computed on THIS dataset.
      - baseline residuals r_base = y_true - μ_m
      - model residuals    r_pred = y_true - yhat
    Returns dict:
      {'names':[], 'residuals': {m: {'baseline': np.ndarray, 'model': np.ndarray}}, 'mu': (M,), 'n_pos': (M,)}
    """
    out = run_inference(model, graphs, device=device)
    y, yhat, X = out["y_true"], out["y_pred"], out["X"]
    M = X.shape[1]
    if len(marker_names) != M:
        raise ValueError("marker_names length must match feature dimension M")

    res = {"names": list(marker_names), "residuals": {}, "mu": np.zeros(M), "n_pos": np.zeros(M, dtype=np.int64)}
    for m in range(M):
        mask = X[:, m] > 0.5
        if not np.any(mask):
            res["residuals"][m] = {"baseline": np.array([]), "model": np.array([])}
            res["mu"][m] = np.nan; res["n_pos"][m] = 0
            continue
        mu_m = float(np.mean(y[mask]))
        r_base = y[mask]  - mu_m
        r_pred = y[mask]  - yhat[mask]
        res["residuals"][m] = {"baseline": r_base, "model": r_pred}
        res["mu"][m] = mu_m; res["n_pos"][m] = int(mask.sum())
    return res


def compute_marker_residual_mse(res):
    """
    From compute_marker_residuals(...) output, compute per-marker MSE for:
      - baseline: mean( (y - μ_m)^2 )
      - model   : mean( (y - ŷ)^2 )
    Also returns SEM (std/sqrt(n_pos)) and the improvement Δ = model - baseline
    (negative Δ means model beats baseline).
    Returns dict:
      {
        'names': [...],
        'mse_baseline': (M,),
        'mse_model': (M,),
        'sem_baseline': (M,),
        'sem_model': (M,),
        'delta': (M,),
        'n_pos': (M,)
      }
    """
    names = res["names"]
    M = len(names)
    mse_b = np.full(M, np.nan, dtype=float)
    mse_m = np.full(M, np.nan, dtype=float)
    sem_b = np.full(M, np.nan, dtype=float)
    sem_m = np.full(M, np.nan, dtype=float)
    npos  = np.asarray(res["n_pos"], dtype=int)

    for m in range(M):
        rb = res["residuals"][m]["baseline"]  # y - μ_m
        rm = res["residuals"][m]["model"]     # y - ŷ
        if rb.size > 0:
            rb2 = rb**2
            rm2 = rm**2
            mse_b[m] = float(np.mean(rb2))
            mse_m[m] = float(np.mean(rm2))
            # SEM of squared residuals (not of residual): std(r^2) / sqrt(n)
            sem_b[m] = float(np.std(rb2) / np.sqrt(max(rb2.size, 1)))
            sem_m[m] = float(np.std(rm2) / np.sqrt(max(rm2.size, 1)))

    delta = mse_m - mse_b
    return {
        "names": names,
        "mse_baseline": mse_b,
        "mse_model": mse_m,
        "sem_baseline": sem_b,
        "sem_model": sem_m,
        "delta": delta,
        "n_pos": npos,
    }


# ---------------------------------------------------------------------
# Plotting (combine up to two splits)
# ---------------------------------------------------------------------

def plot_overall_mean_std(res_train=None, res_val=None, title=""):
    """
    Errorbar plot of overall mean±std for truth and prediction.
    Pass one or both results from compute_overall_mean_std(...).
    """
    xs, ys, es, lbls = [], [], [], []
    off = 0
    if res_train is not None:
        xs += [off, off+1]; lbls += ["train: y", "train: yhat"]
        ys += [res_train["y"][0], res_train["yhat"][0]]
        es += [res_train["y"][1], res_train["yhat"][1]]
        off += 2
    if res_val is not None:
        xs += [off, off+1]; lbls += ["val: y", "val: yhat"]
        ys += [res_val["y"][0], res_val["yhat"][0]]
        es += [res_val["y"][1], res_val["yhat"][1]]

    fig, ax = plt.subplots()
    ax.errorbar(xs, ys, yerr=es, fmt='o', capsize=3)
    for x, label in zip(xs, lbls):
        ax.plot([x], [ys[xs.index(x)]], 'o', label=label)  # anchor legend labels
    ax.set_xticks(xs, lbls, rotation=0)
    ax.set_ylabel("value")
    ax.set_title(title or "Overall mean ± std (truth vs prediction)")
    ax.grid(True, alpha=0.2)
    ax.legend()
    plt.tight_layout(); plt.show()



def plot_per_marker_mean_std(res_train=None, res_val=None, top_k=None, rotate=60, title=""):
    """
    Two subplots: left=train, right=val. Each shows per-marker mean±std for truth (o) and prediction (x).
    Provide result dicts from compute_per_marker_mean_std(...). Selects top_k markers by |val y_mean| if val given,
    else by |train y_mean|.
    """
    # Choose order
    if res_val is not None:
        base_names = res_val["names"]; order_metric = np.abs(res_val["y_mean"])
    elif res_train is not None:
        base_names = res_train["names"]; order_metric = np.abs(res_train["y_mean"])
    else:
        raise ValueError("Provide at least one result dict")

    names = base_names
    if top_k is not None and top_k < len(names):
        idx = np.argsort(order_metric)[::-1][:top_k]
        names = [base_names[i] for i in idx]

    def collect(res):
        if res is None: return None
        name_to_idx = {n:i for i,n in enumerate(res["names"])}
        idxs = np.array([name_to_idx[n] for n in names], dtype=int)
        return {
            "y_mean":  res["y_mean"][idxs],
            "y_std":   res["y_std"][idxs],
            "yh_mean": res["yhat_mean"][idxs],
            "yh_std":  res["yhat_std"][idxs],
        }

    tr = collect(res_train)
    vl = collect(res_val)

    x = np.arange(len(names))
    fig, axes = plt.subplots(1, 2, figsize=(max(10, 0.6*len(names)), 4), sharey=True)

    if tr is not None:
        ax = axes[0]
        ax.errorbar(x, tr["y_mean"],  yerr=tr["y_std"],  fmt='o', capsize=3, label="train: y")
        ax.errorbar(x, tr["yh_mean"], yerr=tr["yh_std"], fmt='x', capsize=3, label="train: yhat")
        ax.set_title("Per-marker (train)")
        ax.set_xticks(x, names, rotation=rotate, ha="right")
        ax.grid(True, axis='y', alpha=0.2)
        ax.legend()

    if vl is not None:
        ax = axes[1]
        ax.errorbar(x, vl["y_mean"],  yerr=vl["y_std"],  fmt='o', capsize=3, label="val: y")
        ax.errorbar(x, vl["yh_mean"], yerr=vl["yh_std"], fmt='x', capsize=3, label="val: yhat")
        ax.set_title("Per-marker (val)")
        ax.set_xticks(x, names, rotation=rotate, ha="right")
        ax.grid(True, axis='y', alpha=0.2)
        ax.legend()

    fig.suptitle(title or "Per-marker mean ± std (truth vs prediction)")
    plt.tight_layout(); plt.show()



def plot_marker_baseline_mse(res_train=None, res_val=None, top_k=None, rotate=60, title=""):
    """
    Errorbar plot of per-marker mean((yhat - μ_m)^2) with SEM error bars.
    If top_k is set, markers are selected by highest validation mse_mean; else by train.
    """
    # Select order
    if res_val is not None:
        base_names = res_val["names"]; order_metric = res_val["mse_mean"]
    elif res_train is not None:
        base_names = res_train["names"]; order_metric = res_train["mse_mean"]
    else:
        raise ValueError("Provide at least one result dict")

    names = base_names
    if top_k is not None and top_k < len(names):
        names, _ = _subset_top_k(base_names, ref_array=order_metric, top_k=top_k)

    def collect(res):
        if res is None: return None
        name_to_idx = {n:i for i,n in enumerate(res["names"])}
        idxs = np.array([name_to_idx[n] for n in names], dtype=int)
        return {"mean": res["mse_mean"][idxs], "sem": res["mse_sem"][idxs]}

    tr = collect(res_train)
    vl = collect(res_val)

    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(max(8, 0.5*len(names)), 4))
    if tr is not None:
        ax.errorbar(x - 0.05, tr["mean"], yerr=tr["sem"], fmt='o', capsize=3, label="train")
    if vl is not None:
        ax.errorbar(x + 0.05, vl["mean"], yerr=vl["sem"], fmt='x', capsize=3, label="val")
    ax.set_xticks(x, names, rotation=rotate, ha="right")
    ax.set_ylabel("mean squared distance to μ_m")
    ax.set_title(title or "Per-marker (yhat - μ_m)^2 (positives only)")
    ax.grid(True, axis='y', alpha=0.2)
    plt.tight_layout(); plt.show()

    
def plot_marker_residual_histograms(
    res,
    bins=40,                # int (count) OR array-like of bin EDGES
    max_markers=None,
    min_pos=10,
    cols=5,
    title_prefix="",
    density=True            # keep density=True by default; set False for counts
):
    """
    For each marker, plot histograms of residuals (positives only):
      baseline: y_true - μ_m
      model   : y_true - yhat

    Args
    ----
    res : dict
        Output of compute_marker_residuals(...)
    bins : int or array-like
        If int, that many bins (matplotlib default behavior).
        If array-like, treated as explicit bin edges (shared across subplots).
    max_markers : int or None
        Cap the number of subplots (chosen by largest n_pos).
    min_pos : int
        Only plot markers with at least this many positives.
    cols : int
        Number of subplot columns.
    title_prefix : str
        Figure suptitle prefix, e.g., "Train" or "Val".
    density : bool
        Whether to normalize histograms to a probability density.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    names = res["names"]
    n_pos = res["n_pos"]

    # Select markers to show (positives only)
    valid_idx = [i for i in range(len(names)) if n_pos[i] >= min_pos]
    valid_idx.sort(key=lambda i: n_pos[i], reverse=True)
    if max_markers is not None:
        valid_idx = valid_idx[:max_markers]

    if not valid_idx:
        print("No markers meet the criteria for plotting.")
        return

    rows = int(np.ceil(len(valid_idx) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols*3.6, rows*3.0), squeeze=False)
    axes = axes.ravel()

    # If bins is array-like, we’ll pass it directly to both hist calls (shared edges)
    shared_bins = bins  # can be int or array-like

    for ax_idx, m in enumerate(valid_idx):
        ax = axes[ax_idx]
        r_base  = res["residuals"][m]["baseline"]
        r_model = res["residuals"][m]["model"]
        if r_base.size == 0:
            ax.axis('off'); continue

        ax.hist(r_base,  bins=shared_bins, density=density, alpha=0.4, label="baseline: y − μ_m")
        ax.hist(r_model, bins=shared_bins, density=density, alpha=0.6, label="model: y − ŷ")
        ax.set_title(f"{names[m]}  (n+={n_pos[m]})", fontsize=10)
        ax.grid(True, alpha=0.2)

    # Hide unused axes
    for k in range(len(valid_idx), len(axes)):
        axes[k].axis('off')

    fig.suptitle(f"{title_prefix} residuals per marker (positives only)", fontsize=12)
    # Shared legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")
    plt.tight_layout(); plt.show()


def plot_marker_residual_mse(res_train=None, res_val=None, top_k=None, rotate=60, title=""):
    """
    Plot per-marker MSE of residuals (positives only), baseline vs model, with SEM error bars.
    Pass outputs from compute_marker_residual_mse(...). You can provide one or both (train/val).
    If top_k is set, markers are chosen by largest validation improvement (|delta|) if val given, else by train.
    """
    if (res_train is None) and (res_val is None):
        raise ValueError("Provide at least one result dict")

    # Decide ordering
    if res_val is not None:
        base_names = res_val["names"]; order_metric = np.abs(res_val["delta"])
    else:
        base_names = res_train["names"]; order_metric = np.abs(res_train["delta"])

    names = base_names
    if top_k is not None and top_k < len(names):
        idx = np.argsort(order_metric)[::-1][:top_k]
        names = [base_names[i] for i in idx]

    def select(res):
        if res is None: return None
        name2i = {n:i for i,n in enumerate(res["names"])}
        ids = np.array([name2i[n] for n in names], dtype=int)
        return {
            "b": res["mse_baseline"][ids],
            "m": res["mse_model"][ids],
            "sb": res["sem_baseline"][ids],
            "sm": res["sem_model"][ids],
        }

    tr = select(res_train)
    vl = select(res_val)

    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(max(8, 0.5*len(names)), 4))
    if tr is not None:
        ax.errorbar(x-0.07, tr["b"], yerr=tr["sb"], fmt='o', capsize=3, label="train: baseline")
        ax.errorbar(x-0.07, tr["m"], yerr=tr["sm"], fmt='x', capsize=3, label="train: model")
    if vl is not None:
        ax.errorbar(x+0.07, vl["b"], yerr=vl["sb"], fmt='o', capsize=3, label="val: baseline")
        ax.errorbar(x+0.07, vl["m"], yerr=vl["sm"], fmt='x', capsize=3, label="val: model")

    ax.set_xticks(x, names, rotation=rotate, ha="right")
    ax.set_ylabel("MSE of residuals")
    ax.set_title(title or "Per-marker MSE (positives only): baseline vs model")
    ax.grid(True, axis='y', alpha=0.2)
    ax.legend()
    plt.tight_layout(); plt.show()
