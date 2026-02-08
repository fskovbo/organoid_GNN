
import numpy as np
import torch
from torch_geometric.loader import DataLoader

# ============================================================
# Inference: y_true, y_pred, X
# ============================================================
@torch.no_grad()
def run_inference(graphs, model, device=None, batch_size=32, num_workers=0, pin_memory=True, return_log_var=False):
    """
    Run model over graphs and return concatenated arrays:
      y_true : (K,) float64
      y_pred : (K,) float64  (mean prediction μ)
      X      : (K, M) float32 (binary markers)

    If return_log_var=True, additionally returns:
      log_var : (K,) float64  (log variance, i.e. log(σ^2))
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device).eval()

    loader = DataLoader(graphs, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=pin_memory)

    Ys, Ymu, Ylv, Xs = [], [], [], []
    for batch in loader:
        batch = batch.to(device, non_blocking=True)

        out, _ = model(batch.x, batch.edge_index)
        # Support both deterministic models (yhat) and Gaussian models ((mu, log_var))
        if isinstance(out, (tuple, list)) and len(out) == 2:
            mu, log_var = out
        else:
            mu, log_var = out, None

        Ys.append(batch.y.detach().cpu().numpy())
        Ymu.append(mu.detach().cpu().numpy())
        if return_log_var:
            if log_var is None:
                raise ValueError("Model did not return (mu, log_var) but return_log_var=True was requested.")
            Ylv.append(log_var.detach().cpu().numpy())
        Xs.append(batch.x.detach().cpu().numpy())

    y_true = np.concatenate(Ys, axis=0).astype(np.float64)
    y_pred = np.concatenate(Ymu, axis=0).astype(np.float64)
    X = np.concatenate(Xs, axis=0).astype(np.float32)

    if return_log_var:
        log_var = np.concatenate(Ylv, axis=0).astype(np.float64)
        return y_true, y_pred, log_var, X
    return y_true, y_pred, X



# ============================================================
# NLL per marker 
# ============================================================

def compute_marker_gaussian_params(y_true, X):
    """
    For each marker m, compute baseline Gaussian params using marker-positive cells:
      mu_pos[m]  = mean(y_true | marker m positive)
      var_pos[m] = var (y_true | marker m positive)  (ddof=0)

    Also compute a fallback for rows with no positives:
      mu_none, var_none from rows where a cell has no positive markers.

    Returns:
      mu_pos  : (M,) float64
      var_pos : (M,) float64 (clipped to >= 1e-12)
      mu_none : float64
      var_none: float64 (clipped to >= 1e-12)
    """
    K, M = X.shape
    mu_pos = np.zeros(M, dtype=np.float64)
    var_pos = np.zeros(M, dtype=np.float64)

    y_all_mu = float(np.mean(y_true))
    y_all_var = float(np.var(y_true)) + 1e-12

    for m in range(M):
        mask = X[:, m] > 0.5
        if np.any(mask):
            ym = y_true[mask]
            mu_pos[m] = float(np.mean(ym))
            var_pos[m] = float(np.var(ym)) + 1e-12
        else:
            mu_pos[m] = y_all_mu
            var_pos[m] = y_all_var

    row_pos = (X > 0.5).sum(axis=1)
    if np.any(row_pos == 0):
        y0 = y_true[row_pos == 0]
        mu_none = float(np.mean(y0))
        var_none = float(np.var(y0)) + 1e-12
    else:
        mu_none = y_all_mu
        var_none = y_all_var

    # final safety clamp
    var_pos = np.maximum(var_pos, 1e-12)
    var_none = max(var_none, 1e-12)
    return mu_pos, var_pos, mu_none, var_none


def per_node_gaussian_nll(y_true, mu, var):
    """
    Per-node Gaussian NLL (includes constant term):
      0.5 * (log(2*pi*var) + (y-mu)^2 / var)
    Inputs can be numpy arrays.
    """
    var = np.maximum(var, 1e-12)
    return 0.5 * (np.log(2.0 * np.pi * var) + (y_true - mu) ** 2 / var)


# def nll_per_marker(y_true, mu_model, log_var_model, X):
#     """
#     Compute per-node NLL arrays per marker (positives only) for:
#       - model: N(mu_model, exp(log_var_model))
#       - baseline: marker-conditional Gaussian N(mu_m, var_m) where
#                  (mu_m, var_m) are estimated from y_true over marker-positive cells.

#     Returns:
#       nll_model_list    : list length M of (n_m,) float64
#       nll_baseline_list : list length M of (n_m,) float64
#       Delta_mean_list   : list length M of (n_m,) float64
#       Delta_var_list    : list length M of (n_m,) float64
#       n_pos             : (M,) int64 positives per marker
#     """
#     y_true = np.asarray(y_true).reshape(-1).astype(np.float64)
#     mu_model = np.asarray(mu_model).reshape(-1).astype(np.float64)
#     log_var_model = np.asarray(log_var_model).reshape(-1).astype(np.float64)
#     X = np.asarray(X)

#     M = X.shape[1]
#     n_pos = np.zeros(M, dtype=np.int64)

#     # Baseline Gaussian params per marker, estimated from y_true
#     mu_pos, var_pos, mu_none, var_none = compute_marker_gaussian_params(y_true, X)

#     # Model per-node variance
#     var_model = np.exp(log_var_model).astype(np.float64)
#     var_model = np.maximum(var_model, 1e-12)

#     nll_model_list, nll_base_list = [], []
#     Delta_mean_list, Delta_var_list = [], []

#     for m in range(M):
#         mask = X[:, m] > 0.5
#         n_pos[m] = int(np.sum(mask))

#         if n_pos[m] == 0:
#             nll_model_list.append(np.zeros((0,), dtype=np.float64))
#             nll_base_list.append(np.zeros((0,), dtype=np.float64))
#             continue

#         # Model NLL on those nodes
#         nll_m = per_node_gaussian_nll(y_true[mask], mu_model[mask], var_model[mask]).astype(np.float64)

#         # Baseline NLL: N(mu_pos[m], var_pos[m]) on those nodes
#         mu_b = mu_pos[m]
#         var_b = var_pos[m]
#         nll_b = per_node_gaussian_nll(y_true[mask], mu_b, var_b).astype(np.float64)

#         nll_model_list.append(nll_m)
#         nll_base_list.append(nll_b)

#         # Compute decomposition of NLL diffrence
#         Delta_mean = ((y_true[mask]-mu_b)**2 - (y_true[mask]-mu_model[mask])**2)/(2*var_b)
#         Delta_var = 0.5*np.log(var_b/var_model[mask])

#         Delta_mean_list.append(Delta_mean)
#         Delta_var_list.append(Delta_var)


#     return nll_model_list, nll_base_list, Delta_mean_list, Delta_var_list, n_pos


import numpy as np

def nll_per_marker(y_true, mu_model, log_var_model, X, eps=1e-12):
    """
    Compute per-node NLL arrays per marker (positives only) for:
      - model: N(mu_model, exp(log_var_model))
      - baseline: marker-conditional Gaussian N(mu_pos[m], var_pos[m])
                  estimated from y_true over marker-positive cells.

    Returns:
      nll_model_list    : list length M of (n_m,) float64
      nll_baseline_list : list length M of (n_m,) float64
      Delta_mean_list   : list length M of (n_m,) float64  (exact component)
      Delta_var_list    : list length M of (n_m,) float64  (exact component)
      n_pos             : (M,) int64 positives per marker

    Notes:
      The decomposition is exact:
        (nll_base - nll_model) = Delta_mean + Delta_var
      where
        Delta_var  = 0.5 * log(var_b / var_model)
        Delta_mean = 0.5 * [ (y-mu_b)^2/var_b - (y-mu_model)^2/var_model ]
    """
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    mu_model = np.asarray(mu_model, dtype=np.float64).reshape(-1)
    log_var_model = np.asarray(log_var_model, dtype=np.float64).reshape(-1)
    X = np.asarray(X)

    M = X.shape[1]
    n_pos = np.zeros(M, dtype=np.int64)

    # Baseline Gaussian params per marker, estimated from y_true
    mu_pos, var_pos, mu_none, var_none = compute_marker_gaussian_params(y_true, X)

    # Robustify baseline variance (vector clamp)
    var_pos = np.asarray(var_pos, dtype=np.float64)
    var_pos = np.maximum(var_pos, eps)

    # Model per-node variance (vector clamp)
    var_model = np.exp(log_var_model).astype(np.float64)
    var_model = np.maximum(var_model, eps)

    nll_model_list, nll_base_list = [], []
    Delta_mean_list, Delta_var_list = [], []

    for m in range(M):
        mask = X[:, m] > 0.5
        n = int(np.sum(mask))
        n_pos[m] = n

        if n == 0:
            nll_model_list.append(np.zeros((0,), dtype=np.float64))
            nll_base_list.append(np.zeros((0,), dtype=np.float64))
            Delta_mean_list.append(np.zeros((0,), dtype=np.float64))
            Delta_var_list.append(np.zeros((0,), dtype=np.float64))
            continue

        y_m = y_true[mask]
        mu_m = mu_model[mask]
        var_m = var_model[mask]  # (n,)

        mu_b = float(mu_pos[m])
        var_b = float(var_pos[m])  # scalar, already clamped

        # Per-node NLLs
        nll_m = per_node_gaussian_nll(y_m, mu_m, var_m).astype(np.float64)
        nll_b = per_node_gaussian_nll(y_m, mu_b, var_b).astype(np.float64)

        nll_model_list.append(nll_m)
        nll_base_list.append(nll_b)

        # Exact decomposition of the per-node NLL difference
        Delta_var = 0.5 * np.log(var_b / var_m)
        Delta_mean = 0.5 * ((y_m - mu_b) ** 2 / var_b - (y_m - mu_m) ** 2 / var_m)

        Delta_mean_list.append(Delta_mean.astype(np.float64))
        Delta_var_list.append(Delta_var.astype(np.float64))

    return nll_model_list, nll_base_list, Delta_mean_list, Delta_var_list, n_pos




# ============================================================
# Residuals per marker 
# ============================================================
def compute_marker_means(y_true, X):
    """
    μ_m = mean(y_true | marker m positive), μ_none for rows with no positives.
    Returns:
      mu_pos  : (M,) float64
      mu_none : float64
    """
    K, M = X.shape
    mu_pos = np.zeros(M, dtype=np.float64)
    for m in range(M):
        mask = X[:, m] > 0.5
        mu_pos[m] = np.mean(y_true[mask]) if np.any(mask) else np.mean(y_true)
    row_pos = (X > 0.5).sum(axis=1)
    mu_none = float(np.mean(y_true[row_pos == 0])) if np.any(row_pos == 0) else float(np.mean(y_true))
    return mu_pos, mu_none

def residuals_per_marker(y_true, y_pred, X):
    """
    Build residual arrays per marker (positives only). Means are computed inside.
      r_model_list[m]    = y_true[+m] - y_pred[+m]
      r_baseline_list[m] = y_true[+m] - μ_m
    Returns:
      r_model_list    : list length M of (n_m,) float64
      r_baseline_list : list length M of (n_m,) float64
      n_pos           : (M,) int64 positives per marker
      mu_pos          : (M,) float64 (returned for convenience)
    """
    mu_pos, _ = compute_marker_means(y_true, X)
    M = X.shape[1]
    r_model_list, r_base_list = [], []
    n_pos = np.zeros(M, dtype=np.int64)
    for m in range(M):
        mask = X[:, m] > 0.5
        n_pos[m] = int(np.sum(mask))
        if n_pos[m] == 0:
            r_model_list.append(np.zeros((0,), dtype=np.float64))
            r_base_list.append(np.zeros((0,), dtype=np.float64))
        else:
            r_model_list.append((y_true[mask] - y_pred[mask]).astype(np.float64))
            r_base_list.append((y_true[mask] - mu_pos[m]).astype(np.float64))
    return r_model_list, r_base_list, n_pos, mu_pos


# ============================================================
# Curvature binning (single or list)
# ============================================================
def _as_list(arr_or_list):
    """Normalize input to a list of 1D float arrays."""
    if isinstance(arr_or_list, (list, tuple)):
        return [np.asarray(a, dtype=float).ravel() for a in arr_or_list]
    return [np.asarray(arr_or_list, dtype=float).ravel()]


def build_bins_from_true(y_true, nbins=12, strategy="quantile", clip_percentiles=(1.0, 99.0)):
    """
    Build bin edges on TRUE curvature (1D array) and corresponding bin centers.

    strategy='quantile' -> nearly equal counts per bin; centers = mean(true y) per bin
    strategy='uniform'  -> equal-width bins in [lo, hi]; centers = midpoints

    Returns:
      edges   : (B+1,) float64
      centers : (B,)   float64
    """
    y = np.asarray(y_true, float).ravel()
    lo, hi = np.percentile(y, [clip_percentiles[0], clip_percentiles[1]])

    if strategy == "uniform":
        edges = np.linspace(lo, hi, nbins + 1)
    elif strategy == "quantile":
        qs = np.linspace(0, 1, nbins + 1)
        edges = np.quantile(y, qs)
        edges = np.unique(edges)
        if edges.size < nbins + 1:
            edges = np.linspace(lo, hi, nbins + 1)
    else:
        raise ValueError("strategy must be 'quantile' or 'uniform'")

    edges = edges.astype(np.float64)

    # Centers
    if strategy == "quantile":
        idx = np.digitize(y, edges, right=False) - 1
        B = edges.size - 1
        centers = np.empty(B, dtype=np.float64)
        for b in range(B):
            sel = (idx == b)
            if np.any(sel):
                centers[b] = float(np.mean(y[sel]))
            else:
                centers[b] = 0.5 * (edges[b] + edges[b + 1])
    else:  # 'uniform'
        centers = 0.5 * (edges[:-1] + edges[1:])

    return edges, centers



def bin_by_curvature(
    y_in,
    values_in,
    edges=None,
    nbins=12,
    strategy="quantile",
    clip_percentiles=(1.0, 99.0),
):
    """
    Bin one or many series by TRUE curvature and compute per-bin means.

    Inputs:
      y_in      : 1D array OR list of 1D arrays (true y)
      values_in : 1D array OR list of 1D arrays (same shapes)
      edges     : None (build per series), a single edges array for all series,
                  or a list of edges arrays matching y_in when y_in is a list.

    Returns:
      If inputs are single arrays:
        counts  : (B,)   int
        means   : (B,)   float64
        edges   : (B+1,) float64
        centers : (B,)   float64  (mean true y per bin for 'quantile'; midpoints for 'uniform')
      If inputs are lists:
        counts_list  : list of (B,)   int
        means_list   : list of (B,)   float64
        edges_list   : list of (B+1,) float64
        centers_list : list of (B,)   float64
    """
    is_list = isinstance(y_in, (list, tuple))
    if is_list:
        assert isinstance(values_in, (list, tuple)) and len(y_in) == len(values_in), \
            "When y_in is a list, values_in must be a list of the same length."
        YL = [np.asarray(y, float).ravel() for y in y_in]
        VL = [np.asarray(v, float).ravel() for v in values_in]

        # edges handling
        if edges is None:
            EL, CL = [], []
            for y in YL:
                e, c = build_bins_from_true(y, nbins=nbins, strategy=strategy, clip_percentiles=clip_percentiles)
                EL.append(e); CL.append(c)
        elif isinstance(edges, (list, tuple)):
            assert len(edges) == len(YL), "edges list must match number of series."
            EL = [np.asarray(e, float).ravel() for e in edges]
            # centers per series depend on data; compute them to match old behavior
            CL = []
            for y, e in zip(YL, EL):
                # recompute centers with current data and strategy
                _, c = build_bins_from_true(y, nbins=len(e)-1, strategy=strategy, clip_percentiles=clip_percentiles)
                CL.append(c)
        else:
            e = np.asarray(edges, float).ravel()
            EL = [e for _ in YL]
            CL = []
            for y in YL:
                _, c = build_bins_from_true(y, nbins=len(e)-1, strategy=strategy, clip_percentiles=clip_percentiles)
                CL.append(c)

        counts_list, means_list = [], []
        for y, v, e in zip(YL, VL, EL):
            idx = np.digitize(y, e, right=False) - 1
            B = e.size - 1
            cnt = np.zeros(B, dtype=np.int64)
            mu  = np.full(B, np.nan, dtype=np.float64)
            for b in range(B):
                sel = (idx == b)
                cnt[b] = int(np.sum(sel))
                if cnt[b] > 0:
                    mu[b] = float(np.mean(v[sel]))
            counts_list.append(cnt)
            means_list.append(mu)

        return counts_list, means_list, EL, CL

    else:
        y = np.asarray(y_in, float).ravel()
        v = np.asarray(values_in, float).ravel()
        if edges is None or isinstance(edges, (list, tuple)):
            e, centers = build_bins_from_true(y, nbins=nbins, strategy=strategy, clip_percentiles=clip_percentiles)
        else:
            e = np.asarray(edges, float).ravel()
            # centers for single series: match behavior (mean true y per quantile bin / midpoints)
            _, centers = build_bins_from_true(y, nbins=len(e)-1, strategy=strategy, clip_percentiles=clip_percentiles)

        idx = np.digitize(y, e, right=False) - 1
        B = e.size - 1
        cnt = np.zeros(B, dtype=np.int64)
        mu  = np.full(B, np.nan, dtype=np.float64)
        for b in range(B):
            sel = (idx == b)
            cnt[b] = int(np.sum(sel))
            if cnt[b] > 0:
                mu[b] = float(np.mean(v[sel]))
        return cnt, mu, e, centers


# ============================================================
# Uniform-weighted MSE (single or list)
# ============================================================

def uniform_weighted_mse(
    y_in, pred_in, nbins=8, strategy="quantile",
    clip_percentiles=(1.0, 99.0), min_count=1, use_shared_edges=False
):
    """
    Compute uniform-in-curvature MSE for one or many series:
      - build bin edges on TRUE y (either per series or shared across all)
      - per-bin MSE = mean((pred - y)^2)
      - final = mean of per-bin MSE across non-empty bins

    Inputs can be single 1D arrays or lists.

    Returns:
      uw_mse : (S,) float64 array of uniform MSE per series (nan if no non-empty bins)
      uw_sem : (S,) float64 array of SEM across per-bin MSEs (0 if <2 non-empty bins)
    """
    YL = _as_list(y_in)
    PL = _as_list(pred_in)
    assert len(YL) == len(PL), "y_in and pred_in must have same number of series"

    # Optional shared edges (for comparability across series)
    shared_edges = None
    if use_shared_edges:
        y_cat = np.concatenate(YL, axis=0)
        shared_edges, _ = build_bins_from_true(
            y_cat, nbins=nbins, strategy=strategy, clip_percentiles=clip_percentiles
        )

    S = len(YL)
    uw = np.full(S, np.nan, dtype=np.float64)
    se = np.full(S, np.nan, dtype=np.float64)

    for i, (y, p) in enumerate(zip(YL, PL)):
        if shared_edges is None:
            e, _ = build_bins_from_true(
                y, nbins=nbins, strategy=strategy, clip_percentiles=clip_percentiles
            )
        else:
            e = shared_edges

        idx = np.digitize(y, e, right=False) - 1
        B = e.size - 1
        bin_mses = []
        for b in range(B):
            sel = (idx == b)
            if np.sum(sel) >= min_count:
                r = p[sel] - y[sel]
                bin_mses.append(float(np.mean(r**2)))
        if bin_mses:
            bin_mses = np.asarray(bin_mses, dtype=np.float64)
            uw[i] = float(bin_mses.mean())
            se[i] = float(bin_mses.std(ddof=1) / np.sqrt(bin_mses.size)) if bin_mses.size >= 2 else 0.0

    return uw, se
