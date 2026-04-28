import numpy as np


def _select_target(a, target_index=0):
    a = np.asarray(a, dtype=np.float64)
    if a.ndim == 2:
        return a[:, int(target_index)]
    return a.reshape(-1)


def append_none_marker_column(X, marker_names, *, name="None"):
    """Append a binary 'no marker positive' column to a marker matrix."""
    X = np.asarray(X, dtype=float)
    none_mask = (X.sum(axis=1) == 0).astype(float)
    X_ext = np.concatenate([X, none_mask[:, None]], axis=1)
    return X_ext, list(marker_names) + [name]


def compute_markerwise_means(y_true, X, target_index=0):
    """
    μ_m = mean(y_true | marker m positive), μ_none for rows with no positives.
    Returns:
      mu_pos  : (M,) float64
      mu_none : float64
    """
    y_true = _select_target(y_true, target_index=target_index)
    K, M = X.shape
    mu_pos = np.zeros(M, dtype=np.float64)
    for m in range(M):
        mask = X[:, m] > 0.5
        mu_pos[m] = np.mean(y_true[mask]) if np.any(mask) else np.mean(y_true)
    row_pos = (X > 0.5).sum(axis=1)
    mu_none = float(np.mean(y_true[row_pos == 0])) if np.any(row_pos == 0) else float(np.mean(y_true))
    return mu_pos, mu_none


def compute_markerwise_residuals(y_true, y_pred, X, target_index=0):
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
    y_true = _select_target(y_true, target_index=target_index)
    y_pred = _select_target(y_pred, target_index=target_index)
    mu_pos, _ = compute_markerwise_means(y_true, X)
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


def estimate_marker_conditional_gaussians(y_true, X, target_index=0):
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


def compute_nodewise_nll(y_true, mu, var):
    """
    Per-node Gaussian NLL (includes constant term):
      0.5 * (log(2*pi*var) + (y-mu)^2 / var)
    Inputs can be numpy arrays.
    """
    var = np.maximum(var, 1e-12)
    return 0.5 * (np.log(2.0 * np.pi * var) + (y_true - mu) ** 2 / var)


def compute_markerwise_nll(y_true, mu_model, log_var_model, X, eps=1e-12, target_index=0):
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
    y_true = _select_target(y_true, target_index=target_index)
    mu_model = _select_target(mu_model, target_index=target_index)
    log_var_model = _select_target(log_var_model, target_index=target_index)
    X = np.asarray(X)

    M = X.shape[1]
    n_pos = np.zeros(M, dtype=np.int64)

    # Baseline Gaussian params per marker, estimated from y_true
    mu_pos, var_pos, mu_none, var_none = estimate_marker_conditional_gaussians(y_true, X)

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
        nll_m = compute_nodewise_nll(y_m, mu_m, var_m).astype(np.float64)
        nll_b = compute_nodewise_nll(y_m, mu_b, var_b).astype(np.float64)

        nll_model_list.append(nll_m)
        nll_base_list.append(nll_b)

        # Exact decomposition of the per-node NLL difference
        Delta_var = 0.5 * np.log(var_b / var_m)
        Delta_mean = 0.5 * ((y_m - mu_b) ** 2 / var_b - (y_m - mu_m) ** 2 / var_m)

        Delta_mean_list.append(Delta_mean.astype(np.float64))
        Delta_var_list.append(Delta_var.astype(np.float64))

    return nll_model_list, nll_base_list, Delta_mean_list, Delta_var_list, n_pos
