import numpy as np

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



# ============================================================
# Neighborhood composition (k-hop)
# ============================================================
def _adj_list_from_edge_index(edge_index, num_nodes: int):
    """Build an undirected adjacency list from a (2, E) edge_index."""
    src = edge_index[0]
    dst = edge_index[1]
    adj = [[] for _ in range(num_nodes)]
    for u, v in zip(src, dst):
        u = int(u); v = int(v)
        if v not in adj[u]:
            adj[u].append(v)
        if u not in adj[v]:
            adj[v].append(u)
    return adj


def _khop_nodes(adj, start: int, hops: int):
    """Return the set of nodes within <=hops of start (including start)."""
    seen = {start}
    frontier = {start}
    for _ in range(hops):
        nxt = set()
        for u in frontier:
            for v in adj[u]:
                if v not in seen:
                    seen.add(v)
                    nxt.add(v)
        frontier = nxt
        if not frontier:
            break
    return seen


def neighborhood_composition(
    graphs,
    hops: int = 1,
    mode: str = "fraction",
    include_center: bool = False,
    threshold: float = 0.5,
):
    """
    Compute per-node neighborhood marker composition up to k hops.

    Parameters
    ----------
    graphs : list[torch_geometric.data.Data]
        Each graph must have .x (N,M) marker matrix and .edge_index.
    hops : int
        Neighborhood radius in hops.
    mode : {'fraction','presence'}
        'fraction' -> feature[m] = fraction of nodes in neighborhood positive for marker m.
        'presence' -> feature[m] = 1 if any node in neighborhood is positive for marker m.
    include_center : bool
        If False, the center node itself is excluded from the neighborhood aggregation.
    threshold : float
        Threshold for considering a marker positive.

    Returns
    -------
    X_center : (K, M) float32
        Center-cell marker matrix (binarized by threshold) for all nodes across graphs.
    X_neigh : (K, M) float32
        Neighborhood composition features for all nodes across graphs.
    """
    if mode not in {"fraction", "presence"}:
        raise ValueError("mode must be 'fraction' or 'presence'")

    Xc_all, Xn_all = [], []
    for g in graphs:
        X = g.x.detach().cpu().numpy()
        edge_index = g.edge_index.detach().cpu().numpy()
        N, M = X.shape

        Xb = (X > threshold).astype(np.float32)
        adj = _adj_list_from_edge_index(edge_index, N)

        Xn = np.zeros((N, M), dtype=np.float32)
        for i in range(N):
            nodes = _khop_nodes(adj, i, hops)
            if not include_center:
                nodes.discard(i)
            if len(nodes) == 0:
                continue
            nb = Xb[list(nodes), :]
            if mode == "fraction":
                Xn[i, :] = nb.mean(axis=0)
            else:  # presence
                Xn[i, :] = (nb.max(axis=0) > 0.0).astype(np.float32)

        Xc_all.append(Xb)
        Xn_all.append(Xn)

    X_center = np.vstack(Xc_all).astype(np.float32)
    X_neigh = np.vstack(Xn_all).astype(np.float32)
    return X_center, X_neigh


# ============================================================
# Sparse linear influence 
# ============================================================
def sparse_neighborhood_influence_improvement(
    y_true,
    y_pred_mu,
    X_center,
    X_neigh,
    marker_names=None,
    alpha: float = 1e-3,
    use_squared_error: bool = True,
):
    """
    For each center marker A, fit a sparse linear model (Lasso) that predicts
    *improvement over baseline* from neighborhood composition.

    Improvement signal per node i (restricted to A-positive nodes):
      Δ_i = err_base_i - err_model_i
    where
      err_base_i  = (y_i - μ_A)^2 or |y_i - μ_A|
      err_model_i = (y_i - yhat_i)^2 or |y_i - yhat_i|
    and μ_A is the mean curvature among A-positive nodes (computed globally
    from the provided y_true/X_center).

    Parameters
    ----------
    y_true : (K,) array
    y_pred_mu : (K,) array
        Mean prediction μ from your Gaussian model (or deterministic model).
    X_center : (K,M) array
        Center marker indicators.
    X_neigh : (K,M) array
        Neighborhood composition features.
    marker_names : list[str] | None
    alpha : float
        L1 strength for Lasso.
    use_squared_error : bool
        If True use squared error; else absolute error.

    Returns
    -------
    result : dict
        result[A] = {
          'n': int,
          'baseline_mean': float,
          'coef': (M,) array,
          'intercept': float,
          'top': list of (marker_name, coef)
        }
    """
    try:
        from sklearn.linear_model import Lasso
        from sklearn.preprocessing import StandardScaler
    except Exception as e:
        raise ImportError("This function requires scikit-learn (sklearn).") from e

    y_true = np.asarray(y_true, float).ravel()
    y_pred_mu = np.asarray(y_pred_mu, float).ravel()
    X_center = np.asarray(X_center, float)
    X_neigh = np.asarray(X_neigh, float)

    K, M = X_center.shape
    if marker_names is None:
        marker_names = [f"m{j}" for j in range(M)]

    # Baseline means per marker (A-positive)
    mu_pos, _ = compute_marker_means(y_true, X_center)

    out = {}
    scaler = StandardScaler(with_mean=True, with_std=True)

    for A in range(M):
        mask = X_center[:, A] > 0.5
        n = int(mask.sum())
        if n < max(50, 5 * M):
            # too few examples for stable sparse fits; still store minimal info
            out[A] = {"n": n, "baseline_mean": float(mu_pos[A]), "coef": np.zeros(M), "intercept": 0.0, "top": []}
            continue

        yA = y_true[mask]
        muA = float(mu_pos[A])
        yhatA = y_pred_mu[mask]
        XA = X_neigh[mask, :]

        if use_squared_error:
            err_base = (yA - muA) ** 2
            err_model = (yA - yhatA) ** 2
        else:
            err_base = np.abs(yA - muA)
            err_model = np.abs(yA - yhatA)

        delta = err_base - err_model  # positive => model improves over baseline

        # Standardize features for Lasso
        XA_z = scaler.fit_transform(XA)
        reg = Lasso(alpha=alpha, fit_intercept=True, max_iter=10000)
        reg.fit(XA_z, delta)

        coef = reg.coef_.astype(float)
        intercept = float(reg.intercept_)

        # Top influences by absolute weight
        top_idx = np.argsort(-np.abs(coef))
        top = [(marker_names[j], float(coef[j])) for j in top_idx[:10] if coef[j] != 0.0]

        out[A] = {
            "n": n,
            "baseline_mean": muA,
            "coef": coef,
            "intercept": intercept,
            "top": top,
        }

    return out


def sparse_neighborhood_influence_uncertainty_reduction(
    y_true,
    X_center,
    X_neigh,
    pred_var,
    marker_names=None,
    alpha: float = 1e-3,
):
    """
    For each center marker A, fit a sparse linear model (Lasso) that predicts
    *uncertainty reduction* from neighborhood composition.

    We define a simple baseline uncertainty for A-positive nodes as the empirical
    residual variance around the marker-conditional mean:
      base_mse_A = mean((y - μ_A)^2 | A+)

    Then for each A-positive node i:
      Δσ²_i = base_mse_A - var_pred_i
    where var_pred_i is the model-predicted variance for node i.

    Parameters
    ----------
    y_true : (K,) array
    X_center : (K,M) array
    X_neigh : (K,M) array
    pred_var : (K,) array
        Predicted variance σ² in the SAME units as y_true.
    marker_names : list[str] | None
    alpha : float
        L1 strength for Lasso.

    Returns
    -------
    result : dict
        result[A] = {
          'n': int,
          'baseline_mse': float,
          'coef': (M,) array,
          'intercept': float,
          'top': list of (marker_name, coef)
        }
    """
    try:
        from sklearn.linear_model import Lasso
        from sklearn.preprocessing import StandardScaler
    except Exception as e:
        raise ImportError("This function requires scikit-learn (sklearn).") from e

    y_true = np.asarray(y_true, float).ravel()
    X_center = np.asarray(X_center, float)
    X_neigh = np.asarray(X_neigh, float)
    pred_var = np.asarray(pred_var, float).ravel()

    K, M = X_center.shape
    if marker_names is None:
        marker_names = [f"m{j}" for j in range(M)]

    mu_pos, _ = compute_marker_means(y_true, X_center)
    scaler = StandardScaler(with_mean=True, with_std=True)
    out = {}

    for A in range(M):
        mask = X_center[:, A] > 0.5
        n = int(mask.sum())
        if n < max(50, 5 * M):
            out[A] = {"n": n, "baseline_mse": np.nan, "coef": np.zeros(M), "intercept": 0.0, "top": []}
            continue

        yA = y_true[mask]
        muA = float(mu_pos[A])
        XA = X_neigh[mask, :]
        varA = pred_var[mask]

        base_mse_A = float(np.mean((yA - muA) ** 2))
        delta = base_mse_A - varA  # positive => model is more certain than baseline

        XA_z = scaler.fit_transform(XA)
        reg = Lasso(alpha=alpha, fit_intercept=True, max_iter=10000)
        reg.fit(XA_z, delta)

        coef = reg.coef_.astype(float)
        intercept = float(reg.intercept_)
        top_idx = np.argsort(-np.abs(coef))
        top = [(marker_names[j], float(coef[j])) for j in top_idx[:10] if coef[j] != 0.0]

        out[A] = {
            "n": n,
            "baseline_mse": base_mse_A,
            "coef": coef,
            "intercept": intercept,
            "top": top,
        }

    return out
