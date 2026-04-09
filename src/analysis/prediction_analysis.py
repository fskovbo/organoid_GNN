import numpy as np
from scipy.stats import spearmanr
from src.inference.predict import predict_targets
from src.analysis.marker_stats import (
    compute_markerwise_residuals,
    compute_markerwise_nll,
    compute_nodewise_nll,
)


def rescale_distribution_outputs(y, mu, log_var=None, center=0.0, scale=1.0):
    y = np.asarray(y, dtype=np.float64) * scale + center
    mu = np.asarray(mu, dtype=np.float64) * scale + center

    if log_var is not None:
        log_var = np.asarray(log_var, dtype=np.float64) + 2.0 * np.log(scale)
            
    return y, mu, log_var


def compute_markerwise_uncertainty_correlation(y, mu, log_var, X):
    """
    For each marker (positives only):
        compute correlation between predicted variance and squared residual.

    Returns
    -------
    rho : (M,) Spearman correlation
    pval : (M,) p-value
    n_pos : (M,) number of samples
    """

    y = np.asarray(y)
    mu = np.asarray(mu)
    var = np.exp(np.asarray(log_var))
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


def compute_global_uncertainty_correlation(y, mu, log_var):
    r2 = (y - mu) ** 2
    v = np.exp(log_var)

    return spearmanr(v, r2)


def _safe_sem(x):
    x = np.asarray(x, dtype=np.float64)
    if x.size <= 1:
        return np.nan
    return float(np.std(x, ddof=1) / np.sqrt(x.size))


def _aggregate_subset_metrics(y, mu, log_var, X, mask, eps=1e-12):
    """
    Aggregate metrics on an arbitrary subset of nodes.

    Baseline mean/variance are estimated from y_true on the same subset.
    """
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

    # simple subset baseline
    mu_b = float(np.mean(y_s))
    var_b = float(np.var(y_s)) + eps

    res_model = y_s - mu_s
    res_base = y_s - mu_b

    mse_model = float(np.mean(res_model**2))
    mse_base = float(np.mean(res_base**2))

    nll_model_node = compute_nodewise_nll(y_s, mu_s, var_s)
    nll_base_node = compute_nodewise_nll(y_s, mu_b, var_b)

    from scipy.stats import spearmanr
    rho, pval = (np.nan, np.nan)
    if n >= 10:
        rho, pval = spearmanr(var_s, res_model**2)

    return {
        "n": n,
        "mse_model": mse_model,
        "sem_mse_model": _safe_sem(res_model**2),
        "mse_base": mse_base,
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


def eval_model_per_marker(model, g_val, device, scale, center, eps=1e-12, center_only=False):
    """
    Evaluate markerwise metrics, plus aggregate metrics for:
      - any_marker: nodes with at least one positive marker
      - no_marker : nodes with no positive markers
      - all_nodes : all evaluated nodes
    """
    y, mu, log_var, X = predict_targets(
        g_val,
        model,
        device=device,
        return_log_var=True,
        center_only=center_only,
    )

    y, mu, log_var = rescale_distribution_outputs(
        y, mu, log_var=log_var, center=center, scale=scale
    )
    var = np.exp(log_var)

    residuals_model, residuals_base, n_pos_resid, mu_pos_resid = compute_markerwise_residuals(y, mu, X)

    mse_model = np.array([np.mean(r**2) if r.size > 0 else np.nan for r in residuals_model])
    mse_base = np.array([np.mean(r**2) if r.size > 0 else np.nan for r in residuals_base])

    sem_mse_model = np.array([
        np.std(r**2, ddof=1) / np.sqrt(r.size) if r.size > 1 else np.nan
        for r in residuals_model
    ])
    sem_mse_base = np.array([
        np.std(r**2, ddof=1) / np.sqrt(r.size) if r.size > 1 else np.nan
        for r in residuals_base
    ])

    Xb = X > 0.5
    var_model = np.array([
        np.mean(var[Xb[:, m]]) if np.any(Xb[:, m]) else np.nan
        for m in range(X.shape[1])
    ])
    sem_var_model = np.array([
        np.std(var[Xb[:, m]], ddof=1) / np.sqrt(np.sum(Xb[:, m]))
        if np.sum(Xb[:, m]) > 1 else np.nan
        for m in range(X.shape[1])
    ])

    var_base = mse_base
    sem_var_base = sem_mse_base

    nll_model_list, nll_base_list, delta_mean_list, delta_var_list, n_pos_nll = compute_markerwise_nll(
        y_true=y,
        mu_model=mu,
        log_var_model=log_var,
        X=X,
        eps=eps,
    )

    nll_model = np.array([np.mean(v) if v.size > 0 else np.nan for v in nll_model_list])
    nll_base = np.array([np.mean(v) if v.size > 0 else np.nan for v in nll_base_list])

    sem_nll_model = np.array([
        np.std(v, ddof=1) / np.sqrt(v.size) if v.size > 1 else np.nan
        for v in nll_model_list
    ])
    sem_nll_base = np.array([
        np.std(v, ddof=1) / np.sqrt(v.size) if v.size > 1 else np.nan
        for v in nll_base_list
    ])

    delta_mean = np.array([np.mean(v) if v.size > 0 else np.nan for v in delta_mean_list])
    delta_var = np.array([np.mean(v) if v.size > 0 else np.nan for v in delta_var_list])

    rho_marker, pval_marker, n_pos_corr = compute_markerwise_uncertainty_correlation(
        y=y,
        mu=mu,
        log_var=log_var,
        X=X,
    )

    rho_global, pval_global = compute_global_uncertainty_correlation(
        y=y,
        mu=mu,
        log_var=log_var,
    )

    # aggregate masks
    row_pos = (X > 0.5).sum(axis=1)
    mask_any = row_pos > 0
    mask_none = row_pos == 0
    mask_all = np.ones_like(row_pos, dtype=bool)

    aggregate = {
        "any_marker": _aggregate_subset_metrics(y, mu, log_var, X, mask_any, eps=eps),
        "no_marker": _aggregate_subset_metrics(y, mu, log_var, X, mask_none, eps=eps),
        "all_nodes": _aggregate_subset_metrics(y, mu, log_var, X, mask_all, eps=eps),
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