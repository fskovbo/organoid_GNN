import numpy as np
from scipy.stats import spearmanr
from src.inference.predict import predict_targets
from src.analysis.marker_stats import compute_markerwise_residuals, compute_markerwise_nll

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



def eval_model_per_marker(model, g_val, device, scale, center, eps=1e-12, center_only=False):
    """
    Evaluate predictive performance and uncertainty calibration of a model
    on a validation graph, aggregated per marker.

    The function runs the model to obtain predictive means and variances,
    rescales outputs back to the original data space, and computes a range
    of marker-wise metrics. These include mean squared error (MSE) and
    negative log-likelihood (NLL) for both the model and a baseline,
    standard errors of these estimates, average predicted variance,
    and diagnostics quantifying how predictive uncertainty correlates
    with squared residuals. It also reports differences in mean and
    variance relative to the baseline.

    Returns a dictionary containing all marker-level performance,
    uncertainty, and calibration statistics, as well as global
    uncertainty–error correlation measures.
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

    # --- uncertainty / squared-residual correlation ---
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
    }