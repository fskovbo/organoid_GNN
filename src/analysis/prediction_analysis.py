import numpy as np
from scipy.stats import spearmanr


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