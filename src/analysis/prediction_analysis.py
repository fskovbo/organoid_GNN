import numpy as np


def rescale_distribution_outputs(y, mu, log_var=None, center=0.0, scale=1.0):
    y = np.asarray(y, dtype=np.float64) * scale + center
    mu = np.asarray(mu, dtype=np.float64) * scale + center

    if log_var is not None:
        log_var = np.asarray(log_var, dtype=np.float64) + 2.0 * np.log(scale)
            
    return y, mu, log_var