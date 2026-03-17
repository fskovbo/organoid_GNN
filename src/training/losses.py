
import torch

def gaussian_nll(mu: torch.Tensor, log_var: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Per-node Gaussian negative log-likelihood (up to an additive constant):
      0.5 * (log_var + (y - mu)^2 / exp(log_var))
    """
    var = torch.exp(log_var)
    return 0.5 * (log_var + (y - mu) ** 2 / var)


def huber_loss(pred, target, delta=1.0):
    """
    Standard Huber loss (smooth L1). Set delta=0 for MSE-ish behavior.
    """
    if delta == 0:
        return 0.5 * (pred - target) ** 2
    err = pred - target
    abs_err = torch.abs(err)
    quad = torch.minimum(abs_err, torch.tensor(delta, device=pred.device))
    return 0.5 * quad**2 + delta * (abs_err - quad)


def calibration_penalty(res: torch.Tensor, y: torch.Tensor, mode: str = "corr", eps: float = 1e-12) -> torch.Tensor:
    """
    Push residuals to be *uncorrelated* with y (i.e., flatten bias vs curvature).
    mode='corr'  -> squared Pearson r between res and y (scale-invariant).
    mode='cov'   -> squared covariance ( (E[(res-mean)(y-mean)])^2 ).
    """
    r = res - res.mean()
    yt = y - y.mean()
    if mode == "corr":
        num = (r * yt).sum()
        den = torch.sqrt((r.square().sum() * yt.square().sum()).clamp_min(eps))
        return (num / den).square()
    elif mode == "cov":
        return ( (r * yt).mean() ).square()
    else:
        raise ValueError("calib_mode must be 'corr' or 'cov'")

@torch.no_grad()
def residual_slope(res: torch.Tensor, y: torch.Tensor, eps: float = 1e-12) -> float:
    """
    Slope of least-squares line residual ~ a + b*y. b≈0 means flat bias curve.
    """
    yt = y - y.mean()
    denom = yt.square().sum().clamp_min(eps)
    b = (res * yt).sum() / denom
    return float(b.detach().cpu())