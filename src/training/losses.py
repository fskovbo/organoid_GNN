import math
import torch
import torch.nn.functional as F


def gaussian_nll(mu: torch.Tensor, log_var: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Per-node Gaussian negative log-likelihood (up to an additive constant):
      0.5 * (log_var + (y - mu)^2 / exp(log_var))
    """
    var = torch.exp(log_var).clamp_min(1e-12)
    return 0.5 * (log_var + (y - mu).square() / var)


def student_t_nll(
    mu: torch.Tensor,
    log_scale2: torch.Tensor,
    y: torch.Tensor,
    df: float = 4.0,
) -> torch.Tensor:
    """
    Per-node Student-t negative log-likelihood.

    Parameterization:
      scale^2 = exp(log_scale2)

    For heavy-tailed targets, this is often more robust than Gaussian NLL.
    """
    if df <= 0:
        raise ValueError("df must be > 0")

    scale2 = torch.exp(log_scale2).clamp_min(1e-12)
    z2 = (y - mu).square() / scale2

    c = (
        torch.lgamma(torch.tensor((df + 1.0) / 2.0, device=y.device, dtype=y.dtype))
        - torch.lgamma(torch.tensor(df / 2.0, device=y.device, dtype=y.dtype))
        - 0.5 * torch.log(torch.tensor(df * math.pi, device=y.device, dtype=y.dtype))
    )

    # log p = c - 0.5 log(scale2) - ((df+1)/2) log(1 + z2/df)
    logp = c - 0.5 * log_scale2 - ((df + 1.0) / 2.0) * torch.log1p(z2 / df)
    return -logp


def huber_loss(pred: torch.Tensor, target: torch.Tensor, delta: float = 1.0) -> torch.Tensor:
    """
    Per-node Huber loss.
    """
    err = pred - target
    abs_err = err.abs()
    if delta <= 0:
        return 0.5 * err.square()

    quad = torch.minimum(abs_err, torch.tensor(delta, device=pred.device, dtype=pred.dtype))
    return 0.5 * quad.square() + delta * (abs_err - quad)


def mae_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (pred - target).abs()


def variance_calibration_nll(
    mu: torch.Tensor,
    log_var: torch.Tensor,
    y: torch.Tensor,
    detach_mean: bool = True,
) -> torch.Tensor:
    """
    Gaussian NLL used only to calibrate the variance head.

    If detach_mean=True, the variance head is trained against frozen residuals,
    so the uncertainty branch does not pull the mean back toward conservative
    Gaussian-NLL behavior.
    """
    mu_used = mu.detach() if detach_mean else mu
    return gaussian_nll(mu_used, log_var, y)


def huber_plus_variance_calibration(
    mu: torch.Tensor,
    log_var: torch.Tensor,
    y: torch.Tensor,
    delta: float = 1.0,
    var_weight: float = 1.0,
    detach_mean_for_var: bool = True,
) -> torch.Tensor:
    """
    Per-node combined loss:
      Huber(mean) + var_weight * GaussianNLL(detached mean, variance)

    This is a simple way to get sharper means while still training a variance head.
    """
    mean_term = huber_loss(mu, y, delta=delta)
    var_term = variance_calibration_nll(
        mu,
        log_var,
        y,
        detach_mean=detach_mean_for_var,
    )
    return mean_term + var_weight * var_term


def calibration_penalty(
    res: torch.Tensor,
    y: torch.Tensor,
    mode: str = "corr",
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Push residuals to be uncorrelated with y.
    """
    r = res - res.mean()
    yt = y - y.mean()
    if mode == "corr":
        num = (r * yt).sum()
        den = torch.sqrt((r.square().sum() * yt.square().sum()).clamp_min(eps))
        return (num / den).square()
    elif mode == "cov":
        return ((r * yt).mean()).square()
    else:
        raise ValueError("calib_mode must be 'corr' or 'cov'")


@torch.no_grad()
def residual_slope(res: torch.Tensor, y: torch.Tensor, eps: float = 1e-12) -> float:
    yt = y - y.mean()
    denom = yt.square().sum().clamp_min(eps)
    b = (res * yt).sum() / denom
    return float(b.detach().cpu())