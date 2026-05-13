"""Utilities for one- or multi-target Gaussian outputs.

Backwards compatibility:
- target_dim=1 with covariance_mode='diagonal' returns old-style 1-D mu/log_var.
- target_dim>1 returns (N, D) mu and either (N, D) log_var or (N, D, D) Cholesky L.
"""
from __future__ import annotations

import copy
from collections.abc import Sequence

import numpy as np
import torch


def normalize_covariance_mode(mode: str) -> str:
    mode = str(mode).lower()
    if mode not in {"diagonal", "full"}:
        raise ValueError("covariance_mode must be 'diagonal' or 'full'")
    return mode


def gaussian_head_output_dim(target_dim: int = 1, covariance_mode: str = "diagonal") -> int:
    target_dim = int(target_dim)
    if target_dim < 1:
        raise ValueError("target_dim must be >= 1")
    covariance_mode = normalize_covariance_mode(covariance_mode)
    if covariance_mode == "diagonal":
        return 2 * target_dim
    return target_dim + target_dim * (target_dim + 1) // 2


def _maybe_squeeze_target_dim(x: torch.Tensor, target_dim: int) -> torch.Tensor:
    return x.reshape(-1) if int(target_dim) == 1 else x


def finalize_gaussian_outputs(
    out: torch.Tensor,
    target_dim: int = 1,
    covariance_mode: str = "diagonal",
    log_scale2_clamp=(-10.0, 10.0),
    diag_eps: float = 1e-6,
):
    """Convert raw head outputs to ``(mu, scale)``.

    In diagonal mode, ``scale`` is log-variance. In full mode, ``scale`` is a
    lower-triangular Cholesky factor L such that covariance = L @ L.T.
    """
    target_dim = int(target_dim)
    covariance_mode = normalize_covariance_mode(covariance_mode)
    expected = gaussian_head_output_dim(target_dim, covariance_mode)
    if out.ndim != 2 or out.shape[1] != expected:
        raise ValueError(f"Expected head output shape (N, {expected}), got {tuple(out.shape)}")

    mu = out[:, :target_dim].contiguous()
    lo, hi = log_scale2_clamp

    if covariance_mode == "diagonal":
        log_var = torch.clamp(out[:, target_dim:target_dim * 2].contiguous(), lo, hi)
        return _maybe_squeeze_target_dim(mu, target_dim), _maybe_squeeze_target_dim(log_var, target_dim)

    raw = out[:, target_dim:]
    L = out.new_zeros((out.shape[0], target_dim, target_dim))
    tril_i, tril_j = torch.tril_indices(target_dim, target_dim, device=out.device)
    L[:, tril_i, tril_j] = raw
    diag_idx = torch.arange(target_dim, device=out.device)
    # Interpret raw diagonal entries as log-variances, matching diagonal mode.
    L[:, diag_idx, diag_idx] = torch.exp(0.5 * torch.clamp(L[:, diag_idx, diag_idx], lo, hi)).clamp_min(diag_eps)
    return mu, L


def as_target_matrix(y: torch.Tensor, target_dim: int | None = None) -> torch.Tensor:
    """Return y as (N, D), accepting old (N,) or new (N, D) targets."""
    if y.ndim == 1:
        y2 = y.unsqueeze(-1)
    elif y.ndim == 2:
        y2 = y
    else:
        y2 = y.reshape(y.shape[0], -1)
    if target_dim is not None and y2.shape[1] != int(target_dim):
        raise ValueError(f"Expected target_dim={target_dim}, got y shape {tuple(y.shape)}")
    return y2


def as_output_matrix(x: torch.Tensor) -> torch.Tensor:
    """Return mu/log_var as (N, D), accepting old (N,) outputs."""
    return x.unsqueeze(-1) if x.ndim == 1 else x


def covariance_from_scale(scale: torch.Tensor) -> torch.Tensor:
    """Return covariance matrix/matrices from diagonal log_var or full Cholesky L."""
    if scale.ndim == 1:
        return torch.diag_embed(torch.exp(scale).unsqueeze(-1))
    if scale.ndim == 2:
        return torch.diag_embed(torch.exp(scale))
    if scale.ndim == 3:
        return scale @ scale.transpose(-1, -2)
    raise ValueError(f"Unsupported scale shape {tuple(scale.shape)}")


def distribution_to_diagonal(mu, scale, *, return_log_var: bool = True):
    """Convert diagonal or full-covariance outputs to old-style independent outputs."""
    if torch.is_tensor(mu):
        mu_out = mu.reshape(-1) if mu.ndim == 2 and mu.shape[1] == 1 else mu
        if scale.ndim == 3:
            cov = covariance_from_scale(scale)
            var = torch.diagonal(cov, dim1=-2, dim2=-1).clamp_min(1e-12)
            lv = torch.log(var)
        else:
            lv = scale
        lv = lv.reshape(-1) if lv.ndim == 2 and lv.shape[1] == 1 else lv
        return (mu_out, lv) if return_log_var else (mu_out, torch.exp(lv))

    mu_np = np.asarray(mu)
    scale_np = np.asarray(scale)
    mu_out = mu_np.reshape(-1) if mu_np.ndim == 2 and mu_np.shape[1] == 1 else mu_np
    if scale_np.ndim == 3:
        cov = scale_np @ np.swapaxes(scale_np, -1, -2)
        var = np.maximum(np.diagonal(cov, axis1=-2, axis2=-1), 1e-12)
        lv = np.log(var)
    else:
        lv = scale_np
    lv = lv.reshape(-1) if lv.ndim == 2 and lv.shape[1] == 1 else lv
    return (mu_out, lv) if return_log_var else (mu_out, np.exp(lv))


def select_target_columns(y, target_indices=None):
    """Select target columns from y while preserving old 1-D behavior for one column."""
    if target_indices is None:
        return y
    y_arr = np.asarray(y)
    y2 = y_arr[:, None] if y_arr.ndim == 1 else y_arr
    if isinstance(target_indices, (int, np.integer)):
        idx = [int(target_indices)]
    else:
        idx = [int(i) for i in target_indices]
    y_sel = y2[:, idx]
    return y_sel[:, 0] if len(idx) == 1 else y_sel


def select_graph_targets(graphs, target_indices=None, *, inplace: bool = False, y_attr: str = "y"):
    """Return graphs with ``g.y`` restricted to selected target columns."""
    if target_indices is None:
        return graphs if inplace else [copy.copy(g) for g in graphs]
    graphs_out = graphs if inplace else [copy.copy(g) for g in graphs]
    for g in graphs_out:
        y = getattr(g, y_attr)
        if torch.is_tensor(y):
            y2 = y.unsqueeze(-1) if y.ndim == 1 else y
            idx = [int(target_indices)] if isinstance(target_indices, int) else [int(i) for i in target_indices]
            sel = y2[:, idx]
            setattr(g, y_attr, sel[:, 0].contiguous() if len(idx) == 1 else sel.contiguous())
        else:
            setattr(g, y_attr, select_target_columns(y, target_indices))
    return graphs_out
