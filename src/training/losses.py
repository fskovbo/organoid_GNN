import math
from dataclasses import dataclass, field
from typing import Callable

import torch


@dataclass
class WeightedLossTerm:
    """Wrap one auxiliary loss term with a name, weight, and fixed parameters."""

    name: str
    fn: Callable
    weight: float = 1.0
    enabled: bool = True
    params: dict = field(default_factory=dict)

    def __call__(self, *, mu: torch.Tensor, log_scale2: torch.Tensor, batch) -> torch.Tensor:
        """Evaluate the weighted auxiliary term on one batched forward pass."""
        if not self.enabled or self.weight == 0.0:
            return mu.new_zeros(())
        val = self.fn(mu=mu, log_scale2=log_scale2, batch=batch, **self.params)
        return self.weight * val


class CompositeLoss:
    """Sum the Gaussian base loss with any number of externally supplied auxiliary terms."""

    def __init__(self, base_loss_fn: Callable, aux_terms=None, return_breakdown: bool = False):
        self.base_loss_fn = base_loss_fn
        self.aux_terms = list(aux_terms) if aux_terms is not None else []
        self.return_breakdown = return_breakdown

    def __call__(self, *, mu: torch.Tensor, log_scale2: torch.Tensor, batch):
        """Compute the full loss, optionally returning an unweighted term breakdown."""
        base = self.base_loss_fn(mu=mu, log_scale2=log_scale2, batch=batch)
        total = base

        if not self.return_breakdown:
            for term in self.aux_terms:
                total = total + term(mu=mu, log_scale2=log_scale2, batch=batch)
            return total

        breakdown = {"base": float(base.detach().cpu())}
        for term in self.aux_terms:
            if not term.enabled or term.weight == 0.0:
                breakdown[term.name] = 0.0
                continue
            raw = term.fn(mu=mu, log_scale2=log_scale2, batch=batch, **term.params)
            total = total + term.weight * raw
            breakdown[term.name] = float(raw.detach().cpu())
        return total, breakdown


def _as_matrix(x: torch.Tensor) -> torch.Tensor:
    return x.unsqueeze(-1) if x.ndim == 1 else x


def gaussian_nll(mu: torch.Tensor, log_var: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Compute diagonal Gaussian NLL per node and target.

    Supports old one-target shapes (N,) and new multi-target shapes (N, D).
    """
    mu = _as_matrix(mu)
    log_var = _as_matrix(log_var)
    y = _as_matrix(y)
    if mu.shape != y.shape:
        raise ValueError(f"mu and y shape mismatch: {tuple(mu.shape)} vs {tuple(y.shape)}")
    var = torch.exp(log_var).clamp_min(1e-12)
    return 0.5 * (log_var + (y - mu).square() / var)


def multivariate_gaussian_nll(mu: torch.Tensor, chol: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Compute full-covariance Gaussian NLL per node using Cholesky factors.

    ``chol`` is (N, D, D), lower triangular, with positive diagonal.
    Constants are omitted, matching ``gaussian_nll``.
    """
    mu = _as_matrix(mu)
    y = _as_matrix(y)
    if chol.ndim != 3:
        raise ValueError(f"Expected chol shape (N, D, D), got {tuple(chol.shape)}")
    if mu.shape != y.shape or chol.shape[0] != y.shape[0] or chol.shape[1] != y.shape[1] or chol.shape[2] != y.shape[1]:
        raise ValueError(f"Incompatible shapes: mu={tuple(mu.shape)}, y={tuple(y.shape)}, chol={tuple(chol.shape)}")
    diff = (y - mu).unsqueeze(-1)
    sol = torch.linalg.solve_triangular(chol, diff, upper=False).squeeze(-1)
    maha = sol.square().sum(dim=-1)
    log_det_cov = 2.0 * torch.log(torch.diagonal(chol, dim1=-2, dim2=-1).clamp_min(1e-12)).sum(dim=-1)
    return 0.5 * (log_det_cov + maha)


def gaussian_loss_from_outputs(mu: torch.Tensor, scale: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Return mean NLL for diagonal or full-covariance outputs."""
    if scale.ndim == 3:
        return multivariate_gaussian_nll(mu, scale, y).mean()
    return gaussian_nll(mu, scale, y).mean()


def make_base_loss(_cfg=None) -> Callable:
    """Build the base training loss, diagonal by default and full when model outputs Cholesky factors."""

    def base_loss(*, mu: torch.Tensor, log_scale2: torch.Tensor, batch) -> torch.Tensor:
        return gaussian_loss_from_outputs(mu, log_scale2, batch.y)

    return base_loss


def edge_difference_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    edge_index: torch.Tensor,
) -> torch.Tensor:
    """Match edgewise prediction differences to edgewise target differences."""
    if edge_index.numel() == 0:
        return pred.new_zeros(())

    src, dst = edge_index
    pred_diff = pred[src] - pred[dst]
    target_diff = target[src] - target[dst]
    err2 = (pred_diff - target_diff).square()
    return err2.mean() if err2.numel() > 0 else pred.new_zeros(())


def weighted_edge_difference_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    edge_index: torch.Tensor,
    alpha: float = 1.0,
    normalize_by: str = "graph_std",
    clip_weight: float | None = 4.0,
    batch_index: torch.Tensor | None = None,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Match edgewise differences while upweighting edges with large true jumps."""
    if edge_index.numel() == 0:
        return pred.new_zeros(())
    if alpha < 0:
        raise ValueError("alpha must be >= 0")
    if normalize_by not in {"global_std", "graph_std", "none"}:
        raise ValueError("normalize_by must be one of: 'global_std', 'graph_std', 'none'")

    src, dst = edge_index
    pred_diff = pred[src] - pred[dst]
    target_diff = target[src] - target[dst]
    err2 = (pred_diff - target_diff).square()
    abs_target_diff = target_diff.abs()

    if normalize_by == "none":
        scale = torch.ones_like(abs_target_diff)
    elif normalize_by == "global_std" or batch_index is None:
        global_scale = target.std(unbiased=False).clamp_min(eps)
        scale = torch.full_like(abs_target_diff, global_scale)
    else:
        n_graphs = int(batch_index.max().item()) + 1
        per_graph_std = torch.empty(n_graphs, device=target.device, dtype=target.dtype)
        for g in range(n_graphs):
            mask = batch_index == g
            per_graph_std[g] = target[mask].std(unbiased=False).clamp_min(eps) if mask.any() else target.new_tensor(1.0)
        scale = per_graph_std[batch_index[src]]

    weights = 1.0 + alpha * (abs_target_diff / scale.clamp_min(eps))
    if clip_weight is not None:
        if clip_weight < 1.0:
            raise ValueError("clip_weight must be >= 1.0 when provided")
        weights = torch.clamp(weights, max=clip_weight)

    return (weights * err2).sum() / weights.sum().clamp_min(eps)


def calibration_penalty(
    res: torch.Tensor,
    y: torch.Tensor,
    mode: str = "corr",
    eps: float = 1e-12,
) -> torch.Tensor:
    """Penalize correlation or covariance between residuals and targets."""
    r = res - res.mean()
    yt = y - y.mean()
    if mode == "corr":
        num = (r * yt).sum()
        den = torch.sqrt((r.square().sum() * yt.square().sum()).clamp_min(eps))
        return (num / den).square()
    if mode == "cov":
        return ((r * yt).mean()).square()
    raise ValueError("mode must be 'corr' or 'cov'")


@torch.no_grad()
def residual_slope(res: torch.Tensor, y: torch.Tensor, eps: float = 1e-12) -> float:
    """Estimate the least-squares slope of residuals regressed on targets."""
    yt = y - y.mean()
    denom = yt.square().sum().clamp_min(eps)
    b = (res * yt).sum() / denom
    return float(b.detach().cpu())


def integrated_curvature_loss(
    pred: torch.Tensor,
    cell_patch_area: torch.Tensor,
    batch_index: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """Penalize deviation of per-graph integrated predicted curvature from 4π."""
    pred = pred.reshape(-1)
    cell_patch_area = cell_patch_area.reshape(-1)
    batch_index = batch_index.reshape(-1)

    if pred.shape[0] != cell_patch_area.shape[0]:
        raise ValueError(
            f"pred and cell_patch_area length mismatch: {pred.shape[0]} vs {cell_patch_area.shape[0]}"
        )
    if pred.shape[0] != batch_index.shape[0]:
        raise ValueError(
            f"pred and batch_index length mismatch: {pred.shape[0]} vs {batch_index.shape[0]}"
        )

    n_graphs = int(batch_index.max().item()) + 1 if batch_index.numel() > 0 else 0
    if n_graphs == 0:
        return pred.new_zeros(())

    integrated_pred = torch.zeros(n_graphs, device=pred.device, dtype=pred.dtype)
    integrated_pred.scatter_add_(0, batch_index, pred * cell_patch_area)

    target_val = pred.new_tensor(4.0 * math.pi)
    per_graph = (integrated_pred - target_val).square()

    if reduction == "mean":
        return per_graph.mean()
    if reduction == "sum":
        return per_graph.sum()
    if reduction == "none":
        return per_graph
    raise ValueError("reduction must be 'mean', 'sum', or 'none'")


def graph_sum_matching_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    batch_index: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """Penalize mismatch between per-graph sums of predictions and targets."""
    pred = _as_matrix(pred)
    target = _as_matrix(target)
    batch_index = batch_index.reshape(-1)

    if pred.shape[0] != target.shape[0]:
        raise ValueError(f"pred and target length mismatch: {pred.shape[0]} vs {target.shape[0]}")
    if pred.shape[0] != batch_index.shape[0]:
        raise ValueError(
            f"pred and batch_index length mismatch: {pred.shape[0]} vs {batch_index.shape[0]}"
        )

    n_graphs = int(batch_index.max().item()) + 1 if batch_index.numel() > 0 else 0
    if n_graphs == 0:
        return pred.new_zeros(())

    pred_sum = torch.zeros((n_graphs, pred.shape[1]), device=pred.device, dtype=pred.dtype)
    true_sum = torch.zeros((n_graphs, target.shape[1]), device=target.device, dtype=target.dtype)
    idx = batch_index.unsqueeze(-1).expand_as(pred)
    pred_sum.scatter_add_(0, idx, pred)
    true_sum.scatter_add_(0, idx, target)

    per_graph = (pred_sum - true_sum).square()

    if reduction == "mean":
        return per_graph.mean()
    if reduction == "sum":
        return per_graph.sum()
    if reduction == "none":
        return per_graph
    raise ValueError("reduction must be 'mean', 'sum', or 'none'")


def edge_loss_term(
    *,
    mu: torch.Tensor,
    log_scale2: torch.Tensor,
    batch,
    weighted: bool = False,
    alpha: float = 1.0,
    normalize_by: str = "graph_std",
    clip_weight: float | None = 4.0,
) -> torch.Tensor:
    """Compute an auxiliary edge-consistency term from the current batch."""
    del log_scale2
    batch_index = getattr(batch, "batch", None)
    if weighted:
        return weighted_edge_difference_loss(
            pred=mu,
            target=batch.y,
            edge_index=batch.edge_index,
            alpha=alpha,
            normalize_by=normalize_by,
            clip_weight=clip_weight,
            batch_index=batch_index,
        )
    return edge_difference_loss(pred=mu, target=batch.y, edge_index=batch.edge_index)


def calibration_term(
    *,
    mu: torch.Tensor,
    log_scale2: torch.Tensor,
    batch,
    mode: str = "corr",
) -> torch.Tensor:
    """Compute an auxiliary residual-calibration penalty from the current batch."""
    del log_scale2
    return calibration_penalty(mu - batch.y, batch.y, mode=mode)


def integrated_curvature_term(
    *,
    mu: torch.Tensor,
    log_scale2: torch.Tensor,
    batch,
    reduction: str = "mean",
    target_index: int = 0,
) -> torch.Tensor:
    """Compute the auxiliary 4π integrated-curvature penalty for each graph."""
    del log_scale2
    if not hasattr(batch, "cell_patch_area"):
        raise AttributeError(
            "Batch has no 'cell_patch_area'. Materialize it onto graphs before training."
        )
    pred = _as_matrix(mu)[:, int(target_index)]
    return integrated_curvature_loss(
        pred=pred,
        cell_patch_area=batch.cell_patch_area,
        batch_index=batch.batch,
        reduction=reduction,
    )


def graph_sum_matching_term(
    *,
    mu: torch.Tensor,
    log_scale2: torch.Tensor,
    batch,
    reduction: str = "mean",
) -> torch.Tensor:
    """Compute the auxiliary per-graph sum-matching penalty for predictions."""
    del log_scale2
    return graph_sum_matching_loss(
        pred=mu,
        target=batch.y,
        batch_index=batch.batch,
        reduction=reduction,
    )