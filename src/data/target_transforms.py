"""
Target transformation utilities for graph-level/node-level curvature targets.

The central idea is to fit a transform on TRAIN targets only, apply it to graph.y
before training, and later invert predictions (including predicted uncertainty) back
to the original target scale.

Typical usage
-------------

    from target_transforms import StandardizeTransform, AsinhStandardizeTransform

    target_transform = AsinhStandardizeTransform(robust=True).fit(train_graphs)
    target_transform.transform_graphs(train_graphs, in_place=True)
    target_transform.transform_graphs(val_graphs, in_place=True)

    # Later, in prediction/analysis code:
    y_true_orig = target_transform.inverse(y_true_transformed)
    mu_orig = target_transform.inverse_mean(mu_transformed)
    log_var_orig = target_transform.inverse_log_var(mu_transformed, log_var_transformed)

Notes
-----
For nonlinear transforms, a Gaussian in transformed space is not exactly Gaussian
in original space. The inverse variance methods below use a first-order delta-method
approximation, which is usually good when predictive uncertainty is not huge.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence, Tuple, Union

import numpy as np
import torch

ArrayLike = Union[float, Sequence[float], np.ndarray, torch.Tensor]
_EPS = 1e-12


def _graph_targets_to_numpy(graphs: Sequence) -> np.ndarray:
    """Concatenate graph.y values into an array of shape (N, D)."""
    if len(graphs) == 0:
        raise ValueError("Expected at least one graph.")

    ys = []
    for g in graphs:
        if not hasattr(g, "y"):
            raise ValueError("All graphs must have a .y attribute.")
        y = g.y.detach().cpu().numpy() if torch.is_tensor(g.y) else np.asarray(g.y)
        ys.append(y[:, None] if y.ndim == 1 else y)

    return np.concatenate(ys, axis=0)


def _fit_center_scale(y: np.ndarray, robust: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """Fit per-target-dimension center/scale."""
    if robust:
        center = np.median(y, axis=0)
        iqr = np.percentile(y, 75, axis=0) - np.percentile(y, 25, axis=0)
        scale = np.where(iqr > _EPS, iqr / 1.349, 1.0)
    else:
        center = np.mean(y, axis=0)
        std = np.std(y, axis=0)
        scale = np.where(std > _EPS, std, 1.0)

    return center.astype(np.float64), scale.astype(np.float64)


def _as_torch_like(x: ArrayLike, like: torch.Tensor) -> torch.Tensor:
    return torch.as_tensor(x, dtype=like.dtype, device=like.device)


def _restore_1d_if_needed(x: Union[np.ndarray, torch.Tensor], was_1d: bool):
    if was_1d and x.ndim == 2 and x.shape[1] == 1:
        return x[:, 0]
    return x


def _diag_log_var_from_scale(scale: ArrayLike) -> Union[np.ndarray, torch.Tensor]:
    """
    Return diagonal log-variance from either diagonal log_var outputs or Cholesky factors.

    Supports arrays/tensors shaped:
      - (N, D): interpreted as diagonal log-variance
      - (N, D, D): interpreted as Cholesky/covariance factor L, using diag(L L^T)
    """
    if torch.is_tensor(scale):
        if scale.ndim == 3:
            cov = scale @ scale.transpose(-1, -2)
            return torch.log(torch.clamp(torch.diagonal(cov, dim1=-2, dim2=-1), min=_EPS))
        return scale

    arr = np.asarray(scale)
    if arr.ndim == 3:
        cov = arr @ np.swapaxes(arr, -1, -2)
        return np.log(np.maximum(np.diagonal(cov, axis1=-2, axis2=-1), _EPS))
    return arr


@dataclass
class TargetTransform(ABC):
    """Abstract base class for invertible target transforms."""

    name: str = field(default="base", init=False)
    fitted: bool = field(default=False, init=False)

    @abstractmethod
    def fit_array(self, y: np.ndarray) -> "TargetTransform":
        """Fit any transform parameters from an array shaped (N, D)."""

    def fit(self, train_graphs: Sequence) -> "TargetTransform":
        """Fit transform parameters from train_graphs only."""
        return self.fit_array(_graph_targets_to_numpy(train_graphs))

    @abstractmethod
    def forward(self, y: ArrayLike) -> ArrayLike:
        """Transform targets from original space to training space."""

    @abstractmethod
    def inverse(self, z: ArrayLike) -> ArrayLike:
        """Transform targets from training space back to original space."""

    @abstractmethod
    def inverse_derivative(self, z: ArrayLike) -> ArrayLike:
        """
        Derivative d inverse(z) / dz.

        Used for approximate uncertainty propagation from transformed space to
        original space with the delta method.
        """

    def transform_graphs(self, graphs: Sequence, in_place: bool = True) -> Sequence:
        """Apply forward transform to graph.y for every graph."""
        if not self.fitted:
            raise RuntimeError("Transform must be fitted before transform_graphs().")

        out = graphs if in_place else [g.clone() for g in graphs]
        for g in out:
            g.y = self.forward(g.y)
        return out

    def fit_transform_graphs(self, train_graphs: Sequence, in_place: bool = True) -> Sequence:
        """Fit on train_graphs and transform them."""
        self.fit(train_graphs)
        return self.transform_graphs(train_graphs, in_place=in_place)

    def inverse_mean(self, mu: ArrayLike) -> ArrayLike:
        """
        Invert a predicted mean.

        For nonlinear transforms this is inverse(E[z]), not exactly E[inverse(z)].
        It is still the most common deterministic summary for plotting predictions.
        """
        return self.inverse(mu)

    def inverse_log_var(self, mu: ArrayLike, log_var: ArrayLike) -> ArrayLike:
        """
        Approximate inverse transform of diagonal log-variance.

        If y = inverse(z), then Var[y] approx (dy/dz)^2 Var[z].
        Therefore log Var[y] approx log Var[z] + 2 log |dy/dz|.
        """
        diag_log_var = _diag_log_var_from_scale(log_var)
        deriv = self.inverse_derivative(mu)

        if torch.is_tensor(diag_log_var) or torch.is_tensor(deriv):
            if not torch.is_tensor(diag_log_var):
                diag_log_var = torch.as_tensor(diag_log_var, dtype=deriv.dtype, device=deriv.device)
            if not torch.is_tensor(deriv):
                deriv = torch.as_tensor(deriv, dtype=diag_log_var.dtype, device=diag_log_var.device)
            return diag_log_var + 2.0 * torch.log(torch.clamp(torch.abs(deriv), min=_EPS))

        return np.asarray(diag_log_var, dtype=np.float64) + 2.0 * np.log(
            np.maximum(np.abs(np.asarray(deriv, dtype=np.float64)), _EPS)
        )

    def inverse_distribution(
        self,
        y: Optional[ArrayLike],
        mu: ArrayLike,
        log_var: Optional[ArrayLike] = None,
    ) -> Tuple[Optional[ArrayLike], ArrayLike, Optional[ArrayLike]]:
        """Invert true targets, predicted means, and optional diagonal log-variance."""
        y_inv = None if y is None else self.inverse(y)
        mu_inv = self.inverse_mean(mu)
        log_var_inv = None if log_var is None else self.inverse_log_var(mu, log_var)
        return y_inv, mu_inv, log_var_inv


@dataclass
class IdentityTransform(TargetTransform):
    """No-op transform, useful when you want a common interface everywhere."""

    name: str = field(default="identity", init=False)

    def fit_array(self, y: np.ndarray) -> "IdentityTransform":
        self.fitted = True
        return self

    def forward(self, y: ArrayLike) -> ArrayLike:
        return y

    def inverse(self, z: ArrayLike) -> ArrayLike:
        return z

    def inverse_derivative(self, z: ArrayLike) -> ArrayLike:
        if torch.is_tensor(z):
            return torch.ones_like(z)
        return np.ones_like(np.asarray(z, dtype=np.float64))


@dataclass
class StandardizeTransform(TargetTransform):
    """Mean/std or median/IQR standardization: z = (y - center) / scale."""

    robust: bool = False
    center: Optional[np.ndarray] = None
    scale: Optional[np.ndarray] = None
    name: str = field(default="standardize", init=False)

    def fit_array(self, y: np.ndarray) -> "StandardizeTransform":
        y = y[:, None] if y.ndim == 1 else y
        self.center, self.scale = _fit_center_scale(y, robust=self.robust)
        self.fitted = True
        return self

    def forward(self, y: ArrayLike) -> ArrayLike:
        if self.center is None or self.scale is None:
            raise RuntimeError("Transform has not been fitted.")
        if torch.is_tensor(y):
            was_1d = y.ndim == 1
            out = (y - _as_torch_like(self.center, y)) / _as_torch_like(self.scale, y)
            return _restore_1d_if_needed(out, was_1d)
        arr = np.asarray(y, dtype=np.float64)
        was_1d = arr.ndim == 1
        out = (arr - self.center) / self.scale
        return _restore_1d_if_needed(out, was_1d)

    def inverse(self, z: ArrayLike) -> ArrayLike:
        if self.center is None or self.scale is None:
            raise RuntimeError("Transform has not been fitted.")
        if torch.is_tensor(z):
            was_1d = z.ndim == 1
            out = z * _as_torch_like(self.scale, z) + _as_torch_like(self.center, z)
            return _restore_1d_if_needed(out, was_1d)
        arr = np.asarray(z, dtype=np.float64)
        was_1d = arr.ndim == 1
        out = arr * self.scale + self.center
        return _restore_1d_if_needed(out, was_1d)

    def inverse_derivative(self, z: ArrayLike) -> ArrayLike:
        if self.scale is None:
            raise RuntimeError("Transform has not been fitted.")
        if torch.is_tensor(z):
            return torch.ones_like(z) * _as_torch_like(self.scale, z)
        return np.ones_like(np.asarray(z, dtype=np.float64)) * self.scale


@dataclass
class RescaleTransform(StandardizeTransform):
    """
    Simple affine rescaling with user-provided center and scale.

    Useful when you already know the normalization constants and do not want to
    estimate them from the training data.
    """

    center_value: ArrayLike = 0.0
    scale_value: ArrayLike = 1.0
    name: str = field(default="rescale", init=False)

    def fit_array(self, y: np.ndarray) -> "RescaleTransform":
        y = y[:, None] if y.ndim == 1 else y
        d = y.shape[1]
        center = np.asarray(self.center_value, dtype=np.float64)
        scale = np.asarray(self.scale_value, dtype=np.float64)
        self.center = np.broadcast_to(center, (d,)).copy()
        self.scale = np.broadcast_to(scale, (d,)).copy()
        self.scale = np.where(np.abs(self.scale) > _EPS, self.scale, 1.0)
        self.fitted = True
        return self


@dataclass
class AsinhTransform(TargetTransform):
    """
    Asinh transform for heavy-tailed targets: z = asinh((y - center) / scale).

    The scale controls where the transform transitions from nearly linear to
    logarithmic compression. With robust=True, center/scale are fit by median/IQR.
    """

    robust: bool = True
    center: Optional[np.ndarray] = None
    scale: Optional[np.ndarray] = None
    name: str = field(default="asinh", init=False)

    def fit_array(self, y: np.ndarray) -> "AsinhTransform":
        y = y[:, None] if y.ndim == 1 else y
        self.center, self.scale = _fit_center_scale(y, robust=self.robust)
        self.fitted = True
        return self

    def forward(self, y: ArrayLike) -> ArrayLike:
        if self.center is None or self.scale is None:
            raise RuntimeError("Transform has not been fitted.")
        if torch.is_tensor(y):
            was_1d = y.ndim == 1
            out = torch.asinh((y - _as_torch_like(self.center, y)) / _as_torch_like(self.scale, y))
            return _restore_1d_if_needed(out, was_1d)
        arr = np.asarray(y, dtype=np.float64)
        was_1d = arr.ndim == 1
        out = np.arcsinh((arr - self.center) / self.scale)
        return _restore_1d_if_needed(out, was_1d)

    def inverse(self, z: ArrayLike) -> ArrayLike:
        if self.center is None or self.scale is None:
            raise RuntimeError("Transform has not been fitted.")
        if torch.is_tensor(z):
            was_1d = z.ndim == 1
            out = _as_torch_like(self.scale, z) * torch.sinh(z) + _as_torch_like(self.center, z)
            return _restore_1d_if_needed(out, was_1d)
        arr = np.asarray(z, dtype=np.float64)
        was_1d = arr.ndim == 1
        out = self.scale * np.sinh(arr) + self.center
        return _restore_1d_if_needed(out, was_1d)

    def inverse_derivative(self, z: ArrayLike) -> ArrayLike:
        if self.scale is None:
            raise RuntimeError("Transform has not been fitted.")
        if torch.is_tensor(z):
            return _as_torch_like(self.scale, z) * torch.cosh(z)
        return self.scale * np.cosh(np.asarray(z, dtype=np.float64))


@dataclass
class AsinhStandardizeTransform(TargetTransform):
    """
    Two-stage transform: first asinh-compress heavy tails, then standardize.

        u = asinh((y - raw_center) / raw_scale)
        z = (u - center) / scale

    This is often a good default for long-tailed curvature targets because the
    model still sees roughly standardized targets, while extreme values are
    compressed before fitting the Gaussian likelihood.
    """

    robust: bool = True
    raw_center: Optional[np.ndarray] = None
    raw_scale: Optional[np.ndarray] = None
    center: Optional[np.ndarray] = None
    scale: Optional[np.ndarray] = None
    name: str = field(default="asinh_standardize", init=False)

    def fit_array(self, y: np.ndarray) -> "AsinhStandardizeTransform":
        y = y[:, None] if y.ndim == 1 else y
        self.raw_center, self.raw_scale = _fit_center_scale(y, robust=self.robust)
        u = np.arcsinh((y - self.raw_center) / self.raw_scale)
        self.center, self.scale = _fit_center_scale(u, robust=False)
        self.fitted = True
        return self

    def forward(self, y: ArrayLike) -> ArrayLike:
        if any(v is None for v in (self.raw_center, self.raw_scale, self.center, self.scale)):
            raise RuntimeError("Transform has not been fitted.")
        if torch.is_tensor(y):
            was_1d = y.ndim == 1
            u = torch.asinh((y - _as_torch_like(self.raw_center, y)) / _as_torch_like(self.raw_scale, y))
            z = (u - _as_torch_like(self.center, y)) / _as_torch_like(self.scale, y)
            return _restore_1d_if_needed(z, was_1d)
        arr = np.asarray(y, dtype=np.float64)
        was_1d = arr.ndim == 1
        u = np.arcsinh((arr - self.raw_center) / self.raw_scale)
        z = (u - self.center) / self.scale
        return _restore_1d_if_needed(z, was_1d)

    def inverse(self, z: ArrayLike) -> ArrayLike:
        if any(v is None for v in (self.raw_center, self.raw_scale, self.center, self.scale)):
            raise RuntimeError("Transform has not been fitted.")
        if torch.is_tensor(z):
            was_1d = z.ndim == 1
            u = z * _as_torch_like(self.scale, z) + _as_torch_like(self.center, z)
            y = _as_torch_like(self.raw_scale, z) * torch.sinh(u) + _as_torch_like(self.raw_center, z)
            return _restore_1d_if_needed(y, was_1d)
        arr = np.asarray(z, dtype=np.float64)
        was_1d = arr.ndim == 1
        u = arr * self.scale + self.center
        y = self.raw_scale * np.sinh(u) + self.raw_center
        return _restore_1d_if_needed(y, was_1d)

    def inverse_derivative(self, z: ArrayLike) -> ArrayLike:
        if any(v is None for v in (self.raw_center, self.raw_scale, self.center, self.scale)):
            raise RuntimeError("Transform has not been fitted.")
        if torch.is_tensor(z):
            u = z * _as_torch_like(self.scale, z) + _as_torch_like(self.center, z)
            return _as_torch_like(self.raw_scale, z) * torch.cosh(u) * _as_torch_like(self.scale, z)
        arr = np.asarray(z, dtype=np.float64)
        u = arr * self.scale + self.center
        return self.raw_scale * np.cosh(u) * self.scale


def standardize_graph_targets(train_graphs, val_graphs=None, robust: bool = False):
    """
    Backward-compatible wrapper for your old standardize_graph_targets().

    Returns (transform, center, scale). Prefer keeping the returned transform and
    passing it to prediction/analysis functions.
    """
    transform = StandardizeTransform(robust=robust).fit(train_graphs)
    transform.transform_graphs(train_graphs, in_place=True)
    if val_graphs is not None:
        transform.transform_graphs(val_graphs, in_place=True)

    center, scale = transform.center, transform.scale
    if center is not None and center.shape[0] == 1:
        return transform, float(center[0]), float(scale[0])
    return transform, center, scale


def inverse_distribution_outputs(
    y: Optional[ArrayLike],
    mu: ArrayLike,
    log_var: Optional[ArrayLike] = None,
    transform: Optional[TargetTransform] = None,
):
    """
    Convenience function for prediction code.

    If transform is None, returns inputs unchanged. Otherwise calls
    transform.inverse_distribution(y, mu, log_var).
    """
    if transform is None:
        return y, mu, log_var
    return transform.inverse_distribution(y, mu, log_var)




def standardize_graph_global_features(
    train_graphs,
    val_graphs=None,
    attr_name="global_feat",
    robust=False,
):
    """
    Standardize graph-level feature vectors using TRAIN graphs only.

    Assumes each graph stores attr_name with shape (1, D).
    Returns (center, scale), each of shape (D,).
    """
    import numpy as np
    import torch

    X_all = np.concatenate([
        getattr(g, attr_name).detach().cpu().numpy()
        for g in train_graphs
    ], axis=0)

    if robust:
        center = np.median(X_all, axis=0)
        iqr = np.percentile(X_all, 75, axis=0) - np.percentile(X_all, 25, axis=0)
        scale = np.where(iqr > 1e-12, iqr / 1.349, 1.0)
    else:
        center = np.mean(X_all, axis=0)
        std = np.std(X_all, axis=0)
        scale = np.where(std > 1e-12, std, 1.0)

    center_t = torch.as_tensor(center, dtype=train_graphs[0].global_feat.dtype)
    scale_t = torch.as_tensor(scale, dtype=train_graphs[0].global_feat.dtype)

    def _apply(graphs):
        for g in graphs:
            x = getattr(g, attr_name)

            if x.ndim == 1:
                x = x.unsqueeze(0)

            if x.ndim != 2 or x.shape[0] != 1:
                raise ValueError(
                    f"{attr_name} must have shape (1, D), got {tuple(x.shape)}"
                )

            x_std = (x - center_t.unsqueeze(0)) / scale_t.unsqueeze(0)
            setattr(g, attr_name, x_std)

    _apply(train_graphs)
    if val_graphs is not None:
        _apply(val_graphs)

    return center, scale