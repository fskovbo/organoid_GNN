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

import copy
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional, Sequence, Tuple, Union

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


def _single_selected_target_values(*values: Optional[ArrayLike]) -> bool:
    """Return True when provided arrays represent one selected target dimension."""
    saw_value = False
    for value in values:
        if value is None:
            continue
        saw_value = True
        shape = tuple(value.shape) if torch.is_tensor(value) else np.asarray(value).shape
        if not (len(shape) == 1 or (len(shape) == 2 and shape[1] == 1)):
            return False
    return saw_value


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
        graphs: Optional[Sequence] = None,
        center_only: bool = False,
        target_index: Optional[int] = None,
    ) -> Tuple[Optional[ArrayLike], ArrayLike, Optional[ArrayLike]]:
        """Invert true targets, predicted means, and optional diagonal log-variance."""
        y_inv = None if y is None else self.inverse(y)
        mu_inv = self.inverse_mean(mu)
        log_var_inv = None if log_var is None else self.inverse_log_var(mu, log_var)
        return y_inv, mu_inv, log_var_inv


@dataclass
class ChainedTargetTransform(TargetTransform):
    """Apply multiple target transforms in sequence.

    The forward direction applies transforms in the order provided. The inverse
    direction applies them in reverse order, including uncertainty propagation.
    This is useful for workflows such as:

        raw target -> baseline residual -> asinh/standardized residual

    Graph-aware transforms are supported because ``fit`` and
    ``transform_graphs`` operate on graph lists, and ``inverse_distribution``
    forwards the optional ``graphs`` context to each inverse stage.
    """

    transforms: Sequence[TargetTransform] = field(default_factory=list)
    name: str = field(default="chain", init=False)

    def __post_init__(self):
        self.transforms = list(self.transforms)
        if len(self.transforms) == 0:
            raise ValueError("ChainedTargetTransform requires at least one transform.")
        self.name = "chain[" + " -> ".join(t.name for t in self.transforms) + "]"

    def fit_array(self, y: np.ndarray) -> "ChainedTargetTransform":
        arr = y[:, None] if y.ndim == 1 else np.asarray(y, dtype=np.float64)
        for transform in self.transforms:
            transform.fit_array(arr)
            arr = transform.forward(arr)
            arr = arr[:, None] if np.asarray(arr).ndim == 1 else np.asarray(arr)
        self.fitted = True
        return self

    def fit(self, train_graphs: Sequence) -> "ChainedTargetTransform":
        tmp_graphs = [copy.deepcopy(g) for g in train_graphs]
        for transform in self.transforms:
            transform.fit(tmp_graphs)
            transform.transform_graphs(tmp_graphs, in_place=True)
        self.fitted = True
        return self

    def forward(self, y: ArrayLike) -> ArrayLike:
        out = y
        for transform in self.transforms:
            out = transform.forward(out)
        return out

    def inverse(self, z: ArrayLike) -> ArrayLike:
        out = z
        for transform in reversed(self.transforms):
            out = transform.inverse(out)
        return out

    def inverse_derivative(self, z: ArrayLike) -> ArrayLike:
        # Compose inverse derivatives by walking backward through the chain.
        out = z
        deriv_total = None
        reversed_transforms = list(reversed(self.transforms))
        for i, transform in enumerate(reversed_transforms):
            deriv = transform.inverse_derivative(out)
            deriv_total = deriv if deriv_total is None else deriv_total * deriv
            if i < len(reversed_transforms) - 1:
                out = transform.inverse(out)
        return deriv_total

    def transform_graphs(self, graphs: Sequence, in_place: bool = True) -> Sequence:
        if not self.fitted:
            raise RuntimeError("Transform must be fitted before transform_graphs().")
        out = graphs if in_place else [copy.deepcopy(g) for g in graphs]
        for transform in self.transforms:
            transform.transform_graphs(out, in_place=True)
        return out

    def inverse_distribution(
        self,
        y: Optional[ArrayLike],
        mu: ArrayLike,
        log_var: Optional[ArrayLike] = None,
        graphs: Optional[Sequence] = None,
        center_only: bool = False,
        target_index: Optional[int] = None,
    ) -> Tuple[Optional[ArrayLike], ArrayLike, Optional[ArrayLike]]:
        y_cur, mu_cur, log_var_cur = y, mu, log_var
        for transform in reversed(self.transforms):
            y_cur, mu_cur, log_var_cur = transform.inverse_distribution(
                y_cur,
                mu_cur,
                log_var=log_var_cur,
                graphs=graphs,
                center_only=center_only,
                target_index=target_index,
            )
        return y_cur, mu_cur, log_var_cur


def _target_dim_from_graphs(graphs: Sequence) -> int:
    y = graphs[0].y.detach().cpu().numpy() if torch.is_tensor(graphs[0].y) else np.asarray(graphs[0].y)
    if y.ndim == 1:
        return 1
    return int(y.shape[1])


def _concat_graph_keys(graphs: Sequence) -> list[Any]:
    keys: list[Any] = []
    for i, g in enumerate(graphs):
        key = getattr(g, "organoid_str", None)
        if key is None:
            key = getattr(g, "graph_path", None)
        if key is None:
            key = id(g)
        n = int(g.y.shape[0])
        keys.extend((key, j) for j in range(n))
    return keys


def _constant_feature_graphs(graphs: Sequence, *, constant_value: float) -> list:
    graphs_out = [copy.deepcopy(g) for g in graphs]
    for g in graphs_out:
        n_nodes = int(g.x.shape[0])
        g.x = torch.full(
            (n_nodes, 1),
            float(constant_value),
            dtype=g.x.dtype,
            device=g.x.device,
        )
    return graphs_out


@dataclass
class GlobalBaselineResidualTransform(TargetTransform):
    """Subtract a train-only global-feature baseline.

    ``fit(train_graphs)`` trains a dedicated ``GlobalFeatureMLP`` baseline using
    graph-level features only. The transform then subtracts baseline predictions
    from graph targets:

        residual = target - baseline_prediction

    During inversion the same baseline model is evaluated for the supplied
    graphs and added back to true/predicted residuals. The transform is affine
    with derivative 1, so it leaves log-variance unchanged.
    """

    hidden_dim: int = 64
    dropout: float = 0.1
    lr: float = 3e-4
    batch_size: int = 128
    max_epochs: int = 2000
    patience: int = 30
    num_workers: int = 1
    model_kwargs: dict[str, Any] = field(default_factory=dict)
    train_kwargs: dict[str, Any] = field(default_factory=dict)
    model: Any = None
    metrics: Any = None
    history: Any = None
    prediction_cache: dict[tuple[Any, int], np.ndarray] = field(default_factory=dict)
    name: str = field(default="global_baseline_residual", init=False)

    def fit_array(self, y: np.ndarray) -> "GlobalBaselineResidualTransform":
        raise RuntimeError(
            "GlobalBaselineResidualTransform is graph-aware and must be fitted with fit(train_graphs)."
        )

    def fit(self, train_graphs: Sequence) -> "GlobalBaselineResidualTransform":
        from src.data.metadata import infer_global_dim
        from src.models.baseline import GlobalFeatureMLP
        from src.training.loop import TrainConfig, train

        baseline_graphs = [copy.deepcopy(g) for g in train_graphs]

        target_dim = _target_dim_from_graphs(baseline_graphs)
        model_kwargs = {
            "global_dim": infer_global_dim(baseline_graphs),
            "hidden_dim": int(self.hidden_dim),
            "dropout": float(self.dropout),
            "target_dim": target_dim,
        }
        model_kwargs.update(self.model_kwargs)

        cfg_kwargs = {
            "lr": self.lr,
            "batch_size": self.batch_size,
            "max_epochs": self.max_epochs,
            "patience": self.patience,
            "num_workers": self.num_workers,
        }
        cfg_kwargs.update(self.train_kwargs)
        cfg = TrainConfig(**cfg_kwargs)

        self.model = GlobalFeatureMLP(**model_kwargs)
        self.model, self.metrics, self.history = train(
            self.model,
            baseline_graphs,
            baseline_graphs,
            cfg,
        )
        self.prediction_cache.clear()
        self.fitted = True
        return self

    def _predict_baseline(
        self,
        graphs: Sequence,
        *,
        device=None,
        batch_size: Optional[int] = None,
        center_only: bool = False,
    ) -> np.ndarray:
        if not self.fitted or self.model is None:
            raise RuntimeError("Baseline residual transform must be fitted before prediction.")

        keys = _concat_graph_keys(graphs)
        missing = [key for key in keys if key not in self.prediction_cache]
        if missing:
            from src.inference.predict import predict_targets

            baseline_graphs = [copy.deepcopy(g) for g in graphs]
            _, mu, _, _ = predict_targets(
                baseline_graphs,
                self.model,
                device=device,
                batch_size=batch_size or self.batch_size,
                return_log_var=True,
                target_transform=None,
            )
            mu_arr = np.asarray(mu, dtype=np.float64)
            if mu_arr.ndim == 1:
                mu_arr = mu_arr[:, None]
            pred_keys = _concat_graph_keys(graphs)
            if len(pred_keys) != len(mu_arr):
                raise RuntimeError(
                    f"Baseline prediction length mismatch: {len(mu_arr)} predictions for {len(pred_keys)} nodes."
                )
            for key, pred in zip(pred_keys, mu_arr):
                self.prediction_cache[key] = np.asarray(pred, dtype=np.float64).copy()

        pred = np.stack([self.prediction_cache[key] for key in keys], axis=0)
        if center_only:
            center_positions = []
            offset = 0
            for g in graphs:
                if not hasattr(g, "center_idx"):
                    raise ValueError(
                        "center_only=True requires graphs with a .center_idx attribute."
                    )
                c = int(g.center_idx.item()) if torch.is_tensor(g.center_idx) else int(g.center_idx)
                center_positions.append(offset + c)
                offset += int(g.y.shape[0])
            pred = pred[np.asarray(center_positions, dtype=np.int64)]
        return pred[:, 0] if pred.shape[1] == 1 else pred

    def transform_graphs(self, graphs: Sequence, in_place: bool = True) -> Sequence:
        if not self.fitted:
            raise RuntimeError("Transform must be fitted before transform_graphs().")

        out = graphs if in_place else [copy.deepcopy(g) for g in graphs]
        baseline = self._predict_baseline(out)
        baseline = baseline[:, None] if np.asarray(baseline).ndim == 1 else np.asarray(baseline)

        offset = 0
        for g in out:
            n = int(g.y.shape[0])
            y = g.y.detach().cpu().numpy() if torch.is_tensor(g.y) else np.asarray(g.y)
            was_1d = y.ndim == 1
            y2 = y[:, None] if was_1d else y
            resid = y2 - baseline[offset:offset + n]
            offset += n
            resid_t = torch.as_tensor(resid, dtype=g.y.dtype, device=g.y.device)
            g.y = resid_t[:, 0] if was_1d and resid_t.ndim == 2 and resid_t.shape[1] == 1 else resid_t
        return out

    def forward(self, y: ArrayLike) -> ArrayLike:
        raise RuntimeError(
            "GlobalBaselineResidualTransform.forward requires graph context; use transform_graphs()."
        )

    def inverse(self, z: ArrayLike) -> ArrayLike:
        raise RuntimeError(
            "GlobalBaselineResidualTransform.inverse requires graph context; use inverse_distribution(..., graphs=...)."
        )

    def inverse_derivative(self, z: ArrayLike) -> ArrayLike:
        if torch.is_tensor(z):
            return torch.ones_like(z)
        return np.ones_like(np.asarray(z, dtype=np.float64))

    def inverse_distribution(
        self,
        y: Optional[ArrayLike],
        mu: ArrayLike,
        log_var: Optional[ArrayLike] = None,
        graphs: Optional[Sequence] = None,
        center_only: bool = False,
        target_index: Optional[int] = None,
    ) -> Tuple[Optional[ArrayLike], ArrayLike, Optional[ArrayLike]]:
        if graphs is None:
            raise RuntimeError(
                "GlobalBaselineResidualTransform.inverse_distribution requires graphs for baseline reconstruction."
            )
        baseline = self._predict_baseline(graphs, center_only=center_only)

        def _add_baseline(values):
            if values is None:
                return None
            base_arr = baseline
            value_shape = tuple(values.shape) if torch.is_tensor(values) else np.asarray(values).shape
            if (
                target_index is not None
                and np.asarray(base_arr).ndim == 2
                and np.asarray(base_arr).shape[1] > 1
                and (len(value_shape) == 1 or (len(value_shape) == 2 and value_shape[1] == 1))
            ):
                base_arr = np.asarray(base_arr)[:, int(target_index)]
            if torch.is_tensor(values):
                if values.ndim == 2 and np.asarray(base_arr).ndim == 1:
                    base_arr = np.asarray(base_arr)[:, None]
                base = _as_torch_like(base_arr, values)
                return values + base
            arr = np.asarray(values, dtype=np.float64)
            if arr.ndim == 2 and np.asarray(base_arr).ndim == 1:
                base_arr = np.asarray(base_arr)[:, None]
            return arr + base_arr

        return _add_baseline(y), _add_baseline(mu), log_var


# Backward-compatible alias for notebooks/scripts created before the rename.
ConstantGlobalBaselineResidualTransform = GlobalBaselineResidualTransform


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

    def _select_target_transform(self, target_index: int) -> "StandardizeTransform":
        if self.center is None or self.scale is None:
            raise RuntimeError("Transform has not been fitted.")
        out = copy.copy(self)
        out.center = np.asarray(self.center, dtype=np.float64)[int(target_index):int(target_index) + 1].copy()
        out.scale = np.asarray(self.scale, dtype=np.float64)[int(target_index):int(target_index) + 1].copy()
        return out

    def inverse_distribution(
        self,
        y: Optional[ArrayLike],
        mu: ArrayLike,
        log_var: Optional[ArrayLike] = None,
        graphs: Optional[Sequence] = None,
        center_only: bool = False,
        target_index: Optional[int] = None,
    ) -> Tuple[Optional[ArrayLike], ArrayLike, Optional[ArrayLike]]:
        if (
            target_index is not None
            and self.center is not None
            and len(np.asarray(self.center)) > 1
            and _single_selected_target_values(y, mu, log_var)
        ):
            return TargetTransform.inverse_distribution(
                self._select_target_transform(int(target_index)),
                y,
                mu,
                log_var=log_var,
                graphs=graphs,
                center_only=center_only,
                target_index=None,
            )
        return super().inverse_distribution(
            y,
            mu,
            log_var=log_var,
            graphs=graphs,
            center_only=center_only,
            target_index=target_index,
        )


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

    def _select_target_transform(self, target_index: int) -> "AsinhTransform":
        if self.center is None or self.scale is None:
            raise RuntimeError("Transform has not been fitted.")
        out = copy.copy(self)
        out.center = np.asarray(self.center, dtype=np.float64)[int(target_index):int(target_index) + 1].copy()
        out.scale = np.asarray(self.scale, dtype=np.float64)[int(target_index):int(target_index) + 1].copy()
        return out

    def inverse_distribution(
        self,
        y: Optional[ArrayLike],
        mu: ArrayLike,
        log_var: Optional[ArrayLike] = None,
        graphs: Optional[Sequence] = None,
        center_only: bool = False,
        target_index: Optional[int] = None,
    ) -> Tuple[Optional[ArrayLike], ArrayLike, Optional[ArrayLike]]:
        if (
            target_index is not None
            and self.center is not None
            and len(np.asarray(self.center)) > 1
            and _single_selected_target_values(y, mu, log_var)
        ):
            return TargetTransform.inverse_distribution(
                self._select_target_transform(int(target_index)),
                y,
                mu,
                log_var=log_var,
                graphs=graphs,
                center_only=center_only,
                target_index=None,
            )
        return super().inverse_distribution(
            y,
            mu,
            log_var=log_var,
            graphs=graphs,
            center_only=center_only,
            target_index=target_index,
        )


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

    def _select_target_transform(self, target_index: int) -> "AsinhStandardizeTransform":
        if any(v is None for v in (self.raw_center, self.raw_scale, self.center, self.scale)):
            raise RuntimeError("Transform has not been fitted.")
        idx = int(target_index)
        out = copy.copy(self)
        out.raw_center = np.asarray(self.raw_center, dtype=np.float64)[idx:idx + 1].copy()
        out.raw_scale = np.asarray(self.raw_scale, dtype=np.float64)[idx:idx + 1].copy()
        out.center = np.asarray(self.center, dtype=np.float64)[idx:idx + 1].copy()
        out.scale = np.asarray(self.scale, dtype=np.float64)[idx:idx + 1].copy()
        return out

    def inverse_distribution(
        self,
        y: Optional[ArrayLike],
        mu: ArrayLike,
        log_var: Optional[ArrayLike] = None,
        graphs: Optional[Sequence] = None,
        center_only: bool = False,
        target_index: Optional[int] = None,
    ) -> Tuple[Optional[ArrayLike], ArrayLike, Optional[ArrayLike]]:
        if (
            target_index is not None
            and self.center is not None
            and len(np.asarray(self.center)) > 1
            and _single_selected_target_values(y, mu, log_var)
        ):
            return TargetTransform.inverse_distribution(
                self._select_target_transform(int(target_index)),
                y,
                mu,
                log_var=log_var,
                graphs=graphs,
                center_only=center_only,
                target_index=None,
            )
        return super().inverse_distribution(
            y,
            mu,
            log_var=log_var,
            graphs=graphs,
            center_only=center_only,
            target_index=target_index,
        )


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
    graphs: Optional[Sequence] = None,
    center_only: bool = False,
    target_index: Optional[int] = None,
):
    """
    Convenience function for prediction code.

    If transform is None, returns inputs unchanged. Otherwise calls
    transform.inverse_distribution(y, mu, log_var).
    """
    if transform is None:
        return y, mu, log_var
    return transform.inverse_distribution(
        y,
        mu,
        log_var,
        graphs=graphs,
        center_only=center_only,
        target_index=target_index,
    )




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
