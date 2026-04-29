import numpy as np
import torch
from torch_geometric.loader import DataLoader


def _as_numpy_diagonal_log_var(scale):
    """Return diagonal log-variance from either log_var or Cholesky outputs."""
    arr = scale.detach().cpu().numpy() if torch.is_tensor(scale) else np.asarray(scale)
    if arr.ndim == 3:
        cov = arr @ np.swapaxes(arr, -1, -2)
        return np.log(np.maximum(np.diagonal(cov, axis1=-2, axis2=-1), 1e-12))
    return arr


def distribution_to_diagonal_outputs(mu, scale):
    """Convert diagonal or full-covariance outputs to old-compatible (mu, log_var)."""
    log_var = _as_numpy_diagonal_log_var(scale)
    mu = mu.detach().cpu().numpy() if torch.is_tensor(mu) else np.asarray(mu)
    if mu.ndim == 2 and mu.shape[1] == 1:
        mu = mu.reshape(-1)
    if log_var.ndim == 2 and log_var.shape[1] == 1:
        log_var = log_var.reshape(-1)
    return mu, log_var


@torch.no_grad()
def predict_targets(
    graphs,
    model,
    device=None,
    batch_size=32,
    num_workers=0,
    pin_memory=True,
    return_log_var=False,
    center_only=False,
    target_transform=None,
):
    """
    Run the model on a list of graphs and return concatenated predictions.

    If target_transform is provided, y_true, y_pred, and optional log_var are
    inverse-transformed back to the original target units using
    target_transform.inverse_distribution(...).
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if target_transform is not None and not hasattr(target_transform, "inverse_distribution"):
        raise TypeError(
            "target_transform must provide inverse_distribution(y, mu, log_var=None)."
        )

    model = model.to(device).eval()

    loader = DataLoader(
        graphs,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    Ys, Ymu, Ylv, Xs = [], [], [], []

    for batch in loader:
        batch = batch.to(device, non_blocking=True)

        try:
            out, _ = model(batch.x, batch.edge_index, data=batch)
        except TypeError:
            out, _ = model(batch.x, batch.edge_index)

        if isinstance(out, (tuple, list)) and len(out) == 2:
            mu, log_var = out
        else:
            mu, log_var = out, None

        if center_only:
            if not hasattr(batch, "center_idx"):
                raise ValueError(
                    "center_only=True requires graphs with a .center_idx attribute"
                )

            centers = batch.ptr[:-1] + batch.center_idx.to(batch.ptr.device)

            y_sel = batch.y[centers]
            mu_sel = mu[centers]
            x_sel = batch.x[centers]

            Ys.append(y_sel.detach().cpu().numpy())
            Ymu.append(mu_sel.detach().cpu().numpy())
            Xs.append(x_sel.detach().cpu().numpy())

            if return_log_var:
                if log_var is None:
                    raise ValueError(
                        "Model did not return (mu, log_var) but return_log_var=True was requested."
                    )
                Ylv.append(_as_numpy_diagonal_log_var(log_var[centers]))

        else:
            Ys.append(batch.y.detach().cpu().numpy())
            Ymu.append(mu.detach().cpu().numpy())
            Xs.append(batch.x.detach().cpu().numpy())

            if return_log_var:
                if log_var is None:
                    raise ValueError(
                        "Model did not return (mu, log_var) but return_log_var=True was requested."
                    )
                Ylv.append(_as_numpy_diagonal_log_var(log_var))

    y_true = np.concatenate(Ys, axis=0).astype(np.float64)
    y_pred = np.concatenate(Ymu, axis=0).astype(np.float64)
    X = np.concatenate(Xs, axis=0).astype(np.float32)

    log_var = None
    if return_log_var:
        log_var = np.concatenate(Ylv, axis=0).astype(np.float64)

    if target_transform is not None:
        y_true, y_pred, log_var = target_transform.inverse_distribution(
            y_true, y_pred, log_var=log_var
        )

    if return_log_var:
        return y_true, y_pred, log_var, X

    return y_true, y_pred, X