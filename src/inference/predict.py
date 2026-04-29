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
    rescale=False,
    center=None,
    scale=None,
):
    """
    Run the model on a list of graphs and return concatenated predictions.

    Parameters
    ----------
    graphs : list
        List of PyG graphs.
    model : torch.nn.Module
        Trained model.
    device : str or None
        Device for inference. If None, choose CUDA when available.
    batch_size : int
        Batch size for DataLoader.
    num_workers : int
        Number of DataLoader workers.
    pin_memory : bool
        Whether to use pinned memory in the DataLoader.
    return_log_var : bool
        If True, also return predicted log-variance.
    center_only : bool
        If True, only return predictions for subgraph center nodes.
        Requires each graph to have a .center_idx attribute.
    rescale : bool
        If True, transform targets and predictions back to original target scale
        using y = y * scale + center, and log_var = log_var + 2*log(scale).
    center : float or array-like or None
        Target center used during standardization.
    scale : float or array-like or None
        Target scale used during standardization.

    Returns
    -------
    if return_log_var:
        y_true, y_pred, log_var, X
    else:
        y_true, y_pred, X
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if rescale and (center is None or scale is None):
        raise ValueError("rescale=True requires both center and scale.")

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

    if rescale:
        if center is None or scale is None:
            raise ValueError(
                "rescale=True requires both 'center' and 'scale' to be provided."
            )

        y_true, y_pred, log_var = rescale_distribution_outputs(
            y_true,
            y_pred,
            log_var=log_var,
            center=center,
            scale=scale,
        )

    if return_log_var:
        return y_true, y_pred, log_var, X

    return y_true, y_pred, X



def rescale_distribution_outputs(y, mu, log_var=None, center=0.0, scale=1.0):
    """Undo target standardization for predictions and optional log-variance."""

    y = np.asarray(y, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64)

    scale = np.asarray(scale, dtype=np.float64)
    center = np.asarray(center, dtype=np.float64)

    y = y * scale + center
    mu = mu * scale + center

    if log_var is not None:
        log_var = np.asarray(log_var, dtype=np.float64)
        log_var = log_var + 2.0 * np.log(scale)

    return y, mu, log_var


