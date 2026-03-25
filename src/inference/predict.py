import numpy as np
import torch
from torch_geometric.loader import DataLoader


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
):
    """
    Run model over graphs and return concatenated arrays.

    Parameters
    ----------
    graphs : list[Data]
    model : torch.nn.Module
    center_only : bool
        If False, return predictions for all nodes.
        If True, return only the center node of each graph/subgraph.
        Requires each graph to have a `.center_idx` attribute.

    Returns
    -------
    If return_log_var=False:
      y_true : (K,) float64
      y_pred : (K,) float64
      X      : (K, M) float32

    If return_log_var=True:
      y_true : (K,) float64
      y_pred : (K,) float64
      log_var: (K,) float64
      X      : (K, M) float32
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

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

        out, _ = model(batch.x, batch.edge_index)
        if isinstance(out, (tuple, list)) and len(out) == 2:
            mu, log_var = out
        else:
            mu, log_var = out, None

        if center_only:
            if not hasattr(batch, "center_idx"):
                raise ValueError("center_only=True requires graphs with a .center_idx attribute")

            # In a PyG batch, center index of graph i is batch.ptr[i] + batch.center_idx[i]
            centers = batch.ptr[:-1] + batch.center_idx.to(batch.ptr.device)

            y_sel = batch.y[centers]
            mu_sel = mu[centers]
            x_sel = batch.x[centers]

            Ys.append(y_sel.detach().cpu().numpy())
            Ymu.append(mu_sel.detach().cpu().numpy())
            Xs.append(x_sel.detach().cpu().numpy())

            if return_log_var:
                if log_var is None:
                    raise ValueError("Model did not return (mu, log_var) but return_log_var=True was requested.")
                Ylv.append(log_var[centers].detach().cpu().numpy())

        else:
            Ys.append(batch.y.detach().cpu().numpy())
            Ymu.append(mu.detach().cpu().numpy())
            Xs.append(batch.x.detach().cpu().numpy())

            if return_log_var:
                if log_var is None:
                    raise ValueError("Model did not return (mu, log_var) but return_log_var=True was requested.")
                Ylv.append(log_var.detach().cpu().numpy())

    y_true = np.concatenate(Ys, axis=0).astype(np.float64)
    y_pred = np.concatenate(Ymu, axis=0).astype(np.float64)
    X = np.concatenate(Xs, axis=0).astype(np.float32)

    if return_log_var:
        log_var = np.concatenate(Ylv, axis=0).astype(np.float64)
        return y_true, y_pred, log_var, X

    return y_true, y_pred, X