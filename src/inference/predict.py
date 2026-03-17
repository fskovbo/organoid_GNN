import numpy as np
import torch
from torch_geometric.loader import DataLoader


@torch.no_grad()
def predict_targets(graphs, model, device=None, batch_size=32, num_workers=0, pin_memory=True, return_log_var=False):
    """
    Run model over graphs and return concatenated arrays:
      y_true : (K,) float64
      y_pred : (K,) float64  (mean prediction μ)
      X      : (K, M) float32 (binary markers)

    If return_log_var=True, additionally returns:
      log_var : (K,) float64  (log variance, i.e. log(σ^2))
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device).eval()

    loader = DataLoader(graphs, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=pin_memory)

    Ys, Ymu, Ylv, Xs = [], [], [], []
    for batch in loader:
        batch = batch.to(device, non_blocking=True)

        out, _ = model(batch.x, batch.edge_index)
        # Support both deterministic models (yhat) and Gaussian models ((mu, log_var))
        if isinstance(out, (tuple, list)) and len(out) == 2:
            mu, log_var = out
        else:
            mu, log_var = out, None

        Ys.append(batch.y.detach().cpu().numpy())
        Ymu.append(mu.detach().cpu().numpy())
        if return_log_var:
            if log_var is None:
                raise ValueError("Model did not return (mu, log_var) but return_log_var=True was requested.")
            Ylv.append(log_var.detach().cpu().numpy())
        Xs.append(batch.x.detach().cpu().numpy())

    y_true = np.concatenate(Ys, axis=0).astype(np.float64)
    y_pred = np.concatenate(Ymu, axis=0).astype(np.float64)
    X = np.concatenate(Xs, axis=0).astype(np.float32)

    if return_log_var:
        log_var = np.concatenate(Ylv, axis=0).astype(np.float64)
        return y_true, y_pred, log_var, X
    return y_true, y_pred, X