import torch
import numpy as np
from copy import deepcopy


def _as_1d_torch(t: torch.Tensor, dtype=torch.float32) -> torch.Tensor:
    """
    Return a contiguous 1-D tensor (N,) of the requested dtype.
    Accepts (N,), (N,1), or anything squeezable to 1-D.
    """
    if not torch.is_tensor(t):
        t = torch.as_tensor(t)
    t = t.to(dtype=dtype)
    t = t.squeeze()              # drop all size-1 dims
    if t.dim() != 1:
        raise ValueError(f"Expected 1-D tensor after squeeze; got shape {tuple(t.shape)}")
    return t.contiguous().view(-1)

def _as_2d_torch(x: torch.Tensor, dtype=torch.float32) -> torch.Tensor:
    """
    Ensure x is 2-D (N,M) float tensor (for features).
    """
    if not torch.is_tensor(x):
        x = torch.as_tensor(x)
    x = x.to(dtype=dtype)
    if x.dim() != 2:
        raise ValueError(f"Expected 2-D features (N,M); got shape {tuple(x.shape)}")
    return x.contiguous()

def compute_marker_baseline_means(train_graphs):
    """
    Compute μ_m = mean(y | marker m positive) using TRAIN graphs only,
    and μ_none for cells with no positive markers.

    Returns:
      dict with:
        'mu_pos': (M,) float32,
        'mu_none': float,
        'counts_pos': (M,) ints,
        'count_none': int
    """
    # Concatenate across train graphs
    ys = []
    Xs = []
    for g in train_graphs:
        ys.append(g.y.detach().cpu().numpy())
        Xs.append(g.x.detach().cpu().numpy())
    y_all = np.concatenate(ys, axis=0)
    X_all = np.concatenate(Xs, axis=0)  # (N_total, M) binary

    M = X_all.shape[1]
    mu_pos = np.zeros(M, dtype=np.float32)
    counts_pos = np.zeros(M, dtype=np.int64)

    for m in range(M):
        mask = X_all[:, m] > 0.5
        if np.any(mask):
            mu_pos[m] = float(np.mean(y_all[mask]))
            counts_pos[m] = int(mask.sum())
        else:
            mu_pos[m] = float(np.nan)  # no positives in train for this marker

    none_mask = (Xs[0][0:0].shape[0] == 0)  # dummy to satisfy linter
    # cells with NO positive markers in that row
    row_pos = (X_all > 0.5).sum(axis=1)
    mask_none = row_pos == 0
    mu_none = float(np.mean(y_all[mask_none])) if np.any(mask_none) else float(np.mean(y_all))

    # replace NaNs in mu_pos with global mean to be safe
    glob = float(np.mean(y_all))
    mu_pos = np.where(np.isfinite(mu_pos), mu_pos, glob).astype(np.float32)

    return {
        "mu_pos": mu_pos,
        "mu_none": mu_none,
        "counts_pos": counts_pos,
        "count_none": int(mask_none.sum()) if y_all.size else 0,
    }


def per_node_baseline_from_markers(X_bin: torch.Tensor,
                                   mu_pos: np.ndarray,
                                   mu_none: float) -> torch.Tensor:
    """
    Baseline b_i = average( μ_m ) over positive markers m in row i,
    else μ_none if a row has no positives. Returns (N,) float tensor.
    """
    X = _as_2d_torch(X_bin, dtype=torch.float32)        # (N,M)
    N, M = X.shape
    mu_pos_t = torch.as_tensor(mu_pos, dtype=X.dtype, device=X.device).view(-1)  # (M,)
    if mu_pos_t.numel() != M:
        raise ValueError(f"mu_pos length {mu_pos_t.numel()} != num markers {M}")

    pos_counts = X.sum(dim=1)                           # (N,)
    weighted   = X @ mu_pos_t                           # (N,)
    # where count>0 -> weighted/pos_counts, else mu_none
    b = torch.where(
        pos_counts > 0.0,
        weighted / pos_counts.clamp_min(1.0),
        torch.as_tensor(mu_none, dtype=X.dtype, device=X.device)
    )
    return _as_1d_torch(b, dtype=torch.float32)         # (N,)


def residualize_graphs(graphs, mu_dict):
    """
    Deep-copy graphs and replace .y with residual r = y - b (all 1-D).
    Also store:
      - y_orig : (N,) original target
      - y_base : (N,) baseline per node
    """
    mu_pos = mu_dict["mu_pos"]; mu_none = mu_dict["mu_none"]
    out = []
    for g in graphs:
        gg = type(g)()
        # copy core
        gg.x = _as_2d_torch(g.x)
        gg.y = _as_1d_torch(g.y)                        # ensure (N,)
        gg.edge_index = g.edge_index.to(dtype=torch.long).contiguous()
        if hasattr(g, "organoid_str"): gg.organoid_str = g.organoid_str
        if hasattr(g, "organoid_id"):  gg.organoid_id  = g.organoid_id

        # compute baseline and residual
        b = per_node_baseline_from_markers(gg.x, mu_pos, mu_none)  # (N,)
        gg.y_orig = gg.y.clone()                                   # (N,)
        gg.y_base = b.clone()                                      # (N,)
        gg.y      = (gg.y - b).contiguous().view(-1)               # (N,)
        out.append(gg)
    return out

# def residualize_graphs(graphs, mu_dict):
#     """
#     Return deep-copied graphs where y is replaced by residual r = y - b.
#     Original y is kept in .y_orig; baseline in .y_base.
#     Ensures all tensors are on CPU (required when using DataLoader(pin_memory=True)).
#     """
#     mu_pos = mu_dict["mu_pos"]; mu_none = mu_dict["mu_none"]
#     out = []
#     for g in graphs:
#         gg = deepcopy(g)                     # start from CPU graphs
#         # make sure x/edge_index are on CPU
#         gg = gg.to('cpu')

#         # compute per-node baseline on CPU
#         b = per_node_baseline_from_markers(gg.x, mu_pos, mu_none)  # returns same-device as gg.x (CPU)
#         gg.y_orig = gg.y.clone()
#         gg.y_base = b.detach().clone()
#         gg.y      = (gg.y - b).clone()

#         # enforce CPU for all tensor attrs (paranoia)
#         gg = gg.to('cpu')
#         out.append(gg)
#     return out


@torch.no_grad()
def predict_with_baseline(model, graphs, mu_dict, device=None, batch_size=8, num_workers=0, pin_memory=True):
    """
    Run model on residualized targets and add baseline back:
      ŷ = b + r̂
    Returns concatenated y_true, y_pred arrays.
    """
    from torch_geometric.loader import DataLoader
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device).eval()

    Ys, Yh = [], []
    loader = DataLoader(graphs, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=pin_memory)
    mu_pos = mu_dict["mu_pos"]; mu_none = mu_dict["mu_none"]

    for batch in loader:
        batch = batch.to(device, non_blocking=True)
        # baseline per node on-the-fly (using current batch.x)
        b = per_node_baseline_from_markers(batch.x, mu_pos, mu_none)
        r_hat, _ = model(batch.x, batch.edge_index)     # model predicts residual
        y_hat = b + r_hat
        # y_true = if graphs were residualized, use original if present:
        y_true = batch.y if not hasattr(batch, "y_orig") else batch.y_orig
        Ys.append(y_true.detach().cpu()); Yh.append(y_hat.detach().cpu())

    return torch.cat(Ys, 0).numpy(), torch.cat(Yh, 0).numpy()
