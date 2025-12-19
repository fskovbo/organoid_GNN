import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader

# -----------------------------------------------------------------------------
# Config & loss
# -----------------------------------------------------------------------------

@dataclass
class TrainConfig:
    """
    Hyperparameters and runtime knobs for the simple training loop.
    """
    batch_size: int = 4
    lr: float = 2e-3
    weight_decay: float = 1e-4
    max_epochs: int = 80
    patience: int = 12
    huber_delta: float = 1.0
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    pin_memory: bool = True
    num_workers: int = 0  # increase if you want background loading

    lambda_calib: float = 0.0        # set >0 to enable, e.g. 0.05–0.5
    calib_mode: str = "corr"         # "corr" (squared Pearson r) or "cov"


def huber_loss(pred, target, delta=1.0):
    """
    Standard Huber loss (smooth L1). Set delta=0 for MSE-ish behavior.
    """
    if delta == 0:
        return 0.5 * (pred - target) ** 2
    err = pred - target
    abs_err = torch.abs(err)
    quad = torch.minimum(abs_err, torch.tensor(delta, device=pred.device))
    return 0.5 * quad**2 + delta * (abs_err - quad)


def calibration_penalty(res: torch.Tensor, y: torch.Tensor, mode: str = "corr", eps: float = 1e-12) -> torch.Tensor:
    """
    Push residuals to be *uncorrelated* with y (i.e., flatten bias vs curvature).
    mode='corr'  -> squared Pearson r between res and y (scale-invariant).
    mode='cov'   -> squared covariance ( (E[(res-mean)(y-mean)])^2 ).
    """
    r = res - res.mean()
    yt = y - y.mean()
    if mode == "corr":
        num = (r * yt).sum()
        den = torch.sqrt((r.square().sum() * yt.square().sum()).clamp_min(eps))
        return (num / den).square()
    elif mode == "cov":
        return ( (r * yt).mean() ).square()
    else:
        raise ValueError("calib_mode must be 'corr' or 'cov'")

@torch.no_grad()
def residual_slope(res: torch.Tensor, y: torch.Tensor, eps: float = 1e-12) -> float:
    """
    Slope of least-squares line residual ~ a + b*y. b≈0 means flat bias curve.
    """
    yt = y - y.mean()
    denom = yt.square().sum().clamp_min(eps)
    b = (res * yt).sum() / denom
    return float(b.detach().cpu())

# -----------------------------------------------------------------------------
# Minimal training utilities
# -----------------------------------------------------------------------------

def make_loaders(train_graphs, val_graphs, cfg):
    """
    Build DataLoaders from (lists of) PyG Data graphs.
    """
    train_loader = DataLoader(train_graphs, batch_size=cfg.batch_size,
                              shuffle=True, pin_memory=cfg.pin_memory,
                              num_workers=cfg.num_workers)
    val_loader   = DataLoader(val_graphs,   batch_size=cfg.batch_size,
                              shuffle=False, pin_memory=cfg.pin_memory,
                              num_workers=cfg.num_workers)
    return train_loader, val_loader


def epoch_pass(model, loader, cfg, optimizer=None):
    """
    One full pass over the loader. If 'optimizer' is provided → training step,
    else evaluation only. Returns (mean_loss, mean_MAE).
    """
    is_train = optimizer is not None
    model.train(is_train)

    total_loss, total_mae, total_n = 0.0, 0.0, 0
    for batch in loader:
        batch = batch.to(cfg.device, non_blocking=True)
        yhat, _ = model(batch.x, batch.edge_index)
        res = yhat - batch.y

        base_loss = huber_loss(yhat, batch.y, cfg.huber_delta).mean()
        calib = calibration_penalty(res, batch.y, mode=cfg.calib_mode)

        loss = base_loss + cfg.lambda_calib * calib

        if is_train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()

        with torch.no_grad():
            mae = torch.mean(torch.abs(yhat - batch.y))

        n = batch.y.numel()
        total_loss += loss.item() * n
        total_mae  += mae.item()  * n
        total_n    += n

    return total_loss / max(total_n, 1), total_mae / max(total_n, 1)


def train(model, train_graphs, val_graphs, cfg=TrainConfig()):
    """
    Train `model` on `train_graphs`, validate on `val_graphs`.
    - Does NOT create/split datasets.
    - Does NOT instantiate the model (you pass it in).
    - Returns (model, metrics, history).
    """
    model = model.to(cfg.device)
    train_loader, val_loader = make_loaders(train_graphs, val_graphs, cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    best_val, best_state, patience_left = math.inf, None, cfg.patience
    hist = {"train_loss": [], "train_mae": [], "val_loss": [], "val_mae": []}

    for epoch in range(1, cfg.max_epochs + 1):
        tr_loss, tr_mae = epoch_pass(model, train_loader, cfg, optimizer=opt)
        vl_loss, vl_mae = epoch_pass(model, val_loader,   cfg, optimizer=None)

        print(f"epoch {epoch:03d} | train loss {tr_loss:.4f} mae {tr_mae:.4f} | "
              f"val loss {vl_loss:.4f} mae {vl_mae:.4f}")

        hist["train_loss"].append(tr_loss); hist["train_mae"].append(tr_mae)
        hist["val_loss"].append(vl_loss);   hist["val_mae"].append(vl_mae)

        if vl_mae < best_val - 1e-7:
            best_val = vl_mae
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = cfg.patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, {"val_mae": best_val}, hist