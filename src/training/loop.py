import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader

from src.training.losses import (
    calibration_penalty,
    gaussian_nll,
    student_t_nll,
    huber_plus_variance_calibration,
)


@dataclass
class TrainConfig:
    """
    Hyperparameters and runtime knobs for the training loop.
    """
    batch_size: int = 4
    lr: float = 2e-3
    weight_decay: float = 1e-4
    max_epochs: int = 80
    patience: int = 12

    # Mean-loss / uncertainty-loss settings
    loss_name: str = "gaussian"   # "gaussian" | "student_t" | "huber_var"
    huber_delta: float = 1.0
    student_t_df: float = 4.0
    var_weight: float = 0.25
    detach_mean_for_var: bool = True

    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    pin_memory: bool = True
    num_workers: int = 0

    # Optional residual calibration penalty
    lambda_calib: float = 0.0
    calib_mode: str = "corr"      # "corr" or "cov"


def make_loaders(train_graphs, val_graphs, cfg):
    """
    Build DataLoaders from (lists of) PyG Data graphs.
    """
    train_loader = DataLoader(
        train_graphs,
        batch_size=cfg.batch_size,
        shuffle=True,
        pin_memory=cfg.pin_memory,
        num_workers=cfg.num_workers,
    )
    val_loader = DataLoader(
        val_graphs,
        batch_size=cfg.batch_size,
        shuffle=False,
        pin_memory=cfg.pin_memory,
        num_workers=cfg.num_workers,
    )
    return train_loader, val_loader


def compute_base_loss(mu, log_scale2, y, cfg):
    """
    Per-node loss, reduced to batch mean.
    """
    if cfg.loss_name == "gaussian":
        per_node = gaussian_nll(mu, log_scale2, y)

    elif cfg.loss_name == "student_t":
        per_node = student_t_nll(mu, log_scale2, y, df=cfg.student_t_df)

    elif cfg.loss_name == "huber_var":
        per_node = huber_plus_variance_calibration(
            mu,
            log_scale2,
            y,
            delta=cfg.huber_delta,
            var_weight=cfg.var_weight,
            detach_mean_for_var=cfg.detach_mean_for_var,
        )

    else:
        raise ValueError(
            f"Unknown loss_name='{cfg.loss_name}'. "
            "Expected one of: 'gaussian', 'student_t', 'huber_var'."
        )

    return per_node.mean()


def epoch_pass(model, loader, cfg, optimizer=None):
    is_train = optimizer is not None
    model.train(is_train)

    total_loss, total_mae, total_n = 0.0, 0.0, 0

    for batch in loader:
        batch = batch.to(cfg.device, non_blocking=True)

        (mu, log_scale2), _ = model(batch.x, batch.edge_index)

        base_loss = compute_base_loss(mu, log_scale2, batch.y, cfg)

        # Optional calibration penalty on mean residuals
        res = mu - batch.y
        calib = calibration_penalty(res, batch.y, mode=cfg.calib_mode)
        loss = base_loss + cfg.lambda_calib * calib

        if is_train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()

        with torch.no_grad():
            mae = torch.mean(torch.abs(mu - batch.y))

        n = batch.y.numel()
        total_loss += loss.item() * n
        total_mae += mae.item() * n
        total_n += n

    return total_loss / max(total_n, 1), total_mae / max(total_n, 1)


def train(model, train_graphs, val_graphs, cfg=TrainConfig()):
    model = model.to(cfg.device)
    train_loader, val_loader = make_loaders(train_graphs, val_graphs, cfg)
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    best_val, best_state, patience_left = math.inf, None, cfg.patience
    hist = {"train_loss": [], "train_mae": [], "val_loss": [], "val_mae": []}

    for epoch in range(1, cfg.max_epochs + 1):
        tr_loss, tr_mae = epoch_pass(model, train_loader, cfg, optimizer=opt)
        vl_loss, vl_mae = epoch_pass(model, val_loader, cfg, optimizer=None)

        print(
            f"epoch {epoch:03d} | "
            f"train loss {tr_loss:.4f} mae {tr_mae:.4f} | "
            f"val loss {vl_loss:.4f} mae {vl_mae:.4f}"
        )

        hist["train_loss"].append(tr_loss)
        hist["train_mae"].append(tr_mae)
        hist["val_loss"].append(vl_loss)
        hist["val_mae"].append(vl_mae)

        if vl_mae < best_val - 1e-7:
            best_val = vl_mae
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }
            patience_left = cfg.patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, {"val_mae": best_val}, hist