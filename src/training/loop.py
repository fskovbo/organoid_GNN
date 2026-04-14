import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader

from src.training.losses import CompositeLoss, make_base_loss


@dataclass
class TrainConfig:
    """Store training hyperparameters and the list of auxiliary loss terms."""

    batch_size: int = 4
    lr: float = 2e-3
    weight_decay: float = 1e-4
    max_epochs: int = 80
    patience: int = 12

    loss_name: str = "gaussian"

    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    pin_memory: bool = True
    num_workers: int = 0

    aux_losses: list = field(default_factory=list)


def make_loaders(train_graphs, val_graphs, cfg):
    """Build PyG data loaders for the training and validation graph lists."""
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


def forward_model(model, batch):
    """Run the model on one batch, supporting models with or without a data= kwarg."""
    try:
        return model(batch.x, batch.edge_index, data=batch)
    except TypeError:
        return model(batch.x, batch.edge_index)


def epoch_pass(model, loader, cfg, loss_fn, optimizer=None):
    """Run one full training or validation pass and return mean loss and MAE."""
    is_train = optimizer is not None
    model.train(is_train)

    total_loss, total_mae, total_n = 0.0, 0.0, 0

    for batch in loader:
        batch = batch.to(cfg.device, non_blocking=True)

        (mu, log_scale2), _ = forward_model(model, batch)
        loss = loss_fn(mu=mu, log_scale2=log_scale2, batch=batch)

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


def train(model, train_graphs, val_graphs, cfg=TrainConfig(), loss_fn=None):
    """Train the model with early stopping and return the best checkpoint by val MAE."""
    model = model.to(cfg.device)
    train_loader, val_loader = make_loaders(train_graphs, val_graphs, cfg)

    if loss_fn is None:
        loss_fn = CompositeLoss(
            base_loss_fn=make_base_loss(cfg),
            aux_terms=cfg.aux_losses,
        )

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    best_val, best_state, patience_left = math.inf, None, cfg.patience
    hist = {"train_loss": [], "train_mae": [], "val_loss": [], "val_mae": []}

    for epoch in range(1, cfg.max_epochs + 1):
        tr_loss, tr_mae = epoch_pass(model, train_loader, cfg, loss_fn, optimizer=opt)
        vl_loss, vl_mae = epoch_pass(model, val_loader, cfg, loss_fn, optimizer=None)

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
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = cfg.patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, {"val_mae": best_val}, hist