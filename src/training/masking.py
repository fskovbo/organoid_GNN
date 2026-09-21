"""Reusable observed/missing fate training."""
from dataclasses import dataclass
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch_geometric.loader import DataLoader
from src.data.fate_masking import random_mask, _forward_with_mask
from src.models.size_models import seed_all
from src.training.losses import CompositeLoss, make_base_loss, WeightedLossTerm, edge_loss_term
from src.training.progress import log_epochs, report_epoch

def rate_tag(rate):
    return f'p{float(rate):g}'


@dataclass(frozen=True)
class MaskTrainingConfig:
    rates: tuple = (0., .01, .02, .05)
    masked_loss_weight: float = .5
    inner_val_fraction: float = .15
    inner_split_seed: int = 8123
    max_epochs: int = 500
    patience: int = 30
    batch_size: int = 64
    num_workers: int = 0
    grad_clip: float | None = 2.

    def __post_init__(self):
        if not self.rates or len(set(self.rates)) != len(self.rates) or any(not 0 <= p < 1 for p in self.rates):
            raise ValueError('Masking rates must be distinct and in [0, 1).')
        if 0. not in self.rates:
            raise ValueError('Include rate 0 as the matched unmasked control.')
        if not any(p > 0 for p in self.rates):
            raise ValueError('Include at least one positive masking rate.')
        if not 0 < self.masked_loss_weight < 1 or not 0 < self.inner_val_fraction < .5:
            raise ValueError('Invalid loss weight or inner validation fraction.')
        if min(self.max_epochs, self.patience, self.batch_size) < 1:
            raise ValueError('Epochs, patience and batch size must be positive.')
        if self.grad_clip is not None and self.grad_clip <= 0:
            raise ValueError('grad_clip must be positive or None.')


def inner_split(n_graphs, fraction, seed):
    if n_graphs < 3:
        raise ValueError('Need at least three outer-training organoids.')
    order = np.random.default_rng(seed).permutation(n_graphs)
    n = min(n_graphs - 1, max(1, int(np.ceil(n_graphs * fraction))))
    return order[n:].tolist(), order[:n].tolist()


def _loss(settings):
    return CompositeLoss(make_base_loss(), [WeightedLossTerm(
        name='edge', fn=edge_loss_term, weight=settings['EDGE_LOSS_WEIGHT'],
        params=settings['EDGE_LOSS_PARAMS'])])


@torch.no_grad()
def intact_validation_mse(model, loader, device):
    model.eval()
    total, n = 0., 0
    for batch in loader:
        batch = batch.to(device)
        (mu, _), _ = _forward_with_mask(model, batch)
        errors = (mu.reshape(-1) - batch.y.reshape(-1)).square()
        total += float(errors.sum()); n += len(errors)
    return total / n


def train_mask_model(model, train_graphs, early_graphs, settings, config, *, rate, seed, device,
                     verbose=None, epoch_callback=None):
    """Matched optimizer steps/BN passes even for rate 0; select on intact inner MSE.

    Backpropagate the two weighted passes separately before one optimizer step,
    bounding activation memory. Validation is always eval-mode and unmasked.
    """
    seed_all(seed)
    loader_rng = torch.Generator().manual_seed(seed + 1000)
    mask_rng = torch.Generator().manual_seed(seed + 2000)
    loader = DataLoader(train_graphs, batch_size=config.batch_size, shuffle=True,
        generator=loader_rng, num_workers=config.num_workers, pin_memory=str(device).startswith('cuda'))
    early = DataLoader(early_graphs, batch_size=config.batch_size, shuffle=False, num_workers=0)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings['LR'], weight_decay=settings['WEIGHT_DECAY'])
    loss_fn = _loss(settings)
    best, best_state, remaining, history = np.inf, None, config.patience, []
    best_epoch = 0
    for epoch in range(1, config.max_epochs + 1):
        model.train(); total_loss = 0.; total_nodes = 0; masked_nodes = 0
        for batch in loader:
            batch = batch.to(device)
            # All targets remain unchanged; flatten a single target to avoid
            # accidental broadcasting in the repository's auxiliary edge loss.
            batch.y = batch.y.reshape(-1)
            mask = random_mask(len(batch.x), rate, mask_rng).to(batch.x.device)
            optimizer.zero_grad(set_to_none=True)
            batch_loss = 0.
            for selected, weight in [(None, 1 - config.masked_loss_weight), (mask, config.masked_loss_weight)]:
                (mu, lv), _ = _forward_with_mask(model, batch, selected)
                loss = loss_fn(mu=mu, log_scale2=lv, batch=batch)
                if not torch.isfinite(loss):
                    raise FloatingPointError('Non-finite masking training loss.')
                (weight * loss).backward()
                batch_loss += weight * float(loss.detach())
            if config.grad_clip is not None:
                nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            total_loss += batch_loss * len(batch.x); total_nodes += len(batch.x)
            masked_nodes += int(mask.sum())
        score = intact_validation_mse(model, early, device)
        if not np.isfinite(score):
            raise FloatingPointError('Non-finite early-stopping score.')
        history.append(dict(epoch=epoch, train_loss=total_loss/total_nodes, inner_intact_mse_z=score,
                            realized_mask_rate=masked_nodes/total_nodes))
        if log_epochs(verbose):
            print(f'{rate_tag(rate)} seed={seed} epoch={epoch}: loss={history[-1]["train_loss"]:.5g}, '
                  f'inner intact MSE={score:.5g}', flush=True)
        if score < best - 1e-7:
            best = score; remaining = config.patience
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            remaining -= 1
        report_epoch(dict(epoch=epoch, max_epochs=config.max_epochs, metric='Inner intact MSE (transformed)',
                          value=score, best_value=best, best_epoch=best_epoch,
                          bad_epochs=config.patience-remaining, patience=config.patience), epoch_callback)
        if remaining <= 0:
            break
    model.load_state_dict(best_state)
    return model, pd.DataFrame(history)
