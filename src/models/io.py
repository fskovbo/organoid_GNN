from pathlib import Path
import torch


def save_model_checkpoint(
    path,
    model,
    optimizer=None,
    epoch=None,
    metrics=None,
    config=None,
):
    """
    Save full training checkpoint.

    Parameters
    ----------
    path : str or Path
    model : torch.nn.Module
    optimizer : torch.optim.Optimizer or None
    epoch : int or None
    metrics : dict or None
    config : dict or None
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    ckpt = {
        "model_state_dict": model.state_dict(),
        "epoch": epoch,
        "metrics": metrics,
        "config": config,
    }

    if optimizer is not None:
        ckpt["optimizer_state_dict"] = optimizer.state_dict()

    torch.save(ckpt, path)


def load_model_checkpoint(
    path,
    model,
    optimizer=None,
    map_location="cpu",
):
    """
    Load full training checkpoint into model (+ optimizer optional).

    Returns
    -------
    checkpoint_dict
    """
    ckpt = torch.load(path, map_location=map_location)

    model.load_state_dict(ckpt["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    return ckpt


def save_model_weights(path, model):
    """
    Save only model weights (lightweight).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def load_model_weights(path, model, map_location="cpu", strict=True):
    """
    Load only weights.
    """
    state = torch.load(path, map_location=map_location)
    model.load_state_dict(state, strict=strict)