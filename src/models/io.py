from pathlib import Path
import torch


def save_model_checkpoint(path, model, optimizer=None, epoch=None, metrics=None, config=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ckpt = {"model_state_dict": model.state_dict(), "epoch": epoch, "metrics": metrics, "config": config}
    if optimizer is not None:
        ckpt["optimizer_state_dict"] = optimizer.state_dict()
    torch.save(ckpt, path)


def load_model_checkpoint(path, model, optimizer=None, map_location="cpu"):
    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return ckpt


def save_model_weights(path, model):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def load_model_weights(path, model, map_location="cpu", strict=True):
    state = torch.load(path, map_location=map_location)
    model.load_state_dict(state, strict=strict)

    