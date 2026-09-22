"""Explicit model specifications and compatibility with historical checkpoints."""
import importlib
import inspect
import json
from pathlib import Path
from collections.abc import Mapping

import torch


def model_spec(model):
    """Describe a repository model using its actual constructor attributes.

    Missing attributes fail explicitly rather than silently using defaults.
    FiLM forwards keyword arguments to GIN, so inspect both constructors.
    """
    cls = type(model)
    if not cls.__module__.startswith("src.models."):
        raise TypeError(f"Register a model specification for {cls.__module__}.{cls.__name__}")
    kwargs = {}
    classes = [cls]
    if any(p.kind == p.VAR_KEYWORD for p in inspect.signature(cls).parameters.values()):
        classes = [c for c in reversed(cls.mro()) if c.__module__.startswith("src.models.")]
    for owner in classes:
        for name, parameter in inspect.signature(owner).parameters.items():
            if parameter.kind in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD):
                continue
            if not hasattr(model, name):
                raise ValueError(f"{cls.__name__} does not record constructor argument {name!r}")
            kwargs[name] = getattr(model, name)
    spec = {"module": cls.__module__, "class": cls.__name__, "kwargs": kwargs}
    # Missing-fate initialization can require a learned residual projection even
    # when augmented input width equals hidden width (normally an identity).
    projection = getattr(model, 'input_proj', None)
    if (projection is not None and getattr(model, 'residual', False)
            and getattr(model, 'n_markers', None) == getattr(model, 'hidden_dim', None)):
        spec['input_projection'] = dict(in_features=projection.in_features,
            out_features=projection.out_features, bias=projection.bias is not None)
    return spec


def build_model(spec):
    if not spec["module"].startswith("src.models."):
        raise ValueError("Model specifications must refer to src.models")
    model = getattr(importlib.import_module(spec["module"]), spec["class"])(**spec["kwargs"])
    if 'input_projection' in spec:
        model.input_proj = torch.nn.Linear(**spec['input_projection'])
    return model


def checkpoint_state(payload):
    """Accept bare weights, model_state_dict, and historical masking model_state."""
    if not isinstance(payload, Mapping):
        raise ValueError("Expected a checkpoint mapping")
    for key in ("model_state_dict", "model_state"):
        if key in payload:
            return payload[key]
    if payload and all(torch.is_tensor(value) for value in payload.values()):
        return payload
    raise ValueError("Unrecognized checkpoint format")


def load_weights(model, path, *, device="cpu"):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint_state(payload), strict=True)
    return model.to(device).eval()


def save_weights_with_spec(model, path):
    """Save portable bare weights and an explicit constructor sidecar.

    Keeping the bare tensor mapping preserves compatibility with older readers.
    The enclosing training run records preprocessing, membership, and settings.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    spec = model_spec(model)
    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, path)
    path.with_suffix(path.suffix + ".json").write_text(json.dumps(spec, indent=2))
