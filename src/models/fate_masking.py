"""Reusable observed/missing fate model initialization."""
from src.models.size_models import make_model
from torch import nn
import torch


def make_mask_model(settings, markers, seed=None):
    """Same base initialization at each rate; new mask columns start at zero."""
    model = make_model(settings, list(markers) + ['fate_missing'])
    # Always project residual inputs: adding the flag can otherwise switch
    # between identity and learned projection when hidden_dim equals input_dim.
    if model.num_layers > 0 and model.residual and model.input_proj is None:
        model.input_proj = nn.Linear(len(markers) + 1, model.hidden_dim, bias=False)
    if seed is None:
        return model
    original = make_model(settings, markers, seed=seed)
    state = model.state_dict()
    if original.num_layers > 0 and original.residual and original.input_proj is None:
        state['input_proj.weight'].zero_()
        state['input_proj.weight'][:, :len(markers)] = torch.eye(len(markers))
    for key, value in original.state_dict().items():
        if state[key].shape == value.shape:
            state[key] = value.clone()
        elif original.num_layers == 0 and key == 'head.net.0.weight':
            # The missingness column precedes the global inputs, not follows them.
            state[key].zero_()
            state[key][:, :len(markers)] = value[:, :len(markers)]
            state[key][:, len(markers) + 1:] = value[:, len(markers):]
        elif key in ('convs.0.nn.0.weight', 'input_proj.weight'):
            if state[key].shape != (value.shape[0], value.shape[1] + 1):
                raise ValueError(f'Unexpected mask-input shape for {key}')
            state[key].zero_(); state[key][:, :-1] = value
        else:
            raise ValueError(f'Unexpected initialization mismatch: {key}')
    model.load_state_dict(state)
    return model
