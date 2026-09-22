"""Reusable observed/missing fate model initialization."""
from src.models.size_models import make_model
from torch import nn
import torch


def with_missing_fate_input(original):
    """Add a zero-initialized missing-fate channel to an explicit GIN/FiLM.

    Preserve the supplied architecture, globals and observed-input predictions.
    No reference run or training settings are consulted.
    """
    from src.models.gnn import GINCurvature
    from src.artifacts.checkpoints import model_spec, build_model
    if not isinstance(original, GINCurvature):
        raise TypeError('Missing-fate initialization supports GIN and FiLM GIN.')
    spec = model_spec(original)
    markers = original.n_markers
    spec['kwargs']['n_markers'] = markers + 1
    model = build_model(spec)
    if model.num_layers > 0 and model.residual and model.input_proj is None:
        model.input_proj = nn.Linear(markers + 1, model.hidden_dim, bias=False)
    state = model.state_dict()
    if original.num_layers > 0 and original.residual and original.input_proj is None:
        state['input_proj.weight'].zero_()
        state['input_proj.weight'][:, :markers] = torch.eye(markers)
    for key, value in original.state_dict().items():
        if state[key].shape == value.shape:
            state[key] = value.detach().cpu().clone()
        elif original.num_layers == 0 and key == 'head.net.0.weight':
            state[key].zero_()
            state[key][:, :markers] = value[:, :markers]
            state[key][:, markers + 1:] = value[:, markers:]
        elif key in ('convs.0.nn.0.weight', 'input_proj.weight'):
            if state[key].shape != (value.shape[0], value.shape[1] + 1):
                raise ValueError(f'Unexpected mask-input shape for {key}')
            state[key].zero_()
            state[key][:, :-1] = value
        else:
            raise ValueError(f'Unexpected initialization mismatch: {key}')
    model.load_state_dict(state)
    return model


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
