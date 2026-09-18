"""Reusable observed/missing fate encoding."""
import copy
import torch
from torch import nn


def encode_fates(x, mask=None):
    """Return a new [fates, missing] matrix; erase ALL fate bits on masked nodes."""
    if x.ndim != 2:
        raise ValueError('Expected a node-by-marker matrix.')
    if mask is None:
        mask = torch.zeros(len(x), dtype=torch.bool, device=x.device)
    else:
        mask = torch.as_tensor(mask, dtype=torch.bool, device=x.device)
    if mask.shape != (len(x),):
        raise ValueError('Mask must contain one boolean per node.')
    result = torch.cat([x.clone(), mask[:, None].to(x.dtype)], dim=1)
    result[mask, :-1] = 0
    return result


def random_mask(n_nodes, rate, generator):
    """CPU RNG independent of model/dropout RNG; rates share nested random draws."""
    if not 0 <= rate < 1:
        raise ValueError('Masking probability must be in [0, 1).')
    return torch.rand(n_nodes, generator=generator) < rate


class ObservedFateAdapter(nn.Module):
    """Use an eight-input network with existing seven-input inference utilities."""
    def __init__(self, network):
        super().__init__()
        self.network = network

    def forward(self, x, edge_index, data=None):
        encoded = encode_fates(x)
        view = copy.copy(data) if data is not None else None
        if view is not None:
            view.x = encoded
        return self.network(encoded, edge_index, data=view)


def _forward_with_mask(model, batch, mask=None):
    # Erase fate in both the explicit x argument and data.x. No original source
    # identity remains accessible to the network through the batch container.
    view = copy.copy(batch)
    view.x = encode_fates(batch.x, mask)
    return model(view.x, view.edge_index, data=view)
