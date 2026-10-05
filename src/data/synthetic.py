"""Synthetic adjacency for controlled tissue-field diagnostics."""
import numpy as np
import torch


def triangular_lattice_edges(side):
    """Return symmetric COO edges for a periodic side×side triangular lattice.

    Every node has exactly six distinct neighbors. Periodicity avoids degree
    changes at boundaries; it does not represent a spherical organoid mesh.
    Nodes are indexed as ``row * side + column``. Side must be at least five
    to avoid the small-torus identification of opposite neighbors.
    """
    if isinstance(side,bool) or not isinstance(side,(int,np.integer)) or side<5:
        raise ValueError('side must be an integer >= 5')
    row,column=np.indices((side,side));source=(row*side+column).reshape(-1)
    targets=[((row+dr)%side*side+(column+dc)%side).reshape(-1)
             for dr,dc in [(1,0),(-1,0),(0,1),(0,-1),(1,-1),(-1,1)]]
    return torch.as_tensor(np.stack((np.tile(source,6),np.concatenate(targets))),dtype=torch.long)
