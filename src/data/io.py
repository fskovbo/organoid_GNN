import os, glob, warnings
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected, remove_self_loops, coalesce


def _as_target_array(a, target_indices=None) -> np.ndarray:
    """Return targets as (N,) for one target or (N,D) for multiple targets.

    target_indices can be None, an int, or a sequence of ints. This allows
    loading only Gaussian curvature (0), only mean curvature (1), or both.
    """
    y = np.asarray(a, dtype=np.float32)
    if y.ndim == 1:
        y = y[:, None]
    elif y.ndim == 2:
        pass
    else:
        y = np.squeeze(y)
        if y.ndim == 1:
            y = y[:, None]
        elif y.ndim != 2:
            raise AssertionError(f"target must be 1-D or 2-D after squeeze; got shape {y.shape}")

    if target_indices is not None:
        if isinstance(target_indices, (int, np.integer)):
            idx = [int(target_indices)]
        else:
            idx = [int(i) for i in target_indices]
        y = y[:, idx]

    if y.shape[1] == 1:
        return y[:, 0]
    return y


# Backwards-compatible name used by older code.
def _as_1d_float(a) -> np.ndarray:
    y = _as_target_array(a)
    assert y.ndim == 1, f"target must be 1-D after squeeze; got shape {y.shape}"
    return y


def _edges_to_edge_index(edges, num_nodes: int) -> torch.Tensor:
    """Convert undirected pairs (E,2) into coalesced PyG edge_index."""
    edges = np.asarray(edges, dtype=np.int64)
    if edges.size == 0:
        ei = torch.empty((2, 0), dtype=torch.long)
    else:
        u = torch.from_numpy(edges[:, 0])
        v = torch.from_numpy(edges[:, 1])
        ei = torch.stack([torch.cat([u, v]), torch.cat([v, u])], dim=0)
    ei = to_undirected(ei, num_nodes=num_nodes)
    ei, _ = remove_self_loops(ei)
    return coalesce(ei, num_nodes=num_nodes)


def build_pyg_graph(markers_bin: np.ndarray, edges, curvature: np.ndarray,
                    dtype=torch.float32, target_indices=None) -> Data:
    """Build a PyG graph, preserving old scalar y when one target is selected."""
    markers_bin = np.asarray(markers_bin)
    assert markers_bin.ndim == 2, f"markers_bin must be (N,M), got {markers_bin.shape}"
    N, _ = markers_bin.shape

    y = _as_target_array(curvature, target_indices=target_indices)
    assert y.shape[0] == N, f"target length {y.shape[0]} != N={N}"
    assert np.isfinite(y).all(), "target contains NaN/inf"

    x = torch.as_tensor(markers_bin, dtype=dtype).contiguous()
    y = torch.as_tensor(y, dtype=dtype).contiguous()
    edge_index = _edges_to_edge_index(edges, num_nodes=N)

    return Data(x=x, y=y, edge_index=edge_index)


def load_organoid_npz(npz_path: str, strict: bool = False, target_indices=None):
    """Load one organoid NPZ with x, y and edges.

    The new y format may be (N, 2), where column 0 is Gaussian curvature and
    column 1 is mean curvature. Use target_indices to keep only selected columns.
    """
    try:
        z = np.load(npz_path, allow_pickle=False)
    except Exception as e:
        if strict: raise
        warnings.warn(f"Failed to load {os.path.basename(npz_path)}: {e}")
        return None

    if "x" not in z.files or "y" not in z.files:
        msg = f"{os.path.basename(npz_path)} missing 'x' or 'y'. Skipping."
        if strict: raise ValueError(msg)
        warnings.warn(msg); return None

    x = np.asarray(z["x"], dtype=np.float32)
    if x.ndim != 2:
        msg = f"{os.path.basename(npz_path)}: x must be 2-D (N,M), got {x.shape}."
        if strict: raise ValueError(msg)
        warnings.warn(msg); return None
    N, M = x.shape

    try:
        y = _as_target_array(z["y"], target_indices=target_indices)
    except Exception as e:
        if strict: raise
        warnings.warn(f"{os.path.basename(npz_path)}: {e}. Skipping.")
        return None

    if y.shape[0] != N:
        msg = f"{os.path.basename(npz_path)}: y length {y.shape[0]} != N={N}."
        if strict: raise ValueError(msg)
        warnings.warn(msg); return None
    if N <= 1:
        msg = f"{os.path.basename(npz_path)}: N <= 1. Skipping."
        if strict: raise ValueError(msg)
        warnings.warn(msg); return None
    if not np.isfinite(y).all():
        msg = f"{os.path.basename(npz_path)}: y has NaN/Inf. Skipping."
        if strict: raise ValueError(msg)
        warnings.warn(msg); return None

    edges = np.asarray(z["edges"], dtype=np.int64) if "edges" in z.files else np.zeros((0, 2), np.int64)
    if edges.ndim != 2 or (edges.size and edges.shape[1] != 2):
        msg = f"{os.path.basename(npz_path)}: edges must be (E,2), got {edges.shape}. Skipping."
        if strict: raise ValueError(msg)
        warnings.warn(msg); return None

    return {"x": x, "y": y, "edges": edges, "N": N, "M": M}


def load_graph_dataset_from_dir(dir_path: str, strict: bool = False, target_indices=None):
    """Load all *.npz graphs from a directory."""
    paths = sorted(glob.glob(os.path.join(dir_path, "*.npz")))
    graphs, skipped = [], []
    for p in paths:
        arrs = load_organoid_npz(p, strict=strict, target_indices=target_indices)
        if arrs is None:
            skipped.append(os.path.basename(p))
            continue
        d = build_pyg_graph(arrs["x"], arrs["edges"], arrs["y"])
        d.organoid_str = os.path.splitext(os.path.basename(p))[0]
        graphs.append(d)
    if skipped:
        print(f"Loaded {len(graphs)} graphs; skipped {len(skipped)} bad files (e.g., {skipped[:3]}...).")
    else:
        print(f"Loaded {len(graphs)} graphs; skipped 0.")
    return graphs


def select_graph_targets(graphs, target_indices=None, *, inplace=False, y_attr="y"):
    """Restrict existing PyG graphs to one or more target columns."""
    import copy
    if target_indices is None:
        return graphs if inplace else [copy.copy(g) for g in graphs]
    graphs_out = graphs if inplace else [copy.copy(g) for g in graphs]
    idx = [int(target_indices)] if isinstance(target_indices, (int, np.integer)) else [int(i) for i in target_indices]
    for g in graphs_out:
        y = getattr(g, y_attr)
        y2 = y.unsqueeze(-1) if torch.is_tensor(y) and y.ndim == 1 else y
        if torch.is_tensor(y2):
            yy = y2[:, idx]
            setattr(g, y_attr, yy[:, 0].contiguous() if len(idx) == 1 else yy.contiguous())
        else:
            yy = _as_target_array(y2, target_indices=idx)
            setattr(g, y_attr, yy)
    return graphs_out
