import os, glob, warnings
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected, remove_self_loops, coalesce

def _as_1d_float(a) -> np.ndarray:
    a = np.asarray(a, dtype=np.float32)
    if a.ndim == 2 and a.shape[1] == 1:
        a = a[:, 0]
    else:
        a = np.squeeze(a)
    assert a.ndim == 1, f"target must be 1-D after squeeze; got shape {a.shape}"
    return a

def _edges_to_edge_index(edges, num_nodes: int) -> torch.Tensor:
    """
    Convert undirected pairs (E,2) into PyG edge_index, make undirected,
    drop self-loops, and coalesce duplicates.
    """
    edges = np.asarray(edges, dtype=np.int64)
    if edges.size == 0:
        ei = torch.empty((2, 0), dtype=torch.long)
    else:
        u = torch.from_numpy(edges[:, 0])
        v = torch.from_numpy(edges[:, 1])
        ei = torch.stack([torch.cat([u, v]), torch.cat([v, u])], dim=0)
    ei = to_undirected(ei, num_nodes=num_nodes)
    ei, _ = remove_self_loops(ei)
    ei = coalesce(ei, num_nodes=num_nodes)

    return ei


def build_pyg_graph(markers_bin: np.ndarray, edges, curvature: np.ndarray,
                   dtype=torch.float32) -> Data:
    """
    Build a minimal PyG Data with only x, y, edge_index.
    - markers_bin: (N,M) float/bool/int -> coerced to float32
    - edges:       (E,2) undirected pairs (u<v) or any int pairs
    - curvature:   (N,) or (N,1) -> coerced to (N,)
    """
    markers_bin = np.asarray(markers_bin)
    assert markers_bin.ndim == 2, f"markers_bin must be (N,M), got {markers_bin.shape}"
    N, M = markers_bin.shape

    y = _as_1d_float(curvature)  # (N,)
    assert y.shape[0] == N, f"curvature length {y.shape[0]} != N={N}"
    assert np.isfinite(y).all(), "curvature contains NaN/inf"

    x = torch.as_tensor(markers_bin, dtype=dtype).contiguous()
    y = torch.as_tensor(y, dtype=dtype).contiguous()
    edge_index = _edges_to_edge_index(edges, num_nodes=N)

    return Data(x=x, y=y, edge_index=edge_index)


def load_organoid_npz(npz_path: str, strict: bool = False):
    """
    Load one organoid exported by Project (A).

    Expects (numeric NPZ keys):
      - 'x'     : (N, M) float32
      - 'y'     : (N,)   float32
      - 'edges' : (E, 2) int64 (u < v)
      - 'N','M' : int scalars (optional; used for sanity if present)

    Returns dict of numpy arrays, or None if invalid (when strict=False).
    """
    try:
        z = np.load(npz_path, allow_pickle=False)
    except Exception as e:
        if strict: raise
        warnings.warn(f"Failed to load {os.path.basename(npz_path)}: {e}")
        return None

    # x
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

    # y
    try:
        y = _as_1d_float(z["y"])
    except AssertionError as e:
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

    # edges (optional but expected)
    edges = np.asarray(z["edges"], dtype=np.int64) if "edges" in z.files else np.zeros((0, 2), np.int64)
    if edges.ndim != 2 or (edges.size and edges.shape[1] != 2):
        msg = f"{os.path.basename(npz_path)}: edges must be (E,2), got {edges.shape}. Skipping."
        if strict: raise ValueError(msg)
        warnings.warn(msg); return None

    return {"x": x, "y": y, "edges": edges, "N": N, "M": M}


def load_graph_dataset_from_dir(dir_path: str, strict: bool = False):
    """
    Load every *.npz in a directory and return list[Data] with:
      - x: (N,M) float32
      - y: (N,)  float32
      - edge_index: (2,E) long

    Skips bad files with a warning (unless strict=True).
    """
    paths = sorted(glob.glob(os.path.join(dir_path, "*.npz")))
    graphs, skipped = [], []
    for p in paths:
        arrs = load_organoid_npz(p, strict=strict)
        if arrs is None:
            skipped.append(os.path.basename(p))
            continue
        d = build_pyg_graph(arrs["x"], arrs["edges"], arrs["y"])
        # Optional: attach identifier (filename stem)
        d.organoid_str = os.path.splitext(os.path.basename(p))[0]
        graphs.append(d)
    if skipped:
        print(f"Loaded {len(graphs)} graphs; skipped {len(skipped)} bad files (e.g., {skipped[:3]}...).")
    else:
        print(f"Loaded {len(graphs)} graphs; skipped 0.")

    return graphs