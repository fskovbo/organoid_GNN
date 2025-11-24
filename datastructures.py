import os, glob, warnings, json
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected, remove_self_loops, coalesce


# -----------------------------------------------------------------------------
# Core graph builders & loaders
# -----------------------------------------------------------------------------

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


def make_pyg_graph(markers_bin: np.ndarray, edges, curvature: np.ndarray,
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


def load_one_npz_arrays(npz_path: str, strict: bool = False):
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


def load_dir_to_graphs(dir_path: str, strict: bool = False):
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
        arrs = load_one_npz_arrays(p, strict=strict)
        if arrs is None:
            skipped.append(os.path.basename(p))
            continue
        d = make_pyg_graph(arrs["x"], arrs["edges"], arrs["y"])
        # Optional: attach identifier (filename stem)
        d.organoid_str = os.path.splitext(os.path.basename(p))[0]
        graphs.append(d)
    if skipped:
        print(f"Loaded {len(graphs)} graphs; skipped {len(skipped)} bad files (e.g., {skipped[:3]}...).")
    else:
        print(f"Loaded {len(graphs)} graphs; skipped 0.")

    return graphs


# -----------------------------------------------------------------------------
# Simple dataset prep utilities (pure data; no model/training)
# -----------------------------------------------------------------------------

def organoid_stat(graph, stat="median"):
    """
    Reduce a graph's node targets y to a scalar summary for filtering.
    stat ∈ {'median','mean'}.
    """
    y = graph.y.detach().cpu().numpy().reshape(-1)
    if stat == "median":
        return float(np.median(y))
    elif stat == "mean":
        return float(np.mean(y))
    else:
        raise ValueError("stat must be 'median' or 'mean'")


def filter_outliers(graphs, low_pct=None, high_pct=None, stat="median"):
    """
    Keep only graphs with organoid_stat in [low_pct, high_pct] percentiles
    computed across *all* organoids. If both are None, returns input unchanged.
    """
    if low_pct is None and high_pct is None:
        return graphs, {"low": None, "high": None}
    vals = np.array([organoid_stat(g, stat=stat) for g in graphs], dtype=np.float64)
    low_thr = np.percentile(vals, low_pct) if low_pct is not None else -np.inf
    high_thr = np.percentile(vals, high_pct) if high_pct is not None else np.inf
    keep = [g for g, v in zip(graphs, vals) if (v >= low_thr) and (v <= high_thr)]
    return keep, {"low": float(low_thr), "high": float(high_thr)}


def split_graphs(graphs, val_frac=0.2, seed=42):
    """
    Random split of a list[Data] into (train_graphs, val_graphs) by organoid.
    """
    rng = np.random.default_rng(seed)
    idx = np.arange(len(graphs))
    rng.shuffle(idx)
    n_val = max(1, int(round(len(graphs) * val_frac)))
    val_idx = set(idx[:n_val].tolist())
    g_train, g_val = [], []
    for i, g in enumerate(graphs):
        (g_val if i in val_idx else g_train).append(g)
    return g_train, g_val


def standardize_targets(train_graphs, val_graphs=None, robust=False):
    """
    Compute global target standardization on TRAIN only and apply to provided lists.
    robust=False → mean/std; robust=True → median/IQR/1.349.
    Returns (center, scale).
    """
    y_all = np.concatenate([g.y.detach().cpu().numpy().reshape(-1) for g in train_graphs], axis=0)
    if robust:
        center = float(np.median(y_all))
        iqr = float(np.percentile(y_all, 75) - np.percentile(y_all, 25))
        scale = float(iqr / 1.349) if iqr > 1e-12 else 1.0
    else:
        center = float(np.mean(y_all))
        std = float(np.std(y_all))
        scale = std if std > 1e-12 else 1.0
    for g in train_graphs:
        g.y = (g.y - center) / scale
    if val_graphs is not None:
        for g in val_graphs:
            g.y = (g.y - center) / scale
    return center, scale


# ---------------------------------------------------------------------
# Marker names and metadata loader 
# ---------------------------------------------------------------------

def load_marker_names_from_dir(dir_path):
    """
    Try to find a *_markers.json sidecar in a directory of NPZs and load marker names.
    Returns: list[str] or None if not found.
    """
    jsons = sorted(glob.glob(os.path.join(dir_path, "*_markers.json")))
    if not jsons:
        return None
    with open(jsons[0], "r") as f:
        names = json.load(f)
    return list(names)


def load_aux_metadata_for_dir(dir_path: str):
    """
    Scan a directory of organoid NPZs and load per-organoid auxiliary metadata
    from sidecars named: organoid_<id>_aux.json.

    Returns
    -------
    meta_by_key : dict[str, dict]
        Maps organoid key (filename stem, e.g. "organoid_day4_A01_007")
        -> {"organoid_id": str, "total_surface_area": float|None,
            "total_volume": float|None, "complexity_score": float|None, ...}

    Notes
    -----
    - Missing sidecars are simply skipped; you’ll still get metadata for the ones present.
    - Values are coerced to Python floats where possible; missing fields are omitted.
    """
    meta = {}
    # infer stems from existing npz files
    for npz_path in sorted(glob.glob(os.path.join(dir_path, "organoid_*.npz"))):
        stem = os.path.splitext(os.path.basename(npz_path))[0]            # organoid_<id>
        aux_path = os.path.join(dir_path, f"{stem}_aux.json")
        if not os.path.exists(aux_path):
            continue
        try:
            with open(aux_path, "r") as f:
                raw = json.load(f)
        except Exception as e:
            print(f"Warning: failed reading {aux_path}: {e}")
            continue

        # Coerce values to plain Python floats/ints where possible
        clean = {"organoid_id": str(raw.get("organoid_id", stem.replace("organoid_", "")))}
        for k, v in raw.items():
            if k == "organoid_id": 
                continue
            try:
                # Handle numpy scalars/0-D arrays robustly
                if isinstance(v, (list, tuple)) and len(v) == 1:
                    v = v[0]
                if hasattr(v, "item"):
                    v = v.item()
                if isinstance(v, (int, float)):
                    clean[k] = float(v)
            except Exception:
                # Keep as-is if not numeric
                clean[k] = v
        meta[stem] = clean
    return meta


def attach_metadata_to_graphs(graphs, meta_by_key: dict, quiet: bool = True):
    """
    Attach metadata dicts to each PyG Data in-place as `data.meta`
    using the graph's `organoid_str` (filename stem) as the key.

    Returns the number of graphs that received metadata.
    Safe: does NOT alter x/y/edge_index; won’t affect training & collate.
    """
    count = 0
    for g in graphs:
        key = getattr(g, "organoid_str", None)
        if key is None:
            # fall back to nothing; you can add keys before calling this
            continue
        md = meta_by_key.get(key)
        if md is None:
            if not quiet:
                print(f"No metadata for {key}")
            continue
        # attach a shallow copy to avoid accidental mutation
        g.meta = dict(md)
        count += 1
    return count


def metadata_dataframe(graphs, extra_cols=("num_nodes",)):
    """
    Build a pandas DataFrame with organoid key + selected metadata fields for quick grouping.
    """
    import pandas as pd
    rows = []
    for g in graphs:
        key = getattr(g, "organoid_str", None)
        md  = getattr(g, "meta", {}) or {}
        row = {"organoid": key}
        row.update({k: md.get(k) for k in ["total_surface_area", "total_volume", "complexity_score"]})
        if "num_nodes" in extra_cols:
            row["num_nodes"] = int(g.x.size(0))
        rows.append(row)
    return pd.DataFrame(rows)



# ---------------------------------------------------------------------
# Add 2-hop edges
# ---------------------------------------------------------------------

def augment_two_hop_edges(edge_index: torch.Tensor,
                          num_nodes: int,
                          include_self_loops: bool = False,
                          keep_one_hop: bool = True) -> torch.Tensor:
    """
    Create (1 ∪ 2)-hop edges by squaring the adjacency in sparse form.

    Args
    ----
    edge_index : (2, E) long tensor, undirected or directed
    num_nodes  : N
    include_self_loops : if True, keep i->i edges in the final graph
    keep_one_hop : if True, include original 1-hop edges in the union

    Returns
    -------
    edge_index_aug : (2, E') long tensor, undirected, deduped, no self-loops by default
    """
    # Make sure it's undirected so A is symmetric
    ei = to_undirected(edge_index, num_nodes=num_nodes)
    ei, _ = remove_self_loops(ei)

    # Build sparse adjacency A (N x N). Values are 1.
    N = num_nodes
    device = ei.device
    A = torch.sparse_coo_tensor(ei, torch.ones(ei.size(1), device=device), (N, N))

    # 2-hop: A2 = A @ A  (sparse @ sparse -> sparse)
    # NOTE: torch.sparse.mm expects (N x N) @ (N x N) dense/sparse mix.
    # Convert to CSR via to_sparse_csr for speed if available.
    if hasattr(A, "to_sparse_csr"):
        A_csr = A.to_sparse_csr()
        A2 = torch.sparse.mm(A_csr, A_csr).to_sparse_coo()
    else:
        A2 = torch.sparse.mm(A, A).to_sparse_coo()

    # Extract indices for 2-hop edges
    two_hop = A2.coalesce().indices()  # (2, E2)

    # Optionally keep only *strictly* 2-hop (remove original 1-hop from two_hop)
    if keep_one_hop:
        union = torch.cat([ei, two_hop], dim=1)
    else:
        # remove edges that are also in 1-hop
        # coalesce will deduplicate after we subtract self-loops anyway
        union = torch.cat([two_hop], dim=1)

    # Clean up: drop/keep self-loops, coalesce, undirected
    union = to_undirected(union, num_nodes=N)
    if not include_self_loops:
        union, _ = remove_self_loops(union)
    union = coalesce(union, num_nodes=N)

    return union


def add_two_hop_edges_inplace(data, include_self_loops: bool = False, keep_one_hop: bool = True):
    """
    Mutates a PyG Data object to include 2-hop edges.
    """
    data.edge_index = augment_two_hop_edges(
        data.edge_index, num_nodes=data.x.size(0),
        include_self_loops=include_self_loops,
        keep_one_hop=keep_one_hop
    )
    return data


# ==== Inverse-task helpers: curvature -> markers (multi-label) =================

def select_marker_indices(marker_names, selected=None):
    """
    Map a user-provided list of marker names to column indices.
    If selected is None, returns indices for ALL markers.
    """
    if selected is None:
        return list(range(len(marker_names)))
    name_to_idx = {n: i for i, n in enumerate(marker_names)}
    idx = []
    for n in selected:
        if n not in name_to_idx:
            raise ValueError(f"Requested marker '{n}' not found in marker_names.")
        idx.append(name_to_idx[n])
    return idx


def build_inverse_graphs(graphs, marker_names, predict_markers=None, keep_two_hop=False):
    """
    Convert graphs for the inverse task:
      input  x := curvature as single-channel feature -> (N,1)
      target y := selected marker columns (multi-label) -> (N,K) in {0,1}

    Args
    ----
    graphs          : list[Data] with original x=(N,M markers), y=(N,) curvature
    marker_names    : list[str] length M
    predict_markers : list[str] or None (None => predict all markers)
    keep_two_hop    : if True, assumes you already augmented edge_index; do nothing special

    Returns
    -------
    inv_graphs : list[Data] with x=(N,1) curvature, y=(N,K) 0/1
    selected_names : list[str] names of predicted markers
    """
    idx = select_marker_indices(marker_names, predict_markers)
    sel_names = [marker_names[i] for i in idx]

    inv = []
    for g in graphs:
        N = g.x.size(0)
        # x := curvature as (N,1)
        x_in = g.y.view(N, 1).clone().detach()
        # y := selected marker columns as float {0,1}
        y_out = g.x[:, idx].clone().detach().float()
        gg = type(g)()
        gg.x = x_in
        gg.y = y_out
        gg.edge_index = g.edge_index
        if hasattr(g, "organoid_str"):
            gg.organoid_str = g.organoid_str
        inv.append(gg)
    return inv, sel_names
