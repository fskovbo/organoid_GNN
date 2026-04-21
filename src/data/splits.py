import copy
import numpy as np


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



def filter_graphs_by_target_percentile(graphs, low_pct=None, high_pct=None, stat="median"):
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



def graph_metadata_key(graph, *, meta_lookup=None, fields=("dataset", "label_uid")):
    """
    Build a stable tuple key from graph metadata.

    Parameters
    ----------
    graph : Data
    meta_lookup : dict or None
        Optional lookup keyed by graph.organoid_str -> metadata dict. This is
        useful after metadata has been stripped from graphs.
    fields : tuple[str, ...]
        Metadata fields to concatenate into the key tuple.

    Returns
    -------
    key : tuple
        Stable identifier such as (dataset, label_uid).
    """
    md = getattr(graph, "meta", None)
    if not isinstance(md, dict):
        if meta_lookup is None:
            raise ValueError(
                "Graph has no attached metadata and no meta_lookup was provided."
            )
        organoid_str = getattr(graph, "organoid_str", None)
        if organoid_str is None:
            raise ValueError("Graph has no .organoid_str")
        md = meta_lookup.get(organoid_str, None)
        if md is None:
            raise KeyError(f"No metadata found for organoid_str={organoid_str!r}")

    out = []
    for field in fields:
        if field not in md:
            raise KeyError(f"Missing metadata field {field!r}")
        out.append(md[field])

    return tuple(out)



def select_graphs_by_keys(graphs, keys, *, key_fn, meta_lookup=None, inplace=False):
    """
    Select graphs whose stable key is in `keys`.

    Parameters
    ----------
    graphs : list[Data]
    keys : iterable
        Stable keys such as {(dataset, label_uid), ...}.
    key_fn : callable
        Function with signature key_fn(graph, meta_lookup=...) -> hashable key.
    meta_lookup : dict or None
        Optional metadata lookup used when graphs have stripped metadata.
    inplace : bool
        If False, return shallow copies of the selected graphs.
    """
    keys = set(keys)
    out = []
    for g in graphs:
        key = key_fn(g, meta_lookup=meta_lookup)
        if key in keys:
            out.append(g if inplace else copy.copy(g))
    return out



def split_graphs_by_keys(graphs, keys, *, key_fn, meta_lookup=None, inplace=False):
    """
    Split a graph list into (selected, remainder) by stable key.

    Parameters
    ----------
    graphs : list[Data]
    keys : iterable
        Stable keys such as {(dataset, label_uid), ...}.
    key_fn : callable
        Function with signature key_fn(graph, meta_lookup=...) -> hashable key.
    meta_lookup : dict or None
        Optional metadata lookup used when graphs have stripped metadata.
    inplace : bool
        If False, return shallow copies.
    """
    keys = set(keys)
    selected, remainder = [], []

    for g in graphs:
        key = key_fn(g, meta_lookup=meta_lookup)
        if key in keys:
            selected.append(g if inplace else copy.copy(g))
        else:
            remainder.append(g if inplace else copy.copy(g))

    return selected, remainder



def train_val_split_graphs(
    graphs,
    val_frac=0.2,
    seed=42,
    *,
    force_val_keys=None,
    key_fn=None,
    inplace=False,
):
    """
    Random split of a list[Data] into (train_graphs, val_graphs) by organoid,
    with optional forced inclusion of selected graphs in validation.

    Parameters
    ----------
    graphs : list[Data]
    val_frac : float
        Fraction of graphs assigned to validation.
    seed : int
        Random seed for the split.
    force_val_keys : iterable or None
        Stable keys identifying graphs that must be placed in validation.
        These graphs always count toward the validation quota.
    key_fn : callable or None
        Function with signature key_fn(graph) -> hashable key. Required if
        force_val_keys is provided.
    inplace : bool
        If False, shallow-copy graphs before splitting.

    Returns
    -------
    g_train : list[Data]
    g_val : list[Data]
    split_info : dict
        Dictionary containing the stable keys and indices of the split:
          - train_keys
          - val_keys
          - forced_val_keys
          - random_val_keys
          - train_indices
          - val_indices
          - forced_val_indices
          - random_val_indices
          - val_frac
          - seed

    Notes
    -----
    No split or subset tags are attached to the graphs themselves. This keeps
    the graphs safe to batch with PyG DataLoader. Use `split_info` together
    with `select_graphs_by_keys(...)` or `split_graphs_by_keys(...)` to recover
    subsets later.
    """
    graphs_out = graphs if inplace else [copy.copy(g) for g in graphs]

    rng = np.random.default_rng(seed)
    n_graphs = len(graphs_out)
    n_val_target = max(1, int(round(n_graphs * val_frac)))

    force_val_keys = set() if force_val_keys is None else set(force_val_keys)

    if force_val_keys and key_fn is None:
        raise ValueError("key_fn must be provided when force_val_keys is used")

    graph_keys = [None] * n_graphs
    forced_val_indices = []

    if key_fn is not None:
        for i, g in enumerate(graphs_out):
            graph_keys[i] = key_fn(g)

        if force_val_keys:
            seen = set()
            for i, key in enumerate(graph_keys):
                if key in force_val_keys:
                    forced_val_indices.append(i)
                    seen.add(key)

            missing = force_val_keys - seen
            if missing:
                raise KeyError(
                    "Some force_val_keys were not found in graphs: "
                    f"{sorted(missing)!r}"
                )

    forced_val_indices = sorted(set(forced_val_indices))
    forced_val_index_set = set(forced_val_indices)

    remaining_indices = np.array(
        [i for i in range(n_graphs) if i not in forced_val_index_set],
        dtype=int,
    )
    rng.shuffle(remaining_indices)

    n_random_val = max(0, n_val_target - len(forced_val_indices))
    random_val_indices = sorted(remaining_indices[:n_random_val].tolist())

    val_index_set = forced_val_index_set | set(random_val_indices)

    g_train, g_val = [], []
    train_indices, val_indices = [], []

    for i, g in enumerate(graphs_out):
        if i in val_index_set:
            g_val.append(g)
            val_indices.append(i)
        else:
            g_train.append(g)
            train_indices.append(i)

    if key_fn is not None:
        train_keys = [graph_keys[i] for i in train_indices]
        val_keys = [graph_keys[i] for i in val_indices]
        forced_keys_found = [graph_keys[i] for i in forced_val_indices]
        random_val_keys = [graph_keys[i] for i in random_val_indices]
    else:
        train_keys = None
        val_keys = None
        forced_keys_found = None
        random_val_keys = None

    split_info = {
        "train_keys": train_keys,
        "val_keys": val_keys,
        "forced_val_keys": forced_keys_found,
        "random_val_keys": random_val_keys,
        "train_indices": train_indices,
        "val_indices": val_indices,
        "forced_val_indices": forced_val_indices,
        "random_val_indices": random_val_indices,
        "val_frac": float(val_frac),
        "seed": seed,
    }

    return g_train, g_val, split_info



def standardize_graph_targets(train_graphs, val_graphs=None, robust=False):
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



def standardize_graph_global_features(
    train_graphs,
    val_graphs=None,
    attr_name="global_feat",
    robust=False,
):
    """
    Standardize graph-level feature vectors using TRAIN graphs only.

    Assumes each graph stores attr_name with shape (1, D).
    Returns (center, scale), each of shape (D,).
    """
    import numpy as np
    import torch

    X_all = np.concatenate([
        getattr(g, attr_name).detach().cpu().numpy()
        for g in train_graphs
    ], axis=0)

    if robust:
        center = np.median(X_all, axis=0)
        iqr = np.percentile(X_all, 75, axis=0) - np.percentile(X_all, 25, axis=0)
        scale = np.where(iqr > 1e-12, iqr / 1.349, 1.0)
    else:
        center = np.mean(X_all, axis=0)
        std = np.std(X_all, axis=0)
        scale = np.where(std > 1e-12, std, 1.0)

    center_t = torch.as_tensor(center, dtype=train_graphs[0].global_feat.dtype)
    scale_t = torch.as_tensor(scale, dtype=train_graphs[0].global_feat.dtype)

    def _apply(graphs):
        for g in graphs:
            x = getattr(g, attr_name)

            if x.ndim == 1:
                x = x.unsqueeze(0)

            if x.ndim != 2 or x.shape[0] != 1:
                raise ValueError(
                    f"{attr_name} must have shape (1, D), got {tuple(x.shape)}"
                )

            x_std = (x - center_t.unsqueeze(0)) / scale_t.unsqueeze(0)
            setattr(g, attr_name, x_std)

    _apply(train_graphs)
    if val_graphs is not None:
        _apply(val_graphs)

    return center, scale