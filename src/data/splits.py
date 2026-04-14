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


def train_val_split_graphs(graphs, val_frac=0.2, seed=42):
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

    # Collect training graph features as an (N_graphs, D) array
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