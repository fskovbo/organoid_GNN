import math
import numpy as np
import torch
from collections import Counter
from torch_geometric.loader import DataLoader

# ---------------------------
# Basic integrity checks
# ---------------------------

def _to_numpy(x):
    return x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)

def check_edge_index_integrity(g):
    """
    Check edge_index bounds, self-loops, duplicate edges, undirected symmetry.
    Returns dict with counts.
    """
    N = g.x.size(0)
    ei = g.edge_index
    assert ei.dim() == 2 and ei.size(0) == 2, "edge_index must be (2, E)"
    u, v = ei[0], ei[1]

    # range
    out_of_range = int(((u < 0) | (u >= N) | (v < 0) | (v >= N)).sum().item())

    # self-loops
    self_loops = int((u == v).sum().item())

    # duplicates (treat undirected pairs as unordered)
    uv = torch.stack([torch.minimum(u, v), torch.maximum(u, v)], dim=0)
    E = uv.size(1)
    key = uv[0] * (N+1) + uv[1]
    _, inv, counts = torch.unique(key, return_inverse=True, return_counts=True)
    duplicates = int((counts[inv] - 1).sum().item())  # over all edges

    # undirected symmetry (how many directed edges have their reverse present)
    # build a set of directed keys and check reverse presence
    dir_key = (u * (N+1) + v).tolist()
    dir_set = set(dir_key)
    rev_key = (v * (N+1) + u).tolist()
    has_rev = sum(1 for k in rev_key if k in dir_set)
    symmetry_ratio = has_rev / max(ei.size(1), 1)

    return {
        "num_nodes": int(N),
        "num_edges": int(ei.size(1)),
        "out_of_range": out_of_range,
        "self_loops": self_loops,
        "duplicate_directed": duplicates,
        "undirected_symmetry_ratio": float(symmetry_ratio),
    }

def degree_stats(g):
    """
    Compute degree for each node (using destination index), summarize.
    """
    N = g.x.size(0)
    deg = torch.bincount(g.edge_index[1], minlength=N)
    d = deg.float().detach().cpu().numpy()
    return {
        "deg0_frac": float((d == 0).mean()),
        "deg_mean": float(d.mean()),
        "deg_p50": float(np.percentile(d, 50)),
        "deg_p90": float(np.percentile(d, 90)),
        "deg_max": float(d.max() if d.size > 0 else 0),
    }

def connected_components(g):
    """
    Return component ids and basic stats (torch-only BFS).
    """
    N = g.x.size(0)
    u, v = g.edge_index
    adj = [[] for _ in range(N)]
    for a, b in zip(u.tolist(), v.tolist()):
        adj[a].append(b); adj[b].append(a)
    comp = -torch.ones(N, dtype=torch.long)
    cid = 0
    for i in range(N):
        if comp[i] != -1: continue
        q = [i]; comp[i] = cid
        for x in q:
            for y in adj[x]:
                if comp[y] == -1:
                    comp[y] = cid; q.append(y)
        cid += 1
    comp = comp.detach().cpu().numpy()
    counts = Counter(comp.tolist())
    small = sum(1 for c in counts.values() if c <= 5)
    return comp, {"num_components": cid, "num_small_<=5": small, "largest": max(counts.values()) if counts else 0}

# ---------------------------
# Neighbor structure / signal
# ---------------------------

def homophily_by_marker(g):
    """
    For each marker m: fraction of edges that connect two nodes
    with the same binary value at column m (on directed edges).
    Also returns per-marker neighbor-positive fraction mean.
    """
    X = g.x.detach().cpu().numpy()
    u, v = g.edge_index
    u = u.detach().cpu().numpy(); v = v.detach().cpu().numpy()
    M = X.shape[1]
    same_frac = np.zeros(M, dtype=np.float64)
    nb_pos_frac = np.zeros(M, dtype=np.float64)

    for m in range(M):
        Xu = X[u, m]; Xv = X[v, m]
        same = (Xu == Xv)
        same_frac[m] = same.mean() if same.size else np.nan

        # neighbor-positive fraction across nodes (mean over nodes of mean neighbor value)
        # compute mean of X[:,m] among neighbors per destination node
        # aggregation:
        sums = np.zeros(X.shape[0], dtype=np.float64)
        cnts = np.zeros(X.shape[0], dtype=np.float64)
        np.add.at(sums, v, Xu)
        np.add.at(cnts, v, 1.0)
        with np.errstate(divide='ignore', invalid='ignore'):
            frac = np.where(cnts > 0, sums / cnts, np.nan)
        nb_pos_frac[m] = np.nanmean(frac)
    return {"same_frac": same_frac, "nb_pos_frac_mean": nb_pos_frac}

def two_hop_reachability_fraction(g):
    """
    Fraction of nodes reachable within 2 steps (including 1-step) from each node,
    averaged over nodes (proxy for how quickly information can mix).
    """
    N = g.x.size(0)
    u, v = g.edge_index
    nbrs = [[] for _ in range(N)]
    for a, b in zip(u.tolist(), v.tolist()):
        nbrs[a].append(b); nbrs[b].append(a)
    cover = []
    for i in range(N):
        one = set(nbrs[i])
        two = set()
        for j in one:
            two.update(nbrs[j])
        reach = set([i]) | one | two
        cover.append(len(reach) / N)
    return float(np.mean(cover))

def moran_like_I_y(g):
    """
    Moran-like autocorrelation of target y along edges.
    Positive -> neighbors have similar y.
    """
    y = g.y.detach().cpu()
    y = (y - y.mean())
    denom = (y**2).sum() + 1e-12
    u, v = g.edge_index
    W = u.numel()
    I = ( (y[u]*y[v]).sum() * g.x.size(0) / (denom * max(W,1)) ).item()
    return float(I)

# ---------------------------
# Model-level ablations (ensemble)
# ---------------------------

@torch.no_grad()
def ablation_edges_and_shuffle(model, graphs, device=None, batch_size=8):
    """
    Compare predictions under:
      (A) normal graph,
      (B) no edges,
      (C) shuffled node features (permute rows of x per graph).
    Returns dict of global MAE / MSE for A,B,C.
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device).eval()

    def eval_graphs(modifier=None):
        maes, mses, n_all = [], [], 0
        for g in graphs:
            gg = type(g)()
            gg.x = g.x.clone()
            gg.y = g.y.clone()
            gg.edge_index = g.edge_index.clone()
            if modifier is not None:
                modifier(gg)
            gg = gg.to(device)
            yhat, _ = model(gg.x, gg.edge_index)
            res = (yhat - gg.y).detach().cpu().numpy()
            maes.append(np.abs(res).sum())
            mses.append((res**2).sum())
            n_all += res.size
        return float(sum(maes)/n_all), float(sum(mses)/n_all)

    mae_A, mse_A = eval_graphs(None)
    mae_B, mse_B = eval_graphs(lambda gg: setattr(gg, "edge_index", torch.empty((2,0), dtype=torch.long, device=gg.edge_index.device)))
    def _shuffle(gg):
        idx = torch.randperm(gg.x.size(0))
        gg.x = gg.x[idx]
    mae_C, mse_C = eval_graphs(_shuffle)

    return {
        "normal": {"MAE": mae_A, "MSE": mse_A},
        "no_edges": {"MAE": mae_B, "MSE": mse_B},
        "shuffled_x": {"MAE": mae_C, "MSE": mse_C},
    }

# ---------------------------
# Full audit entry points
# ---------------------------

def audit_one_graph(g, marker_names=None, verbose=True):
    """
    Run integrity & signal checks on a single graph Data.
    """
    rep = {}
    rep["edge_integrity"] = check_edge_index_integrity(g)
    rep["degree"] = degree_stats(g)
    comp, comp_stats = connected_components(g)
    rep["components"] = comp_stats
    rep["two_hop_cover_frac"] = two_hop_reachability_fraction(g)
    rep["moran_I_y"] = moran_like_I_y(g)

    if marker_names is not None and len(marker_names) != g.x.size(1):
        rep["marker_alignment_error"] = f"marker_names={len(marker_names)} vs x.shape[1]={g.x.size(1)}"
    else:
        rep["marker_alignment_error"] = None

    homo = homophily_by_marker(g)
    rep["homophily_same_frac_mean"] = float(np.nanmean(homo["same_frac"]))
    rep["nb_pos_frac_mean_mean"] = float(np.nanmean(homo["nb_pos_frac_mean"]))

    if verbose:
        e = rep["edge_integrity"]
        d = rep["degree"]
        c = rep["components"]
        print(f"[Graph audit]")
        print(f"- nodes={e['num_nodes']} edges={e['num_edges']} | "
              f"out_of_range={e['out_of_range']} self_loops={e['self_loops']} dup_dir={e['duplicate_directed']} "
              f"undirected_symmetry={e['undirected_symmetry_ratio']:.3f}")
        print(f"- deg0_frac={d['deg0_frac']:.3f} deg_mean={d['deg_mean']:.2f} p90={d['deg_p90']:.1f} max={d['deg_max']:.0f}")
        print(f"- components={c['num_components']} (small<=5: {c['num_small_<=5']}) largest={c['largest']}")
        print(f"- two_hop_cover_frac≈{rep['two_hop_cover_frac']:.3f} | Moran-I(y)≈{rep['moran_I_y']:.3f}")
        if rep["marker_alignment_error"]:
            print("! Marker name alignment error:", rep["marker_alignment_error"])
        print(f"- homophily same_frac(mean across markers)≈{rep['homophily_same_frac_mean']:.3f}")
    return rep

def audit_dataset(graphs, sample_k=5, marker_names=None, seed=0):
    """
    Sample a few graphs and run `audit_one_graph`. Also return global degree & component stats.
    """
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(graphs), size=min(sample_k, len(graphs)), replace=False)
    reps = []
    for i in idx:
        print(f"\n=== Graph #{i} ===")
        reps.append(audit_one_graph(graphs[i], marker_names=marker_names, verbose=True))
    return reps
