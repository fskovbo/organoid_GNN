import torch
import torch.nn as nn


# ---------------------------------------------------------------------
# Graph utilities
# ---------------------------------------------------------------------

def _build_adj_list(edge_index: torch.Tensor, num_nodes: int) -> list[list[int]]:
    edge_index = edge_index.detach().cpu()
    src = edge_index[0].tolist()
    dst = edge_index[1].tolist()

    adj = [[] for _ in range(num_nodes)]
    for u, v in zip(src, dst):
        if v not in adj[u]:
            adj[u].append(v)
        if u not in adj[v]:
            adj[v].append(u)
    return adj


def _ring_nodes_for_center(adj: list[list[int]], center: int, k_hops: int) -> list[list[int]]:
    visited = {center}
    frontier = {center}
    rings = [[center]]

    for _ in range(1, k_hops + 1):
        nxt = set()
        for u in frontier:
            for v in adj[u]:
                if v not in visited:
                    nxt.add(v)

        rings.append(sorted(nxt))
        visited |= nxt
        frontier = nxt

        if not frontier:
            while len(rings) < k_hops + 1:
                rings.append([])
            break

    return rings


# ---------------------------------------------------------------------
# Feature construction
# ---------------------------------------------------------------------

def _compute_ring_fraction_features(x, edge_index, k_hops):
    device = x.device
    N, M = x.shape

    adj = _build_adj_list(edge_index, N)

    out = torch.zeros((N, (k_hops + 1) * M), dtype=x.dtype, device=device)

    for c in range(N):
        rings = _ring_nodes_for_center(adj, c, k_hops)

        feats = []
        for nodes in rings:
            if len(nodes) == 0:
                feats.append(torch.zeros(M, dtype=x.dtype, device=device))
            else:
                idx = torch.tensor(nodes, dtype=torch.long, device=device)
                feats.append(x[idx].float().mean(dim=0))

        out[c] = torch.cat(feats, dim=0)

    return out


def _compute_pooled_khop_fraction_features(x, edge_index, k_hops, include_center=True):
    device = x.device
    N, M = x.shape

    adj = _build_adj_list(edge_index, N)
    out = torch.zeros((N, M), dtype=x.dtype, device=device)

    for c in range(N):
        rings = _ring_nodes_for_center(adj, c, k_hops)

        if include_center:
            nodes = [u for ring in rings for u in ring]
        else:
            nodes = [u for ring in rings[1:] for u in ring]

        if len(nodes) == 0:
            continue

        idx = torch.tensor(sorted(set(nodes)), dtype=torch.long, device=device)
        out[c] = x[idx].float().mean(dim=0)

    return out


# ---------------------------------------------------------------------
# Shared MLP encoder
# ---------------------------------------------------------------------

class _MLPHead(nn.Module):
    def __init__(self, in_dim, hidden_dim, dropout, norm):
        super().__init__()

        if norm == "layer":
            norm_layer = nn.LayerNorm(hidden_dim)
        elif norm == "batch":
            norm_layer = nn.BatchNorm1d(hidden_dim)
        else:
            norm_layer = nn.Identity()

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            norm_layer,
            nn.ReLU(),
            nn.Dropout(dropout if dropout > 0 else 0.0),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout if dropout > 0 else 0.0),
        )

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------
# Ring-fraction model (radial information preserved)
# ---------------------------------------------------------------------

class RingFractionMLP(nn.Module):

    def __init__(
        self,
        n_markers,
        k_hops=3,
        hidden_dim=64,
        dropout=0.2,
        norm="layer",
        feature_attr="x_ring",
        log_scale2_clamp=(-10.0, 10.0),
    ):
        super().__init__()

        self.k_hops = k_hops
        self.feature_attr = feature_attr
        self.log_scale2_clamp = log_scale2_clamp

        in_dim = (k_hops + 1) * n_markers

        self.encoder = _MLPHead(in_dim, hidden_dim, dropout, norm)
        self.head = nn.Linear(hidden_dim, 2)

    def forward(self, x, edge_index, data=None):
        if data is not None and hasattr(data, self.feature_attr):
            ring_x = getattr(data, self.feature_attr)
        else:
            ring_x = _compute_ring_fraction_features(x, edge_index, self.k_hops)

        h = self.encoder(ring_x)
        out = self.head(h)

        mu = out[:, 0]
        log_scale2 = out[:, 1]

        lo, hi = self.log_scale2_clamp
        log_scale2 = torch.clamp(log_scale2, lo, hi)

        return (mu, log_scale2), h


# ---------------------------------------------------------------------
# Pooled k-hop model (composition only)
# ---------------------------------------------------------------------

class PooledKHopMLP(nn.Module):

    def __init__(
        self,
        n_markers,
        k_hops=3,
        hidden_dim=64,
        dropout=0.2,
        norm="layer",
        feature_attr="x_pool",
        include_center=True,
        log_scale2_clamp=(-10.0, 10.0),
    ):
        super().__init__()

        self.k_hops = k_hops
        self.feature_attr = feature_attr
        self.include_center = include_center
        self.log_scale2_clamp = log_scale2_clamp

        self.encoder = _MLPHead(n_markers, hidden_dim, dropout, norm)
        self.head = nn.Linear(hidden_dim, 2)

    def forward(self, x, edge_index, data=None):
        if data is not None and hasattr(data, self.feature_attr):
            pooled_x = getattr(data, self.feature_attr)
        else:
            pooled_x = _compute_pooled_khop_fraction_features(
                x,
                edge_index,
                self.k_hops,
                include_center=self.include_center,
            )

        h = self.encoder(pooled_x)
        out = self.head(h)

        mu = out[:, 0]
        log_scale2 = out[:, 1]

        lo, hi = self.log_scale2_clamp
        log_scale2 = torch.clamp(log_scale2, lo, hi)

        return (mu, log_scale2), h