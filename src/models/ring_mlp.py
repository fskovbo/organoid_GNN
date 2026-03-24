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

def _compute_ring_fraction_features_and_sizes(x, edge_index, k_hops):
    device = x.device
    N, M = x.shape

    adj = _build_adj_list(edge_index, N)

    x_ring = torch.zeros((N, (k_hops + 1) * M), dtype=torch.float32, device=device)
    ring_sizes = torch.zeros((N, k_hops + 1), dtype=torch.float32, device=device)

    for c in range(N):
        rings = _ring_nodes_for_center(adj, c, k_hops)

        feats = []
        sizes = []
        for nodes in rings:
            sizes.append(float(len(nodes)))
            if len(nodes) == 0:
                feats.append(torch.zeros(M, dtype=torch.float32, device=device))
            else:
                idx = torch.tensor(nodes, dtype=torch.long, device=device)
                feats.append(x[idx].float().mean(dim=0))

        x_ring[c] = torch.cat(feats, dim=0)
        ring_sizes[c] = torch.tensor(sizes, dtype=torch.float32, device=device)

    return x_ring, ring_sizes


def _pooled_from_ring_features(x_ring, ring_sizes, n_markers, k_hops, include_center=True):
    """
    Exact pooled <= k-hop fractions reconstructed from exact-ring fractions and ring sizes.
    """
    N = x_ring.shape[0]

    start_h = 0 if include_center else 1
    end_h = k_hops + 1

    xr = x_ring[:, : end_h * n_markers].reshape(N, end_h, n_markers)
    rs = ring_sizes[:, :end_h]

    xr = xr[:, start_h:, :]
    rs = rs[:, start_h:].unsqueeze(-1)

    weighted_sum = (xr * rs).sum(dim=1)
    denom = rs.sum(dim=1).clamp_min(1.0)

    return weighted_sum / denom


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

        self.n_markers = n_markers
        self.k_hops = k_hops
        self.feature_attr = feature_attr
        self.log_scale2_clamp = log_scale2_clamp

        in_dim = (k_hops + 1) * n_markers

        self.encoder = _MLPHead(in_dim, hidden_dim, dropout, norm)
        self.head = nn.Linear(hidden_dim, 2)

    def forward(self, x, edge_index, data=None):
        if data is not None and hasattr(data, self.feature_attr):
            ring_x_full = getattr(data, self.feature_attr)
            n_keep = (self.k_hops + 1) * self.n_markers
            ring_x = ring_x_full[:, :n_keep]
        else:
            ring_x, _ = _compute_ring_fraction_features_and_sizes(x, edge_index, self.k_hops)

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
        feature_attr="x_ring",
        include_center=True,
        log_scale2_clamp=(-10.0, 10.0),
    ):
        super().__init__()

        self.n_markers = n_markers
        self.k_hops = k_hops
        self.feature_attr = feature_attr
        self.include_center = include_center
        self.log_scale2_clamp = log_scale2_clamp

        self.encoder = _MLPHead(n_markers, hidden_dim, dropout, norm)
        self.head = nn.Linear(hidden_dim, 2)

    def forward(self, x, edge_index, data=None):
        if data is not None and hasattr(data, self.feature_attr) and hasattr(data, "ring_sizes"):
            x_ring = getattr(data, self.feature_attr)
            ring_sizes = getattr(data, "ring_sizes")
        else:
            x_ring, ring_sizes = _compute_ring_fraction_features_and_sizes(x, edge_index, self.k_hops)

        pooled_x = _pooled_from_ring_features(
            x_ring,
            ring_sizes,
            n_markers=self.n_markers,
            k_hops=self.k_hops,
            include_center=self.include_center,
        )

        h = self.encoder(pooled_x)
        out = self.head(h)

        mu = out[:, 0]
        log_scale2 = out[:, 1]

        lo, hi = self.log_scale2_clamp
        log_scale2 = torch.clamp(log_scale2, lo, hi)

        return (mu, log_scale2), h
    


# ---------------------------------------------------------------------
# Ring-size model (omits marker information)
# ---------------------------------------------------------------------

class RingSizeMLP(nn.Module):
    """
    MLP that uses:
      - center-node fate markers
      - ring sizes only (number of nodes in each exact hop ring)

    It does NOT use neighborhood marker composition.

    Input per node:
      [x_center_markers, ring_sizes_0, ring_sizes_1, ..., ring_sizes_k]

    If `data.ring_sizes` is present, it uses that.
    Otherwise it recomputes ring sizes from (x, edge_index).
    """

    def __init__(
        self,
        n_markers,
        k_hops=3,
        hidden_dim=64,
        dropout=0.2,
        norm="layer",
        ring_sizes_attr="ring_sizes",
        log_scale2_clamp=(-10.0, 10.0),
    ):
        super().__init__()

        self.n_markers = n_markers
        self.k_hops = k_hops
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.norm = norm
        self.ring_sizes_attr = ring_sizes_attr
        self.log_scale2_clamp = log_scale2_clamp

        in_dim = n_markers + (k_hops + 1)

        if norm == "layer":
            norm_layer = nn.LayerNorm(hidden_dim)
        elif norm == "batch":
            norm_layer = nn.BatchNorm1d(hidden_dim)
        else:
            norm_layer = nn.Identity()

        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            norm_layer,
            nn.ReLU(),
            nn.Dropout(dropout if dropout > 0 else 0.0),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout if dropout > 0 else 0.0),
        )

        self.head = nn.Linear(hidden_dim, 2)

    def forward(self, x, edge_index, data=None):
        # center-node marker identity is always taken from x directly
        x_center = x.float()

        # use precomputed ring sizes if available, otherwise recompute
        if data is not None and hasattr(data, self.ring_sizes_attr):
            ring_sizes_full = getattr(data, self.ring_sizes_attr)
            ring_sizes = ring_sizes_full[:, : self.k_hops + 1].float()
        else:
            _, ring_sizes = _compute_ring_fraction_features_and_sizes(x, edge_index, self.k_hops)
            ring_sizes = ring_sizes.float()

        feats = torch.cat([x_center, ring_sizes], dim=1)

        h = self.encoder(feats)
        out = self.head(h)

        mu = out[:, 0]
        log_scale2 = out[:, 1]

        lo, hi = self.log_scale2_clamp
        log_scale2 = torch.clamp(log_scale2, lo, hi)

        return (mu, log_scale2), h

    def num_parameters(self, trainable_only=True):
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())