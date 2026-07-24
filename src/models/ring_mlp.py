import torch
import torch.nn as nn

from src.models.gnn import _FinalHead


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


def _concat_global_features(h, data, global_dim, global_attr):
    if global_dim <= 0:
        return h

    gfeat_node = _broadcast_global_features(data, global_attr)
    if gfeat_node.shape[1] != global_dim:
        raise ValueError(
            f"Expected {global_dim} global features, got {gfeat_node.shape[1]}"
        )
    return torch.cat([h, gfeat_node], dim=1)




def _broadcast_global_features(data, global_attr: str = "global_feat"):
    """Broadcast graph-level features to nodes using the PyG batch vector."""
    if data is None:
        raise ValueError(
            "Global features were requested, but no `data` object was passed to the model."
        )

    if not hasattr(data, global_attr):
        raise AttributeError(
            f"Batch has no attribute {global_attr!r}. Attach it to each graph before training."
        )

    if not hasattr(data, "batch"):
        raise AttributeError("Batch has no 'batch' attribute")

    gfeat = getattr(data, global_attr)
    if gfeat.ndim == 1:
        gfeat = gfeat.unsqueeze(-1)

    if gfeat.ndim != 2:
        raise ValueError(
            f"Expected {global_attr!r} to be 2D after batching, got shape {tuple(gfeat.shape)}"
        )

    return gfeat[data.batch]


# ---------------------------------------------------------------------
# Gaussian output helpers
# ---------------------------------------------------------------------

def _normalize_covariance_mode(mode: str) -> str:
    mode = str(mode).lower()
    if mode not in {"diagonal", "full"}:
        raise ValueError("covariance_mode must be 'diagonal' or 'full'")
    return mode


def _gaussian_head_output_dim(target_dim: int = 1, covariance_mode: str = "diagonal") -> int:
    target_dim = int(target_dim)
    if target_dim < 1:
        raise ValueError("target_dim must be >= 1")
    covariance_mode = _normalize_covariance_mode(covariance_mode)
    return 2 * target_dim if covariance_mode == "diagonal" else target_dim + target_dim * (target_dim + 1) // 2


def _finalize_distribution_outputs(out, log_scale2_clamp, target_dim: int = 1, covariance_mode: str = "diagonal"):
    target_dim = int(target_dim)
    covariance_mode = _normalize_covariance_mode(covariance_mode)
    mu = out[:, :target_dim].contiguous()
    lo, hi = log_scale2_clamp
    if covariance_mode == "diagonal":
        log_scale2 = torch.clamp(out[:, target_dim:2 * target_dim].contiguous(), lo, hi)
        if target_dim == 1:
            return mu.reshape(-1), log_scale2.reshape(-1)
        return mu, log_scale2
    raw = out[:, target_dim:]
    L = out.new_zeros((out.shape[0], target_dim, target_dim))
    tril_i, tril_j = torch.tril_indices(target_dim, target_dim, device=out.device)
    L[:, tril_i, tril_j] = raw
    diag = torch.arange(target_dim, device=out.device)
    L[:, diag, diag] = torch.exp(0.5 * torch.clamp(L[:, diag, diag], lo, hi)).clamp_min(1e-6)
    return mu, L


def distribution_to_diagonal_outputs(mu, scale):
    if scale.ndim == 3:
        cov = scale @ scale.transpose(-1, -2)
        scale = torch.log(torch.diagonal(cov, dim1=-2, dim2=-1).clamp_min(1e-12))
    if mu.ndim == 2 and mu.shape[1] == 1:
        mu = mu.reshape(-1)
    if scale.ndim == 2 and scale.shape[1] == 1:
        scale = scale.reshape(-1)
    return mu, scale

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
        global_dim=0,
        global_attr="global_feat",
        target_dim=1,
        covariance_mode="diagonal",
    ):
        super().__init__()

        self.n_markers = n_markers
        self.k_hops = k_hops
        self.feature_attr = feature_attr
        self.log_scale2_clamp = log_scale2_clamp
        self.global_dim = global_dim
        self.global_attr = global_attr
        self.target_dim = int(target_dim)
        self.covariance_mode = _normalize_covariance_mode(covariance_mode)

        in_dim = (k_hops + 1) * n_markers

        self.encoder = _MLPHead(in_dim, hidden_dim, dropout, norm)
        self.head = _FinalHead(
            hidden_dim + global_dim,
            hidden_dim,
            dropout,
            target_dim=self.target_dim,
            covariance_mode=self.covariance_mode,
        )

    def forward(self, x, edge_index, data=None):
        if data is not None and hasattr(data, self.feature_attr):
            ring_x_full = getattr(data, self.feature_attr)
            n_keep = (self.k_hops + 1) * self.n_markers
            ring_x = ring_x_full[:, :n_keep]
        else:
            ring_x, _ = _compute_ring_fraction_features_and_sizes(x, edge_index, self.k_hops)

        h = self.encoder(ring_x)

        h_out = _concat_global_features(h, data, self.global_dim, self.global_attr)

        out = self.head(h_out)

        mu, log_scale2 = _finalize_distribution_outputs(
            out,
            self.log_scale2_clamp,
            self.target_dim,
            self.covariance_mode,
        )

        return (mu, log_scale2), h

    def num_parameters(self, trainable_only=True):
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())


# ---------------------------------------------------------------------
# Ring-fraction + ring-size model (radial marker composition and topology)
# ---------------------------------------------------------------------

class RingFractionSizeMLP(nn.Module):
    """
    MLP that receives exact-hop marker fractions and exact-hop ring sizes.

    This control keeps the radial marker-composition input of ``RingFractionMLP``
    but also exposes the number of cells in each hop shell, making it a closer
    match to the count-sensitive signal available to a one-layer GIN.
    """

    def __init__(
        self,
        n_markers,
        k_hops=3,
        hidden_dim=64,
        dropout=0.2,
        norm="layer",
        feature_attr="x_ring",
        ring_sizes_attr="ring_sizes",
        log_scale2_clamp=(-10.0, 10.0),
        global_dim=0,
        global_attr="global_feat",
        target_dim=1,
        covariance_mode="diagonal",
    ):
        super().__init__()

        self.n_markers = n_markers
        self.k_hops = k_hops
        self.feature_attr = feature_attr
        self.ring_sizes_attr = ring_sizes_attr
        self.log_scale2_clamp = log_scale2_clamp
        self.global_dim = global_dim
        self.global_attr = global_attr
        self.target_dim = int(target_dim)
        self.covariance_mode = _normalize_covariance_mode(covariance_mode)

        in_dim = (k_hops + 1) * n_markers + (k_hops + 1)

        self.encoder = _MLPHead(in_dim, hidden_dim, dropout, norm)
        self.head = _FinalHead(
            hidden_dim + global_dim,
            hidden_dim,
            dropout,
            target_dim=self.target_dim,
            covariance_mode=self.covariance_mode,
        )

    def forward(self, x, edge_index, data=None):
        if (
            data is not None
            and hasattr(data, self.feature_attr)
            and hasattr(data, self.ring_sizes_attr)
        ):
            ring_x_full = getattr(data, self.feature_attr)
            ring_sizes_full = getattr(data, self.ring_sizes_attr)
            n_keep = (self.k_hops + 1) * self.n_markers
            ring_x = ring_x_full[:, :n_keep].float()
            ring_sizes = ring_sizes_full[:, : self.k_hops + 1].float()
        else:
            ring_x, ring_sizes = _compute_ring_fraction_features_and_sizes(
                x,
                edge_index,
                self.k_hops,
            )
            ring_x = ring_x.float()
            ring_sizes = ring_sizes.float()

        feats = torch.cat([ring_x, ring_sizes], dim=1)
        h = self.encoder(feats)
        h_out = _concat_global_features(h, data, self.global_dim, self.global_attr)
        out = self.head(h_out)

        mu, log_scale2 = _finalize_distribution_outputs(
            out,
            self.log_scale2_clamp,
            self.target_dim,
            self.covariance_mode,
        )

        return (mu, log_scale2), h

    def num_parameters(self, trainable_only=True):
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())


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
        global_dim=0,
        global_attr="global_feat",
        target_dim=1,
        covariance_mode="diagonal",
    ):
        super().__init__()

        self.n_markers = n_markers
        self.k_hops = k_hops
        self.feature_attr = feature_attr
        self.include_center = include_center
        self.log_scale2_clamp = log_scale2_clamp
        self.global_dim = global_dim
        self.global_attr = global_attr
        self.target_dim = int(target_dim)
        self.covariance_mode = _normalize_covariance_mode(covariance_mode)

        self.encoder = _MLPHead(n_markers, hidden_dim, dropout, norm)
        self.head = _FinalHead(
            hidden_dim + global_dim,
            hidden_dim,
            dropout,
            target_dim=self.target_dim,
            covariance_mode=self.covariance_mode,
        )

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

        h_out = _concat_global_features(h, data, self.global_dim, self.global_attr)

        out = self.head(h_out)

        mu, log_scale2 = _finalize_distribution_outputs(
            out,
            self.log_scale2_clamp,
            self.target_dim,
            self.covariance_mode,
        )

        return (mu, log_scale2), h

    def num_parameters(self, trainable_only=True):
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())
    


# ---------------------------------------------------------------------
# Ring-size model (omits marker information)
# ---------------------------------------------------------------------

class RingSizeMLP(nn.Module):
    """
    MLP that uses:
      - optionally the center-node fate markers
      - ring sizes only (number of nodes in each exact hop ring)

    It does NOT use neighborhood marker composition.

    Input per node:
      if use_center_markers=True:
          [x_center_markers, ring_sizes_0, ring_sizes_1, ..., ring_sizes_k]
      else:
          [ring_sizes_0, ring_sizes_1, ..., ring_sizes_k]

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
        use_center_markers=True,
        log_scale2_clamp=(-10.0, 10.0),
        global_dim=0,
        global_attr="global_feat",
        target_dim=1,
        covariance_mode="diagonal",
    ):
        super().__init__()

        self.n_markers = n_markers
        self.k_hops = k_hops
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.norm = norm
        self.ring_sizes_attr = ring_sizes_attr
        self.use_center_markers = use_center_markers
        self.log_scale2_clamp = log_scale2_clamp
        self.global_dim = global_dim
        self.global_attr = global_attr
        self.target_dim = int(target_dim)
        self.covariance_mode = _normalize_covariance_mode(covariance_mode)

        in_dim = (k_hops + 1) + (n_markers if use_center_markers else 0)

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

        self.head = _FinalHead(
            hidden_dim + global_dim,
            hidden_dim,
            dropout,
            target_dim=self.target_dim,
            covariance_mode=self.covariance_mode,
        )

    def forward(self, x, edge_index, data=None):
        # use precomputed ring sizes if available, otherwise recompute
        if data is not None and hasattr(data, self.ring_sizes_attr):
            ring_sizes_full = getattr(data, self.ring_sizes_attr)
            ring_sizes = ring_sizes_full[:, : self.k_hops + 1].float()
        else:
            _, ring_sizes = _compute_ring_fraction_features_and_sizes(x, edge_index, self.k_hops)
            ring_sizes = ring_sizes.float()

        if self.use_center_markers:
            x_center = x.float()
            feats = torch.cat([x_center, ring_sizes], dim=1)
        else:
            feats = ring_sizes

        h = self.encoder(feats)

        h_out = _concat_global_features(h, data, self.global_dim, self.global_attr)

        out = self.head(h_out)

        mu, log_scale2 = _finalize_distribution_outputs(
            out,
            self.log_scale2_clamp,
            self.target_dim,
            self.covariance_mode,
        )

        return (mu, log_scale2), h

    def num_parameters(self, trainable_only=True):
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())
