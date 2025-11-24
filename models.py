
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import degree

# -----------------------------------------------------------------------------
# Explicit "feature library"
# -----------------------------------------------------------------------------

def neighbor_mean(features: torch.Tensor, edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """
    Compute a 1-hop neighbor MEAN for node features (per feature column).
    For each directed edge j->i, we add features[j] to a running sum at i, then divide by the in-degree of i.
    Args:
        features: (N, F) float tensor with node features (here: binary marker calls)
        edge_index: (2, E) long tensor; edge_index[0] = src (j), edge_index[1] = dst (i)
        num_nodes: N, number of nodes
    Returns:
        (N, F) tensor of neighbor-mean features
    """
    row, col = edge_index  # edges j->i (src=row, dst=col)
    sum_aggr = torch.zeros_like(features)
    # Scatter-add features from src nodes into dst slots
    sum_aggr.index_add_(0, col, features[row])
    # Count neighbors per node (in-degree); clamp to avoid div-by-zero
    deg = degree(col, num_nodes=num_nodes, dtype=features.dtype).clamp_min(1.0).view(-1, 1)
    return sum_aggr / deg

def build_explicit_features(x_bin: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    """
    Build interpretable, hand-crafted features per node:
      - self markers (M)
      - 1-hop neighbor fractions (M)  [= neighbor means on 0/1 inputs]
      - elementwise self × neighbor_fraction (M)
    Result shape: (N, 3M)
    """
    N, M = x_bin.shape
    frac1 = neighbor_mean(x_bin, edge_index, num_nodes=N)   # (N, M)
    cross = x_bin * frac1                                   # (N, M)
    return torch.cat([x_bin, frac1, cross], dim=1)          # (N, 3M)

# -----------------------------------------------------------------------------
# A tiny GraphSAGE branch to learn short-range "motifs"
# -----------------------------------------------------------------------------

class SmallSAGE(nn.Module):
    """
    Minimal GraphSAGE encoder operating ONLY on the binary marker inputs.
    Keep this small and regularized; the linear head decides how much to trust it.
    """
    def __init__(self, in_dim: int, hidden: int = 32, out_dim: int = 16, dropout: float = 0.2):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden, aggr='mean')
        self.conv2 = SAGEConv(hidden, out_dim, aggr='mean')
        self.dropout = dropout

    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.conv2(h, edge_index)
        h = F.relu(h)
        return h  # (N, out_dim)

# -----------------------------------------------------------------------------
# Full model = Explicit features  ⊕  SmallSAGE  →  Sparse Linear Head
# -----------------------------------------------------------------------------

class LocalCurvatureGNN(nn.Module):
    """
    Final model:
      - Branch A (explicit):  3M features (self, neighbor fraction, self×neighbor)
      - Branch B (GNN):       SmallSAGE embedding of size Dg
      - Head:                 Linear(3M + Dg → 1)

    We expose utilities to:
      * return masks for [explicit | gnn] slices of head weights
      * apply L1 penalties selectively (e.g., only explicit coefficients)
      * freeze/unfreeze model parts for the two-phase L1 procedure
    """
    def __init__(self, n_markers: int, gnn_hidden: int = 32, gnn_out: int = 16):
        super().__init__()
        self.n_markers = n_markers
        self.explicit_dim = 3 * n_markers
        self.gnn_out = gnn_out

        self.gnn = SmallSAGE(in_dim=n_markers, hidden=gnn_hidden, out_dim=gnn_out)
        self.head = nn.Linear(self.explicit_dim + gnn_out, 1)

        # Pre-compute boolean-like masks as buffers (on the same device as the model)
        self.register_buffer("mask_explicit", torch.zeros(self.explicit_dim + gnn_out))
        self.register_buffer("mask_gnn", torch.zeros(self.explicit_dim + gnn_out))
        self.mask_explicit[: self.explicit_dim] = 1.0
        if gnn_out > 0:
            self.mask_gnn[self.explicit_dim :] = 1.0

    # ---- Forward ----
    def forward(self, x_bin, edge_index):
        # Explicit features (no learned parameters = fully auditable)
        feats = build_explicit_features(x_bin, edge_index)         # (N, 3M)

        # Learned GNN embedding (optional if gnn_out=0)
        if self.gnn_out > 0:
            emb = self.gnn(x_bin, edge_index)                      # (N, Dg)
            z = torch.cat([feats, emb], dim=1)                     # (N, 3M + Dg)
        else:
            z = feats                                              # (N, 3M)
        yhat = self.head(z).squeeze(-1)                            # (N,)
        return yhat, z

    # ---- Penalties ----
    def l1_penalty(self, lambda_explicit: float = 0.0, lambda_gnn: float = 0.0) -> torch.Tensor:
        """
        Generic L1 on head weights for both slices. If you want to penalize only explicit
        coefficients (phase-2 sparse retrain), set lambda_gnn = 0.
        """
        w = self.head.weight.squeeze(0)  # shape: (3M + Dg,)
        l1 = 0.0
        if lambda_explicit > 0.0:
            l1 = l1 + lambda_explicit * (torch.abs(w) * self.mask_explicit).sum()
        if self.gnn_out > 0 and lambda_gnn > 0.0:
            l1 = l1 + lambda_gnn * (torch.abs(w) * self.mask_gnn).sum()
        return l1 if isinstance(l1, torch.Tensor) else torch.tensor(0.0, device=w.device)

    # ---- Freezing helpers for two-phase L1 training ----
    def freeze_all_but_head(self):
        """Freeze explicit builder (none to freeze), GNN encoder, and bias; only head weights/bias remain trainable."""
        # GNN params
        for p in self.gnn.parameters():
            p.requires_grad = False
        # Head params stay trainable by default
        for p in self.head.parameters():
            p.requires_grad = True

    def unfreeze_all(self):
        """Make the whole model trainable again."""
        for p in self.parameters():
            p.requires_grad = True

    # ---- Introspection ----
    def explicit_slices(self):
        """
        Return index slices for the three explicit families (useful for reporting):
          self:   [0:M)
          nb:     [M:2M)
          cross:  [2M:3M)
        """
        M = self.n_markers
        return slice(0, M), slice(M, 2*M), slice(2*M, 3*M)

    def unpack_head_coefficients(self):
        """
        Return a dict with numpy arrays of head coefficients split into:
          - 'self', 'nb', 'cross', and 'gnn' (if present).
        """
        with torch.no_grad():
            w = self.head.weight.squeeze(0).detach().cpu()  # (3M + Dg,)
            M = self.n_markers
            coeffs = {
                "self":  w[:M],
                "nb":    w[M:2*M],
                "cross": w[2*M:3*M],
            }
            if self.gnn_out > 0:
                coeffs["gnn"] = w[3*M:]
            # Convert to numpy for easy logging
            return {k: v.numpy() for k, v in coeffs.items()}
        

class PureSAGECurvature(nn.Module):
    """
    Pure GNN model (no explicit features). Stacks GraphSAGE layers to capture
    multi-hop context and predicts a scalar per node (curvature proxy).

    Args
    ----
    n_markers : int
        Input feature dimension = number of markers (binary channels).
    hidden_dim : int
        Width of hidden node embeddings.
    num_layers : int
        Number of SAGEConv layers (>= 2 recommended). Receptive field is ~num_layers hops.
    dropout : float
        Dropout probability applied after each hidden activation (except output).
    residual : bool
        If True, add skip connections: h_{l+1} += proj(h_l) (with shape matching).
    norm : {'layer', 'batch', None}
        Optional normalization after each conv ('layer' = LayerNorm, 'batch' = BatchNorm).

    Notes
    -----
    - Uses mean aggregation (SAGEConv with aggr='mean').
    - Returns (yhat, h) where yhat is (N,) and h is final embedding (N, hidden_dim).
    - If you want *more* non-locality without very deep stacks, try increasing hidden_dim,
      or enable residuals and light dropout to reduce oversmoothing.
    """
    def __init__(
        self,
        n_markers: int,
        hidden_dim: int = 64,
        num_layers: int = 3,
        dropout: float = 0.2,
        residual: bool = True,
        norm: str = 'layer',   # 'layer' | 'batch' | None
    ):
        super().__init__()
        assert num_layers >= 1, "num_layers must be >= 1"

        self.n_markers = n_markers
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.residual = residual

        # Build SAGE stack
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        in_dim = n_markers
        for layer in range(num_layers):
            conv = SAGEConv(in_dim, hidden_dim, aggr='mean')
            self.convs.append(conv)

            if norm == 'layer':
                self.norms.append(nn.LayerNorm(hidden_dim))
            elif norm == 'batch':
                # BatchNorm1d expects (N, C)
                self.norms.append(nn.BatchNorm1d(hidden_dim))
            else:
                self.norms.append(nn.Identity())

            in_dim = hidden_dim

        # If residuals and the first input dim != hidden_dim, add a projection
        self.input_proj = None
        if residual and n_markers != hidden_dim:
            self.input_proj = nn.Linear(n_markers, hidden_dim, bias=False)

        # Final linear head to scalar
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x, edge_index):
        """
        x: (N, n_markers) float tensor
        edge_index: (2, E) long tensor
        returns:
          yhat: (N,) predictions
          h:    (N, hidden_dim) final node embeddings
        """
        h = x
        h_in0 = x  # stash for first residual if sizes differ

        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            h_new = conv(h, edge_index)
            h_new = norm(h_new)
            h_new = F.relu(h_new, inplace=False)

            if self.residual:
                if i == 0:
                    # First block: either identity or project input to hidden_dim
                    if self.input_proj is not None:
                        h_new = h_new + self.input_proj(h_in0)
                    else:
                        # Only add if shapes match
                        if h_in0.shape[1] == h_new.shape[1]:
                            h_new = h_new + h_in0
                else:
                    # Following blocks: identity residual
                    if h.shape[1] == h_new.shape[1]:
                        h_new = h_new + h

            if self.dropout > 0:
                h_new = F.dropout(h_new, p=self.dropout, training=self.training)

            h = h_new

        yhat = self.head(h).squeeze(-1)
        return yhat, h

    # Small utility to count parameters (optional)
    def num_parameters(self, trainable_only: bool = True) -> int:
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())



class SAGEForMarkers(nn.Module):
    """
    GraphSAGE stack that takes (N, 1) curvature features and predicts K binary markers per node.
    Output are *logits*; use BCEWithLogitsLoss.

    Args
    ----
    in_dim     : usually 1 (curvature channel); you can add more channels later
    out_dim    : number of markers to predict (K)
    hidden_dim : width of hidden embeddings
    num_layers : number of SAGEConv layers
    dropout    : dropout after activations
    residual   : residual connections between layers
    norm       : {'layer','batch',None} normalization per hidden layer
    """
    def __init__(self, in_dim=1, out_dim=1, hidden_dim=64, num_layers=3,
                 dropout=0.1, residual=True, norm='layer'):
        super().__init__()
        assert num_layers >= 1
        self.dropout = dropout
        self.residual = residual

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        dim_in = in_dim
        for l in range(num_layers):
            self.convs.append(SAGEConv(dim_in, hidden_dim, aggr='mean'))
            if norm == 'layer':
                self.norms.append(nn.LayerNorm(hidden_dim))
            elif norm == 'batch':
                self.norms.append(nn.BatchNorm1d(hidden_dim))
            else:
                self.norms.append(nn.Identity())
            dim_in = hidden_dim

        self.input_proj = None
        if residual and in_dim != hidden_dim:
            self.input_proj = nn.Linear(in_dim, hidden_dim, bias=False)

        self.head = nn.Linear(hidden_dim, out_dim)  # logits per node

    def forward(self, x, edge_index):
        h = x
        h0 = x
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            h_new = conv(h, edge_index)
            h_new = norm(h_new)
            h_new = F.relu(h_new, inplace=False)
            if self.residual:
                if i == 0:
                    if self.input_proj is not None:
                        h_new = h_new + self.input_proj(h0)
                    elif h0.shape[1] == h_new.shape[1]:
                        h_new = h_new + h0
                else:
                    if h.shape[1] == h_new.shape[1]:
                        h_new = h_new + h
            if self.dropout > 0:
                h_new = F.dropout(h_new, p=self.dropout, training=self.training)
            h = h_new
        logits = self.head(h)  # (N, K)
        return logits, h