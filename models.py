
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import degree
        

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
    

    from torch_geometric.nn import GINConv

from torch_geometric.nn import GINConv

class GINCurvature(nn.Module):
    def __init__(self, n_markers, hidden_dim=128, num_layers=3, dropout=0.2):
        super().__init__()
        self.dropout = dropout

        mlps = []
        in_dim = n_markers
        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            mlps.append(GINConv(mlp))
            in_dim = hidden_dim

        self.layers = nn.ModuleList(mlps)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x, edge_index):
        for conv in self.layers:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        out = self.head(x).squeeze(-1)
        return out, None  # to match your training API