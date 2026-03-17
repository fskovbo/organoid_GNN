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
    """
    def __init__(
        self,
        n_markers: int,
        hidden_dim: int = 64,
        num_layers: int = 3,
        dropout: float = 0.2,
        residual: bool = True,
        norm: str = 'layer',   # 'layer' | 'batch' | None
        log_var_clamp: tuple[float, float] = (-10.0, 10.0),  # stability
    ):
        super().__init__()
        assert num_layers >= 1, "num_layers must be >= 1"

        self.n_markers = n_markers
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.residual = residual
        self.log_var_clamp = log_var_clamp

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        in_dim = n_markers
        for _ in range(num_layers):
            self.convs.append(SAGEConv(in_dim, hidden_dim, aggr='mean'))

            if norm == 'layer':
                self.norms.append(nn.LayerNorm(hidden_dim))
            elif norm == 'batch':
                self.norms.append(nn.BatchNorm1d(hidden_dim))
            else:
                self.norms.append(nn.Identity())

            in_dim = hidden_dim

        self.input_proj = None
        if residual and n_markers != hidden_dim:
            self.input_proj = nn.Linear(n_markers, hidden_dim, bias=False)

        # 2 outputs: mean and log-variance
        self.head = nn.Linear(hidden_dim, 2)

    def forward(self, x, edge_index):
        h = x
        h_in0 = x

        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            h_new = conv(h, edge_index)
            h_new = norm(h_new)
            h_new = F.relu(h_new, inplace=False)

            if self.residual:
                if i == 0:
                    if self.input_proj is not None:
                        h_new = h_new + self.input_proj(h_in0)
                    else:
                        if h_in0.shape[1] == h_new.shape[1]:
                            h_new = h_new + h_in0
                else:
                    if h.shape[1] == h_new.shape[1]:
                        h_new = h_new + h

            if self.dropout > 0:
                h_new = F.dropout(h_new, p=self.dropout, training=self.training)

            h = h_new

        out = self.head(h)                  # (N, 2)
        mu = out[:, 0].contiguous()         # (N,)
        log_var = out[:, 1].contiguous()    # (N,)

        # clamp for numerical stability (prevents exp overflow/underflow)
        lo, hi = self.log_var_clamp
        log_var = torch.clamp(log_var, lo, hi)

        return (mu, log_var), h

    def num_parameters(self, trainable_only: bool = True) -> int:
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())
