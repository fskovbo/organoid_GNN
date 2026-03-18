import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


class PureSAGECurvature(nn.Module):
    """
    GraphSAGE-based node model for mean/log-variance prediction.

    If num_layers == 0:
        no message passing is used, and predictions depend only on the
        target node's own marker vector via the MLP head.

    If num_layers > 0:
        GraphSAGE layers aggregate neighborhood information before the head.

    Args
    ----
    n_markers : int
        Input feature dimension.
    hidden_dim : int
        Hidden width for GNN / head.
    num_layers : int
        Number of GraphSAGE layers. If 0, no neighborhood information is used.
    dropout : float
        Dropout probability.
    residual : bool
        Whether to use residual connections in the GNN stack.
    norm : {'layer', 'batch', None}
        Normalization after each conv layer.
    log_var_clamp : tuple[float, float]
        Clamp range for predicted log-variance.
    """
    def __init__(
        self,
        n_markers: int,
        hidden_dim: int = 64,
        num_layers: int = 3,
        dropout: float = 0.2,
        residual: bool = True,
        norm: str = "layer",
        log_var_clamp: tuple[float, float] = (-10.0, 10.0),
    ):
        super().__init__()
        assert num_layers >= 0, "num_layers must be >= 0"

        self.n_markers = n_markers
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.residual = residual
        self.log_var_clamp = log_var_clamp

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        if num_layers > 0:
            in_dim = n_markers
            for _ in range(num_layers):
                self.convs.append(SAGEConv(in_dim, hidden_dim, aggr="mean"))

                if norm == "layer":
                    self.norms.append(nn.LayerNorm(hidden_dim))
                elif norm == "batch":
                    self.norms.append(nn.BatchNorm1d(hidden_dim))
                else:
                    self.norms.append(nn.Identity())

                in_dim = hidden_dim

            self.input_proj = None
            if residual and n_markers != hidden_dim:
                self.input_proj = nn.Linear(n_markers, hidden_dim, bias=False)

            head_in_dim = hidden_dim

        else:
            # No message passing: operate directly on node features
            self.input_proj = None
            head_in_dim = n_markers

        # Slightly richer head than a single linear layer
        self.head = nn.Sequential(
            nn.Linear(head_in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout if dropout > 0 else 0.0),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x, edge_index):
        if self.num_layers == 0:
            h = x
        else:
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

        out = self.head(h)               # (N, 2)
        mu = out[:, 0].contiguous()      # (N,)
        log_var = out[:, 1].contiguous() # (N,)

        lo, hi = self.log_var_clamp
        log_var = torch.clamp(log_var, lo, hi)

        return (mu, log_var), h

    def num_parameters(self, trainable_only: bool = True) -> int:
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())