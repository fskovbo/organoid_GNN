import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import SAGEConv, GINConv, GATConv, JumpingKnowledge


# ---------------------------------------------------------------------
# Generic parameter helpers (works for all models, including ring MLPs)
# ---------------------------------------------------------------------

def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """Count the number of parameters in a model."""
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def parameter_summary(model: nn.Module) -> dict:
    """Return a compact parameter-count summary for a model."""
    return {
        "model_class": type(model).__name__,
        "n_parameters_trainable": count_parameters(model, trainable_only=True),
        "n_parameters_total": count_parameters(model, trainable_only=False),
    }


def print_parameter_summary(model: nn.Module) -> None:
    """Print a compact parameter-count summary for a model."""
    s = parameter_summary(model)
    print(
        f"{s['model_class']}: "
        f"trainable={s['n_parameters_trainable']:,} | "
        f"total={s['n_parameters_total']:,}"
    )


# ---------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------

def _make_norm(norm: str, hidden_dim: int) -> nn.Module:
    """Build a normalization layer for hidden node embeddings."""
    if norm == "layer":
        return nn.LayerNorm(hidden_dim)
    elif norm == "batch":
        return nn.BatchNorm1d(hidden_dim)
    return nn.Identity()


def _apply_residual(h_new, h_prev, h_in0, i, residual, input_proj):
    """Apply residual connections across message-passing layers."""
    if not residual:
        return h_new

    if i == 0:
        if input_proj is not None:
            return h_new + input_proj(h_in0)
        elif h_in0.shape[1] == h_new.shape[1]:
            return h_new + h_in0
        return h_new

    if h_prev.shape[1] == h_new.shape[1]:
        return h_new + h_prev
    return h_new


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
        raise AttributeError("Batch has no 'batch' attribute, so graph features cannot be broadcast.")

    gfeat = getattr(data, global_attr)
    if gfeat.ndim == 1:
        gfeat = gfeat.unsqueeze(-1)

    return gfeat[data.batch]


class _FinalHead(nn.Module):
    """Map node embeddings to mean and log-variance outputs."""
    def __init__(self, in_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout if dropout > 0 else 0.0),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, h):
        """Predict per-node mean and log-variance from the input embedding."""
        return self.net(h)


def _finalize_distribution_outputs(out: torch.Tensor, log_scale2_clamp):
    """Split head outputs into mean and clamped log-variance."""
    mu = out[:, 0].contiguous()
    log_scale2 = out[:, 1].contiguous()

    lo, hi = log_scale2_clamp
    log_scale2 = torch.clamp(log_scale2, lo, hi)
    return mu, log_scale2


# ---------------------------------------------------------------------
# 1) Existing GraphSAGE mean model
# ---------------------------------------------------------------------

class PureSAGECurvature(nn.Module):
    """GraphSAGE node regressor with optional graph-level conditioning features."""
    def __init__(
        self,
        n_markers: int,
        hidden_dim: int = 64,
        num_layers: int = 3,
        dropout: float = 0.2,
        residual: bool = True,
        norm: str = "layer",
        log_scale2_clamp: tuple[float, float] = (-10.0, 10.0),
        global_dim: int = 0,
        global_attr: str = "global_feat",
    ):
        super().__init__()
        assert num_layers >= 0, "num_layers must be >= 0"

        self.n_markers = n_markers
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.residual = residual
        self.norm = norm
        self.log_scale2_clamp = log_scale2_clamp
        self.global_dim = global_dim
        self.global_attr = global_attr

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        if num_layers > 0:
            in_dim = n_markers
            for _ in range(num_layers):
                self.convs.append(SAGEConv(in_dim, hidden_dim, aggr="mean"))
                self.norms.append(_make_norm(norm, hidden_dim))
                in_dim = hidden_dim

            self.input_proj = None
            if residual and n_markers != hidden_dim:
                self.input_proj = nn.Linear(n_markers, hidden_dim, bias=False)

            node_head_dim = hidden_dim
        else:
            self.input_proj = None
            node_head_dim = n_markers

        head_in_dim = node_head_dim + global_dim
        self.head = _FinalHead(head_in_dim, hidden_dim, dropout)

    def forward(self, x, edge_index, data=None):
        """Run GraphSAGE message passing and predict nodewise mean and log-variance."""
        if self.num_layers == 0:
            h = x
        else:
            h = x
            h_in0 = x

            for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
                h_new = conv(h, edge_index)
                h_new = norm(h_new)
                h_new = F.relu(h_new, inplace=False)

                h_new = _apply_residual(
                    h_new=h_new,
                    h_prev=h,
                    h_in0=h_in0,
                    i=i,
                    residual=self.residual,
                    input_proj=self.input_proj,
                )

                if self.dropout > 0:
                    h_new = F.dropout(h_new, p=self.dropout, training=self.training)

                h = h_new

        if self.global_dim > 0:
            gfeat_node = _broadcast_global_features(data, self.global_attr)
            if gfeat_node.shape[1] != self.global_dim:
                raise ValueError(
                    f"Expected {self.global_dim} global features, got {gfeat_node.shape[1]}"
                )
            h = torch.cat([h, gfeat_node], dim=1)

        out = self.head(h)
        mu, log_scale2 = _finalize_distribution_outputs(out, self.log_scale2_clamp)
        return (mu, log_scale2), h

    def num_parameters(self, trainable_only: bool = True) -> int:
        """Count the number of model parameters."""
        return count_parameters(self, trainable_only=trainable_only)


# ---------------------------------------------------------------------
# 3) GIN
# ---------------------------------------------------------------------

class GINCurvature(nn.Module):
    """GIN node regressor with optional graph-level conditioning features."""
    def __init__(
        self,
        n_markers: int,
        hidden_dim: int = 64,
        num_layers: int = 3,
        dropout: float = 0.2,
        residual: bool = True,
        norm: str = "layer",
        train_eps: bool = True,
        log_scale2_clamp: tuple[float, float] = (-10.0, 10.0),
        global_dim: int = 0,
        global_attr: str = "global_feat",
    ):
        super().__init__()
        assert num_layers >= 0, "num_layers must be >= 0"

        self.n_markers = n_markers
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.residual = residual
        self.norm = norm
        self.train_eps = train_eps
        self.log_scale2_clamp = log_scale2_clamp
        self.global_dim = global_dim
        self.global_attr = global_attr

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        if num_layers > 0:
            in_dim = n_markers
            for _ in range(num_layers):
                gin_mlp = nn.Sequential(
                    nn.Linear(in_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                )
                self.convs.append(GINConv(gin_mlp, train_eps=train_eps))
                self.norms.append(_make_norm(norm, hidden_dim))
                in_dim = hidden_dim

            self.input_proj = None
            if residual and n_markers != hidden_dim:
                self.input_proj = nn.Linear(n_markers, hidden_dim, bias=False)

            node_head_dim = hidden_dim
        else:
            self.input_proj = None
            node_head_dim = n_markers

        head_in_dim = node_head_dim + global_dim
        self.head = _FinalHead(head_in_dim, hidden_dim, dropout)

    def forward(self, x, edge_index, data=None):
        """Run GIN message passing and predict nodewise mean and log-variance."""
        if self.num_layers == 0:
            h = x
        else:
            h = x
            h_in0 = x

            for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
                h_new = conv(h, edge_index)
                h_new = norm(h_new)
                h_new = F.relu(h_new, inplace=False)

                h_new = _apply_residual(
                    h_new=h_new,
                    h_prev=h,
                    h_in0=h_in0,
                    i=i,
                    residual=self.residual,
                    input_proj=self.input_proj,
                )

                if self.dropout > 0:
                    h_new = F.dropout(h_new, p=self.dropout, training=self.training)

                h = h_new

        if self.global_dim > 0:
            gfeat_node = _broadcast_global_features(data, self.global_attr)
            if gfeat_node.shape[1] != self.global_dim:
                raise ValueError(
                    f"Expected {self.global_dim} global features, got {gfeat_node.shape[1]}"
                )
            h = torch.cat([h, gfeat_node], dim=1)

        out = self.head(h)
        mu, log_scale2 = _finalize_distribution_outputs(out, self.log_scale2_clamp)
        return (mu, log_scale2), h

    def num_parameters(self, trainable_only: bool = True) -> int:
        """Count the number of model parameters."""
        return count_parameters(self, trainable_only=trainable_only)


# ---------------------------------------------------------------------
# 4) GAT
# ---------------------------------------------------------------------

class GATCurvature(nn.Module):
    """GAT node regressor with optional graph-level conditioning features."""
    def __init__(
        self,
        n_markers: int,
        hidden_dim: int = 64,
        num_layers: int = 3,
        dropout: float = 0.2,
        residual: bool = True,
        norm: str = "layer",
        heads: int = 4,
        attn_dropout: float = 0.0,
        log_scale2_clamp: tuple[float, float] = (-10.0, 10.0),
        global_dim: int = 0,
        global_attr: str = "global_feat",
    ):
        super().__init__()
        assert num_layers >= 0, "num_layers must be >= 0"
        assert heads >= 1, "heads must be >= 1"

        self.n_markers = n_markers
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.residual = residual
        self.norm = norm
        self.heads = heads
        self.attn_dropout = attn_dropout
        self.log_scale2_clamp = log_scale2_clamp
        self.global_dim = global_dim
        self.global_attr = global_attr

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        if num_layers > 0:
            in_dim = n_markers
            for _ in range(num_layers):
                self.convs.append(
                    GATConv(
                        in_dim,
                        hidden_dim,
                        heads=heads,
                        concat=False,
                        dropout=attn_dropout,
                    )
                )
                self.norms.append(_make_norm(norm, hidden_dim))
                in_dim = hidden_dim

            self.input_proj = None
            if residual and n_markers != hidden_dim:
                self.input_proj = nn.Linear(n_markers, hidden_dim, bias=False)

            node_head_dim = hidden_dim
        else:
            self.input_proj = None
            node_head_dim = n_markers

        head_in_dim = node_head_dim + global_dim
        self.head = _FinalHead(head_in_dim, hidden_dim, dropout)

    def forward(self, x, edge_index, data=None):
        """Run GAT message passing and predict nodewise mean and log-variance."""
        if self.num_layers == 0:
            h = x
        else:
            h = x
            h_in0 = x

            for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
                h_new = conv(h, edge_index)
                h_new = norm(h_new)
                h_new = F.relu(h_new, inplace=False)

                h_new = _apply_residual(
                    h_new=h_new,
                    h_prev=h,
                    h_in0=h_in0,
                    i=i,
                    residual=self.residual,
                    input_proj=self.input_proj,
                )

                if self.dropout > 0:
                    h_new = F.dropout(h_new, p=self.dropout, training=self.training)

                h = h_new

        if self.global_dim > 0:
            gfeat_node = _broadcast_global_features(data, self.global_attr)
            if gfeat_node.shape[1] != self.global_dim:
                raise ValueError(
                    f"Expected {self.global_dim} global features, got {gfeat_node.shape[1]}"
                )
            h = torch.cat([h, gfeat_node], dim=1)

        out = self.head(h)
        mu, log_scale2 = _finalize_distribution_outputs(out, self.log_scale2_clamp)
        return (mu, log_scale2), h

    def num_parameters(self, trainable_only: bool = True) -> int:
        """Count the number of model parameters."""
        return count_parameters(self, trainable_only=trainable_only)


# ---------------------------------------------------------------------
# 5) Jumping Knowledge on GraphSAGE
# ---------------------------------------------------------------------

class JKGraphSAGECurvature(nn.Module):
    """Jumping-Knowledge GraphSAGE regressor with optional graph conditioning."""
    def __init__(
        self,
        n_markers: int,
        hidden_dim: int = 64,
        num_layers: int = 3,
        dropout: float = 0.2,
        residual: bool = True,
        norm: str = "layer",
        jk_mode: str = "cat",
        log_scale2_clamp: tuple[float, float] = (-10.0, 10.0),
        global_dim: int = 0,
        global_attr: str = "global_feat",
    ):
        super().__init__()
        assert num_layers >= 0, "num_layers must be >= 0"
        assert jk_mode in {"cat", "max", "lstm"}

        self.n_markers = n_markers
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.residual = residual
        self.norm = norm
        self.jk_mode = jk_mode
        self.log_scale2_clamp = log_scale2_clamp
        self.global_dim = global_dim
        self.global_attr = global_attr

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        if num_layers > 0:
            in_dim = n_markers
            for _ in range(num_layers):
                self.convs.append(SAGEConv(in_dim, hidden_dim, aggr="mean"))
                self.norms.append(_make_norm(norm, hidden_dim))
                in_dim = hidden_dim

            self.input_proj = None
            if residual and n_markers != hidden_dim:
                self.input_proj = nn.Linear(n_markers, hidden_dim, bias=False)

            self.jk = JumpingKnowledge(mode=jk_mode, channels=hidden_dim, num_layers=num_layers)

            if jk_mode == "cat":
                node_head_dim = num_layers * hidden_dim
            else:
                node_head_dim = hidden_dim
        else:
            self.input_proj = None
            self.jk = None
            node_head_dim = n_markers

        head_in_dim = node_head_dim + global_dim
        self.head = _FinalHead(head_in_dim, hidden_dim, dropout)

    def forward(self, x, edge_index, data=None):
        """Run JK-GraphSAGE message passing and predict nodewise mean and log-variance."""
        if self.num_layers == 0:
            h = x
        else:
            h = x
            h_in0 = x
            layer_outputs = []

            for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
                h_new = conv(h, edge_index)
                h_new = norm(h_new)
                h_new = F.relu(h_new, inplace=False)

                h_new = _apply_residual(
                    h_new=h_new,
                    h_prev=h,
                    h_in0=h_in0,
                    i=i,
                    residual=self.residual,
                    input_proj=self.input_proj,
                )

                if self.dropout > 0:
                    h_new = F.dropout(h_new, p=self.dropout, training=self.training)

                h = h_new
                layer_outputs.append(h)

            h = self.jk(layer_outputs)

        if self.global_dim > 0:
            gfeat_node = _broadcast_global_features(data, self.global_attr)
            if gfeat_node.shape[1] != self.global_dim:
                raise ValueError(
                    f"Expected {self.global_dim} global features, got {gfeat_node.shape[1]}"
                )
            h = torch.cat([h, gfeat_node], dim=1)

        out = self.head(h)
        mu, log_scale2 = _finalize_distribution_outputs(out, self.log_scale2_clamp)
        return (mu, log_scale2), h

    def num_parameters(self, trainable_only: bool = True) -> int:
        """Count the number of model parameters."""
        return count_parameters(self, trainable_only=trainable_only)


class JKGINCurvature(nn.Module):
    """Jumping-Knowledge GIN regressor with optional graph conditioning."""
    def __init__(
        self,
        n_markers: int,
        hidden_dim: int = 64,
        num_layers: int = 3,
        dropout: float = 0.2,
        residual: bool = True,
        norm: str = "layer",
        train_eps: bool = True,
        jk_mode: str = "cat",
        log_scale2_clamp: tuple[float, float] = (-10.0, 10.0),
        global_dim: int = 0,
        global_attr: str = "global_feat",
    ):
        super().__init__()
        assert num_layers >= 0, "num_layers must be >= 0"
        assert jk_mode in {"cat", "max", "lstm"}

        self.n_markers = n_markers
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.residual = residual
        self.norm = norm
        self.train_eps = train_eps
        self.jk_mode = jk_mode
        self.log_scale2_clamp = log_scale2_clamp
        self.global_dim = global_dim
        self.global_attr = global_attr

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        if num_layers > 0:
            in_dim = n_markers
            for _ in range(num_layers):
                gin_mlp = nn.Sequential(
                    nn.Linear(in_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                )
                self.convs.append(GINConv(gin_mlp, train_eps=train_eps))
                self.norms.append(_make_norm(norm, hidden_dim))
                in_dim = hidden_dim

            self.input_proj = None
            if residual and n_markers != hidden_dim:
                self.input_proj = nn.Linear(n_markers, hidden_dim, bias=False)

            self.jk = JumpingKnowledge(mode=jk_mode, channels=hidden_dim, num_layers=num_layers)

            if jk_mode == "cat":
                node_head_dim = num_layers * hidden_dim
            else:
                node_head_dim = hidden_dim
        else:
            self.input_proj = None
            self.jk = None
            node_head_dim = n_markers

        head_in_dim = node_head_dim + global_dim
        self.head = _FinalHead(head_in_dim, hidden_dim, dropout)

    def forward(self, x, edge_index, data=None):
        """Run JK-GIN message passing and predict nodewise mean and log-variance."""
        if self.num_layers == 0:
            h = x
            layer_outputs = None
        else:
            h = x
            h_in0 = x
            layer_outputs = []

            for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
                h_new = conv(h, edge_index)
                h_new = norm(h_new)
                h_new = F.relu(h_new, inplace=False)

                h_new = _apply_residual(
                    h_new=h_new,
                    h_prev=h,
                    h_in0=h_in0,
                    i=i,
                    residual=self.residual,
                    input_proj=self.input_proj,
                )

                if self.dropout > 0:
                    h_new = F.dropout(h_new, p=self.dropout, training=self.training)

                h = h_new
                layer_outputs.append(h)

            h = self.jk(layer_outputs)

        if self.global_dim > 0:
            gfeat_node = _broadcast_global_features(data, self.global_attr)
            if gfeat_node.shape[1] != self.global_dim:
                raise ValueError(
                    f"Expected {self.global_dim} global features, got {gfeat_node.shape[1]}"
                )
            h = torch.cat([h, gfeat_node], dim=1)

        out = self.head(h)
        mu, log_scale2 = _finalize_distribution_outputs(out, self.log_scale2_clamp)
        return (mu, log_scale2), h

    def num_parameters(self, trainable_only: bool = True) -> int:
        """Count the number of model parameters."""
        return count_parameters(self, trainable_only=trainable_only)