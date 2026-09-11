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
    """Map node embeddings to Gaussian distribution outputs."""
    def __init__(self, in_dim: int, hidden_dim: int, dropout: float, target_dim: int = 1, covariance_mode: str = "diagonal"):
        super().__init__()
        self.target_dim = int(target_dim)
        self.covariance_mode = _normalize_covariance_mode(covariance_mode)
        self.out_dim = _gaussian_head_output_dim(self.target_dim, self.covariance_mode)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout if dropout > 0 else 0.0),
            nn.Linear(hidden_dim, self.out_dim),
        )

    def forward(self, h):
        return self.net(h)


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


def _finalize_distribution_outputs(out: torch.Tensor, log_scale2_clamp, target_dim: int = 1, covariance_mode: str = "diagonal"):
    """Split head outputs into means and either log-variances or Cholesky factors."""
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


def distribution_to_diagonal_outputs(mu: torch.Tensor, scale: torch.Tensor):
    """Convert diagonal or full-covariance outputs to independent (mu, log_var) outputs."""
    if scale.ndim == 3:
        cov = scale @ scale.transpose(-1, -2)
        scale = torch.log(torch.diagonal(cov, dim1=-2, dim2=-1).clamp_min(1e-12))
    if mu.ndim == 2 and mu.shape[1] == 1:
        mu = mu.reshape(-1)
    if scale.ndim == 2 and scale.shape[1] == 1:
        scale = scale.reshape(-1)
    return mu, scale


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
        target_dim: int = 1,
        covariance_mode: str = "diagonal",
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
        self.target_dim = int(target_dim)
        self.covariance_mode = _normalize_covariance_mode(covariance_mode)

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
        self.head = _FinalHead(head_in_dim, hidden_dim, dropout, target_dim=self.target_dim, covariance_mode=self.covariance_mode)

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
        mu, log_scale2 = _finalize_distribution_outputs(out, self.log_scale2_clamp, self.target_dim, self.covariance_mode)
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
        target_dim: int = 1,
        covariance_mode: str = "diagonal",
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
        self.target_dim = int(target_dim)
        self.covariance_mode = _normalize_covariance_mode(covariance_mode)

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
        self.head = _FinalHead(head_in_dim, hidden_dim, dropout, target_dim=self.target_dim, covariance_mode=self.covariance_mode)

    def _condition_update(self, h, layer_index, data):
        """Optional conditioning after normalization and before activation."""
        return h

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
                h_new = self._condition_update(h_new, i, data)
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
        mu, log_scale2 = _finalize_distribution_outputs(out, self.log_scale2_clamp, self.target_dim, self.covariance_mode)
        return (mu, log_scale2), h

    def num_parameters(self, trainable_only: bool = True) -> int:
        """Count the number of model parameters."""
        return count_parameters(self, trainable_only=trainable_only)


class SizeFiLMGINCurvature(GINCurvature):
    """GIN conditioned on one standardized log-cell-count feature per graph.

    Each layer computes ``(1 + delta_gamma(t)) * norm(h) + beta(t)``
    before ReLU and the existing residual connection. FiLM starts as the
    identity. Conditioning is computed once per graph, then broadcast to
    nodes; cell count is never included in the GIN neighbor sum.

    ``global_dim`` controls the features supplied to the prediction head.
    FiLM uses only column ``size_feature_index`` of that vector, allowing
    additional global features to enter the head without conditioning FiLM.
    Set ``global_dim=0`` for a conditioning-only head ablation; in that case
    ``data.<global_attr>`` must still contain one size feature per graph.
    """

    def __init__(
        self, n_markers: int, *, film_hidden_dim: int = 16,
        global_dim: int = 1, global_attr: str = "global_feat",
        size_feature_index: int = 0, **kwargs,
    ):
        if global_dim < 0:
            raise ValueError("global_dim must be nonnegative.")
        if not 0 <= size_feature_index < max(1, global_dim):
            raise ValueError("size_feature_index must select a global size feature.")
        if film_hidden_dim < 1:
            raise ValueError("film_hidden_dim must be positive.")
        super().__init__(
            n_markers, global_dim=global_dim, global_attr=global_attr, **kwargs,
        )
        self.film_hidden_dim = int(film_hidden_dim)
        self.size_feature_index = int(size_feature_index)
        self.film_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1, self.film_hidden_dim),
                nn.SiLU(),
                nn.Linear(self.film_hidden_dim, 2 * self.hidden_dim),
            )
            for _ in range(self.num_layers)
        ])
        for film in self.film_layers:
            nn.init.zeros_(film[-1].weight)
            nn.init.zeros_(film[-1].bias)

    def _condition_update(self, h, layer_index, data):
        if data is None or not hasattr(data, self.global_attr):
            raise ValueError("Size FiLM requires a graph-level log-cell-count feature.")
        if getattr(data, "batch", None) is None:
            raise ValueError("Size FiLM requires the PyG batch vector.")
        size = getattr(data, self.global_attr)
        if size.ndim == 1:
            size = size.unsqueeze(-1)
        expected_dim = max(1, self.global_dim)
        if size.ndim != 2 or size.shape[1] != expected_dim:
            raise ValueError(f"Size FiLM expects exactly {expected_dim} global features per graph.")
        size = size[:, self.size_feature_index:self.size_feature_index + 1]
        delta_gamma, beta = self.film_layers[layer_index](size).chunk(2, dim=-1)
        return (1.0 + delta_gamma[data.batch]) * h + beta[data.batch]


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
        target_dim: int = 1,
        covariance_mode: str = "diagonal",
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
        self.target_dim = int(target_dim)
        self.covariance_mode = _normalize_covariance_mode(covariance_mode)

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
        self.head = _FinalHead(head_in_dim, hidden_dim, dropout, target_dim=self.target_dim, covariance_mode=self.covariance_mode)

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
        mu, log_scale2 = _finalize_distribution_outputs(out, self.log_scale2_clamp, self.target_dim, self.covariance_mode)
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
        target_dim: int = 1,
        covariance_mode: str = "diagonal",
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
        self.target_dim = int(target_dim)
        self.covariance_mode = _normalize_covariance_mode(covariance_mode)

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
        self.head = _FinalHead(head_in_dim, hidden_dim, dropout, target_dim=self.target_dim, covariance_mode=self.covariance_mode)

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
        mu, log_scale2 = _finalize_distribution_outputs(out, self.log_scale2_clamp, self.target_dim, self.covariance_mode)
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
        target_dim: int = 1,
        covariance_mode: str = "diagonal",
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
        self.target_dim = int(target_dim)
        self.covariance_mode = _normalize_covariance_mode(covariance_mode)

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
        self.head = _FinalHead(head_in_dim, hidden_dim, dropout, target_dim=self.target_dim, covariance_mode=self.covariance_mode)

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
        mu, log_scale2 = _finalize_distribution_outputs(out, self.log_scale2_clamp, self.target_dim, self.covariance_mode)
        return (mu, log_scale2), h

    def num_parameters(self, trainable_only: bool = True) -> int:
        """Count the number of model parameters."""
        return count_parameters(self, trainable_only=trainable_only)
