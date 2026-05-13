import torch
import torch.nn as nn

from src.models.gnn import _finalize_distribution_outputs, _normalize_covariance_mode


class GlobalFeatureMLP(nn.Module):
    """Nodewise baseline that predicts only from graph-level features.

    The model follows the same call/return convention as the GNN models so it
    can be trained with the shared training loop:

        (mu, log_var), h = model(batch.x, batch.edge_index, data=batch)

    ``x`` and ``edge_index`` are accepted for API compatibility but ignored.
    Graph-level features are broadcast to nodes using ``data.batch``.
    """

    def __init__(
        self,
        global_dim: int,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        global_attr: str = "global_feat",
        target_dim: int = 1,
        covariance_mode: str = "diagonal",
        log_scale2_clamp: tuple[float, float] = (-10.0, 10.0),
    ):
        super().__init__()
        if int(global_dim) < 1:
            raise ValueError("GlobalFeatureMLP requires at least one global feature.")

        self.global_dim = int(global_dim)
        self.hidden_dim = int(hidden_dim)
        self.dropout = float(dropout)
        self.global_attr = global_attr
        self.target_dim = int(target_dim)
        self.covariance_mode = _normalize_covariance_mode(covariance_mode)
        self.log_scale2_clamp = log_scale2_clamp

        self.net = nn.Sequential(
            nn.Linear(self.global_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout if self.dropout > 0 else 0.0),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout if self.dropout > 0 else 0.0),
            nn.Linear(
                self.hidden_dim,
                2 * self.target_dim
                if self.covariance_mode == "diagonal"
                else self.target_dim + self.target_dim * (self.target_dim + 1) // 2,
            ),
        )

    def _node_global_features(self, data):
        if data is None:
            raise ValueError("GlobalFeatureMLP requires the PyG batch object via data=.")
        if not hasattr(data, self.global_attr):
            raise AttributeError(
                f"Batch has no attribute {self.global_attr!r}. Attach global features before training."
            )
        if not hasattr(data, "batch"):
            raise AttributeError("Batch has no 'batch' vector, so graph features cannot be broadcast.")

        gfeat = getattr(data, self.global_attr)
        if gfeat.ndim == 1:
            gfeat = gfeat.unsqueeze(-1)
        if gfeat.shape[1] != self.global_dim:
            raise ValueError(f"Expected {self.global_dim} global features, got {gfeat.shape[1]}.")
        return gfeat[data.batch]

    def forward(self, x, edge_index=None, data=None):
        h = self._node_global_features(data)
        out = self.net(h)
        mu, log_scale2 = _finalize_distribution_outputs(
            out,
            self.log_scale2_clamp,
            self.target_dim,
            self.covariance_mode,
        )
        return (mu, log_scale2), h

    def num_parameters(self, trainable_only: bool = True) -> int:
        params = self.parameters()
        if trainable_only:
            return sum(p.numel() for p in params if p.requires_grad)
        return sum(p.numel() for p in params)
