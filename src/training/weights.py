import torch 


def per_graph_weighted_mean(
    per_node_loss: torch.Tensor,
    batch_index: torch.Tensor,
    weights: torch.Tensor | None = None,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Compute:
        mean_over_graphs( weighted_mean_over_nodes_in_graph(loss) )

    This prevents large graphs from dominating the batch loss.
    """
    if batch_index is None:
        if weights is None:
            return per_node_loss.mean()
        return (weights * per_node_loss).sum() / weights.sum().clamp_min(eps)

    n_graphs = int(batch_index.max().item()) + 1

    if weights is None:
        weights = torch.ones_like(per_node_loss)

    graph_losses = []
    for g in range(n_graphs):
        mask = (batch_index == g)
        if not mask.any():
            continue

        l_g = per_node_loss[mask]
        w_g = weights[mask]

        graph_loss = (w_g * l_g).sum() / w_g.sum().clamp_min(eps)
        graph_losses.append(graph_loss)

    if len(graph_losses) == 0:
        return per_node_loss.mean()

    return torch.stack(graph_losses).mean()


def soft_tail_weights(
    y: torch.Tensor,
    batch_index: torch.Tensor,
    alpha: float = 2.0,
    quantile: float = 0.9,
    temperature: float = 1.0,
    min_scale: float = 1e-3,
) -> torch.Tensor:
    """
    Per-node weights that softly upweight the upper tail of y *within each graph*.

    Weight formula:
        w_i = 1 + alpha * sigmoid((y_i - q_g) / (temperature * scale_g))

    where q_g is the graph-specific quantile and scale_g is the graph-specific std.
    """
    if batch_index is None:
        q = torch.quantile(y, quantile)
        scale = y.std(unbiased=False).clamp_min(min_scale)
        z = (y - q) / (temperature * scale)
        return 1.0 + alpha * torch.sigmoid(z)

    weights = torch.ones_like(y)
    n_graphs = int(batch_index.max().item()) + 1

    for g in range(n_graphs):
        mask = (batch_index == g)
        if not mask.any():
            continue

        y_g = y[mask]
        q_g = torch.quantile(y_g, quantile)
        scale_g = y_g.std(unbiased=False).clamp_min(min_scale)

        z_g = (y_g - q_g) / (temperature * scale_g)
        weights[mask] = 1.0 + alpha * torch.sigmoid(z_g)

    return weights


def hard_tail_weights(
    y: torch.Tensor,
    batch_index: torch.Tensor,
    alpha: float = 2.0,
    quantile: float = 0.9,
) -> torch.Tensor:
    """
    Simpler alternative:
        w_i = 1 + alpha if y_i > q_g else 1
    """
    if batch_index is None:
        q = torch.quantile(y, quantile)
        return torch.where(y > q, 1.0 + alpha, 1.0).to(y.dtype)

    weights = torch.ones_like(y)
    n_graphs = int(batch_index.max().item()) + 1

    for g in range(n_graphs):
        mask = (batch_index == g)
        if not mask.any():
            continue

        y_g = y[mask]
        q_g = torch.quantile(y_g, quantile)
        weights[mask] = torch.where(y_g > q_g, 1.0 + alpha, 1.0).to(y.dtype)

    return weights