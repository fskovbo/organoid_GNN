import copy
import torch


def build_adj_list(edge_index: torch.Tensor, num_nodes: int) -> list[list[int]]:
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


def compute_hop_rings_from_adj(adj: list[list[int]], center: int, k_hops: int) -> list[list[int]]:
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


def compute_ring_fraction_features_and_sizes(
    x: torch.Tensor,
    edge_index: torch.Tensor,
    k_hops: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns
    -------
    x_ring : (N, (k_hops + 1) * M) float32
        Exact-ring marker fractions.
    ring_sizes : (N, k_hops + 1) float32
        Number of nodes in each exact ring for each center.
    """
    device = x.device
    N, M = x.shape
    adj = build_adj_list(edge_index, N)

    x_ring = torch.zeros((N, (k_hops + 1) * M), dtype=torch.float32, device=device)
    ring_sizes = torch.zeros((N, k_hops + 1), dtype=torch.float32, device=device)

    for c in range(N):
        rings = compute_hop_rings_from_adj(adj, c, k_hops)

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


def attach_precomputed_ring_features(
    graphs,
    k_hops: int,
    ring_attr: str = "x_ring",
    ring_sizes_attr: str = "ring_sizes",
    inplace: bool = False,
):
    graphs_out = graphs if inplace else [copy.copy(g) for g in graphs]

    for g in graphs_out:
        x = g.x
        edge_index = g.edge_index

        x_ring, ring_sizes = compute_ring_fraction_features_and_sizes(x, edge_index, k_hops)
        setattr(g, ring_attr, x_ring)
        setattr(g, ring_sizes_attr, ring_sizes)

    return graphs_out