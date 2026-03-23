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


def compute_ring_fraction_features(x: torch.Tensor, edge_index: torch.Tensor, k_hops: int) -> torch.Tensor:
    device = x.device
    N, M = x.shape
    adj = build_adj_list(edge_index, N)

    out = torch.zeros((N, (k_hops + 1) * M), dtype=torch.float32, device=device)

    for c in range(N):
        rings = compute_hop_rings_from_adj(adj, c, k_hops)
        feats = []
        for nodes in rings:
            if len(nodes) == 0:
                feats.append(torch.zeros(M, dtype=torch.float32, device=device))
            else:
                idx = torch.tensor(nodes, dtype=torch.long, device=device)
                feats.append(x[idx].float().mean(dim=0))
        out[c] = torch.cat(feats, dim=0)

    return out


def compute_pooled_khop_fraction_features(
    x: torch.Tensor,
    edge_index: torch.Tensor,
    k_hops: int,
    include_center: bool = True,
) -> torch.Tensor:
    device = x.device
    N, M = x.shape
    adj = build_adj_list(edge_index, N)

    out = torch.zeros((N, M), dtype=torch.float32, device=device)

    for c in range(N):
        rings = compute_hop_rings_from_adj(adj, c, k_hops)

        if include_center:
            nodes = [u for ring in rings for u in ring]
        else:
            nodes = [u for ring in rings[1:] for u in ring]

        if len(nodes) == 0:
            continue

        idx = torch.tensor(sorted(set(nodes)), dtype=torch.long, device=device)
        out[c] = x[idx].float().mean(dim=0)

    return out


def attach_precomputed_ring_features(
    graphs,
    k_hops: int,
    ring_attr: str = "x_ring",
    pool_attr: str = "x_pool",
    include_center_in_pool: bool = True,
    inplace: bool = False,
):
    graphs_out = graphs if inplace else [copy.copy(g) for g in graphs]

    for g in graphs_out:
        x = g.x
        edge_index = g.edge_index

        setattr(g, ring_attr, compute_ring_fraction_features(x, edge_index, k_hops))
        setattr(
            g,
            pool_attr,
            compute_pooled_khop_fraction_features(
                x,
                edge_index,
                k_hops,
                include_center=include_center_in_pool,
            ),
        )

    return graphs_out