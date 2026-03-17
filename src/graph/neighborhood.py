def compute_hop_rings(edge_index, center_idx, max_hops):
    """
    Inputs:
      edge_index  : torch.LongTensor shape (2, E)
      center_idx  : int, center node index in this subgraph
      max_hops    : int

    Outputs:
      rings       : list of lists; rings[h] = list of node indices at exact hop distance h
                   includes rings[0] = [center_idx]
    """
    src = edge_index[0].tolist()
    dst = edge_index[1].tolist()

    # Treat as undirected for "neighborhood" (common in tissue adjacency)
    adj = {}
    for u, v in zip(src, dst):
        adj.setdefault(u, set()).add(v)
        adj.setdefault(v, set()).add(u)

    visited = set([center_idx])
    frontier = set([center_idx])
    rings = [[center_idx]]

    for h in range(1, max_hops + 1):
        nxt = set()
        for u in frontier:
            for v in adj.get(u, ()):
                if v not in visited:
                    nxt.add(v)
        rings.append(sorted(nxt))
        visited |= nxt
        frontier = nxt
        if len(frontier) == 0:
            # pad remaining hops with empty rings
            for _ in range(h + 1, max_hops + 1):
                rings.append([])
            break

    return rings