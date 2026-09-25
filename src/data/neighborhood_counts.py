"""Exclusive identities and unique-cell counts in exact, undirected hop shells."""
import numpy as np


def fate_identities(x):
    """Return marker-column IDs, with the last ID reserved for Unassigned."""
    values = x.detach().cpu().numpy() if hasattr(x, 'detach') else np.asarray(x)
    if values.ndim != 2 or values.shape[1] < 1:
        raise ValueError('Expected a nonempty marker panel.')
    if not np.isin(values, [0, 1]).all() or np.any(values.sum(axis=1) > 1):
        raise ValueError('Exclusive binary fates are required; apply exclusive_markers first.')
    return np.where(values.sum(axis=1) == 0, values.shape[1], values.argmax(axis=1))


def exact_hop_counts(x, edge_index, radius):
    """Count each cell once at shortest-path distance 1..radius (never walks).

    Edges are treated as undirected; duplicate edges and self loops have no
    effect. Empty/disconnected shells contain zero counts. Inputs are unchanged.
    """
    if not isinstance(radius, int) or isinstance(radius, bool) or radius < 0:
        raise ValueError('radius must be a nonnegative integer.')
    identities = fate_identities(x)
    n = len(identities)
    edges = edge_index.detach().cpu().numpy() if hasattr(edge_index, 'detach') else np.asarray(edge_index)
    if edges.ndim != 2 or edges.shape[0] != 2 or not np.issubdtype(edges.dtype, np.integer):
        raise ValueError('Expected integer edge_index with shape (2, edges).')
    if edges.size and (edges.min() < 0 or edges.max() >= n):
        raise ValueError('Edge index outside the node array.')
    adjacency = [set() for _ in range(n)]
    for u, v in edges.T:
        if u != v:
            adjacency[u].add(v)
            adjacency[v].add(u)
    counts = np.zeros((n, radius, x.shape[1] + 1), dtype=np.int32)
    for center in range(n):
        seen, frontier = {center}, {center}
        for r in range(radius):
            frontier = set().union(*(adjacency[u] for u in frontier)) - seen if frontier else set()
            if not frontier:
                break
            seen.update(frontier)
            counts[center, r] = np.bincount(identities[list(frontier)], minlength=counts.shape[2])
    return counts
