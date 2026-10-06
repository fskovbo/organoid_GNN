"""Node-aligned measured labels and exact-ring fate features, without target leakage."""
from pathlib import Path
import numpy as np
from scipy import sparse


def measured_labels(graph, shared_names, target_names):
    """Read processed stains through organograph; -1 means unmeasured, not negative.

    Validate cell mapping, projections, topology and shared channels before using
    labels. Never apply exclusivity: doing so with the hidden stains leaks labels
    into retained channels. The original graph and PyG inputs are not mutated.
    """
    from organograph.graph.io import load_cell_graph
    from organograph.graph.access import graph_get, graph_get_marker_bin
    if set(shared_names) & set(target_names):
        raise ValueError('Prediction targets must be absent from every input channel.')
    md = graph.meta
    original = load_cell_graph(Path(md['graph_path']))
    if 'kept_node_ids' in md:
        nodes = list(md['kept_node_ids'])
    elif 'new_to_old_index' in md:
        ordered = sorted(original.nodes)
        nodes = [ordered[int(i)] for i in md['new_to_old_index']]
    else:
        raise ValueError('An explicit original-cell mapping is required.')
    if len(nodes) != len(graph.x) or len(set(nodes)) != len(nodes):
        raise ValueError('Cell mapping has wrong length or duplicate IDs.')
    if 'proj_vertex_ids' in md:
        np.testing.assert_array_equal(graph_get(original, 'proj_vertex', nodes=nodes), md['proj_vertex_ids'])
    index = {node: i for i, node in enumerate(nodes)}
    expected = {tuple(sorted((index[a], index[b]))) for a, b in original.edges if a in index and b in index and a != b}
    actual = {tuple(sorted((int(a), int(b)))) for a, b in graph.edge_index.numpy().T if a != b}
    if expected != actual:
        raise ValueError('Original graph and GNN adjacency disagree.')
    panel = list(original.graph['marker_names'])
    for j, name in enumerate(shared_names):
        if name in panel:
            np.testing.assert_array_equal(graph.x[:, j].numpy(), graph_get_marker_bin(original, name, nodes=nodes))
    columns = []
    for name in target_names:
        values = np.asarray(graph_get_marker_bin(original, name, nodes=nodes)).reshape(-1) if name in panel else np.full(len(nodes), -1)
        if not np.isin(values, [-1, 0, 1]).all():
            raise ValueError('Expected measured binary labels or unmeasured sentinel.')
        columns.append(values)
    return np.column_stack(columns).astype(np.int8)


def ring_features(x, edge_index, radius):
    """Center channels followed by exact shortest-path ring fractions, hops 1..R.

    No ring sizes, curvature, target markers or global shape enter these features.
    Empty rings have zero fractions. Coexpression is retained and fractions need
    not sum to one. Boolean reachability prevents path-count weighting.
    """
    x = np.asarray(x, dtype=np.float32)
    if radius < 0 or int(radius) != radius or x.ndim != 2:
        raise ValueError('Expected a nonnegative integer radius and node-feature matrix.')
    if not np.isin(x, [0, 1]).all():
        raise ValueError('Shared fate inputs must be binary.')
    n = len(x)
    edges = np.asarray(edge_index, dtype=int)
    adjacency = sparse.csr_matrix((np.ones(edges.shape[1], bool), edges), shape=(n, n))
    adjacency = adjacency.maximum(adjacency.T)
    adjacency.setdiag(False)
    adjacency.eliminate_zeros()
    frontier = sparse.eye(n, format='csr', dtype=bool)
    seen = frontier.copy()
    parts = [x]
    for _ in range(int(radius)):
        reached = (frontier @ adjacency).astype(bool)
        frontier = reached.astype(np.int8) - reached.multiply(seen).astype(np.int8)
        frontier.eliminate_zeros()
        frontier = frontier.astype(bool)
        sizes = np.asarray(frontier.sum(axis=1)).reshape(-1, 1)
        parts.append(np.asarray(frontier @ x) / np.maximum(sizes, 1))
        seen = seen.maximum(frontier)
    return np.concatenate(parts, axis=1).astype(np.float32)


def select_features(fates, curvature, n_markers, radius, use_curvature):
    """Select a prefix of fate rings, optionally adding center-cell curvature."""
    selected = np.asarray(fates[:, :(radius + 1) * n_markers], dtype=np.float32)
    if use_curvature:
        selected = np.column_stack([selected, curvature]).astype(np.float32)
    return selected
