"""Restore omitted marker channels from original cell graphs, retaining panel availability."""
from pathlib import Path
import numpy as np
from .marker_exclusivity import exclusive_markers


def restore_marker_channels(x, marker_names, metadata, markers, *, edges=None, graph=None):
    """Return exclusive features, names and per-marker measurement availability.

    Read processed ``markers_bin`` through organograph. Original GNN node order
    is recovered from export metadata, including cells removed during export.
    Missing channels remain zero-filled but are explicitly marked unmeasured;
    this does not impute their identities or correct experiment confounding.
    Inputs and the original cell graph are never mutated.
    """
    from organograph.graph.io import load_cell_graph
    from organograph.graph.access import graph_get, graph_get_marker_bin
    values=np.asarray(x)
    names=list(marker_names);markers=list(markers)
    if values.ndim!=2 or values.shape[1]!=len(names) or not np.isin(values,[0,1]).all():
        raise ValueError('Expected binary node-by-marker features.')
    if len(set(names+markers))!=len(names)+len(markers):
        raise ValueError('Restored markers must be distinct and absent from the existing panel.')
    if graph is None:graph=load_cell_graph(Path(metadata['graph_path']))
    if 'kept_node_ids' in metadata:
        nodes=list(metadata['kept_node_ids'])
    elif 'new_to_old_index' in metadata:
        original=sorted(graph.nodes)
        nodes=[original[int(i)] for i in metadata['new_to_old_index']]
    else:raise ValueError('Original node mapping is required for marker restoration.')
    if len(nodes)!=len(values) or len(set(nodes))!=len(nodes) or not all(u in graph for u in nodes):
        raise ValueError('Original cell IDs do not match the GNN nodes.')
    if 'proj_vertex_ids' in metadata:
        if not np.array_equal(graph_get(graph,'proj_vertex',nodes=nodes),metadata['proj_vertex_ids']):
            raise ValueError('Mesh projection IDs disagree with the original cell mapping.')
    if edges is not None:
        index={u:i for i,u in enumerate(nodes)}
        original_edges={tuple(sorted((index[u],index[v]))) for u,v in graph.edges if u in index and v in index and u!=v}
        supplied_edges={tuple(sorted((int(u),int(v)))) for u,v in np.asarray(edges).reshape(-1,2) if u!=v}
        if original_edges!=supplied_edges:raise ValueError('Original and exported graph edges disagree.')
    panel=list(graph.graph['marker_names'])
    # Derived channels (e.g. Cyclin A/D -> KI67) need not exist in the original panel.
    for j,name in enumerate(names):
        if name in panel and not np.array_equal(values[:,j],graph_get_marker_bin(graph,name,nodes=nodes)):
            raise ValueError(f'Existing marker disagrees with original graph: {name}')
    availability={name:name in panel for name in markers}
    columns=[graph_get_marker_bin(graph,name,nodes=nodes) if availability[name] else np.zeros(len(nodes),int) for name in markers]
    expanded=np.column_stack([values,*columns]).astype(values.dtype,copy=False)
    return exclusive_markers(expanded,names+markers),names+markers,availability
