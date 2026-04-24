import networkx as nx
import numpy as np
from organograph.mesh.OrganoidMesh import OrganoidMesh

from src.data.metadata import get_graph_metadata


def pyg_graph_to_nx(pyg_graph, *, marker_names=None):
    """Convert one PyG graph to an undirected NetworkX graph without centroids."""
    import networkx as nx
    import numpy as np

    G = nx.Graph()
    n_nodes = int(pyg_graph.x.shape[0])

    G.graph["organoid_str"] = getattr(pyg_graph, "organoid_str", None)
    if marker_names is not None:
        G.graph["marker_names"] = list(marker_names)

    X = pyg_graph.x.detach().cpu().numpy()

    for i in range(n_nodes):
        G.add_node(
            i,
            x=X[i].copy(),
            marker_bin=X[i].copy(),
            markers_bin=X[i].copy(),
        )

    edge_index = pyg_graph.edge_index.detach().cpu().numpy()
    for u, v in edge_index.T:
        u, v = int(u), int(v)
        if u != v:
            G.add_edge(u, v)

    return G


def pyg_graph_to_nx_with_centroids(
    pyg_graph,
    *,
    marker_names,
    meta_lookup=None,
    normalize_mesh=True,
):
    """Convert one PyG graph to NetworkX and add centroids from mesh projection IDs."""
    md = get_graph_metadata(pyg_graph, meta_lookup=meta_lookup, strict=True)

    mesh_path = md.get("mesh_path")
    proj_vertex_ids = md.get("proj_vertex_ids")

    if mesh_path is None:
        raise ValueError(f"{getattr(pyg_graph, 'organoid_str', None)} missing meta['mesh_path']")
    if proj_vertex_ids is None:
        raise ValueError(f"{getattr(pyg_graph, 'organoid_str', None)} missing meta['proj_vertex_ids']")

    proj_vertex_ids = np.asarray(proj_vertex_ids, dtype=np.int64).reshape(-1)
    n_nodes = int(pyg_graph.x.shape[0])

    if proj_vertex_ids.shape[0] != n_nodes:
        raise ValueError(
            f"proj_vertex_ids length mismatch: {proj_vertex_ids.shape[0]} vs {n_nodes}"
        )

    mesh = OrganoidMesh(mesh_path)
    if normalize_mesh:
        mesh.normalize_inplace()

    centroids = np.asarray(mesh.v[proj_vertex_ids], dtype=float)

    G = pyg_graph_to_nx(pyg_graph, marker_names=marker_names)

    for i in range(n_nodes):
        G.nodes[i]["centroid"] = centroids[i, :3]

    return G