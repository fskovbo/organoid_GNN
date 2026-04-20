import numpy as np
from organograph.mesh.OrganoidMesh import OrganoidMesh

from src.analysis.prediction_analysis import get_graph_slice_bounds
from src.data.metadata import get_graph_metadata


"""Project node-wise quantities from PyG graphs onto organoid meshes."""



def project_node_quantities_to_mesh(
    graph,
    *,
    node_pred,
    node_true,
    meta_lookup=None,
    mesh_path=None,
    vertex_owner=None,
    proj_vertex_ids=None,
    fill_value=np.nan,
    normalize_mesh=True,
    return_debug=False,
):
    """Map node-level targets/predictions to mesh vertices using stored ownership arrays."""
    md = get_graph_metadata(graph, meta_lookup=meta_lookup, strict=True)

    if mesh_path is None:
        mesh_path = md.get("mesh_path")
    if vertex_owner is None:
        vertex_owner = md.get("voronoi_vertex_owner")
    if proj_vertex_ids is None:
        proj_vertex_ids = md.get("proj_vertex_ids")

    if mesh_path is None:
        raise ValueError("mesh_path could not be resolved from inputs or metadata")
    if vertex_owner is None:
        raise ValueError("vertex_owner could not be resolved from inputs or metadata")

    mesh = OrganoidMesh(mesh_path)
    if normalize_mesh:
        mesh.normalize_inplace()

    vertex_owner = np.asarray(vertex_owner, dtype=np.int64).reshape(-1)
    node_pred = np.asarray(node_pred, dtype=np.float32).reshape(-1)
    node_true = np.asarray(node_true, dtype=np.float32).reshape(-1)

    if vertex_owner.shape[0] != mesh.v.shape[0]:
        raise ValueError(
            f"vertex_owner length mismatch: {vertex_owner.shape[0]} vs {mesh.v.shape[0]} mesh vertices"
        )

    n_nodes_meta = int(np.max(vertex_owner)) + 1 if np.any(vertex_owner >= 0) else 0
    n_nodes_graph = int(graph.y.shape[0]) if hasattr(graph, "y") else None

    if len(node_pred) != n_nodes_meta:
        raise ValueError(f"Prediction length mismatch: {len(node_pred)} vs {n_nodes_meta}")
    if len(node_true) != n_nodes_meta:
        raise ValueError(f"Target length mismatch: {len(node_true)} vs {n_nodes_meta}")
    if n_nodes_graph is not None and n_nodes_graph != n_nodes_meta:
        raise ValueError(f"Graph node count mismatch: {n_nodes_graph} vs {n_nodes_meta}")

    mesh_pred = np.full(vertex_owner.shape, fill_value, dtype=np.float32)
    mesh_true = np.full(vertex_owner.shape, fill_value, dtype=np.float32)
    valid = vertex_owner >= 0
    mesh_pred[valid] = node_pred[vertex_owner[valid]]
    mesh_true[valid] = node_true[vertex_owner[valid]]

    result = {
        "mesh": mesh,
        "mesh_pred": mesh_pred,
        "mesh_true": mesh_true,
        "vertex_owner": vertex_owner,
        "mesh_path": mesh_path,
        "organoid_str": getattr(graph, "organoid_str", None),
    }

    if return_debug:
        result["proj_vertex_ids"] = None if proj_vertex_ids is None else np.asarray(proj_vertex_ids, dtype=np.int64)
        result["meta"] = md

    return result



def project_predictions_to_mesh(graph_index, graphs, y_true_all, y_pred_all, *, meta_lookup=None, fill_value=np.nan, return_debug=False):
    """Project one graph from concatenated prediction arrays onto its mesh."""
    graph = graphs[graph_index]
    start, end = get_graph_slice_bounds(graphs, graph_index)
    return project_node_quantities_to_mesh(
        graph,
        node_true=y_true_all[start:end],
        node_pred=y_pred_all[start:end],
        meta_lookup=meta_lookup,
        fill_value=fill_value,
        return_debug=return_debug,
    )