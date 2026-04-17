import numpy as np
from organograph.mesh.OrganoidMesh import OrganoidMesh


def _get_graph_metadata(graph, metadata_lookup=None):
    md = getattr(graph, "meta", None)
    if isinstance(md, dict):
        return md

    if metadata_lookup is None:
        raise ValueError(
            "No metadata attached to graph and no metadata_lookup provided."
        )

    organoid_str = getattr(graph, "organoid_str", None)
    if organoid_str is None:
        raise ValueError("graph has no .organoid_str")

    info = metadata_lookup.get(organoid_str, None)
    if info is None:
        raise KeyError(f"No metadata for organoid_str={organoid_str!r}")

    return info


def project_node_quantities_to_mesh(
    graph,
    node_pred,
    node_true,
    *,
    metadata_lookup=None,
    mesh_path=None,
    vertex_owner=None,
    proj_vertex_ids=None,
    fill_value=np.nan,
    return_debug=False,
):
    md = _get_graph_metadata(graph, metadata_lookup=metadata_lookup)

    if mesh_path is None:
        mesh_path = md.get("mesh_path")

    if vertex_owner is None:
        vertex_owner = md.get("voronoi_vertex_owner", None)

    if proj_vertex_ids is None:
        proj_vertex_ids = md.get("proj_vertex_ids", None)

    if mesh_path is None:
        raise ValueError("mesh_path could not be resolved from inputs/metadata")
    if vertex_owner is None:
        raise ValueError(
            "vertex_owner could not be resolved. Expected metadata field "
            "'voronoi_vertex_owner'."
        )

    mesh = OrganoidMesh(mesh_path)
    mesh.normalize_inplace()

    vertex_owner = np.asarray(vertex_owner, dtype=np.int64).reshape(-1)
    if vertex_owner.shape[0] != mesh.v.shape[0]:
        raise ValueError(
            f"vertex_owner length mismatch: {vertex_owner.shape[0]} vs "
            f"number of mesh vertices {mesh.v.shape[0]}"
        )

    if proj_vertex_ids is not None:
        proj_vertex_ids = np.asarray(proj_vertex_ids, dtype=np.int64).reshape(-1)

    node_pred = np.asarray(node_pred, dtype=np.float32).reshape(-1)
    node_true = np.asarray(node_true, dtype=np.float32).reshape(-1)

    n_nodes_meta = int(np.max(vertex_owner)) + 1 if np.any(vertex_owner >= 0) else 0
    n_nodes_graph = int(graph.y.shape[0]) if hasattr(graph, "y") else None

    if node_pred.shape[0] != n_nodes_meta:
        raise ValueError(
            f"Prediction length mismatch: {node_pred.shape[0]} vs {n_nodes_meta} "
            f"(inferred from vertex_owner)"
        )
    if node_true.shape[0] != n_nodes_meta:
        raise ValueError(
            f"Ground-truth length mismatch: {node_true.shape[0]} vs {n_nodes_meta} "
            f"(inferred from vertex_owner)"
        )

    if n_nodes_graph is not None and n_nodes_graph != n_nodes_meta:
        raise ValueError(
            f"Node count mismatch between val_graph ({n_nodes_graph}) and "
            f"vertex_owner-derived node count ({n_nodes_meta})"
        )

    if proj_vertex_ids is not None and proj_vertex_ids.shape[0] != n_nodes_meta:
        raise ValueError(
            f"proj_vertex_ids length mismatch: {proj_vertex_ids.shape[0]} vs {n_nodes_meta}"
        )

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
        "dist_mat": None,
        "mesh_path": mesh_path,
        "organoid_str": getattr(graph, "organoid_str", None),
    }

    if return_debug:
        result["proj_vertex_ids"] = proj_vertex_ids
        result["meta"] = md

    return result


def get_graph_slice_bounds(graphs, graph_index):
    sizes = [int(g.y.shape[0]) for g in graphs]
    start = sum(sizes[:graph_index])
    end = start + sizes[graph_index]
    return start, end


def project_predictions_to_mesh(
    graph_index,
    graphs,
    y_true_all,
    y_pred_all,
    graph_info=None,
    *,
    fill_value=np.nan,
    return_debug=False,
):
    g = graphs[graph_index]

    start, end = get_graph_slice_bounds(graphs, graph_index)

    node_true = y_true_all[start:end]
    node_pred = y_pred_all[start:end]

    metadata_lookup = None
    if graph_info is not None:
        if isinstance(graph_info, dict):
            metadata_lookup = graph_info
        else:
            row = graph_info[graph_index]
            organoid_str = getattr(g, "organoid_str", None)
            if organoid_str is not None:
                metadata_lookup = {organoid_str: row}

    return project_node_quantities_to_mesh(
        g,
        node_pred=node_pred,
        node_true=node_true,
        metadata_lookup=metadata_lookup,
        fill_value=fill_value,
        return_debug=return_debug,
    )