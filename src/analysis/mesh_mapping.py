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



def project_node_categories_to_mesh(
    graph,
    node_categories,
    *,
    meta_lookup=None,
    mesh_path=None,
    vertex_owner=None,
    normalize_mesh=True,
    missing_category=None,
):
    """Project one categorical node label per cell onto mesh vertices via vertex ownership.

    Parameters
    ----------
    missing_category : object or None
        Category assigned to mesh vertices with no owning node (vertex_owner < 0).
        If None, those vertices remain unassigned.
    """
    md = get_graph_metadata(graph, meta_lookup=meta_lookup, strict=True)

    if mesh_path is None:
        mesh_path = md.get("mesh_path")
    if vertex_owner is None:
        vertex_owner = md.get("voronoi_vertex_owner")

    if mesh_path is None:
        raise ValueError("Missing mesh_path")
    if vertex_owner is None:
        raise ValueError("Missing voronoi_vertex_owner")

    mesh = OrganoidMesh(mesh_path)
    if normalize_mesh:
        mesh.normalize_inplace()

    vertex_owner = np.asarray(vertex_owner, dtype=np.int64).reshape(-1)
    node_categories = np.asarray(node_categories, dtype=object).reshape(-1)

    n_nodes = int(graph.y.shape[0])
    if node_categories.shape[0] != n_nodes:
        raise ValueError(
            f"node_categories has length {node_categories.shape[0]}, expected {n_nodes}"
        )

    mesh_categories = np.empty(vertex_owner.shape[0], dtype=object)
    mesh_categories[:] = missing_category

    valid = vertex_owner >= 0
    mesh_categories[valid] = node_categories[vertex_owner[valid]]

    return {
        "mesh": mesh,
        "mesh_categories": mesh_categories,
        "vertex_owner": vertex_owner,
        "organoid_str": getattr(graph, "organoid_str", None),
        "missing_category": missing_category,
    }



def categories_to_vertex_regions(mesh_categories, category_order=None):
    """Convert per-vertex categorical labels into vertex regions for plot_mesh_by_regions.

    Parameters
    ----------
    mesh_categories : array-like, shape (V,)
        One categorical label per mesh vertex. Missing/unassigned entries can be None.
    category_order : sequence or None
        If provided, regions are returned in this order. Otherwise the order of first
        appearance among non-missing categories is used.

    Returns
    -------
    regions : list[np.ndarray]
        One vertex-index array per category.
    category_order : list
        Category names corresponding to `regions`.
    """
    cats = np.asarray(mesh_categories, dtype=object).reshape(-1)

    if category_order is None:
        seen = []
        seen_set = set()
        for c in cats:
            if c is None:
                continue
            if c not in seen_set:
                seen.append(c)
                seen_set.add(c)
        category_order = seen
    else:
        category_order = list(category_order)

    regions = [np.where(cats == cat)[0].astype(np.int64) for cat in category_order]
    return regions, category_order