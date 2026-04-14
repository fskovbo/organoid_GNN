import os, glob, json
import copy
import torch
import numpy as np


def load_marker_names_from_dir(dir_path):
    """
    Try to find a *_markers.json sidecar in a directory of NPZs and load marker names.
    Returns: list[str] or None if not found.
    """
    jsons = sorted(glob.glob(os.path.join(dir_path, "*_markers.json")))
    if not jsons:
        return None
    with open(jsons[0], "r") as f:
        names = json.load(f)
    return list(names)


def load_aux_metadata_for_dir(dir_path: str):
    """
    Scan a directory of organoid NPZs and load per-organoid auxiliary metadata
    from sidecars named: organoid_<id>_aux.json.

    Returns
    -------
    meta_by_key : dict[str, dict]
        Maps organoid key (filename stem, e.g. "organoid_day4_A01_007")
        -> {"organoid_id": str, "total_surface_area": float|None,
            "total_volume": float|None, "complexity_score": float|None, ...}

    Notes
    -----
    - Missing sidecars are simply skipped; you’ll still get metadata for the ones present.
    - Values are coerced to Python floats where possible; missing fields are omitted.
    """
    meta = {}

    for npz_path in sorted(glob.glob(os.path.join(dir_path, "organoid_*.npz"))):
        stem = os.path.splitext(os.path.basename(npz_path))[0]
        aux_path = os.path.join(dir_path, f"{stem}_aux.json")

        if not os.path.exists(aux_path):
            continue

        try:
            with open(aux_path, "r") as f:
                raw = json.load(f)
        except Exception as e:
            print(f"Warning: failed reading {aux_path}: {e}")
            continue

        clean = {"organoid_id": str(raw.get("organoid_id", stem.replace("organoid_", "")))}

        for k, v in raw.items():
            if k == "organoid_id":
                continue

            try:
                if isinstance(v, (list, tuple)) and len(v) == 1:
                    v = v[0]
                if hasattr(v, "item"):
                    v = v.item()
            except Exception:
                pass

            # keep numeric values as Python floats
            if isinstance(v, (int, float, np.number)):
                clean[k] = float(v)
            else:
                # keep strings / bools / lists / etc. too
                clean[k] = v

        meta[stem] = clean

    return meta


def attach_metadata_to_graphs(graphs, meta_by_stem, include_keys=None, exclude_keys=None):
    """
    Attach metadata dicts to graphs as g.meta.

    Parameters
    ----------
    graphs : list[Data]
    meta_by_stem : dict
        Output of load_aux_metadata_for_dir(...)
    include_keys : iterable[str] or None
        If given, only these metadata keys are attached.
    exclude_keys : iterable[str] or None
        If given, these keys are removed before attaching.

    Returns
    -------
    attached : int
        Number of graphs that received metadata.
    """
    if include_keys is not None and exclude_keys is not None:
        raise ValueError("Use only one of include_keys or exclude_keys")

    include_keys = set(include_keys) if include_keys is not None else None
    exclude_keys = set(exclude_keys) if exclude_keys is not None else set()

    attached = 0

    for g in graphs:
        stem = getattr(g, "organoid_str", None)
        if stem is None:
            continue

        md = meta_by_stem.get(stem, None)
        if md is None:
            continue

        md = dict(md)

        if include_keys is not None:
            md = {k: v for k, v in md.items() if k in include_keys}
        else:
            md = {k: v for k, v in md.items() if k not in exclude_keys}

        g.meta = md
        attached += 1

    return attached


def metadata_dataframe(
    graphs,
    *,
    extra_cols=("num_nodes",),
    include_meta=True,
    prefix_meta=False,
):
    """
    Build a dataframe from graphs + metadata.

    Parameters
    ----------
    graphs : list[Data]
    extra_cols : iterable[str]
        Extra computed columns (e.g. "num_nodes")
    include_meta : bool
        If True, include ALL metadata fields dynamically
    prefix_meta : bool
        If True, prefix metadata columns with 'meta_'

    Returns
    -------
    df : pandas.DataFrame
    """
    import pandas as pd

    rows = []

    for g in graphs:
        key = getattr(g, "organoid_str", None)
        md = getattr(g, "meta", {}) or {}

        row = {
            "organoid": key,
        }

        # --- include all metadata dynamically ---
        if include_meta:
            for k, v in md.items():
                col = f"meta_{k}" if prefix_meta else k
                row[col] = v

        # --- optional computed columns ---
        if "num_nodes" in extra_cols:
            if hasattr(g, "x"):
                row["num_nodes"] = int(g.x.size(0))
            elif hasattr(g, "y"):
                row["num_nodes"] = int(g.y.shape[0])

        rows.append(row)

    return pd.DataFrame(rows)



def ensure_metadata_keys(
    graphs,
    required_keys=None,
    defaults=None,
    inplace=False,
):
    """
    Ensure every graph has a .meta dict with the same keys.

    Parameters
    ----------
    graphs : list[Data]
    required_keys : iterable[str] or None
        Keys that must exist in every g.meta.
    defaults : dict or None
        Default values to use for missing keys.
    inplace : bool
        If False, returns shallow copies of graphs.

    Returns
    -------
    graphs_out : list[Data]
    """
    if required_keys is None:
        required_keys = []
    if defaults is None:
        defaults = {}

    graphs_out = graphs if inplace else [copy.copy(g) for g in graphs]

    for g in graphs_out:
        md = getattr(g, "meta", None)
        if md is None:
            md = {}
        else:
            md = dict(md)

        for key in required_keys:
            if key not in md:
                md[key] = defaults.get(key, np.nan)

        # also make sure any explicitly provided defaults exist
        for key, value in defaults.items():
            if key not in md:
                md[key] = value

        g.meta = md

    return graphs_out


def filter_graphs_by_metadata(
    graphs,
    key,
    keep_values=None,
    drop_values=None,
    missing="keep",   # "keep" | "drop"
    inplace=False,
):
    """
    Filter graphs based on metadata field g.meta[key].

    Parameters
    ----------
    graphs : list[Data]
    key : str
        Metadata key to filter on (e.g. "timepoint")
    keep_values : iterable or None
        If given, only graphs with meta[key] in keep_values are kept.
    drop_values : iterable or None
        If given, graphs with meta[key] in drop_values are removed.
    missing : str
        How to treat graphs where key is missing or None.
        "keep" → keep them
        "drop" → remove them
    inplace : bool
        If False, returns shallow copies.

    Returns
    -------
    filtered_graphs : list[Data]
    """

    if keep_values is not None and drop_values is not None:
        raise ValueError("Specify only one of keep_values or drop_values")

    if keep_values is not None:
        keep_values = set(keep_values)

    if drop_values is not None:
        drop_values = set(drop_values)

    out = []

    for g in graphs:
        md = getattr(g, "meta", {}) or {}
        val = md.get(key, None)

        if val is None:
            if missing == "keep":
                out.append(g if inplace else copy.copy(g))
            continue

        keep_flag = True

        if keep_values is not None:
            keep_flag = val in keep_values

        if drop_values is not None:
            keep_flag = val not in drop_values

        if keep_flag:
            out.append(g if inplace else copy.copy(g))

    return out


def strip_graph_metadata(graphs, attr_names=("meta",), inplace=False):
    """
    Remove non-batchable metadata attributes from PyG Data objects.
    """
    graphs_out = graphs if inplace else [copy.copy(g) for g in graphs]

    for g in graphs_out:
        for attr in attr_names:
            if hasattr(g, attr):
                delattr(g, attr)

    return graphs_out


def restore_graph_metadata(
    graphs,
    meta_lookup,
    *,
    key_attr="organoid_str",
    meta_attr="meta",
    inplace=False,
    strict=True,
):
    """
    Reattach metadata dicts to graphs from a lookup.

    Parameters
    ----------
    graphs : list[Data]
    meta_lookup : dict[str, dict]
        Output of snapshot_graph_metadata(...)
    key_attr : str
        Graph attribute used as stable identifier
    meta_attr : str
        Attribute to write metadata into
    inplace : bool
        If False, shallow-copy graphs before attaching metadata
    strict : bool
        If True, raise if a graph key is missing from meta_lookup

    Returns
    -------
    graphs_out : list[Data]
    """
    graphs_out = graphs if inplace else [copy.copy(g) for g in graphs]

    for i, g in enumerate(graphs_out):
        key = getattr(g, key_attr, None)
        if key is None:
            if strict:
                raise ValueError(f"Graph at index {i} has no {key_attr!r}")
            continue

        md = meta_lookup.get(key, None)
        if md is None:
            if strict:
                raise KeyError(f"No metadata found for graph key {key!r}")
            continue

        setattr(g, meta_attr, copy.deepcopy(md))

    return graphs_out



def print_graph_and_metadata_fields(graphs):
    """
    Print:
      - graph attribute names
      - metadata keys (inside g.meta)
    """
    graph_keys = set()
    meta_keys = set()

    for g in graphs:
        graph_keys.update(vars(g).keys())

        md = getattr(g, "meta", None)
        if isinstance(md, dict):
            meta_keys.update(md.keys())

    print("Graph fields:")
    for k in sorted(graph_keys):
        print(f"  - {k}")

    print("\nMetadata fields (g.meta):")
    for k in sorted(meta_keys):
        print(f"  - {k}")



def snapshot_graph_metadata(graphs, key_attr="organoid_str", meta_attr="meta"):
    """
    Save all metadata dicts from graphs into a lookup keyed by `key_attr`.

    Parameters
    ----------
    graphs : list[Data]
    key_attr : str
        Graph attribute used as stable identifier, usually 'organoid_str'
    meta_attr : str
        Attribute containing metadata dict, usually 'meta'

    Returns
    -------
    meta_lookup : dict[str, dict]
        Maps graph id -> deep-copied metadata dict
    """
    out = {}

    for i, g in enumerate(graphs):
        key = getattr(g, key_attr, None)
        if key is None:
            raise ValueError(f"Graph at index {i} has no {key_attr!r}")

        md = getattr(g, meta_attr, None)
        if md is None:
            out[key] = {}
        elif not isinstance(md, dict):
            raise TypeError(
                f"Graph {key!r} has non-dict {meta_attr!r}: {type(md)}"
            )
        else:
            out[key] = copy.deepcopy(md)

    return out



def promote_metadata_to_graph_tensors(
    graphs,
    field_specs,
    *,
    meta_attr="meta",
    inplace=False,
):
    """
    Copy selected metadata fields from g.meta onto PyG graphs as tensor attributes.

    This is useful for quantities that should survive metadata stripping and be
    batchable during training, such as:
      - node-level fields: cell_patch_area
      - graph-level scalars: total_surface_area
      - graph-level vectors: global_feat built from multiple metadata fields

    Parameters
    ----------
    graphs : list[Data]
        PyG graphs.
    field_specs : list[dict]
        Each spec must be one of the following forms.

        1) Single metadata field -> one graph attribute
           {
               "meta_key": "cell_patch_area",
               "attr_name": "cell_patch_area",   # optional; defaults to meta_key
               "kind": "node" | "graph",
               "dtype": torch.float32,           # optional
           }

        2) Multiple metadata fields -> one graph-level vector attribute
           {
               "meta_keys": ["total_surface_area", "total_volume"],
               "attr_name": "global_feat",
               "kind": "graph_vector",
               "dtype": torch.float32,           # optional
           }

    meta_attr : str
        Name of the metadata attribute on each graph, usually "meta".
    inplace : bool
        If False, returns shallow copies of the graphs.

    Returns
    -------
    graphs_out : list[Data]
        Graphs with new tensor attributes added.

    Notes
    -----
    - "node" fields must have length equal to the number of nodes.
    - "graph" fields are stored as shape (1,) tensors so PyG batches them cleanly.
    - "graph_vector" fields are stored as shape (D,) tensors.
    """
    graphs_out = graphs if inplace else [copy.copy(g) for g in graphs]

    for i, g in enumerate(graphs_out):
        md = getattr(g, meta_attr, None)
        if not isinstance(md, dict):
            raise ValueError(f"Graph {i} has no dict metadata in {meta_attr!r}")

        n_nodes = None
        if hasattr(g, "x") and g.x is not None:
            n_nodes = int(g.x.shape[0])
        elif hasattr(g, "y") and g.y is not None:
            n_nodes = int(g.y.shape[0])

        for spec in field_specs:
            dtype = spec.get("dtype", torch.float32)
            kind = spec.get("kind", None)

            # ----------------------------------------------------------
            # Case 1: one metadata field -> one tensor attribute
            # ----------------------------------------------------------
            if "meta_key" in spec:
                meta_key = spec["meta_key"]
                attr_name = spec.get("attr_name", meta_key)

                if kind is None:
                    raise ValueError(
                        f"Spec for meta_key={meta_key!r} is missing required 'kind'"
                    )

                if meta_key not in md:
                    raise KeyError(f"Graph {i} missing metadata field {meta_key!r}")

                value = md[meta_key]

                if kind == "node":
                    arr = np.asarray(value).reshape(-1)

                    if n_nodes is None:
                        raise ValueError(
                            f"Graph {i}: cannot validate node field {meta_key!r} "
                            "because neither x nor y is present"
                        )

                    if arr.shape[0] != n_nodes:
                        raise ValueError(
                            f"Graph {i}: node field {meta_key!r} has length {arr.shape[0]} "
                            f"but graph has {n_nodes} nodes"
                        )

                    tensor = torch.as_tensor(arr, dtype=dtype)

                elif kind == "graph":
                    arr = np.asarray([value])
                    tensor = torch.as_tensor(arr, dtype=dtype)

                else:
                    raise ValueError(
                        f"Unknown kind={kind!r} for single-field spec; "
                        "expected 'node' or 'graph'"
                    )

                setattr(g, attr_name, tensor)

            # ----------------------------------------------------------
            # Case 2: many metadata fields -> one graph vector
            # ----------------------------------------------------------
            elif "meta_keys" in spec:
                meta_keys = list(spec["meta_keys"])
                attr_name = spec.get("attr_name", None)

                if kind != "graph_vector":
                    raise ValueError(
                        "Specs with 'meta_keys' must use kind='graph_vector'"
                    )
                if not attr_name:
                    raise ValueError(
                        "Specs with 'meta_keys' must provide 'attr_name'"
                    )
                if len(meta_keys) == 0:
                    raise ValueError("meta_keys must not be empty")

                values = []
                for key in meta_keys:
                    if key not in md:
                        raise KeyError(f"Graph {i} missing metadata field {key!r}")
                    values.append(md[key])

                arr = np.asarray(values).reshape(1, -1)
                tensor = torch.as_tensor(arr, dtype=dtype)
                setattr(g, attr_name, tensor)

            else:
                raise ValueError(
                    "Each field spec must contain either 'meta_key' or 'meta_keys'"
                )

    return graphs_out


def infer_global_dim(graphs, attr_name="global_feat"):
    """
    Infer the dimensionality of graph-level features.
    Returns 0 if the attribute does not exist.
    """
    if len(graphs) == 0:
        return 0

    g0 = graphs[0]

    if not hasattr(g0, attr_name):
        return 0

    x = getattr(g0, attr_name)

    if x is None:
        return 0

    # handle shapes: (D,), (1, D)
    if x.ndim == 1:
        return int(x.shape[0])
    elif x.ndim == 2:
        return int(x.shape[-1])
    else:
        raise ValueError(
            f"{attr_name} has unexpected shape {tuple(x.shape)}"
        )