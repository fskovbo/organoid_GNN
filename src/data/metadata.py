import os, glob, json
import copy
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


def metadata_dataframe(graphs, extra_cols=("num_nodes",)):
    import pandas as pd
    rows = []
    for g in graphs:
        key = getattr(g, "organoid_str", None)
        md  = getattr(g, "meta", {}) or {}
        row = {"organoid": key}
        row.update({
            "complexity": md.get("complexity"),
            "dataset": md.get("dataset"),
            "timepoint": md.get("timepoint"),
            "num_nodes_meta": md.get("num_nodes"),
        })
        if "num_nodes" in extra_cols:
            row["num_nodes"] = int(g.x.size(0))
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