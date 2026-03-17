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
    # infer stems from existing npz files
    for npz_path in sorted(glob.glob(os.path.join(dir_path, "organoid_*.npz"))):
        stem = os.path.splitext(os.path.basename(npz_path))[0]            # organoid_<id>
        aux_path = os.path.join(dir_path, f"{stem}_aux.json")
        if not os.path.exists(aux_path):
            continue
        try:
            with open(aux_path, "r") as f:
                raw = json.load(f)
        except Exception as e:
            print(f"Warning: failed reading {aux_path}: {e}")
            continue

        # Coerce values to plain Python floats/ints where possible
        clean = {"organoid_id": str(raw.get("organoid_id", stem.replace("organoid_", "")))}
        for k, v in raw.items():
            if k == "organoid_id": 
                continue
            try:
                # Handle numpy scalars/0-D arrays robustly
                if isinstance(v, (list, tuple)) and len(v) == 1:
                    v = v[0]
                if hasattr(v, "item"):
                    v = v.item()
                if isinstance(v, (int, float)):
                    clean[k] = float(v)
            except Exception:
                # Keep as-is if not numeric
                clean[k] = v
        meta[stem] = clean
    return meta


def attach_metadata_to_graphs(graphs, meta_by_key: dict, quiet: bool = True):
    """
    Attach metadata dicts to each PyG Data in-place as `data.meta`
    using the graph's `organoid_str` (filename stem) as the key.

    Returns the number of graphs that received metadata.
    Safe: does NOT alter x/y/edge_index; won’t affect training & collate.
    """
    count = 0
    for g in graphs:
        key = getattr(g, "organoid_str", None)
        if key is None:
            # fall back to nothing; you can add keys before calling this
            continue
        md = meta_by_key.get(key)
        if md is None:
            if not quiet:
                print(f"No metadata for {key}")
            continue
        # attach a shallow copy to avoid accidental mutation
        g.meta = dict(md)
        count += 1
    return count


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