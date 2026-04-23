import copy
import glob
import json
import os
from typing import Any

import numpy as np
import torch


"""Utilities for attaching, validating, querying, and promoting graph metadata."""


# -----------------------------------------------------------------------------
# IO / attachment
# -----------------------------------------------------------------------------

def load_marker_names_from_dir(dir_path: str):
    """Load marker names from a *_markers.json sidecar if present."""
    jsons = sorted(glob.glob(os.path.join(dir_path, "*_markers.json")))
    if not jsons:
        return None
    with open(jsons[0], "r") as f:
        names = json.load(f)
    return list(names)



def load_aux_metadata_for_dir(dir_path: str):
    """Load auxiliary per-organoid JSON metadata from a dataset directory."""
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

            if isinstance(v, (int, float, np.number)):
                clean[k] = float(v)
            else:
                clean[k] = v

        meta[stem] = clean

    return meta



def attach_metadata_to_graphs(graphs, meta_by_stem, include_keys=None, exclude_keys=None):
    """Attach metadata dicts to graphs as ``g.meta``."""
    if include_keys is not None and exclude_keys is not None:
        raise ValueError("Use only one of include_keys or exclude_keys")

    include_keys = set(include_keys) if include_keys is not None else None
    exclude_keys = set(exclude_keys) if exclude_keys is not None else set()
    attached = 0

    for g in graphs:
        stem = getattr(g, "organoid_str", None)
        if stem is None:
            continue

        md = meta_by_stem.get(stem)
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


# -----------------------------------------------------------------------------
# Metadata normalization / lookup
# -----------------------------------------------------------------------------

def ensure_metadata_keys(graphs, required_keys=None, defaults=None, inplace=False):
    """Ensure every graph has a ``.meta`` dict containing the requested keys."""
    required_keys = [] if required_keys is None else list(required_keys)
    defaults = {} if defaults is None else dict(defaults)
    graphs_out = graphs if inplace else [copy.copy(g) for g in graphs]

    for g in graphs_out:
        md = getattr(g, "meta", None)
        md = {} if md is None else dict(md)

        for key in required_keys:
            if key not in md:
                md[key] = defaults.get(key, np.nan)
        for key, value in defaults.items():
            if key not in md:
                md[key] = value

        g.meta = md

    return graphs_out



def snapshot_graph_metadata(graphs, key_attr="organoid_str", meta_attr="meta"):
    """Create a deep-copied lookup ``{graph_key: metadata_dict}`` from graphs."""
    out = {}
    for i, g in enumerate(graphs):
        key = getattr(g, key_attr, None)
        if key is None:
            raise ValueError(f"Graph at index {i} has no {key_attr!r}")

        md = getattr(g, meta_attr, None)
        if md is None:
            out[key] = {}
        elif not isinstance(md, dict):
            raise TypeError(f"Graph {key!r} has non-dict {meta_attr!r}: {type(md)}")
        else:
            out[key] = copy.deepcopy(md)
    return out



def restore_graph_metadata(
    graphs,
    meta_lookup,
    *,
    key_attr="organoid_str",
    meta_attr="meta",
    inplace=False,
    strict=True,
):
    """Reattach metadata dicts to graphs from a lookup."""
    graphs_out = graphs if inplace else [copy.copy(g) for g in graphs]

    for i, g in enumerate(graphs_out):
        key = getattr(g, key_attr, None)
        if key is None:
            if strict:
                raise ValueError(f"Graph at index {i} has no {key_attr!r}")
            continue

        md = meta_lookup.get(key)
        if md is None:
            if strict:
                raise KeyError(f"No metadata found for graph key {key!r}")
            continue

        setattr(g, meta_attr, copy.deepcopy(md))

    return graphs_out



def get_graph_metadata(
    graph,
    meta_lookup=None,
    *,
    key_attr="organoid_str",
    meta_attr="meta",
    strict=True,
    default=None,
):
    """Resolve metadata either from ``graph.meta`` or from ``meta_lookup``."""
    md = getattr(graph, meta_attr, None)
    if isinstance(md, dict):
        return md

    if meta_lookup is None:
        if strict:
            raise ValueError("Graph has no attached metadata and no metadata_lookup was provided.")
        return {} if default is None else default

    key = getattr(graph, key_attr, None)
    if key is None:
        if strict:
            raise ValueError(f"Graph has no {key_attr!r} attribute.")
        return {} if default is None else default

    md = meta_lookup.get(key)
    if md is None:
        if strict:
            raise KeyError(f"No metadata found for graph key {key!r}")
        return {} if default is None else default

    return md



def get_graph_metadata_value(
    graph,
    key,
    meta_lookup=None,
    *,
    default=None,
    strict=False,
    cast=None,
):
    """Resolve one metadata field from a graph, optionally casting the value."""
    md = get_graph_metadata(graph, meta_lookup=meta_lookup, strict=strict)
    value = md.get(key, default)
    if cast is not None and value is not None:
        value = cast(value)
    return value



def _graph_num_nodes(graph) -> int:
    if hasattr(graph, "x") and graph.x is not None:
        return int(graph.x.shape[0])
    if hasattr(graph, "y") and graph.y is not None:
        return int(graph.y.shape[0])
    raise ValueError("Could not infer number of nodes from graph")



def get_metadata_array(graphs, key, meta_lookup=None, *, dtype=float, missing_value=np.nan):
    """Extract one scalar metadata value per graph as a dense array."""
    out = []
    for g in graphs:
        value = get_graph_metadata_value(g, key, meta_lookup=meta_lookup, default=None, strict=False)
        if value is None:
            out.append(missing_value)
        else:
            out.append(value)
    return np.asarray(out, dtype=dtype)



# -----------------------------------------------------------------------------
# Filtering helpers
# -----------------------------------------------------------------------------

import copy
from collections import defaultdict

import numpy as np

# assumes get_graph_metadata_value is already available


def _print_filter_summary(graphs, kept_mask, *, meta_lookup=None, label="Filter summary"):
    """
    Print kept/total counts per (dataset, timepoint).

    Parameters
    ----------
    graphs : list
    kept_mask : array-like[bool], shape (len(graphs),)
    meta_lookup : dict or None
    label : str
    """
    kept_mask = np.asarray(kept_mask, dtype=bool)
    if kept_mask.shape[0] != len(graphs):
        raise ValueError("kept_mask must have length len(graphs)")

    counts = defaultdict(lambda: {"kept": 0, "total": 0})

    for keep, g in zip(kept_mask, graphs):
        dataset = get_graph_metadata_value(
            g, "dataset", meta_lookup=meta_lookup, default="MISSING", strict=False
        )
        timepoint = get_graph_metadata_value(
            g, "timepoint", meta_lookup=meta_lookup, default="MISSING", strict=False
        )
        key = (dataset, timepoint)

        counts[key]["total"] += 1
        if keep:
            counts[key]["kept"] += 1

    n_kept = int(kept_mask.sum())
    n_total = len(graphs)

    print(f"{label}: kept {n_kept} / {n_total} graphs\n")
    print(f"{'dataset':<16} {'timepoint':<16} {'kept':>6} {'total':>6} {'frac':>8}")
    print("-" * 58)

    for (dataset, timepoint) in sorted(counts.keys(), key=lambda x: (str(x[0]), str(x[1]))):
        kept = counts[(dataset, timepoint)]["kept"]
        total = counts[(dataset, timepoint)]["total"]
        frac = kept / total if total > 0 else np.nan
        print(f"{str(dataset):<16} {str(timepoint):<16} {kept:>6} {total:>6} {frac:>8.3f}")


def filter_graphs_by_metadata(
    graphs,
    key,
    keep_values=None,
    drop_values=None,
    missing="keep",
    inplace=False,
    *,
    meta_lookup=None,
    print_summary=False,
):
    """
    Filter graphs by a categorical metadata field.

    Parameters
    ----------
    graphs : list
    key : str
        Metadata key to filter on.
    keep_values : iterable or None
        Keep only graphs whose metadata value is in this set.
    drop_values : iterable or None
        Drop graphs whose metadata value is in this set.
    missing : {"keep", "drop"}
        What to do when the metadata key is missing.
    inplace : bool
    meta_lookup : dict or None
        Optional metadata lookup.
    print_summary : bool
        If True, print kept/total counts per (dataset, timepoint).
    """
    if keep_values is not None and drop_values is not None:
        raise ValueError("Specify only one of keep_values or drop_values")
    if missing not in ("keep", "drop"):
        raise ValueError("missing must be 'keep' or 'drop'")

    keep_values = set(keep_values) if keep_values is not None else None
    drop_values = set(drop_values) if drop_values is not None else None

    out = []
    kept_mask = []

    for g in graphs:
        val = get_graph_metadata_value(
            g, key, meta_lookup=meta_lookup, default=None, strict=False
        )

        if val is None:
            keep_flag = (missing == "keep")
        else:
            keep_flag = True
            if keep_values is not None:
                keep_flag = val in keep_values
            if drop_values is not None:
                keep_flag = val not in drop_values

        kept_mask.append(keep_flag)
        if keep_flag:
            out.append(g if inplace else copy.copy(g))

    if print_summary:
        _print_filter_summary(
            graphs,
            kept_mask,
            meta_lookup=meta_lookup,
            label=f"filter_graphs_by_metadata(key={key!r})",
        )

    return out


def filter_graphs_by_numeric_metadata(
    graphs,
    key,
    meta_lookup=None,
    *,
    min_value=None,
    max_value=None,
    allow_missing=False,
    inplace=False,
    print_summary=False,
):
    """
    Filter graphs by a numeric metadata field such as complexity.

    Parameters
    ----------
    graphs : list
    key : str
        Metadata key to filter on.
    meta_lookup : dict or None
    min_value : float or None
    max_value : float or None
    allow_missing : bool
        Whether missing / non-finite values should be kept.
    inplace : bool
    print_summary : bool
        If True, print kept/total counts per (dataset, timepoint).
    """
    out = []
    kept_mask = []

    for g in graphs:
        val = get_graph_metadata_value(
            g, key, meta_lookup=meta_lookup, default=None, strict=False
        )
        keep = True

        if val is None:
            keep = allow_missing
        else:
            val = float(val)
            if not np.isfinite(val):
                keep = allow_missing
            else:
                if min_value is not None:
                    keep = keep and (val >= float(min_value))
                if max_value is not None:
                    keep = keep and (val <= float(max_value))

        kept_mask.append(keep)
        if keep:
            out.append(g if inplace else copy.copy(g))

    if print_summary:
        _print_filter_summary(
            graphs,
            kept_mask,
            meta_lookup=meta_lookup,
            label=(
                f"filter_graphs_by_numeric_metadata("
                f"key={key!r}, min_value={min_value}, max_value={max_value})"
            ),
        )

    return out


def filter_graphs_by_sphericity(
    graphs,
    meta_lookup=None,
    *,
    area_key="total_surface_area",
    volume_key="total_volume",
    max_sphericity=0.95,
    allow_missing=False,
    inplace=False,
    print_summary=False,
    return_scores=False,
):
    """
    Filter out graphs that are too close to being a sphere using the
    isoperimetric quotient

        Q = 36 * pi * V^2 / A^3

    where Q = 1 for a perfect sphere and Q < 1 otherwise.

    Parameters
    ----------
    graphs : list
    meta_lookup : dict or None
    area_key : str
        Metadata key for surface area.
    volume_key : str
        Metadata key for volume.
    max_sphericity : float
        Keep only graphs with Q < max_sphericity.
        Example: max_sphericity=0.95 removes graphs with Q >= 0.95.
    allow_missing : bool
        If True, keep graphs with missing/non-finite metadata.
    inplace : bool
    print_summary : bool
        If True, print kept/total counts per (dataset, timepoint).
    return_scores : bool
        If True, also return the per-graph Q values.

    Returns
    -------
    filtered_graphs : list
    scores : np.ndarray, optional
        Returned only if return_scores=True.
    """
    out = []
    kept_mask = []
    scores = []

    for g in graphs:
        area = get_graph_metadata_value(
            g, area_key, meta_lookup=meta_lookup, default=None, strict=False
        )
        volume = get_graph_metadata_value(
            g, volume_key, meta_lookup=meta_lookup, default=None, strict=False
        )

        keep = True
        q = np.nan

        if area is None or volume is None:
            keep = allow_missing
        else:
            area = float(area)
            volume = float(volume)

            if (not np.isfinite(area)) or (not np.isfinite(volume)) or area <= 0.0 or volume <= 0.0:
                keep = allow_missing
            else:
                q = 36.0 * np.pi * (volume ** 2) / (area ** 3)
                keep = (q < float(max_sphericity))

        scores.append(q)
        kept_mask.append(keep)

        if keep:
            out.append(g if inplace else copy.copy(g))

    if print_summary:
        _print_filter_summary(
            graphs,
            kept_mask,
            meta_lookup=meta_lookup,
            label=f"filter_graphs_by_sphericity(max_sphericity={max_sphericity})",
        )

    scores = np.asarray(scores, dtype=float)
    if return_scores:
        return out, scores
    return out



def fill_missing_metadata_for_group(
    graphs,
    field,
    fill_value,
    *,
    dataset,
    timepoint,
    inplace=False,
):
    """
    Fill missing metadata field for graphs belonging to a specific (dataset, timepoint).

    Parameters
    ----------
    graphs : list
    field : str
        Metadata field to fill.
    fill_value : any
        Value to assign if the field is missing or None.
    dataset : str
    timepoint : str
    inplace : bool

    Returns
    -------
    list
        Updated graphs.
    """
    out = []

    for g in graphs:
        md = getattr(g, "meta", None)
        if md is None:
            md = {}
            g.meta = md

        g_dataset = md.get("dataset", None)
        g_timepoint = md.get("timepoint", None)

        g_out = g if inplace else copy.copy(g)

        # Only act on specified group
        if g_dataset == dataset and g_timepoint == timepoint:
            val = md.get(field, None)
            if val is None:
                g_out.meta[field] = fill_value

        out.append(g_out)

    return out

# -----------------------------------------------------------------------------
# Graph attribute utilities
# -----------------------------------------------------------------------------

def strip_graph_metadata(graphs, attr_names=("meta",), inplace=False):
    """Remove non-batchable metadata attributes from PyG Data objects."""
    graphs_out = graphs if inplace else [copy.copy(g) for g in graphs]
    for g in graphs_out:
        for attr in attr_names:
            if hasattr(g, attr):
                delattr(g, attr)
    return graphs_out



def add_log_metadata_features(
    graphs,
    *,
    meta_lookup=None,
    area_key="total_surface_area",
    volume_key="total_volume",
    num_nodes_key="num_nodes",
    eps=1e-12,
    inplace=False,
):
    """Add commonly used log-transformed global metadata fields into ``g.meta``."""
    graphs_out = graphs if inplace else [copy.copy(g) for g in graphs]

    for g in graphs_out:
        md = dict(get_graph_metadata(g, meta_lookup=meta_lookup, strict=True))
        area = float(md[area_key])
        volume = float(md[volume_key])
        num_nodes = float(md[num_nodes_key])

        md["log_surface_area"] = np.log(area + eps)
        md["log_volume"] = np.log(volume + eps)
        md["log_volume_over_area"] = np.log((volume + eps) / (area + eps))
        md["log_num_cells"] = np.log(num_nodes + eps)
        g.meta = md

    return graphs_out



def promote_metadata_to_graph_tensors(graphs, field_specs, *, meta_attr="meta", inplace=False):
    """Copy selected metadata fields from ``g.meta`` to batchable tensor attributes."""
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

            if "meta_key" in spec:
                meta_key = spec["meta_key"]
                attr_name = spec.get("attr_name", meta_key)
                if kind is None:
                    raise ValueError(f"Spec for meta_key={meta_key!r} is missing required 'kind'")
                if meta_key not in md:
                    raise KeyError(f"Graph {i} missing metadata field {meta_key!r}")

                value = md[meta_key]
                if kind == "node":
                    arr = np.asarray(value).reshape(-1)
                    if n_nodes is None:
                        raise ValueError(f"Graph {i}: cannot validate node field {meta_key!r}")
                    if arr.shape[0] != n_nodes:
                        raise ValueError(
                            f"Graph {i}: node field {meta_key!r} has length {arr.shape[0]} but graph has {n_nodes} nodes"
                        )
                    tensor = torch.as_tensor(arr, dtype=dtype)
                elif kind == "graph":
                    tensor = torch.as_tensor(np.asarray([value]), dtype=dtype)
                else:
                    raise ValueError(f"Unknown kind={kind!r}; expected 'node' or 'graph'")
                setattr(g, attr_name, tensor)

            elif "meta_keys" in spec:
                meta_keys = list(spec["meta_keys"])
                attr_name = spec.get("attr_name", None)
                if kind != "graph_vector":
                    raise ValueError("Specs with 'meta_keys' must use kind='graph_vector'")
                if not attr_name:
                    raise ValueError("Specs with 'meta_keys' must provide 'attr_name'")
                values = [md[key] for key in meta_keys]
                tensor = torch.as_tensor(np.asarray(values).reshape(1, -1), dtype=dtype)
                setattr(g, attr_name, tensor)
            else:
                raise ValueError("Each field spec must contain either 'meta_key' or 'meta_keys'")

    return graphs_out



def infer_global_dim(graphs, attr_name="global_feat"):
    """Infer the dimensionality of a graph-level tensor attribute."""
    if len(graphs) == 0:
        return 0
    g0 = graphs[0]
    if not hasattr(g0, attr_name):
        return 0
    x = getattr(g0, attr_name)
    if x is None:
        return 0
    if x.ndim == 1:
        return int(x.shape[0])
    if x.ndim == 2:
        return int(x.shape[-1])
    raise ValueError(f"{attr_name} has unexpected shape {tuple(x.shape)}")


# -----------------------------------------------------------------------------
# Tabular inspection
# -----------------------------------------------------------------------------

def metadata_dataframe(graphs, *, extra_cols=("num_nodes",), include_meta=True, prefix_meta=False, meta_lookup=None):
    """Build a dataframe from graphs and their metadata."""
    import pandas as pd

    rows = []
    for g in graphs:
        row = {"organoid": getattr(g, "organoid_str", None)}

        if include_meta:
            md = get_graph_metadata(g, meta_lookup=meta_lookup, strict=False, default={})
            for k, v in md.items():
                row[f"meta_{k}" if prefix_meta else k] = v

        if "num_nodes" in extra_cols:
            row["num_nodes"] = _graph_num_nodes(g)

        rows.append(row)

    return pd.DataFrame(rows)



def print_graph_and_metadata_fields(graphs):
    """Print graph attribute names and the union of all metadata keys."""
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