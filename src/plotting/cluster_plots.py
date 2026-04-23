import inspect
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.data.metadata import get_graph_metadata_value


"""Reusable matplotlib plots for cluster-level summaries."""


def plot_cluster_marker_heatmap(
    values,
    marker_names,
    *,
    cluster_labels=None,
    figsize=(6, 4),
    colorbar_label="fraction positive",
    title=None,
    ax=None,
    vmin=0.0,
    vmax=1.0,
    cmap=None,
    show_colorbar=True,
):
    """Plot a single cluster-by-marker heatmap and return the figure."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 2:
        raise ValueError(f"values must be 2-D, got shape {values.shape}")

    K = values.shape[0]
    if cluster_labels is None:
        cluster_labels = [f"C{k}" for k in range(K)]

    created_fig = ax is None
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    im = ax.imshow(values, aspect="auto", vmin=vmin, vmax=vmax, cmap=cmap)
    ax.set_xlabel("marker")
    ax.set_ylabel("cluster")
    ax.set_xticks(range(len(marker_names)))
    ax.set_xticklabels(marker_names, rotation=45, ha="right")
    ax.set_yticks(range(K))
    ax.set_yticklabels(cluster_labels)
    if title is not None:
        ax.set_title(title)

    if show_colorbar:
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label(colorbar_label)

    if created_fig:
        fig.tight_layout()
    return fig, ax


def plot_cluster_boxplots(
    data_list,
    labels,
    *,
    data_labels=None,
    colors=None,
    clusters=None,
    cluster_order=None,
    show_outliers=False,
    figsize=(12, 5),
    ax=None,
    ylabel=None,
    title=None,
    xlabel="cluster",
):
    """
    General cluster-wise boxplot function.

    Parameters
    ----------
    data_list : list of array-like
        List of arrays to plot, e.g. [y_true, y_pred, log_var].
        Each array must have shape (N,).
    labels : array-like
        Cluster labels of shape (N,).
    data_labels : list of str, optional
        Labels for legend. Must match len(data_list).
    colors : list, optional
        Colors for each dataset. Must match len(data_list).
    clusters : array-like, optional
        Subset of clusters to plot. If None, plot all clusters present.
    cluster_order : array-like, optional
        Order in which to plot the selected clusters, e.g. output of
        `cluster_order_from_values(...)`.
        If provided together with `clusters`, only the clusters in `clusters`
        are kept, in the order specified by `cluster_order`.
    show_outliers : bool
        Whether to show outlier points in the boxplots.
    figsize : tuple
        Figure size if ax is None.
    ax : matplotlib.axes.Axes, optional
        Existing axis to plot into.
    ylabel : str, optional
        Y-axis label.
    title : str, optional
        Plot title.
    xlabel : str, optional
        X-axis label.

    Returns
    -------
    fig, ax
    """
    labels = np.asarray(labels, dtype=int)
    data_list = [np.asarray(d) for d in data_list]

    n = len(labels)
    for i, d in enumerate(data_list):
        if len(d) != n:
            raise ValueError(f"data_list[{i}] has length {len(d)}, expected {n}")

    if clusters is None:
        clusters = np.array(sorted(np.unique(labels)), dtype=int)
    else:
        clusters = np.asarray(clusters, dtype=int)

    if cluster_order is not None:
        cluster_order = np.asarray(cluster_order, dtype=int)
        cluster_set = set(clusters.tolist())
        clusters = np.array([k for k in cluster_order if k in cluster_set], dtype=int)

    if len(clusters) == 0:
        raise ValueError("No clusters selected for plotting.")

    K = len(data_list)

    if data_labels is None:
        data_labels = [f"data_{i}" for i in range(K)]
    if len(data_labels) != K:
        raise ValueError("data_labels must have the same length as data_list")

    if colors is None:
        cmap = plt.get_cmap("tab10")
        colors = [cmap(i % cmap.N) for i in range(K)]
    if len(colors) != K:
        raise ValueError("colors must have the same length as data_list")

    created_fig = ax is None
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    # Collect per-cluster data in the requested order
    clustered_data = [
        [d[labels == k] for k in clusters]
        for d in data_list
    ]

    # Use ordinal x positions so arbitrary cluster order works cleanly
    x = np.arange(len(clusters), dtype=float)

    width = 0.8 / K
    offsets = (np.arange(K) - (K - 1) / 2.0) * width

    for data_k, offset, color, label in zip(clustered_data, offsets, colors, data_labels):
        pos = x + offset

        bp = ax.boxplot(
            data_k,
            positions=pos,
            widths=width * 0.9,
            patch_artist=True,
            showfliers=show_outliers,
            manage_ticks=False,
        )

        for box in bp["boxes"]:
            box.set_facecolor(color)

        ax.plot([], [], color=color, linewidth=8, label=label)

    ax.set_xticks(x)
    ax.set_xticklabels([f"C{k}" for k in clusters])
    ax.set_xlabel(xlabel)

    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)

    ax.legend()

    if created_fig:
        fig.tight_layout()

    return fig, ax


def _normalize_fields(fields):
    if isinstance(fields, str):
        return [fields]
    return list(fields)


def _is_scalar_like(value):
    if isinstance(value, str):
        return True
    arr = np.asarray(value, dtype=object)
    return arr.ndim == 0 or arr.size == 1


def _call_transform(transform, raw_values, n_nodes):
    if transform is None:
        if len(raw_values) != 1:
            raise ValueError(
                "When multiple fields are requested, transform must be provided."
            )
        return raw_values[0]

    try:
        sig = inspect.signature(transform)
        if "n_nodes" in sig.parameters:
            return transform(*raw_values, n_nodes=n_nodes)
    except Exception:
        pass
    return transform(*raw_values)


def _coerce_to_node_values(value, n_nodes):
    """Turn one graph-level transformed value into exactly one value per node."""
    if _is_scalar_like(value):
        scalar = value if isinstance(value, str) else np.asarray(value).reshape(()).item()
        return np.full(n_nodes, scalar, dtype=object if isinstance(scalar, str) else None)

    arr = np.asarray(value)

    if arr.ndim == 1:
        if arr.shape[0] == n_nodes:
            return arr
        raise ValueError(
            f"Transform returned shape {arr.shape}; expected scalar or length {n_nodes}."
        )

    raise ValueError(
        "Transform must reduce metadata to either a scalar (broadcasted to all nodes) "
        f"or a 1-D array of length n_nodes={n_nodes}. Got shape {arr.shape}."
    )


def _extract_raw_metadata_value(graph, field, *, meta_lookup=None, strict=True):
    value = get_graph_metadata_value(
        graph,
        field,
        meta_lookup=meta_lookup,
        default=None,
        strict=strict,
    )
    if value is None and strict:
        organoid_str = getattr(graph, "organoid_str", None)
        raise KeyError(f"Graph {organoid_str!r} missing metadata field {field!r}")
    return value


def _infer_field_kind(value, n_nodes):
    if value is None or isinstance(value, str):
        return "broadcast"

    arr = np.asarray(value, dtype=object)
    if arr.ndim == 0 or arr.size == 1:
        return "broadcast"
    if arr.size == 0:
        return "empty"
    if arr.ndim >= 1 and any(dim >= n_nodes for dim in arr.shape):
        return "nodewise"
    return "broadcast"


def _build_cluster_metadata_dataframe(
    graphs,
    labels,
    *,
    fields,
    meta_lookup=None,
    transform=None,
    strict=True,
):
    """Build a node-level dataframe from graph metadata.

    Rule for raw metadata fields:
    - if the raw field has a dimension >= n_nodes, it is treated as node-wise metadata
    - otherwise it is treated as graph-level metadata and broadcast to all nodes

    The optional `transform` is applied per graph, after raw values are fetched.
    It should return either:
    - a scalar -> broadcast to all nodes in that graph
    - a 1-D array of length n_nodes -> used directly

    Examples
    --------
    d_crypts_graph minimum over regions:
        transform=lambda x: np.min(x, axis=0) if x.shape[1] >= x.shape[0] else np.min(x, axis=1)

    exponentiate graph-level log metadata:
        transform=lambda x: np.exp(x)

    combine dataset and timepoint:
        transform=lambda dataset, timepoint: f"{dataset} | {timepoint}"
    """
    labels = np.asarray(labels)
    if labels.ndim != 1:
        raise ValueError("labels must be 1-D")

    field_list = _normalize_fields(fields)
    rows = []

    cursor = 0
    for gi, graph in enumerate(graphs):
        n_nodes = int(graph.y.shape[0]) if hasattr(graph, "y") and graph.y is not None else int(graph.x.shape[0])
        graph_labels = labels[cursor: cursor + n_nodes]
        if graph_labels.shape[0] != n_nodes:
            raise ValueError(
                f"labels do not align with graph sizes at graph index {gi}: expected {n_nodes}, got {graph_labels.shape[0]}"
            )

        raw_values = [
            _extract_raw_metadata_value(graph, field, meta_lookup=meta_lookup, strict=strict)
            for field in field_list
        ]

        transformed = _call_transform(transform, raw_values, n_nodes)
        node_values = _coerce_to_node_values(transformed, n_nodes)

        # Preserve original raw fields in the dataframe where practical.
        row_dict = {"cluster": graph_labels, "value": node_values}
        for field, raw in zip(field_list, raw_values):
            kind = _infer_field_kind(raw, n_nodes)
            if kind == "broadcast":
                arr = np.asarray(raw, dtype=object)

                # scalar / string / single-value metadata -> broadcast
                if isinstance(raw, str) or arr.ndim == 0 or arr.size == 1:
                    scalar = raw if isinstance(raw, str) else arr.reshape(()).item()
                    row_dict[field] = np.full(
                        n_nodes,
                        scalar,
                        dtype=object if isinstance(scalar, str) else None,
                    )

                # empty metadata -> fill with NaN
                elif arr.size == 0:
                    row_dict[field] = np.full(n_nodes, np.nan)

                # anything else should not be broadcast
                else:
                    raise ValueError(
                        f"Field {field!r} looked broadcast-like but had shape {arr.shape}; "
                        "cannot safely broadcast it to nodes."
                    )

        rows.append(pd.DataFrame(row_dict))
        cursor += n_nodes

    if cursor != labels.shape[0]:
        raise ValueError(
            f"labels length {labels.shape[0]} does not match total nodes consumed {cursor}"
        )

    return pd.concat(rows, axis=0, ignore_index=True)


def _values_by_cluster(df, value_col, cluster_order, *, include_all=False):
    arrays = []
    labels = []

    if include_all:
        arrays.append(df[value_col].values)
        labels.append("All")

    for k in cluster_order:
        arrays.append(df.loc[df["cluster"] == k, value_col].values)
        labels.append(f"C{k}")

    return arrays, labels


def plot_cluster_metadata_boxplots(
    graphs,
    labels,
    *,
    fields,
    meta_lookup=None,
    transform=None,
    cluster_order=None,
    include_all=False,
    ylabel=None,
    title=None,
    show_outliers=False,
    facecolors="lightgray",
    ax=None,
    figsize=(11, 5),
    boxplot_kwargs=None,
    strict=True,
    return_dataframe=False,
):
    """Plot cluster-wise boxplots directly from graph metadata.

    Parameters
    ----------
    graphs : list[Data]
    labels : array-like, shape (total_num_nodes,)
    fields : str or sequence[str]
        Metadata field(s) to fetch from each graph.
    transform : callable or None
        Applied per graph after fetching raw metadata values.
        Return either a scalar or a 1-D array of length n_nodes.
        If omitted, a single field is used directly.
    return_dataframe : bool
        If True, also return the internally built node-level dataframe.

    Notes
    -----
    This function is intended for numeric values. If the transformed values are
    non-numeric (for example combined measurement labels), the dataframe can still
    be returned, but boxplotting will fail.
    """
    df = _build_cluster_metadata_dataframe(
        graphs,
        labels,
        fields=fields,
        meta_lookup=meta_lookup,
        transform=transform,
        strict=strict,
    )

    if not np.issubdtype(np.asarray(df["value"]).dtype, np.number):
        raise TypeError(
            "plot_cluster_metadata_boxplots requires numeric values after transformation. "
            "For categorical transformations, use return_dataframe=True and build a different plot from the dataframe."
        )

    if cluster_order is None:
        cluster_order = np.sort(pd.unique(df["cluster"]))
    else:
        cluster_order = np.asarray(cluster_order)

    box_data, tick_labels = _values_by_cluster(
        df,
        "value",
        cluster_order,
        include_all=include_all,
    )

    # Drop NaNs from each box.
    box_data = [vals[np.isfinite(vals)] for vals in box_data]

    created_fig = ax is None
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    boxplot_kwargs = {} if boxplot_kwargs is None else dict(boxplot_kwargs)
    bp = ax.boxplot(
        box_data,
        showfliers=show_outliers,
        patch_artist=True,
        **boxplot_kwargs,
    )

    if isinstance(facecolors, str):
        colors = [facecolors] * len(bp["boxes"])
    else:
        colors = list(facecolors)
        if len(colors) != len(bp["boxes"]):
            raise ValueError(
                f"facecolors must have length {len(bp['boxes'])}, got {len(colors)}"
            )
    for box, color in zip(bp["boxes"], colors):
        box.set_facecolor(color)

    ax.set_xticks(np.arange(1, len(tick_labels) + 1))
    ax.set_xticklabels(tick_labels, rotation=45 if len(tick_labels) > 8 else 0)
    ax.set_xlabel("cluster")
    ax.set_ylabel(ylabel or (_normalize_fields(fields)[0] if isinstance(fields, str) or len(_normalize_fields(fields)) == 1 else "value"))
    default_title = ylabel or (_normalize_fields(fields)[0] if len(_normalize_fields(fields)) == 1 else "metadata")
    ax.set_title(title or f"Distribution of {default_title} by cluster")

    if created_fig:
        fig.tight_layout()

    if return_dataframe:
        return fig, ax, df
    return fig, ax

