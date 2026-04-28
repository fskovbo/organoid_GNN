import math
import matplotlib.pyplot as plt
import numpy as np



def select_target_from_result(values, target_index=None, *, expected_ndim=None, name="values"):
    """Select trailing target dimension from depth-scan result arrays.

    Existing arrays with no target dimension are returned unchanged. If an array
    has one extra trailing target dimension, pass ``target_index``.
    """
    arr = np.asarray(values, dtype=float)
    if expected_ndim is not None and arr.ndim == expected_ndim:
        return arr
    if expected_ndim is not None and arr.ndim == expected_ndim + 1:
        if arr.shape[-1] == 1 and target_index is None:
            return arr[..., 0]
        if target_index is None:
            raise ValueError(
                f"{name} has a target dimension with shape {arr.shape}; pass target_index."
            )
        return arr[..., int(target_index)]
    if target_index is not None and arr.ndim >= 1:
        return arr[..., int(target_index)]
    return arr


def plot_markerwise_metric_vs_depth(
    all_results,
    depths,
    marker_names,
    *,
    metric_key,
    ylabel,
    title,
    sem_key=None,
    baseline_key=None,
    baseline_sem_key=None,
    selected_idx=None,
    family_order=None,
    family_colors=None,
    baseline_color="#888888",
    n_cols=2,
    panel_height=3.4,
    panel_width=5.0,
    sharex=True,
    sharey=False,
    baseline_hline=None,
    target_index=None,
    legend_fontsize=7,
    fill_alpha=0.14,
    baseline_fill_alpha=0.18,
    marker_label_fontsize=11,
    marker_label_bbox=True,
):
    """
    Plot marker-wise metric vs depth for multiple model families.
    """

    depths = np.asarray(depths)

    if family_order is None:
        family_order = list(all_results.keys())

    if family_colors is None:
        family_colors = {}

    n_markers = len(marker_names)

    if selected_idx is None:
        selected_idx = list(range(n_markers))
    else:
        selected_idx = list(selected_idx)

    selected_marker_names = [marker_names[i] for i in selected_idx]

    n_sel = len(selected_idx)
    n_rows = math.ceil(n_sel / n_cols)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(panel_width * n_cols, panel_height * n_rows),
        sharex=sharex,
        sharey=sharey,
    )

    axes = np.atleast_1d(axes).ravel()

    available_families = [f for f in family_order if f in all_results]
    if len(available_families) == 0:
        raise ValueError("No requested families present in all_results.")

    ref_family = available_families[0]

    for ax, mi, mname in zip(axes, selected_idx, selected_marker_names):

        for family in available_families:
            y_all = select_target_from_result(
                all_results[family][metric_key],
                target_index=target_index,
                expected_ndim=2,
                name=f"{family}.{metric_key}",
            )
            y = y_all[:, mi]
            color = family_colors.get(family, None)

            ax.plot(
                depths,
                y,
                "-o",
                linewidth=2,
                markersize=4,
                color=color,
                label=family,
            )

            if sem_key is not None:
                s_all = select_target_from_result(
                    all_results[family][sem_key],
                    target_index=target_index,
                    expected_ndim=2,
                    name=f"{family}.{sem_key}",
                )
                s = s_all[:, mi]
                ax.fill_between(
                    depths,
                    y - s,
                    y + s,
                    color=color,
                    alpha=fill_alpha,
                )

        # baseline
        if baseline_key is not None:
            y0_all = select_target_from_result(
                all_results[ref_family][baseline_key],
                target_index=target_index,
                expected_ndim=1,
                name=f"{ref_family}.{baseline_key}",
            )
            y0 = y0_all[mi]

            ax.axhline(
                y0,
                linestyle="--",
                color=baseline_color,
                alpha=0.9,
                label="baseline",
            )

            if baseline_sem_key is not None:
                s0_all = select_target_from_result(
                    all_results[ref_family][baseline_sem_key],
                    target_index=target_index,
                    expected_ndim=1,
                    name=f"{ref_family}.{baseline_sem_key}",
                )
                s0 = s0_all[mi]
                ax.fill_between(
                    depths,
                    y0 - s0,
                    y0 + s0,
                    color=baseline_color,
                    alpha=baseline_fill_alpha,
                )

        elif baseline_hline is not None:
            ax.axhline(
                baseline_hline,
                linestyle="--",
                color=baseline_color,
                alpha=0.8,
            )

        # marker label inside panel
        text_kwargs = dict(
            x=0.03,
            y=0.97,
            s=mname,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=marker_label_fontsize,
        )
        if marker_label_bbox:
            text_kwargs["bbox"] = dict(
                facecolor="white",
                edgecolor="none",
                alpha=0.75,
                boxstyle="round,pad=0.2",
            )
        ax.text(**text_kwargs)

        # no external title anymore
        ax.set_ylabel(ylabel)
        ax.set_xlabel("depth / k-hops")
        ax.set_xticks(depths)

        # force x tick labels on all panels, even with sharex=True
        ax.tick_params(axis="x", which="both", labelbottom=True)

        ax.legend(fontsize=legend_fontsize)

    for ax in axes[n_sel:]:
        ax.set_visible(False)

    fig.suptitle(title, y=1.02)
    plt.tight_layout()

    return fig, axes



def plot_aggregate_metric_vs_depth(
    all_results,
    depths,
    *,
    aggregate_key,
    metric_key,
    ylabel,
    title,
    sem_key=None,
    baseline_key=None,
    baseline_sem_key=None,
    family_order=None,
    family_colors=None,
    baseline_color="#888888",
    baseline_hline=None,
    target_index=None,
    legend_fontsize=8,
    fill_alpha=0.14,
    baseline_fill_alpha=0.18,
    figsize=(7, 4),
):
    """
    Plot one aggregate metric vs depth for multiple model families.

    Parameters
    ----------
    all_results : dict
        Results dict with structure:
        all_results[family]["aggregate"][aggregate_key][metric_key] -> (n_depths,)
    depths : array-like
        X-axis values.
    aggregate_key : str
        One of e.g. "all_nodes", "any_marker", "no_marker".
    metric_key : str
        Metric to plot, e.g. "mse_model", "var_model", "nll_model", "rho".
    ylabel : str
        Y-axis label.
    title : str
        Figure title.
    sem_key : str or None
        Aggregate SEM key for uncertainty band.
    baseline_key : str or None
        Aggregate baseline key for horizontal reference.
    baseline_sem_key : str or None
        Aggregate SEM for baseline band.
    """
    depths = np.asarray(depths)

    if family_order is None:
        family_order = list(all_results.keys())

    if family_colors is None:
        family_colors = {}

    available_families = [f for f in family_order if f in all_results]
    if len(available_families) == 0:
        raise ValueError("No requested families present in all_results.")

    ref_family = available_families[0]

    fig, ax = plt.subplots(figsize=figsize)

    for family in available_families:
        y = select_target_from_result(
            all_results[family]["aggregate"][aggregate_key][metric_key],
            target_index=target_index,
            expected_ndim=1,
            name=f"{family}.aggregate.{aggregate_key}.{metric_key}",
        )
        color = family_colors.get(family, None)

        ax.plot(
            depths,
            y,
            "-o",
            linewidth=2,
            markersize=4,
            color=color,
            label=family,
        )

        if sem_key is not None:
            s = select_target_from_result(
                all_results[family]["aggregate"][aggregate_key][sem_key],
                target_index=target_index,
                expected_ndim=1,
                name=f"{family}.aggregate.{aggregate_key}.{sem_key}",
            )
            ax.fill_between(
                depths,
                y - s,
                y + s,
                color=color,
                alpha=fill_alpha,
            )

    # baseline
    if baseline_key is not None:
        y0_arr = select_target_from_result(
            all_results[ref_family]["aggregate"][aggregate_key][baseline_key],
            target_index=target_index,
            expected_ndim=1,
            name=f"{ref_family}.aggregate.{aggregate_key}.{baseline_key}",
        )
        y0 = float(y0_arr[0]) if np.ndim(y0_arr) > 0 else float(y0_arr)

        ax.axhline(
            y0,
            linestyle="--",
            color=baseline_color,
            alpha=0.9,
            label="baseline",
        )

        if baseline_sem_key is not None:
            s0_arr = select_target_from_result(
                all_results[ref_family]["aggregate"][aggregate_key][baseline_sem_key],
                target_index=target_index,
                expected_ndim=1,
                name=f"{ref_family}.aggregate.{aggregate_key}.{baseline_sem_key}",
            )
            s0 = float(s0_arr[0]) if np.ndim(s0_arr) > 0 else float(s0_arr)
            ax.fill_between(
                depths,
                y0 - s0,
                y0 + s0,
                color=baseline_color,
                alpha=baseline_fill_alpha,
            )

    elif baseline_hline is not None:
        ax.axhline(
            baseline_hline,
            linestyle="--",
            color=baseline_color,
            alpha=0.8,
        )

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("depth / k-hops")
    ax.set_xticks(depths)
    ax.legend(fontsize=legend_fontsize)

    plt.tight_layout()
    return fig, ax


def plot_all_aggregates_metric_vs_depth(
    all_results,
    depths,
    *,
    aggregate_keys=("all_nodes", "any_marker", "no_marker"),
    metric_key,
    ylabel,
    title,
    sem_key=None,
    baseline_key=None,
    baseline_sem_key=None,
    family_order=None,
    family_colors=None,
    baseline_color="#888888",
    baseline_hline=None,
    target_index=None,
    n_cols=2,
    panel_height=3.6,
    panel_width=5.2,
    sharex=True,
    sharey=False,
    legend_fontsize=7,
    fill_alpha=0.14,
    baseline_fill_alpha=0.18,
):
    """
    Plot one aggregate metric vs depth for several aggregate subsets
    (e.g. all_nodes / any_marker / no_marker), one panel per subset.
    """
    depths = np.asarray(depths)

    if family_order is None:
        family_order = list(all_results.keys())

    if family_colors is None:
        family_colors = {}

    aggregate_keys = list(aggregate_keys)
    n_panels = len(aggregate_keys)
    n_rows = math.ceil(n_panels / n_cols)

    available_families = [f for f in family_order if f in all_results]
    if len(available_families) == 0:
        raise ValueError("No requested families present in all_results.")

    ref_family = available_families[0]

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(panel_width * n_cols, panel_height * n_rows),
        sharex=sharex,
        sharey=sharey,
    )
    axes = np.atleast_1d(axes).ravel()

    pretty_names = {
        "all_nodes": "all nodes",
        "any_marker": "marker-positive nodes",
        "no_marker": "marker-negative nodes",
    }

    for ax, aggregate_key in zip(axes, aggregate_keys):
        for family in available_families:
            y = select_target_from_result(
                all_results[family]["aggregate"][aggregate_key][metric_key],
                target_index=target_index,
                expected_ndim=1,
                name=f"{family}.aggregate.{aggregate_key}.{metric_key}",
            )
            color = family_colors.get(family, None)

            ax.plot(
                depths,
                y,
                "-o",
                linewidth=2,
                markersize=4,
                color=color,
                label=family,
            )

            if sem_key is not None:
                s = select_target_from_result(
                    all_results[family]["aggregate"][aggregate_key][sem_key],
                    target_index=target_index,
                    expected_ndim=1,
                    name=f"{family}.aggregate.{aggregate_key}.{sem_key}",
                )
                ax.fill_between(
                    depths,
                    y - s,
                    y + s,
                    color=color,
                    alpha=fill_alpha,
                )

        if baseline_key is not None:
            y0_arr = select_target_from_result(
                all_results[ref_family]["aggregate"][aggregate_key][baseline_key],
                target_index=target_index,
                expected_ndim=1,
                name=f"{ref_family}.aggregate.{aggregate_key}.{baseline_key}",
            )
            y0 = float(y0_arr[0]) if np.ndim(y0_arr) > 0 else float(y0_arr)

            ax.axhline(
                y0,
                linestyle="--",
                color=baseline_color,
                alpha=0.9,
                label="baseline",
            )

            if baseline_sem_key is not None:
                s0_arr = select_target_from_result(
                    all_results[ref_family]["aggregate"][aggregate_key][baseline_sem_key],
                    target_index=target_index,
                    expected_ndim=1,
                    name=f"{ref_family}.aggregate.{aggregate_key}.{baseline_sem_key}",
                )
                s0 = float(s0_arr[0]) if np.ndim(s0_arr) > 0 else float(s0_arr)
                ax.fill_between(
                    depths,
                    y0 - s0,
                    y0 + s0,
                    color=baseline_color,
                    alpha=baseline_fill_alpha,
                )

        elif baseline_hline is not None:
            ax.axhline(
                baseline_hline,
                linestyle="--",
                color=baseline_color,
                alpha=0.8,
            )

        ax.set_title(pretty_names.get(aggregate_key, aggregate_key))
        ax.set_ylabel(ylabel)
        ax.set_xlabel("depth / k-hops")
        ax.set_xticks(depths)
        ax.legend(fontsize=legend_fontsize)

    for ax in axes[n_panels:]:
        ax.set_visible(False)

    fig.suptitle(title, y=1.02)
    plt.tight_layout()
    return fig, axes