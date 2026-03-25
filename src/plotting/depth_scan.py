import math
import matplotlib.pyplot as plt
import numpy as np


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
    legend_fontsize=7,
    fill_alpha=0.14,
    baseline_fill_alpha=0.18,
):
    """
    Plot marker-wise metric vs depth for multiple model families.

    Parameters
    ----------
    all_results : dict
        Nested results dict: all_results[family][metric_key]
        is typically shape (n_depths, n_markers).
    depths : array-like
        X-axis values (e.g. model depth or k-hop radius).
    marker_names : list[str]
        Names of all markers.
    metric_key : str
        Key of metric to plot (e.g. "mse_model", "var_model").
    ylabel : str
        Y-axis label.
    title : str
        Figure title.

    Optional
    --------
    sem_key : str
        Key for uncertainty bands.
    baseline_key : str
        Key for baseline horizontal reference.
    baseline_sem_key : str
        Key for baseline uncertainty band.
    selected_idx : sequence[int] or None
        If None → plot all markers.
        Otherwise plot only given marker indices.
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

            y = np.asarray(all_results[family][metric_key])[:, mi]
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
                s = np.asarray(all_results[family][sem_key])[:, mi]
                ax.fill_between(
                    depths,
                    y - s,
                    y + s,
                    color=color,
                    alpha=fill_alpha,
                )

        # --- baseline ---
        if baseline_key is not None:
            y0 = np.asarray(all_results[ref_family][baseline_key])[mi]

            ax.axhline(
                y0,
                linestyle="--",
                color=baseline_color,
                alpha=0.9,
                label="baseline",
            )

            if baseline_sem_key is not None:
                s0 = np.asarray(all_results[ref_family][baseline_sem_key])[mi]
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

        ax.set_title(mname)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("depth / k-hops")
        ax.set_xticks(depths)
        ax.legend(fontsize=legend_fontsize)

    for ax in axes[n_sel:]:
        ax.set_visible(False)

    fig.suptitle(title, y=1.02)
    plt.tight_layout()

    return fig, axes