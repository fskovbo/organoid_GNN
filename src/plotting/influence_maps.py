import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np


def _mask_low_count_values(values, counts=None, min_count=None):
    values = np.asarray(values, dtype=float)
    if min_count is None:
        return values
    if counts is None:
        raise ValueError("counts must be provided when min_count is not None.")

    counts = np.asarray(counts)
    if counts.shape != values.shape:
        raise ValueError(f"counts shape {counts.shape} does not match values shape {values.shape}.")

    masked = values.copy()
    masked[counts < min_count] = np.nan
    return masked


def _min_count_for_hop(min_count, hop, index):
    if min_count is None:
        return None
    if isinstance(min_count, dict):
        return min_count.get(hop, min_count.get(str(hop)))
    if np.ndim(min_count) > 0 and not np.isscalar(min_count):
        return list(min_count)[index]
    return min_count


def _cmap_with_bad_color(cmap, bad_color):
    cmap_obj = plt.get_cmap(cmap).copy() if isinstance(cmap, str) else cmap.copy()
    cmap_obj.set_bad(bad_color)
    return cmap_obj


def _draw_heatmap(ax, data, *, aspect="auto", **kwargs):
    n_rows, n_cols = np.asarray(data).shape
    mesh = ax.pcolormesh(
        np.arange(n_cols + 1),
        np.arange(n_rows + 1),
        data,
        shading="flat",
        edgecolors="white",
        linewidth=0.6,
        antialiased=False,
        **kwargs,
    )
    ax.set_xlim(0, n_cols)
    ax.set_ylim(n_rows, 0)
    ax.set_aspect(aspect)
    ax.grid(False)
    return mesh


def _apply_heatmap_cell_grid(ax, n_rows, n_cols, *, color="white", linewidth=0.6):
    """Keep style-level grids from cutting through heatmap cells."""
    ax.grid(False)
    ax.tick_params(which="minor", bottom=False, left=False)


def _overlay_heatmap_hatches(
    ax,
    mask,
    *,
    hatch="////",
    edgecolor="0.50",
    linewidth=0.0,
):
    mask = np.asarray(mask, dtype=bool)
    for row_index, col_index in np.argwhere(mask):
        ax.add_patch(
            Rectangle(
                (col_index, row_index),
                1,
                1,
                facecolor=(1, 1, 1, 0),
                edgecolor=edgecolor,
                hatch=hatch,
                linewidth=linewidth,
                zorder=3,
            )
        )


def select_target_from_matrix(mat, target_index=None, *, expected_ndim=None, name="mat"):
    """Select a target column from influence matrices when they include one.

    For existing single-target matrices, this is a no-op. For multi-target
    matrices with one extra trailing dimension, pass ``target_index``.
    """
    arr = np.asarray(mat)
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


def plot_influence_heatmap(
    mat,
    hops,
    marker_names,
    title,
    *,
    target_index=None,
    counts=None,
    min_count=None,
    bad_color="white",
    cmap=None,
):
    """
    Inputs:
      mat          : np.ndarray (H, M)
      hops         : list[int]
      marker_names : list[str]
      title        : str
      target_index : int or None
        Target column to select if mat has shape (H, M, T).
      counts       : np.ndarray or None
        Count matrix with the same shape as mat after target selection.
      min_count    : int, dict, sequence, or None
        Values with counts below this threshold are shown as bad_color. If a
        dict is passed, keys are hop values. If a sequence is passed, values
        are matched to the order of ``hops``.

    Output:
      fig, ax
    """

    mat = select_target_from_matrix(mat, target_index=target_index, expected_ndim=2)
    mat = np.asarray(mat, dtype=float).copy()
    if counts is not None:
        counts = select_target_from_matrix(counts, target_index=target_index, expected_ndim=2)
    if min_count is not None and counts is None:
        raise ValueError("counts must be provided when min_count is not None.")
    if min_count is not None:
        for hi, hop in enumerate(hops):
            hop_min_count = _min_count_for_hop(min_count, hop, hi)
            if hop_min_count is not None:
                mat[hi] = _mask_low_count_values(
                    mat[hi],
                    counts=counts[hi],
                    min_count=hop_min_count,
                )
    cmap = _cmap_with_bad_color(cmap or plt.rcParams["image.cmap"], bad_color)

    fig, ax = plt.subplots(
        figsize=(0.7 * len(marker_names) + 3, 0.7 * len(hops) + 2)
    )

    low_count_mask = None
    if min_count is not None:
        low_count_mask = np.zeros_like(counts, dtype=bool)
        for hi, hop in enumerate(hops):
            hop_min_count = _min_count_for_hop(min_count, hop, hi)
            if hop_min_count is not None:
                low_count_mask[hi] = counts[hi] < hop_min_count
    im = _draw_heatmap(ax, mat, aspect="auto", cmap=cmap)
    if low_count_mask is not None:
        _overlay_heatmap_hatches(ax, low_count_mask)
    _apply_heatmap_cell_grid(ax, mat.shape[0], mat.shape[1])
    fig.colorbar(im, ax=ax)

    ax.set_yticks(np.arange(len(hops)) + 0.5)
    ax.set_yticklabels([f"hop {h}" for h in hops])

    ax.set_xticks(np.arange(len(marker_names)) + 0.5)
    ax.set_xticklabels(marker_names, rotation=60, ha="right")

    ax.set_title(title)
    fig.tight_layout()

    return fig, ax


def plot_influence_center_resolved(
    mat,
    hops,
    marker_names,
    title,
    target_index=None,
    center_zero=False,
    sort_center=True,
    cmap="viridis",
    counts=None,
    min_count=None,
    bad_color="white",
    vmin=None,
    vmax=None,
    cbar_ticks=None,
    cbar_label=None,
):
    """
    Inputs:
      mat          : np.ndarray (H, M_center, M_pert)
      hops         : list[int]
      marker_names : list[str]
      title        : str
      center_zero  : bool
      sort_center  : bool
      cmap         : str or matplotlib colormap
      target_index : int or None
        Target column to select if mat has shape (H, M_center, M_pert, T).
      counts       : np.ndarray or None
        Count tensor with the same shape as mat after target selection.
      min_count    : int, dict, sequence, or None
        Values with counts below this threshold are shown as bad_color. If a
        dict is passed, keys are hop values. If a sequence is passed, values
        are matched to the order of ``hops``.

    Output:
      fig, axes
    """

    mat = select_target_from_matrix(mat, target_index=target_index, expected_ndim=3)
    mat = np.asarray(mat)
    if counts is not None:
        counts = select_target_from_matrix(counts, target_index=target_index, expected_ndim=3)
        counts = np.asarray(counts)
        if counts.shape != mat.shape:
            raise ValueError(f"counts shape {counts.shape} does not match mat shape {mat.shape}.")
    H, My, Mx = mat.shape
    assert H == len(hops)
    assert My == len(marker_names)
    assert Mx == len(marker_names)

    # Sort center markers globally (shared across hops)
    order = np.arange(My)
    if sort_center:
        score = np.nanmean(np.abs(mat), axis=(0, 2))  # (My,)
        order = np.argsort(-score)

    mat_s = mat[:, order, :]
    counts_s = None if counts is None else counts[:, order, :]
    center_names = [marker_names[i] for i in order]
    cmap = _cmap_with_bad_color(cmap, bad_color)

    fig, axes = plt.subplots(
        1, H,
        figsize=(0.50 * len(marker_names) * H + 2.5,
                 0.35 * len(marker_names) + 2),
        squeeze=False
    )
    axes = axes[0]

    for hi, hop in enumerate(hops):
        ax = axes[hi]
        hop_min_count = _min_count_for_hop(min_count, hop, hi)
        data = _mask_low_count_values(
            mat_s[hi],
            counts=None if counts_s is None else counts_s[hi],
            min_count=hop_min_count,
        )
        low_count_mask = (
            None
            if hop_min_count is None or counts_s is None
            else counts_s[hi] < hop_min_count
        )

        if vmin is not None or vmax is not None:
            im = _draw_heatmap(
                ax,
                data,
                aspect="auto",
                vmin=vmin,
                vmax=vmax,
                cmap=cmap,
            )
        elif center_zero and np.any(np.isfinite(data)):
            hop_vmax = np.nanmax(np.abs(data))
            hop_vmin = -hop_vmax
            im = _draw_heatmap(
                ax,
                data,
                aspect="auto",
                vmin=hop_vmin,
                vmax=hop_vmax,
                cmap=cmap,
            )
        else:
            im = _draw_heatmap(
                ax,
                data,
                aspect="auto",
                cmap=cmap,
            )

        if low_count_mask is not None:
            _overlay_heatmap_hatches(ax, low_count_mask)
        _apply_heatmap_cell_grid(ax, data.shape[0], data.shape[1])

        ax.set_xlabel("perturbation marker X", fontsize=13)
        ax.set_ylabel("center marker Y", fontsize=13)

        ax.set_xticks(np.arange(len(marker_names)) + 0.5)
        ax.set_xticklabels(marker_names, rotation=60, ha="right", fontsize=11)

        ax.set_yticks(np.arange(len(center_names)) + 0.5)
        ax.set_yticklabels(center_names, fontsize=11)

        ax.set_title(rf"$d_{{pert}} = {hop}$", fontsize=14)

        cbar = fig.colorbar(
            im,
            ax=ax,
            fraction=0.046,
            pad=0.04,
            ticks=cbar_ticks,
        )
        if cbar_label is not None:
            cbar.set_label(cbar_label, fontsize=12)
        cbar.ax.tick_params(labelsize=11, length=4, width=0.8, direction="out")

    fig.suptitle(title, fontsize=16)
    fig.tight_layout()

    return fig, axes
