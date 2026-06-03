import matplotlib.pyplot as plt
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


def _cmap_with_bad_color(cmap, bad_color):
    cmap_obj = plt.get_cmap(cmap).copy() if isinstance(cmap, str) else cmap.copy()
    cmap_obj.set_bad(bad_color)
    return cmap_obj


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
      min_count    : int or None
        Values with counts below this threshold are shown as bad_color.

    Output:
      fig, ax
    """

    mat = select_target_from_matrix(mat, target_index=target_index, expected_ndim=2)
    if counts is not None:
        counts = select_target_from_matrix(counts, target_index=target_index, expected_ndim=2)
    mat = _mask_low_count_values(mat, counts=counts, min_count=min_count)
    cmap = _cmap_with_bad_color(cmap or plt.rcParams["image.cmap"], bad_color)

    fig, ax = plt.subplots(
        figsize=(0.7 * len(marker_names) + 3, 0.7 * len(hops) + 2)
    )

    im = ax.imshow(mat, aspect="auto", cmap=cmap)
    fig.colorbar(im, ax=ax)

    ax.set_yticks(range(len(hops)))
    ax.set_yticklabels([f"hop {h}" for h in hops])

    ax.set_xticks(range(len(marker_names)))
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
      min_count    : int or None
        Values with counts below this threshold are shown as bad_color.

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
        data = _mask_low_count_values(
            mat_s[hi],
            counts=None if counts_s is None else counts_s[hi],
            min_count=min_count,
        )

        if center_zero and np.any(np.isfinite(data)):
            vmax = np.nanmax(np.abs(data))
            vmin = -vmax
            im = ax.imshow(
                data,
                aspect="auto",
                vmin=vmin,
                vmax=vmax,
                cmap=cmap,
            )
        else:
            im = ax.imshow(
                data,
                aspect="auto",
                cmap=cmap,
            )

        ax.set_xlabel("perturbation marker X", fontsize=13)
        ax.set_ylabel("center marker Y", fontsize=13)

        ax.set_xticks(range(len(marker_names)))
        ax.set_xticklabels(marker_names, rotation=60, ha="right", fontsize=11)

        ax.set_yticks(range(len(center_names)))
        ax.set_yticklabels(center_names, fontsize=11)

        ax.set_title(f"hop {hop}", fontsize=14)

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=11)

    fig.suptitle(title, fontsize=16)
    fig.tight_layout()

    return fig, axes
