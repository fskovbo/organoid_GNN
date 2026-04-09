import matplotlib.pyplot as plt
import numpy as np


def plot_influence_heatmap(mat, hops, marker_names, title):
    """
    Inputs:
      mat          : np.ndarray (H, M)
      hops         : list[int]
      marker_names : list[str]
      title        : str

    Output:
      fig, ax
    """

    fig, ax = plt.subplots(
        figsize=(0.7 * len(marker_names) + 3, 0.7 * len(hops) + 2)
    )

    im = ax.imshow(mat, aspect="auto")
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
    center_zero=False,
    sort_center=True,
    cmap="viridis",
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

    Output:
      fig, axes
    """

    mat = np.asarray(mat)
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
    center_names = [marker_names[i] for i in order]

    fig, axes = plt.subplots(
        1, H,
        figsize=(0.50 * len(marker_names) * H + 2.5,
                 0.35 * len(marker_names) + 2),
        squeeze=False
    )
    axes = axes[0]

    for hi, hop in enumerate(hops):
        ax = axes[hi]
        data = mat_s[hi]

        if center_zero:
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