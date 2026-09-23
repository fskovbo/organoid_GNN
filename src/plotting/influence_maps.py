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


def plot_pair_heatmaps(summary, *, title='', min_organoids=1, min_cases=None,
                       limit=None, hops=None, marker_names=None):
    """Recipient/source heatmaps with historical white, hatched low-support cells.

    ``min_cases`` accepts a scalar or per-hop dict and uses distinct retained
    physical cases (``n_cases``), not rows replicated across fitted model seeds.
    An additional organoid threshold can be enforced independently.
    """
    if summary.empty:
        raise ValueError('No supported effects to plot.')
    hops = sorted(summary.hop.unique()) if hops is None else list(hops)
    recipients = sorted(summary.center_marker.unique()) if marker_names is None else list(marker_names)
    sources = sorted(summary.source_marker_name.unique()) if marker_names is None else list(marker_names)
    limit = (float(summary['mean'].abs().max()) or 1.) if limit is None else limit
    fig, axes = plt.subplots(1, len(hops), figsize=(5*len(hops), 5), squeeze=False)
    for hi, (ax, hop) in enumerate(zip(axes[0], hops)):
        part = summary[summary.hop == hop]
        def matrix(column, fill=np.nan):
            return part.pivot(index='center_marker', columns='source_marker_name', values=column).reindex(
                index=recipients, columns=sources).fillna(fill).to_numpy(float)
        values = matrix('mean')
        low = (matrix('n_organoids', 0) < min_organoids) | ~np.isfinite(values)
        threshold = _min_count_for_hop(min_cases, hop, hi)
        if threshold is not None:
            low |= matrix('n_cases', 0) < threshold
        values[low] = np.nan
        im = _draw_heatmap(ax, values, cmap=_cmap_with_bad_color('RdBu_r', 'white'), vmin=-limit, vmax=limit)
        _overlay_heatmap_hatches(ax, low)
        ax.set(title=f'Hop {hop}', xlabel='Source fate', ylabel='Recipient fate',
            xticks=np.arange(len(sources))+.5, xticklabels=sources,
            yticks=np.arange(len(recipients))+.5, yticklabels=recipients)
        ax.tick_params(axis='x', rotation=90)
        fig.colorbar(im, ax=ax, label='Edited − intact prediction')
    fig.suptitle(title + '\nHatched: insufficient retained cases or organoids')
    fig.tight_layout()
    return fig


def plot_pair_count_heatmaps(counts, *, hops, marker_names, title='Retained sample counts', vmax=None):
    """Annotated pair counts per hop, with a shared logarithmic blue scale.

    Expects one row per (hop, center_marker, source_marker_name), with n_cases
    counting unique physical cases. Zeros are white but still annotated; low
    counts remain visible irrespective of effect-plot display thresholds.
    """
    from matplotlib.colors import LogNorm
    hops, names = list(hops), list(marker_names)
    if not hops or not names:
        raise ValueError('Provide at least one hop and marker identity.')
    keys = ['hop', 'center_marker', 'source_marker_name']
    if counts.duplicated(keys).any():
        raise ValueError('Select one method and observed/sweep cohort before plotting counts.')
    matrices = []
    for hop in hops:
        part = counts[counts.hop == hop]
        matrix = part.pivot(index='center_marker', columns='source_marker_name', values='n_cases').reindex(
            index=names, columns=names).fillna(0).to_numpy(float)
        if not np.isfinite(matrix).all() or (matrix < 0).any() or (matrix != np.floor(matrix)).any():
            raise ValueError('Sample counts must be finite nonnegative integers.')
        matrices.append(matrix)
    # LogNorm needs a nondegenerate positive range, including all-zero inputs.
    maximum = max(float(m.max()) for m in matrices) if vmax is None else float(vmax)
    norm = LogNorm(vmin=1, vmax=max(2, maximum))
    fig, axes = plt.subplots(1, len(hops), figsize=(max(4.5, .6*len(names))*len(hops),
        max(4, .5*len(names)+1.5)), squeeze=False, constrained_layout=True)
    for ax, hop, matrix in zip(axes[0], hops, matrices):
        image = _draw_heatmap(ax, np.ma.masked_less_equal(matrix, 0),
            cmap=_cmap_with_bad_color('Blues', 'white'), norm=norm)
        for row, column in np.ndindex(matrix.shape):
            value = int(matrix[row, column])
            rgba = (1, 1, 1, 1) if value == 0 else image.cmap(norm(value))
            luminance = .299*rgba[0] + .587*rgba[1] + .114*rgba[2]
            ax.text(column+.5, row+.5, str(value), ha='center', va='center',
                color='white' if luminance < .45 else 'black', fontsize=8.5)
        ax.set(title=f'Hop {hop}', xlabel='Perturbation marker (source)', ylabel='Center marker (recipient)',
            xticks=np.arange(len(names))+.5, xticklabels=names,
            yticks=np.arange(len(names))+.5, yticklabels=names)
        ax.tick_params(axis='x', rotation=60)
        plt.setp(ax.get_xticklabels(), ha='right')
    fig.colorbar(image, ax=list(axes[0]), fraction=.046, pad=.04,
                 label='Retained unique cases (log color scale)')
    fig.suptitle(title)
    return fig


def plot_pair_size_curves(summary, *, hop, title='', min_organoids=1, min_cases=None,
                          ylabel='Edited − intact prediction'):
    """Pair curves with distinct method/mode colors and an explicit hop title."""
    import matplotlib.pyplot as plt
    part = summary[summary.hop==hop].copy()
    if part.empty:
        raise ValueError(f'No supported effects at hop {hop}.')
    if 'label' not in part:
        part['label'] = 'Effect'
    centers = sorted(part.center_marker.unique());sources = sorted(part.source_marker_name.unique())
    labels = list(part.label.unique())
    palette = plt.get_cmap('tab10' if len(labels) <= 5 else 'tab20')
    colors = {(label, mode): palette((2*i+j) % palette.N)
              for i, label in enumerate(labels) for j, mode in enumerate(('observed', 'sweep'))}
    fig,axes = plt.subplots(len(centers),len(sources),figsize=(3*len(sources),2.5*len(centers)),squeeze=False)
    for i,center in enumerate(centers):
        for j,source in enumerate(sources):
            ax=axes[i,j]
            selected=part[(part.center_marker==center)&(part.source_marker_name==source)].copy()
            low = selected.n_organoids < min_organoids
            threshold = _min_count_for_hop(min_cases, hop, sorted(summary.hop.unique()).index(hop))
            if threshold is not None:
                low |= selected.n_cases < threshold
            # Gaps, rather than lines joining across unsupported size bins.
            selected.loc[low, ['mean', 'ci_low', 'ci_high']] = np.nan
            for (label,mode),curve in selected.groupby(['label','analysis'],sort=False):
                curve=curve.sort_values('N')
                color = colors[label, mode]
                ax.plot(curve.N,curve['mean'],'o-' if mode=='observed' else 's--',color=color,label=f'{label}: {mode}')
                ax.fill_between(curve.N,curve.ci_low,curve.ci_high,color=color,alpha=.13)
            ax.axhline(0,color='grey',lw=.6)
            ax.set_xlim(part.N.min()*.9,part.N.max()*1.1)
            ax.set(xscale='log',xlabel='Cell count N',ylabel=ylabel,title=f'{center} ← {source}')
    handles={}
    for ax in axes.ravel():
        h,l=ax.get_legend_handles_labels();handles.update(zip(l,h))
    fig.legend(handles.values(),handles.keys(),loc='upper center',ncol=max(1,min(4,len(handles))))
    fig.suptitle(f'{title} — Hop {hop}' if title else f'Hop {hop}',y=1.02)
    fig.tight_layout(rect=(0,0,1,.96))
    return fig
