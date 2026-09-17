"""Paired old/replacement contrasts with explicit identity and support labels."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from src.analysis.replacement_ablation import METHODS

LABELS = {'zero': 'Zero → unassigned', 'replacement_fixed': 'Replacement: fixed weights',
          'replacement_size_dependent': 'Replacement: N-dependent weights',
          'masking': 'Hide fate (trained mask)'}
COLORS = dict(zip(METHODS, ['#777777', '#2878b5', '#d45a31']))
COLORS['masking'] = '#7b3faa'
METRICS = {'delta_relative': 'Δ curvature / (4π / Aref(N))',
           'delta_mu': 'Δ curvature', 'delta_z': 'Δ transformed curvature'}


def plot_pair_grid(result, *, analysis='sweep', hop=1, metric='delta_relative', min_organoids=5):
    """Rows = unchanged recipient; columns = original source identity.

    Every panel uses the same cases for all three methods. Sweep cohorts also
    stay fixed across N. Panel y scales differ to preserve visible curve shape.
    """
    names = result['config']['markers']
    summary = result['summary']
    methods = result.get('methods', METHODS)
    k = len(names)
    fig, axes = plt.subplots(k, k, figsize=(2.8*k, 2.25*k), squeeze=False, sharex=True)
    for i, center in enumerate(names):
        for j, source in enumerate(names):
            ax = axes[i, j]
            ax.axhline(0, color='.8', lw=.7)
            if not summary.empty:
                frame = summary[(summary.analysis == analysis) & (summary.hop == hop) &
                    (summary.metric == metric) & (summary.center_marker == center) &
                    (summary.source_marker_name == source)].sort_values('n')
                for method in methods:
                    part = frame[frame.method == method].copy()
                    ok = part.n_organoids >= min_organoids
                    # Keep NaN gaps so unsupported bins are not connected across.
                    y = part['mean'].where(ok)
                    ax.plot(part.n, y, color=COLORS[method], lw=1.4,
                            ls='--' if method == 'replacement_size_dependent' else '-', marker='.', ms=3)
                    ax.fill_between(part.n, part.ci_low.where(ok), part.ci_high.where(ok),
                                    color=COLORS[method], alpha=.10)
                if frame.empty or not (frame.n_organoids >= min_organoids).any():
                    ax.text(.5, .5, 'Insufficient support', ha='center', transform=ax.transAxes, fontsize=8)
                else:
                    counts = frame.n_organoids
                    ax.text(.03, .96, f'org: {counts.min()}–{counts.max()}', va='top',
                            transform=ax.transAxes, fontsize=7, color='.4')
            else:
                ax.text(.5, .5, 'Insufficient support', ha='center', transform=ax.transAxes, fontsize=8)
            if i == 0:
                ax.set_title(f'Source: {source}', fontsize=10)
            if j == 0:
                ax.set_ylabel(f'Center: {center}', fontsize=10)
            if i == k - 1:
                ax.set_xlabel('Supplied N' if analysis == 'sweep' else 'Observed N (bin median)')
            ax.set_xscale('log'); ax.tick_params(labelsize=7)
    handles = [Line2D([0], [0], color=COLORS[m], label=LABELS[m],
                      ls='--' if m == 'replacement_size_dependent' else '-') for m in methods]
    fig.legend(handles=handles, loc='upper center', ncol=len(methods), bbox_to_anchor=(.5, .978))
    mode = 'Fixed neighborhoods, size sweep' if analysis == 'sweep' else 'Static: observed size'
    model_label = result.get('model_label', 'Original exclusive FiLM')
    fig.suptitle(f'{model_label}: {mode} • exact hop {hop}\n'
                 f'{METRICS[metric]} = intervened − intact prediction', y=1.01)
    fig.text(.5, .003, 'Unassigned-source zeroing is a no-op. Per-panel y scales. '
             'Shading: organoid bootstrap, conditional on fitted models and replacement references.', ha='center', fontsize=10)
    fig.tight_layout(rect=[0, .02, 1, .955])
    return fig


def plot_support(result):
    names = result['config']['markers']; support = result['support']
    fig, axes = plt.subplots(2, 2, figsize=(16, 13), squeeze=False)
    for row, analysis in enumerate(['observed', 'sweep']):
        for col, hop in enumerate([1, 2]):
            ax = axes[row, col]
            part = support[(support.analysis == analysis) & (support.hop == hop)]
            grouped = part.groupby(['center_marker', 'source_marker_name'])[['n_supported', 'n_cases']].sum()
            fraction = (grouped.n_supported / grouped.n_cases).unstack().reindex(index=names, columns=names)
            image = ax.imshow(fraction.to_numpy(), vmin=0, vmax=1, cmap='viridis')
            ax.set_xticks(range(len(names)), names, rotation=45, ha='right')
            ax.set_yticks(range(len(names)), names)
            ax.set_xlabel('Original source identity'); ax.set_ylabel('Unchanged center identity')
            ax.set_title(f'{analysis}, hop {hop}: fraction retained in paired comparison')
            for i in range(len(names)):
                for j in range(len(names)):
                    value = fraction.iloc[i, j]
                    ax.text(j, i, '—' if not np.isfinite(value) else f'{value:.0%}', ha='center', va='center',
                            color='white' if np.isfinite(value) and value < .5 else 'black', fontsize=9)
    fig.suptitle('Coverage: sweeps require support at every N; blank ≠ zero response')
    fig.subplots_adjust(wspace=.4, hspace=.45, right=.86, top=.93, bottom=.10)
    colorbar_ax = fig.add_axes([.90, .28, .015, .45])
    fig.colorbar(image, cax=colorbar_ax, label='Supported fraction')
    return fig


def plot_reference_weights(result, *, center, source, hop=1):
    """Optional inspection: identities contributing to each mixture at each N."""
    frame = result['paired_cases']
    frame = frame[(frame.analysis == 'sweep') & (frame.center_marker == center) &
                  (frame.source_marker_name == source) & (frame.hop == hop)]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, mode in zip(axes, ['fixed', 'dependent']):
        columns = [f'p_{mode}_{j}' for j in range(len(result['config']['markers']))]
        if not frame.empty:
            # Average repeated seeds and cases within organoid first.
            p = frame.groupby(['evaluated_n', 'organoid_str'])[columns].mean().groupby('evaluated_n').mean()
            ax.stackplot(p.index, p.to_numpy().T, labels=result['config']['markers'])
        ax.set_xscale('log'); ax.set_ylim(0, 1); ax.set_xlabel('Supplied N'); ax.set_title(mode)
    axes[0].set_ylabel('Replacement probability, equal organoid weight')
    if not frame.empty:
        axes[-1].legend(loc='upper left', bbox_to_anchor=(1, 1))
    else:
        for ax in axes:
            ax.text(.5, .5, 'No supported cases', ha='center', transform=ax.transAxes)
    fig.suptitle(f'Center {center}, source {source}, exact hop {hop}')
    fig.tight_layout()
    return fig
