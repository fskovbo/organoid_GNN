"""Focused displays for fixed-pair, one-hop KI67 sweeps."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D


COUNTS = (80, 300, 700)
N_COLORS = {80: '#2166ac', 300: '#7b3294', 700: '#d6604d'}


def _read(out, name):
    return pd.read_csv(Path(out) / name)


def _size_axis(ax, ylabel=None):
    ax.set_xscale('log')
    ax.set_xticks([80, 220, 300, 400, 700], ['80', '220', '300', '400', '700'])
    ax.axvline(300, color='.5', lw=.8, ls=':')
    ax.set_xlabel('Supplied cell count N (log scale)')
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.grid(alpha=.15)


def _curve(ax, curves, metric, label, color, band=True, **kwargs):
    p = curves[curves.metric == metric].sort_values('n')
    ax.plot(p.n, p['mean'], label=label, color=color, **kwargs)
    if band:
        ax.fill_between(p.n, p.low, p.high, color=color, alpha=.13)
    return p


def response_patterns(out):
    curves = _read(out, 'tables/curves.csv')
    seeds = _read(out, 'tables/seed_curves.csv')
    individual = _read(out, 'tables/individual_curves.csv.gz')
    patterns = _read(out, 'tables/individual_patterns.csv')
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3), layout='constrained')
    for _, p in seeds.groupby('seed'):
        axes[0].plot(p.n, p.delta_relative, color='.65', lw=.9, alpha=.8)
    _curve(axes[0], curves, 'delta_relative', 'Mean; 95% organoid bootstrap', '#2166ac')
    axes[0].axhline(0, color='.3', lw=.8)
    _size_axis(axes[0], 'Geometrically normalized ablation effect')
    axes[0].set_title('A  Population response (all 15 models)')
    axes[0].legend(fontsize=8)
    pivot = individual.pivot(index=['fold', 'case_id'], columns='n', values='delta_relative')
    order = patterns.sort_values(['minimum_n', 'fold', 'case_id']).set_index(['fold', 'case_id']).index
    matrix = pivot.reindex(order)
    limit = np.quantile(np.abs(matrix.to_numpy()), .97)
    im = axes[1].imshow(matrix, aspect='auto', cmap='RdBu_r', vmin=-limit, vmax=limit, interpolation='nearest')
    axes[1].set_xticks(range(len(matrix.columns)), matrix.columns, rotation=45)
    axes[1].set_xlabel('Supplied N (columns equally spaced)')
    axes[1].set_ylabel('Fixed source–center pairs, sorted by minimum N')
    axes[1].set_title('B  Individual responses (mean across seeds)')
    fig.colorbar(im, ax=axes[1], label='Normalized effect (colors clipped at 97th percentile)')
    return fig


def _cluster_colors(out, fold=0, seed=42):
    order = _read(out, f'fold_{fold}/seed_{seed}/curvature_order.csv')
    occupied = order.loc[~order.empty_at_reference, 'cluster'].tolist()
    colors = {c: plt.get_cmap('plasma')(v) for c, v in zip(occupied, np.linspace(.12, .82, len(occupied)))}
    return {int(c): colors.get(c, '.6') for c in order.cluster}


def state_map(out, fold=0, seed=42):
    points = _read(out, f'fold_{fold}/seed_{seed}/pca_points.csv.gz')
    variance = _read(out, f'fold_{fold}/seed_{seed}/pca_variance.csv').variance_ratio
    colors = _cluster_colors(out, fold, seed)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharex=True, sharey=True, layout='constrained')
    for ax, n in zip(axes, COUNTS):
        p = points[points.n == n]
        for c, part in p.groupby('cluster'):
            ax.scatter(part.pc1, part.pc2, s=10, alpha=.7, color=colors[c], label=f'C{c+1}')
        ax.set_title(f'Intact centers: N={n}')
        ax.set_xlabel(f'PC1 ({variance.iloc[0]:.1%} state variance)')
    axes[0].set_ylabel(f'PC2 ({variance.iloc[1]:.1%} state variance)')
    handles = [Line2D([], [], color=color, marker='o', ls='', label=f'C{c+1}') for c, color in colors.items()]
    axes[-1].legend(handles=handles, title='Fixed cluster labels', fontsize=7, ncol=2)
    return fig


def ablation_movements(out, fold=0, seed=42):
    points = _read(out, f'fold_{fold}/seed_{seed}/pca_points.csv.gz')
    response = _read(out, f'fold_{fold}/seed_{seed}/response_points.csv.gz')
    reference = points[points.n == 300]
    selected = reference.groupby('reference_cluster', group_keys=False).sample(frac=.12, random_state=73).case_id
    p = points[points.case_id.isin(selected)]
    r = response[response.case_id.isin(selected)]
    limit = max(np.quantile(np.abs(p.delta_relative), .97), 1e-8)
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), sharex='row', sharey='row', layout='constrained')
    for j, n in enumerate(COUNTS):
        part = p[p.n == n]
        axes[0, j].scatter(part.pc1, part.pc2, color='.65', s=8, alpha=.6)
        q = axes[0, j].quiver(part.pc1, part.pc2, part.ablated_pc1-part.pc1, part.ablated_pc2-part.pc2,
            part.delta_relative, cmap='RdBu_r', norm=norm, angles='xy', scale_units='xy', scale=1, width=.003)
        part = r[r.n == n]
        axes[1, j].quiver(np.zeros(len(part)), np.zeros(len(part)), part.response1, part.response2,
            part.delta_relative, cmap='RdBu_r', norm=norm, angles='xy', scale_units='xy', scale=1, width=.003, alpha=.7)
        axes[0, j].set_title(f'N={n}: intact → KI67 removed')
        axes[0, j].set_xlabel('State PC1')
        axes[1, j].set_xlabel('Response axis 1')
        for ax in axes[:, j]:
            ax.axhline(0, color='.8', lw=.6)
            ax.axvline(0, color='.8', lw=.6)
            ax.set_aspect('equal', adjustable='box')
    for row, xs, ys in [(0, np.r_[p.pc1, p.ablated_pc1], np.r_[p.pc2, p.ablated_pc2]),
                        (1, np.r_[0, r.response1], np.r_[0, r.response2])]:
        for ax in axes[row]:
            dx, dy = max(np.ptp(xs), 1e-5), max(np.ptp(ys), 1e-5)
            ax.set_xlim(xs.min()-.08*dx, xs.max()+.08*dx)
            ax.set_ylim(ys.min()-.08*dy, ys.max()+.08*dy)
    axes[0, 0].set_ylabel('State PC2')
    axes[1, 0].set_ylabel('Response axis 2\n(origin = no ablation)')
    fig.colorbar(q, ax=axes.ravel().tolist(), shrink=.8, label='Full-model normalized effect')
    return fig


def projection_fidelity(out):
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.7), layout='constrained')
    for name, style in [('projection_fidelity', '-'), ('response_projection_fidelity', '--')]:
        table = _read(out, f'tables/{name}.csv')
        for n in COUNTS:
            part = table[table.n == n].sort_values('dimensions')
            for ax, metric in zip(axes, ['retained_energy', 'relative_rmse', 'sign_agreement']):
                ax.plot(part.dimensions, part[metric], style, color=N_COLORS[n], marker='.', ms=5)
    for ax, title, ylabel in zip(axes,
        ['A  Hidden displacement retained', 'B  Error in normalized response', 'C  Response sign retained'],
        ['Mean squared-norm fraction', 'RMSE / RMS(full effect)', 'Fraction with matching sign']):
        ax.set_xscale('log', base=2)
        ax.set_xticks([2, 8, 32, 128, 256], ['2', '8', '32', '128', '256'])
        ax.set_xlabel('Retained dimensions')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(alpha=.15)
    handles = [Line2D([], [], color=c, label=f'N={n}') for n, c in N_COLORS.items()]
    handles += [Line2D([], [], color='.3', ls=ls, label=label) for ls, label in [('-', 'State PCA'), ('--', 'Response SVD')]]
    axes[0].legend(handles=handles, fontsize=7)
    return fig


def readout_mechanism(out):
    curves = _read(out, 'tables/curves.csv')
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5), layout='constrained')
    _curve(axes[0, 0], curves, 'delta_z', 'Actual response', '#2166ac')
    _curve(axes[0, 0], curves, 'frozen_reference_readout', 'Readout fixed at N=300 (algebraic probe)', '#b35806')
    axes[0, 0].set_title('A  Does the embedding displacement itself turn?')
    a = _curve(axes[0, 1], curves, 'response_change_from_reference', 'Change in embedding displacement', '#2166ac')
    b = _curve(axes[0, 1], curves, 'readout_change_from_reference', 'Change in effective readout', '#b35806')
    axes[0, 1].plot(a.n, a['mean'].to_numpy()+b['mean'].to_numpy(), color='black', ls='--', label='Sum = response change from N=300')
    axes[0, 1].set_title('B  Exact finite-step decomposition (full 256-D)')
    _curve(axes[1, 0], curves, 'hidden_cosine', 'KI67 displacement direction', '#2166ac')
    _curve(axes[1, 0], curves, 'readout_cosine', 'Effective readout direction', '#b35806')
    axes[1, 0].set_title('C  Direction relative to N=300')
    _curve(axes[1, 1], curves, 'positive_units', 'Positive head-unit contributions', '#b2182b')
    _curve(axes[1, 1], curves, 'negative_units', 'Negative head-unit contributions', '#2166ac')
    _curve(axes[1, 1], curves, 'delta_z', 'Net response', 'black')
    axes[1, 1].set_title('D  Opposing contributions inside the head')
    for ax, ylabel in zip(axes.ravel(), ['Δ transformed curvature', 'Change in Δ transformed curvature', 'Mean cosine similarity', 'Δ transformed curvature']):
        _size_axis(ax, ylabel)
        ax.legend(fontsize=7)
        if 'cosine' not in ylabel:
            ax.axhline(0, color='.5', lw=.7)
    return fig


def neighborhood_context(out, fold=0, seed=42):
    profiles = _read(out, f'fold_{fold}/seed_{seed}/neighborhood_profiles.csv').sort_values('cluster')
    curves = _read(out, 'tables/cluster_curves.csv')
    curves = curves[(curves.fold == fold) & (curves.seed == seed)]
    colors = _cluster_colors(out, fold, seed)
    markers = ['Agr2', 'AldoB', 'Chroma', 'KI67', 'LGR5', 'Lysozyme', 'Serotonin', 'Unmarked']
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3), layout='constrained')
    matrix = profiles[[f'hop1_fraction_{m}' for m in markers]].to_numpy()
    im = axes[0].imshow(matrix, vmin=0, vmax=1, cmap='Blues', aspect='auto')
    for i in range(len(profiles)):
        for j in range(len(markers)):
            axes[0].text(j, i, f'{matrix[i,j]:.0%}', ha='center', va='center', fontsize=8, color='white' if matrix[i,j]>.55 else 'black')
    axes[0].set_xticks(range(len(markers)), markers, rotation=45, ha='right')
    axes[0].set_yticks(range(len(profiles)), [f'C{r.cluster+1} ({r.n_organoids} organoids)' for r in profiles.itertuples()])
    axes[0].set_title('A  Neighbor composition of fixed N=300 groups')
    fig.colorbar(im, ax=axes[0], shrink=.8, label='Mean fraction of hop-1 neighbors')
    for c, part in curves.groupby('cluster'):
        part = part[part.sufficient_support].sort_values('n')
        if part.empty:
            continue
        axes[1].plot(part.n, part['mean'], label=f'C{c+1}', color=colors[c])
        axes[1].fill_between(part.n, part.low, part.high, color=colors[c], alpha=.1)
    axes[1].axhline(0, color='.5', lw=.7)
    _size_axis(axes[1], 'Normalized KI67 ablation effect')
    axes[1].set_title('B  Same centers and group membership at every N')
    axes[1].legend(fontsize=8)
    return fig


def other_marker_probes(out, fold=0, seed=42):
    frame = _read(out, f'fold_{fold}/seed_{seed}/marker_probes.csv.gz')
    org = frame.groupby(['marker', 'organoid_str', 'n']).mean(numeric_only=True).reset_index()
    means = org.groupby(['marker', 'n']).mean(numeric_only=True).reset_index()
    supports = frame.groupby('marker').organoid_str.nunique()
    colors = dict(zip(sorted(supports.index), plt.get_cmap('tab10').colors))
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), layout='constrained')
    arrows = []
    for marker, part in means.groupby('marker'):
        color = colors[marker]
        p = part[part.n == 300].iloc[0]
        for x, y, alpha in [(p.delta_pc1, p.delta_pc2, 1), (p.ki67_pc1, p.ki67_pc2, .3)]:
            axes[0].quiver(0, 0, x, y, color=color, alpha=alpha, angles='xy', scale_units='xy', scale=1, width=.009)
            arrows.append((x, y))
        axes[1].plot(part.n, part.full_space_cosine_to_KI67, color=color, marker='o', label=f'{marker} ({supports[marker]} org.)')
        axes[2].plot(part.n, part.retained_energy_2pc, color=color, marker='o')
    arrows = np.vstack([[0, 0], arrows])
    for dimension, setter in enumerate([axes[0].set_xlim, axes[0].set_ylim]):
        lo, hi = arrows[:, dimension].min(), arrows[:, dimension].max()
        setter(lo-.2*(hi-lo), hi+.2*(hi-lo))
    axes[0].set_aspect('equal', adjustable='box')
    axes[0].set_xlabel('Δ state PC1')
    axes[0].set_ylabel('Δ state PC2')
    axes[0].set_title('A  Mean arrows at N=300\nFaint = KI67 on matching centers')
    for ax, title, label in zip(axes[1:], ['B  Paired direction comparison (256-D)', 'C  Visibility in the state PCA'],
        ['Mean cosine to KI67 displacement', 'Mean retained squared-norm fraction']):
        _size_axis(ax, label)
        ax.set_xticks(list(COUNTS), [str(n) for n in COUNTS])
        ax.set_title(title)
    axes[1].legend(fontsize=7)
    return fig
