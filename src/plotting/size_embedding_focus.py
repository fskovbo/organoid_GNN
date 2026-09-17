"""Five figures addressing non-monotonic responses, without marker rankings."""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from src.analysis.size_embedding_focus import FACTORS, load_config

FACTOR_LABELS = {"ablation_response": "Change in ablation response Δh", "intact_embedding": "Change in intact embedding h",
                 "head_size": "Explicit size input to head", "normalization": "Geometric normalization"}
FACTOR_COLORS = {"ablation_response": "#3977b3", "intact_embedding": "#db8c36", "head_size": "#765ca6", "normalization": "#72954d"}


def _axis(ax, config, ylabel):
    ax.set_xscale("log")
    ax.set_xticks(config.anchors, [str(n) for n in config.anchors])
    ax.minorticks_off()
    ax.set_xlabel("Supplied cell count N")
    ax.set_ylabel(ylabel)
    ax.axhline(0, color=".6", lw=.7)
    ax.grid(alpha=.15)


def _line(ax, part, **kwargs):
    part = part.sort_values("n")
    if part.empty:
        return
    mean = part["mean"].where(part.sufficient_support)
    ax.plot(part.n, mean, **kwargs)
    ax.fill_between(part.n, part.low, part.high, color=kwargs.get("color", "C0"), alpha=.12)


def response_curves(source):
    """Each marker uses its own fixed eligible cohort; no relative importance score."""
    source = Path(source)
    config = load_config(source)
    means = pd.read_csv(source / "tables/overall_curves.csv")
    means = means[(means.cohort == "eligible") & (means.route == "full") & (means.metric == "delta_relative")]
    seeds = pd.read_csv(source / "tables/seed_curves.csv")
    seeds = seeds[(seeds.cohort == "eligible") & (seeds.route == "full")]
    fig, axes = plt.subplots(len(config.hops), len(config.markers), figsize=(11, 6.8), squeeze=False, layout="constrained")
    for r, hop in enumerate(config.hops):
        for c, marker in enumerate(config.markers):
            ax = axes[r, c]
            part = means[(means.marker == marker) & (means.hop == hop)]
            for _, group in seeds[(seeds.marker == marker) & (seeds.hop == hop)].groupby("seed"):
                ax.plot(group.n, group.delta_relative, color=".65", lw=.8, alpha=.7)
            _line(ax, part, color="#286995", lw=2, label="Mean ± organoid bootstrap interval")
            _axis(ax, config, "Normalized ablation effect")
            ax.set_title(f"Remove neighbor {marker} · hop {hop}")
            if hop == 1 and not part.empty:
                index = part["mean"].idxmin() if marker == "KI67" else part["mean"].idxmax()
                row = part.loc[index]
                ax.plot(row.n, row["mean"], "o", color="black", ms=4)
                ax.annotate(f"N={int(row.n)}", (row.n, row["mean"]), xytext=(10, 12), textcoords="offset points")
    fig.suptitle("1 · Where do the responses turn?\nCenter LGR5 and neighborhoods stay fixed; gray curves show the three seeds")
    return fig


def mechanism_changes(source, hop=1):
    source = Path(source)
    config = load_config(source)
    table = pd.read_csv(source / "focused_analysis/tables/mechanism_summary.csv")
    table = table[table.hop == hop]
    fig, axes = plt.subplots(len(config.markers), 2, figsize=(13, 7), squeeze=False, layout="constrained")
    for r, marker in enumerate(config.markers):
        for c, scale in enumerate(("delta_z", "delta_relative")):
            ax = axes[r, c]
            part = table[(table.marker == marker) & (table.scale == scale)]
            intervals = part[["from_n", "n"]].drop_duplicates().sort_values("from_n")
            bottom_pos, bottom_neg = np.zeros(len(intervals)), np.zeros(len(intervals))
            total = np.zeros(len(intervals))
            for factor in FACTORS:
                values = []
                for interval in intervals.itertuples():
                    row = part[(part.from_n == interval.from_n) & (part.n == interval.n) & (part.factor == factor)].iloc[0]
                    values.append(row["mean"] if row.sufficient_support else np.nan)
                values = np.asarray(values)
                bottom = np.where(values >= 0, bottom_pos, bottom_neg)
                ax.bar(np.arange(len(intervals)), values, bottom=bottom, color=FACTOR_COLORS[factor], label=FACTOR_LABELS[factor], width=.7)
                bottom_pos += np.maximum(values, 0)
                bottom_neg += np.minimum(values, 0)
                total += values
            ax.plot(np.arange(len(intervals)), total, "ko", label="Net response change", ms=4)
            ax.axhline(0, color=".4", lw=.7)
            ax.set_xticks(np.arange(len(intervals)), [f"{int(a)}→{int(b)}" for a, b in intervals.to_numpy()])
            ax.set_xlabel("Interval in supplied N")
            ax.set_ylabel("Change in ablation effect across interval")
            ax.set_title(f"{marker}, hop {hop} · {'transformed curvature' if scale == 'delta_z' else 'normalized curvature'}")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3, fontsize=9)
    fig.suptitle("2 · What produces the rise or fall?\nSigned contributions from cached-head substitutions; interactions shared symmetrically between factors")
    return fig


def _order(source, fold, seed):
    folder = Path(source) / f"focused_analysis/fold_{fold}/seed_{seed}/lgr5"
    order = pd.read_csv(folder / "curvature_order.csv").sort_values("cluster")
    count = (~order.empty_at_reference).sum()
    palette = {int(r.cluster): (".55" if r.empty_at_reference else plt.get_cmap("plasma")(.1 + .7*r.cluster / max(1, count-1))) for r in order.itertuples()}
    return folder, order, palette


def embedding_map(source, fold=0, seed=42):
    config = load_config(source)
    folder, order, palette = _order(source, fold, seed)
    table = pd.read_csv(folder / "tsne.csv.gz")
    table = table[table.condition == "intact"]
    fig, axes = plt.subplots(1, len(config.anchors), figsize=(15, 4.1), sharex=True, sharey=True, layout="constrained")
    for ax, n in zip(np.atleast_1d(axes), config.anchors):
        for cluster, group in table[table.n == n].groupby("cluster"):
            ax.scatter(group.tsne_1, group.tsne_2, s=6, alpha=.55, color=palette[cluster], rasterized=True)
        ax.set_title(f"N = {n}")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("t-SNE 1")
    axes[0].set_ylabel("t-SNE 2")
    handles = [Line2D([], [], ls="", marker="o", color=palette[r.cluster], label=f"C{r.cluster}" + (" (empty at reference)" if r.empty_at_reference else "")) for r in order.itertuples()]
    fig.legend(handles=handles, loc="outside lower center", ncol=len(handles), fontsize=8)
    fig.suptitle(f"3 · How does the representation change with N?\nFold {fold}, seed {seed}; low→high median predicted residual curvature at N={config.reference_n}; ordering fixed")
    return fig


def neighborhood_content(source, fold=0, seed=42):
    source = Path(source)
    config = load_config(source)
    folder, order, palette = _order(source, fold, seed)
    markers = json_markers(source)
    profiles = pd.read_csv(folder / "profiles.csv")
    profiles = profiles[profiles.state == f"N{config.reference_n}"].set_index("cluster")
    ids = order.loc[~order.empty_at_reference, "cluster"].to_numpy()
    fig, axes = plt.subplots(1, len(config.hops), figsize=(12, 4.5), squeeze=False, layout="constrained")
    for ax, hop in zip(axes[0], config.hops):
        values = profiles.reindex(ids)[[f"hop{hop}_fraction_{m}" for m in markers]]
        im = ax.imshow(values, vmin=0, vmax=1, cmap="viridis", aspect="auto")
        ax.set_xticks(range(len(markers)), markers, rotation=45, ha="right")
        ax.set_yticks(range(len(ids)), [f"C{k} ({int(profiles.loc[k,'n_organoids'])} org)" for k in ids])
        for label, k in zip(ax.get_yticklabels(), ids):
            label.set_color(palette[k])
        ax.set_title(f"Exact hop {hop}")
    fig.colorbar(im, ax=axes[0].tolist(), label="Mean neighbor fraction (equal organoid weight)", shrink=.75)
    fig.suptitle(f"4 · Which fixed neighborhoods do these clusters describe?\nMembership fixed at N={config.reference_n}; rows ordered by predicted residual curvature")
    return fig


def json_markers(source):
    import json
    return [*json.loads((Path(source) / "settings.json").read_text())["markers"], "Unmarked"]


def neighborhood_responses(source, fold=0, seed=42):
    source = Path(source)
    config = load_config(source)
    _, order, palette = _order(source, fold, seed)
    ids = order.loc[~order.empty_at_reference, "cluster"].to_numpy()
    table = pd.read_csv(source / "focused_analysis/tables/fixed_cluster_curves.csv")
    table = table[(table.fold == fold) & (table.seed == seed) & (table.cohort == "eligible") & (table.metric == "delta_relative")]
    fig, axes = plt.subplots(len(config.markers), len(ids), figsize=(3.2 * len(ids), 6.7), squeeze=False, layout="constrained", sharey="row")
    for r, marker in enumerate(config.markers):
        for c, cluster in enumerate(ids):
            ax = axes[r, c]
            part = table[(table.marker == marker) & (table.reference_cluster == cluster)]
            supported = False
            for hop, style in [(1, "-"), (2, "--")]:
                group = part[part.hop == hop]
                if not group.empty and group.sufficient_support.any():
                    supported = True
                    _line(ax, group, color=palette[cluster], ls=style, lw=1.8, label=f"Hop {hop} ({int(group.n_organoids.min())} org)")
            _axis(ax, config, f"Remove {marker}\nNormalized ablation effect" if c == 0 else "")
            ax.set_title(f"C{cluster}")
            if supported:
                ax.legend(fontsize=7)
            else:
                ax.text(.5, .5, "Insufficient source support", ha="center", transform=ax.transAxes, fontsize=8)
    fig.suptitle(f"5 · Which neighborhood groups show the bends?\nFixed N={config.reference_n} groups; each marker uses its own eligible sources; fold {fold}, seed {seed}")
    return fig


def all_figures(source):
    config = load_config(source)
    fold, seed = config.tsne_checkpoint
    return {
        "01_response_curves": response_curves(source),
        "02_sources_of_response_change": mechanism_changes(source),
        "03_curvature_ordered_embeddings": embedding_map(source, fold, seed),
        "04_fixed_neighborhood_content": neighborhood_content(source, fold, seed),
        "05_fixed_neighborhood_responses": neighborhood_responses(source, fold, seed),
    }
