"""Figures for the checkpoint-specific, paired N embedding analysis."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import networkx as nx
import numpy as np
import pandas as pd

COLORS = {"KI67": "#b35a20", "LGR5": "#3277b3"}
LABELS = dict(delta_relative="ΔK × Â(N)/(4π)", delta_raw="ΔK (physical curvature)",
    delta_z="Δ transformed curvature", delta_fixed_reference="ΔK × Â(reference N)/(4π)",
    hidden_norm="‖Δh‖", hidden_ratio="‖Δh(N)‖ / ‖Δh(reference)‖",
    hidden_cosine="cos(Δh(N), Δh(reference))", share="Mean per-center absolute-effect share")


def _axis(ax, config, metric, xlabel="Supplied N"):
    ax.set_xscale("log")
    ax.set_xticks(config.anchors, labels=[str(n) for n in config.anchors])
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel(xlabel)
    ax.set_ylabel(LABELS.get(metric, metric))
    ax.grid(alpha=.15)
    for n in config.anchors[1:-1]:
        ax.axvline(n, color=".6", lw=.7, ls=":")


def _line(ax, frame, label, color, style="-"):
    frame = frame.sort_values("n")
    if frame.empty:
        return
    # Unsupported points remain in tables, not presented as robust curves.
    y = frame["mean"].where(frame.sufficient_support) if "sufficient_support" in frame else frame["mean"]
    ax.plot(frame.n, y, style, color=color, label=label, lw=1.7, marker=".")
    if {"low", "high"} <= set(frame):
        ax.fill_between(frame.n, frame.low, frame.high, color=color, alpha=.13)


def plot_curves(output_dir, config, metric="delta_relative", cohort="matched"):
    table = pd.read_csv(Path(output_dir) / "tables/overall_curves.csv")
    table = table[(table.metric == metric) & (table.cohort == cohort)]
    fig, axes = plt.subplots(len(config.hops), len(config.markers), figsize=(11, 4 * len(config.hops)), squeeze=False, layout="constrained")
    for r, hop in enumerate(config.hops):
        for c, marker in enumerate(config.markers):
            ax = axes[r, c]
            for route, style, color in [("full", "-", COLORS.get(marker, "C0")), ("film_only", "--", "#568a47"), ("head_only", ":", ".4")]:
                _line(ax, table[(table.marker == marker) & (table.hop == hop) & (table.route == route)], route.replace("_", " "), color, style)
            _axis(ax, config, metric)
            ax.set_title(f"Neighbor {marker}, hop {hop}; center LGR5 unchanged")
            ax.legend()
    fig.suptitle(f"{cohort.title()} support | paired organoid bootstrap; seeds averaged within organoid")
    return fig


def plot_tsne(checkpoint_dir, nodes, config, population="lgr5", color_by="cluster", ablation=None):
    frame = pd.read_csv(Path(checkpoint_dir) / population / "tsne.csv.gz")
    frame = frame.merge(nodes[["node_id", "center_marker", "region"]], on="node_id", validate="many_to_one")
    intact = frame[frame.condition == "intact"]
    categories = sorted(intact[color_by].unique())
    palette = {name: plt.get_cmap("tab10")(i % 10) for i, name in enumerate(categories)}
    fig, axes = plt.subplots(1, len(config.anchors), figsize=(5 * len(config.anchors), 4.7), sharex=True, sharey=True, layout="constrained")
    axes = np.atleast_1d(axes)
    chosen = sorted(frame.loc[frame.condition == ablation, "node_id"].unique())[:12] if ablation else []
    for ax, n in zip(axes, config.anchors):
        part = intact[intact.n == n]
        for label in categories:
            g = part[part[color_by] == label]
            ax.scatter(g.tsne_1, g.tsne_2, s=7, alpha=.55, color=palette[label], label=str(label), rasterized=True)
        if ablation:
            edits = frame[(frame.n == n) & (frame.condition == ablation) & frame.node_id.isin(chosen)].set_index("node_id")
            base = part.set_index("node_id")
            for i, edit in edits.iterrows():
                ax.annotate("", (edit.tsne_1, edit.tsne_2), (base.loc[i].tsne_1, base.loc[i].tsne_2),
                    arrowprops=dict(arrowstyle="->", color="black", lw=1, alpha=.65))
        ax.set_title(f"N = {n}")
        ax.set_xlabel("t-SNE 1")
        ax.set_xticks([])
        ax.set_yticks([])
    axes[0].set_ylabel("t-SNE 2")
    axes[-1].legend(markerscale=2, fontsize=8, loc="best")
    detail = f"; arrows remove one neighboring {ablation} signal (hop 1)" if ablation else ""
    fig.suptitle(f"{population}: one joint map and fixed sampled cells{detail}\nMap distances are for visualization; quantitative changes use hidden/PCA coordinates")
    return fig


def plot_profiles(checkpoint_dir, config, markers, population="lgr5", profile="hop1_fraction_"):
    frame = pd.read_csv(Path(checkpoint_dir) / population / "profiles.csv")
    cols = [profile + marker for marker in [*markers, "Unmarked"]]
    fig, axes = plt.subplots(1, len(config.anchors), figsize=(5 * len(config.anchors), 4.5), layout="constrained", squeeze=False)
    for ax, n in zip(axes[0], config.anchors):
        part = frame[frame.state == f"N{n}"].set_index("cluster").reindex(range(config.n_clusters))
        values = part[cols].to_numpy()
        im = ax.imshow(values, aspect="auto", vmin=0, vmax=1, cmap="viridis")
        ax.set_xticks(range(len(cols)), [*markers, "Unmarked"], rotation=70)
        ax.set_yticks(range(config.n_clusters), [f"C{k} ({int(part.loc[k, 'n_organoids']) if pd.notna(part.loc[k, 'n_organoids']) else 0} org)" for k in range(config.n_clusters)])
        ax.set_title(f"N = {n}")
    fig.colorbar(im, ax=axes[0].tolist(), label="Mean fraction; equal organoid weights within cluster", shrink=.7)
    fig.suptitle(f"{population}: {profile.replace('_', ' ')} | dynamic membership, fixed cluster definitions")
    return fig


def plot_transitions(checkpoint_dir, config, population="lgr5"):
    frame = pd.read_csv(Path(checkpoint_dir) / population / "transitions.csv")
    fig, axes = plt.subplots(1, len(config.anchors) - 1, figsize=(5 * (len(config.anchors) - 1), 4), squeeze=False, layout="constrained")
    for ax, (a, b) in zip(axes[0], zip(config.anchors[:-1], config.anchors[1:])):
        part = frame[(frame.from_n == a) & (frame.to_n == b)].pivot(index="source", columns="target", values="conditional_fraction")
        im = ax.imshow(part, vmin=0, vmax=1, cmap="Blues")
        ax.set_title(f"N = {a} → {b}")
        ax.set_xlabel("Later supplied-N cluster")
        ax.set_ylabel("Earlier supplied-N cluster")
        ax.set_xticks(range(config.n_clusters))
        ax.set_yticks(range(config.n_clusters))
    fig.colorbar(im, ax=axes[0].tolist(), label="Conditional transition fraction", shrink=.75)
    fig.suptitle(f"{population}: same-cell cluster transitions; supplied N is not tracked biological time")
    return fig


def plot_geometry(checkpoint_dir, config):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), layout="constrained")
    metrics = ["translation_norm", "common_positive_gain", "relative_shape_mismatch"]
    for population in ["all", "lgr5"]:
        frame = pd.read_csv(Path(checkpoint_dir) / population / "representation_change.csv").sort_values("n")
        for ax, metric in zip(axes, metrics):
            ax.plot(frame.n, frame[metric], marker=".", label=population)
            _axis(ax, config, metric.replace("_", " "))
            ax.legend()
    fig.suptitle("Embedding changes relative to the anchor: common translation, scale, and remaining shape change")
    return fig


def plot_fixed_clusters(output_dir, config, fold=0, seed=42, metric="delta_relative", cohort="matched"):
    table = pd.read_csv(Path(output_dir) / "tables/fixed_cluster_curves.csv")
    frame = table[(table.fold == fold) & (table.seed == seed) & (table.metric == metric) & (table.cohort == cohort)]
    fig, axes = plt.subplots(len(config.hops), config.n_clusters, figsize=(3.3 * config.n_clusters, 4 * len(config.hops)), squeeze=False, layout="constrained")
    for r, hop in enumerate(config.hops):
        for cluster, ax in enumerate(axes[r]):
            for marker in config.markers:
                part = frame[(frame.reference_cluster == cluster) & (frame.hop == hop) & (frame.marker == marker)]
                _line(ax, part, marker, COLORS.get(marker, "C0"))
            _axis(ax, config, metric)
            ax.set_title(f"C{cluster} at N={config.reference_n}; hop {hop}")
            if ax.lines:
                ax.legend(fontsize=8)
    fig.suptitle(f"Fixed LGR5-center groups, {cohort} support | fold {fold}, seed {seed}; cluster IDs belong to this checkpoint")
    return fig


def plot_observed(output_dir, config, metric="delta_relative"):
    tables = Path(output_dir) / "tables"
    observed = pd.read_csv(tables / "observed_curves.csv")
    local = pd.read_csv(tables / "local_support_sweep_curves.csv")
    overall = pd.read_csv(tables / "overall_curves.csv")
    overall = overall[(overall.route == "full") & (overall.cohort == "eligible")]
    fig, axes = plt.subplots(len(config.hops), len(config.markers), figsize=(11, 4 * len(config.hops)), squeeze=False, layout="constrained")
    for r, hop in enumerate(config.hops):
        for c, marker in enumerate(config.markers):
            ax = axes[r, c]
            for frame, label, color, style in [(overall, "Sweep: fixed full cohort", ".65", "-"),
                    (local, "Sweep: nearby observed-size cases", "#3277b3", "-"),
                    (observed, "Observed: those same cases", "#b35a20", "--")]:
                _line(ax, frame[(frame.marker == marker) & (frame.hop == hop) & (frame.metric == metric)], label, color, style)
            _axis(ax, config, metric, "Supplied N / observed-window anchor")
            ax.set_title(f"{marker}, hop {hop}")
            ax.legend(fontsize=8)
    fig.suptitle(f"Observed windows: nearest anchor within ×{config.observed_window_fold}; support varies between windows")
    return fig


def plot_seed_extrema(output_dir, config, metric="delta_relative", cohort="matched"):
    tables = Path(output_dir) / "tables"
    seeds = pd.read_csv(tables / "seed_curves.csv")
    boot = pd.read_csv(tables / "grid_extrema_bootstrap.csv")
    fig, axes = plt.subplots(2, len(config.markers), figsize=(11, 7.5), squeeze=False, layout="constrained")
    for c, marker in enumerate(config.markers):
        for seed, group in seeds[(seeds.route == "full") & (seeds.cohort == cohort) & (seeds.marker == marker) & (seeds.hop == 1)].groupby("seed"):
            axes[0, c].plot(group.n, group[metric], marker=".", label=f"seed {seed}")
        _axis(axes[0, c], config, metric)
        axes[0, c].set_title(f"{marker}, hop 1, {cohort} support")
        axes[0, c].legend()
        kind = "minimum" if marker == "KI67" else "maximum"
        group = boot[(boot.marker == marker) & (boot.hop == 1) & (boot.cohort == cohort) & (boot.metric == metric) & (boot.kind == kind)]
        axes[1, c].plot(group.n, group.probability, marker="o")
        _axis(axes[1, c], config, f"Bootstrap probability of grid {kind}")
    fig.suptitle("Extrema are selected on the evaluated grid; endpoint mass indicates a boundary extremum")
    return fig


def plot_shares(output_dir, config):
    fig, axes = plt.subplots(1, len(config.hops), figsize=(10, 4), squeeze=False, layout="constrained")
    for ax, hop in zip(axes[0], config.hops):
        for marker in config.markers:
            path = Path(output_dir) / f"tables/sensitivity_share_{marker}.csv"
            if path.exists():
                table = pd.read_csv(path)
                _line(ax, table[table.hop == hop], marker, COLORS.get(marker, "C0"))
        ax.axhline(.5, color=".5", ls=":")
        ax.set_ylim(0, 1)
        _axis(ax, config, "share")
        ax.set_title(f"Matched source availability, hop {hop}")
        ax.legend()
    fig.suptitle("Absolute response share among the selected marker ablations; not a fraction of the whole prediction")
    return fig


def plot_exemplars(checkpoint_dir, nodes, neighborhoods, config, markers):
    assigned = pd.read_csv(Path(checkpoint_dir) / "lgr5/assignments.csv.gz")
    assigned = assigned[assigned.state == f"N{config.reference_n}"]
    assigned = assigned.merge(nodes, on="node_id", validate="one_to_one")
    fig, axes = plt.subplots(1, config.n_clusters, figsize=(3.6 * config.n_clusters, 4), squeeze=False, layout="constrained")
    palette = [plt.get_cmap("tab10")(i) for i in range(len(markers) + 1)]
    for cluster, ax in enumerate(axes[0]):
        candidates = assigned[assigned.cluster == cluster].sort_values(["confidence", "node_id"], ascending=[False, True])
        if candidates.empty:
            ax.set_title(f"C{cluster}: empty")
            ax.axis("off")
            continue
        row = candidates.iloc[0]
        sub = neighborhoods[str(int(row.node_id))]
        graph = nx.Graph()
        graph.add_nodes_from(range(len(sub["orig_nodes"])))
        graph.add_edges_from(np.asarray(sub["edges"]).T.tolist())
        center = sub["center_idx"]
        positions = nx.spring_layout(graph, seed=config.sampling_seed)
        nx.draw_networkx(graph, pos=positions, ax=ax, with_labels=False,
            node_color=[palette[i] for i in sub["marker_index"]],
            node_size=[160 if j == center else 45 for j in graph],
            edgecolors=["black" if j == center else "white" for j in graph], width=.5)
        ax.set_title(f"C{cluster}; {row.region}\n{row.organoid_str}\ncell {row.orig_center}, observed N={row.observed_n}", fontsize=8)
        ax.axis("off")
    handles = [plt.Line2D([], [], marker="o", ls="", color=palette[i], label=m) for i, m in enumerate([*markers, "Unmarked"])]
    fig.legend(handles=handles, loc="outside lower center", ncol=len(handles))
    fig.suptitle(f"One high-confidence neighborhood per cluster at N={config.reference_n}; enlarged node is the unchanged LGR5 center")
    return fig


def save_figure(fig, output_dir, name):
    folder = Path(output_dir) / "figures"
    folder.mkdir(exist_ok=True, parents=True)
    for extension in ("png", "pdf"):
        fig.savefig(folder / f"{name}.{extension}", dpi=170, bbox_inches="tight")
    return fig
