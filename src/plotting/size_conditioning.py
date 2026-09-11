"""Compare observed-size and fixed-neighborhood ablations from saved tables."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import NullFormatter
import numpy as np
import pandas as pd


MODEL_LABELS = {
    "gin_head_size": "GIN: N in head",
    "gin_film_size": "GIN: N in FiLM and head",
    "gin_head_size_geometry": "GIN: N + geometry in head",
    "gin_film_size_geometry": "GIN: N in FiLM; N + geometry in head",
}
CURVE_STYLES = {
    "observed": dict(color="#2878b5", linestyle="-", marker="o", label="Observed size"),
    "sweep": dict(color="#d45a31", linestyle="--", marker="s", label="Sweep size"),
}


def load_size_overlay_tables(run_dir):
    """Load original-unit effects and the exact observed-bin x coordinates.

    This function reads CSVs only. No graph data or checkpoints are loaded.
    The validation summary saved the same cohort-bin medians that were used
    to plot observed effects in the original experiment.
    """
    tables = Path(run_dir) / "tables"
    centers = pd.read_csv(tables / "binned_mse.csv")[["size_bin", "n_center"]].drop_duplicates()
    if centers["size_bin"].duplicated().any():
        raise ValueError("Saved size-bin medians disagree between models.")
    observed = pd.read_csv(tables / "observed_delta_mu_summary.csv")
    observed = observed.merge(centers, on="size_bin", how="left", validate="many_to_one")
    observed = observed.rename(columns={"n_center": "n"}).assign(analysis="observed")
    sweep = pd.read_csv(tables / "sweep_delta_mu_summary.csv")
    sweep = sweep.rename(columns={"evaluated_n": "n"}).assign(analysis="sweep")
    combined = pd.concat([observed, sweep], ignore_index=True)
    required = ["model", "center_marker", "source_marker_name", "hop", "n", "mean", "n_organoids"]
    if combined[required].isna().any().any() or (combined["n"] <= 0).any():
        raise ValueError("Missing overlay data or invalid cell-count coordinates.")
    keys = ["model", "center_marker", "source_marker_name", "hop", "analysis", "n"]
    if combined.duplicated(keys).any():
        raise ValueError("Duplicate effect-summary rows.")
    return combined


def _pair_rows(table, center, source, hop):
    return table[(table.center_marker == center) &
                 (table.source_marker_name == source) & (table.hop == hop)]


def _draw_overlay(ax, panel, *, min_organoids, show_confidence, x_scale):
    ax.axhline(0, color="0.65", linewidth=0.65, zorder=0)
    available = False
    for kind, style in CURVE_STYLES.items():
        group = panel[panel.analysis == kind].sort_values("n")
        valid = group.n_organoids >= min_organoids
        # Keep masked rows to leave gaps where support is inadequate.
        ax.plot(group.n, group["mean"].where(valid), linewidth=1.6,
                markersize=3.5, **style)
        if show_confidence:
            ax.fill_between(group.n, group.ci_low.where(valid), group.ci_high.where(valid),
                            color=style["color"], alpha=0.13, linewidth=0)
        available = available or bool(valid.any())
    if not available:
        ax.text(0.5, 0.5, "Insufficient support", transform=ax.transAxes,
                ha="center", va="center", color="0.5", fontsize=8)
    ax.set_xscale(x_scale)
    if x_scale == "log":
        counts = panel.n.dropna()
        if len(counts):
            ticks = np.unique(np.rint(np.geomspace(counts.min(), counts.max(), 3)).astype(int))
            ax.set_xticks(ticks, labels=[str(n) for n in ticks])
        ax.xaxis.set_minor_formatter(NullFormatter())
    ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
    ax.tick_params(labelsize=8)
    ax.spines[["top", "right"]].set_visible(False)


def _match_limits(ax, pair, min_organoids, show_confidence):
    valid = pair[pair.n_organoids >= min_organoids]
    columns = ["mean", "ci_low", "ci_high"] if show_confidence else ["mean"]
    values = valid[columns].to_numpy().reshape(-1)
    values = values[np.isfinite(values)]
    if len(values):
        low, high = min(0., float(values.min())), max(0., float(values.max()))
        margin = max(high - low, 1e-8) * 0.08
        ax.set_ylim(low - margin, high + margin)


def _legend(fig):
    handles = [Line2D([], [], linewidth=1.6, markersize=4, **style)
               for style in CURVE_STYLES.values()]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.965),
               ncol=2, frameon=False)


def plot_size_overlay_grid(
    table, model, hop, *, min_organoids=5, show_confidence=True,
    x_scale="log", include_unmarked=True, match_model_limits=True,
    ylabel="Δ curvature",
):
    """One center-by-source grid for a model, with both analyses on each axis."""
    if x_scale not in ("log", "linear"):
        raise ValueError("x_scale must be 'log' or 'linear'.")
    if model not in set(table.model) or hop not in set(table.hop):
        raise ValueError("Requested model or hop is absent from the saved tables.")
    sources = sorted(table.source_marker_name.unique())
    centers = [marker for marker in sources if marker in set(table.center_marker)]
    if include_unmarked and "unmarked" in set(table.center_marker):
        centers.append("unmarked")
    fig, axes = plt.subplots(len(centers), len(sources), squeeze=False, sharex=True,
                             figsize=(3.0 * len(sources), 2.45 * len(centers)))
    for i, center in enumerate(centers):
        for j, source in enumerate(sources):
            ax = axes[i, j]
            pair = _pair_rows(table, center, source, hop)
            _draw_overlay(ax, pair[pair.model == model], min_organoids=min_organoids,
                          show_confidence=show_confidence, x_scale=x_scale)
            if match_model_limits:
                _match_limits(ax, pair, min_organoids, show_confidence)
            if i == 0:
                ax.set_title(f"Ablate {source}", fontsize=10)
            if j == 0:
                ax.set_ylabel(f"Center: {center}\n{ylabel}", fontsize=9)
            if i == len(centers) - 1:
                ax.set_xlabel("Cell count N", fontsize=9)
    fig.suptitle(f"{MODEL_LABELS.get(model, model)} — exact hop {hop}\n"
                 "Observed-size and fixed-neighborhood sweep effects", fontsize=15, y=0.998)
    _legend(fig)
    fig.tight_layout(rect=(0, 0, 1, 0.94), h_pad=1.5)
    return fig


def plot_geometric_normalization_comparison(table, pair, models, *, min_organoids=5):
    """Raw and dimensionless overlays side by side, with one row per model."""
    center, source, hop = pair
    fig, axes = plt.subplots(len(models), 2, squeeze=False, sharex=True, sharey='col',
                             figsize=(11, 3.5 * len(models)))
    for row, model in enumerate(models):
        for col, (metric, label) in enumerate([
            ('delta_mu', 'Δ curvature (original units)'),
            ('delta_relative', 'ΔK / Kref(N) (dimensionless)'),
        ]):
            panel = _pair_rows(table, center, source, hop)
            panel = panel[(panel.model == model) & (panel.metric == metric)]
            _draw_overlay(axes[row, col], panel, min_organoids=min_organoids,
                          show_confidence=True, x_scale='log')
            axes[row, col].set_ylabel(label)
            axes[row, col].set_title(MODEL_LABELS.get(model, model), fontsize=11)
            if row == len(models) - 1:
                axes[row, col].set_xlabel('Cell count N')
    fig.suptitle(f'Ablate {source} near {center}, exact hop {hop}\n'
                 'Raw versus geometric normalization', y=0.995)
    _legend(fig)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    return fig


def plot_size_overlay_examples(
    table, pairs, *, models=("gin_head_size", "gin_film_size"),
    min_organoids=5, show_confidence=True, x_scale="log",
):
    """Selected (center, source, hop) panels, with one row per model."""
    if not pairs:
        raise ValueError("Provide at least one (center, source, hop) tuple.")
    if x_scale not in ("log", "linear"):
        raise ValueError("x_scale must be 'log' or 'linear'.")
    for center, source, hop in pairs:
        if _pair_rows(table, center, source, hop).empty:
            raise ValueError(f"No saved effects for {(center, source, hop)!r}.")
    fig, axes = plt.subplots(len(models), len(pairs), squeeze=False, sharex=True,
                             sharey="col", figsize=(4.1 * len(pairs), 3.2 * len(models)))
    for i, model in enumerate(models):
        for j, (center, source, hop) in enumerate(pairs):
            ax = axes[i, j]
            panel = _pair_rows(table, center, source, hop)
            _draw_overlay(ax, panel[panel.model == model], min_organoids=min_organoids,
                          show_confidence=show_confidence, x_scale=x_scale)
            if i == 0:
                ax.set_title(f"Ablate {source} near {center}\nExact hop {hop}", fontsize=11)
            if j == 0:
                ax.set_ylabel(f"{MODEL_LABELS.get(model, model)}\nΔ curvature", fontsize=10)
            if i == len(models) - 1:
                ax.set_xlabel("Cell count N")
    fig.suptitle("Observed-size versus sweep-size ablations", fontsize=15, y=0.995)
    _legend(fig)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    return fig
