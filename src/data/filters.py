
import copy
from collections import defaultdict

import numpy as np
import torch

from src.data.metadata import get_graph_metadata_value


# -----------------------------------------------------------------------------
# Generic helpers
# -----------------------------------------------------------------------------


def _copy_graph(g, inplace=False):
    return g if inplace else copy.copy(g)


def split_graphs_by_mask(graphs, kept_mask, *, inplace=False):
    """Split graphs into ``(kept, rejected)`` according to a boolean mask."""
    kept_mask = np.asarray(kept_mask, dtype=bool)
    if kept_mask.shape[0] != len(graphs):
        raise ValueError("kept_mask must have length len(graphs)")

    kept = []
    rejected = []
    for g, keep in zip(graphs, kept_mask):
        if keep:
            kept.append(_copy_graph(g, inplace=inplace))
        else:
            rejected.append(_copy_graph(g, inplace=inplace))
    return kept, rejected


def _finish_filter(graphs, kept_mask, *, inplace=False, return_rejected=False):
    kept, rejected = split_graphs_by_mask(graphs, kept_mask, inplace=inplace)
    if return_rejected:
        return kept, rejected
    return kept


def _print_filter_summary(graphs, kept_mask, *, meta_lookup=None, label="Filter summary"):
    """Print kept/total counts per ``(dataset, timepoint)`` metadata group."""
    kept_mask = np.asarray(kept_mask, dtype=bool)
    if kept_mask.shape[0] != len(graphs):
        raise ValueError("kept_mask must have length len(graphs)")

    counts = defaultdict(lambda: {"kept": 0, "total": 0})

    for keep, g in zip(kept_mask, graphs):
        dataset = get_graph_metadata_value(
            g, "dataset", meta_lookup=meta_lookup, default="MISSING", strict=False
        )
        timepoint = get_graph_metadata_value(
            g, "timepoint", meta_lookup=meta_lookup, default="MISSING", strict=False
        )
        key = (dataset, timepoint)

        counts[key]["total"] += 1
        if keep:
            counts[key]["kept"] += 1

    n_kept = int(kept_mask.sum())
    n_total = len(graphs)

    print(f"{label}: kept {n_kept} / {n_total} graphs\n")
    print(f"{'dataset':<16} {'timepoint':<16} {'kept':>6} {'total':>6} {'frac':>8}")
    print("-" * 58)

    for dataset, timepoint in sorted(counts.keys(), key=lambda x: (str(x[0]), str(x[1]))):
        kept = counts[(dataset, timepoint)]["kept"]
        total = counts[(dataset, timepoint)]["total"]
        frac = kept / total if total > 0 else np.nan
        print(f"{str(dataset):<16} {str(timepoint):<16} {kept:>6} {total:>6} {frac:>8.3f}")


# -----------------------------------------------------------------------------
# Metadata filters
# -----------------------------------------------------------------------------


def filter_graphs_by_metadata(
    graphs,
    key,
    keep_values=None,
    drop_values=None,
    missing="keep",
    inplace=False,
    *,
    meta_lookup=None,
    print_summary=False,
    return_rejected=False,
):
    """Filter graphs by a categorical metadata field.

    Set ``return_rejected=True`` to return ``(kept, rejected)``.
    """
    if keep_values is not None and drop_values is not None:
        raise ValueError("Specify only one of keep_values or drop_values")
    if missing not in ("keep", "drop"):
        raise ValueError("missing must be 'keep' or 'drop'")

    keep_values = set(keep_values) if keep_values is not None else None
    drop_values = set(drop_values) if drop_values is not None else None
    kept_mask = []

    for g in graphs:
        val = get_graph_metadata_value(
            g, key, meta_lookup=meta_lookup, default=None, strict=False
        )

        if val is None:
            keep = missing == "keep"
        else:
            keep = True
            if keep_values is not None:
                keep = val in keep_values
            if drop_values is not None:
                keep = val not in drop_values

        kept_mask.append(keep)

    if print_summary:
        _print_filter_summary(
            graphs,
            kept_mask,
            meta_lookup=meta_lookup,
            label=f"filter_graphs_by_metadata(key={key!r})",
        )

    return _finish_filter(
        graphs, kept_mask, inplace=inplace, return_rejected=return_rejected
    )


def filter_graphs_by_numeric_metadata(
    graphs,
    key,
    meta_lookup=None,
    *,
    min_value=None,
    max_value=None,
    allow_missing=False,
    inplace=False,
    print_summary=False,
    return_rejected=False,
):
    """Filter graphs by a numeric metadata field such as ``complexity``.

    Set ``return_rejected=True`` to return ``(kept, rejected)``.
    """
    kept_mask = []

    for g in graphs:
        val = get_graph_metadata_value(
            g, key, meta_lookup=meta_lookup, default=None, strict=False
        )
        keep = True

        if val is None:
            keep = allow_missing
        else:
            val = float(val)
            if not np.isfinite(val):
                keep = allow_missing
            else:
                if min_value is not None:
                    keep = keep and val >= float(min_value)
                if max_value is not None:
                    keep = keep and val <= float(max_value)

        kept_mask.append(keep)

    if print_summary:
        _print_filter_summary(
            graphs,
            kept_mask,
            meta_lookup=meta_lookup,
            label=(
                f"filter_graphs_by_numeric_metadata("
                f"key={key!r}, min_value={min_value}, max_value={max_value})"
            ),
        )

    return _finish_filter(
        graphs, kept_mask, inplace=inplace, return_rejected=return_rejected
    )


def filter_graphs_by_sphericity(
    graphs,
    meta_lookup=None,
    *,
    area_key="total_surface_area",
    volume_key="total_volume",
    max_sphericity=0.95,
    allow_missing=False,
    inplace=False,
    print_summary=False,
    return_scores=False,
    return_rejected=False,
):
    """Filter out graphs that are too close to a sphere.

    Sphericity is measured by the isoperimetric quotient

        Q = 36 * pi * V^2 / A^3

    where Q = 1 for a perfect sphere and Q < 1 otherwise. This filter keeps
    graphs with ``Q < max_sphericity``.

    Returns
    -------
    kept : list
        Default return value.
    kept, rejected : tuple[list, list]
        Returned when ``return_rejected=True``.
    kept, scores : tuple[list, ndarray]
        Returned when ``return_scores=True``.
    kept, rejected, scores : tuple[list, list, ndarray]
        Returned when both flags are True.
    """
    kept_mask = []
    scores = []

    for g in graphs:
        area = get_graph_metadata_value(
            g, area_key, meta_lookup=meta_lookup, default=None, strict=False
        )
        volume = get_graph_metadata_value(
            g, volume_key, meta_lookup=meta_lookup, default=None, strict=False
        )

        keep = True
        q = np.nan

        if area is None or volume is None:
            keep = allow_missing
        else:
            area = float(area)
            volume = float(volume)

            if (
                not np.isfinite(area)
                or not np.isfinite(volume)
                or area <= 0.0
                or volume <= 0.0
            ):
                keep = allow_missing
            else:
                q = 36.0 * np.pi * volume ** 2 / area ** 3
                keep = q < float(max_sphericity)

        scores.append(q)
        kept_mask.append(keep)

    if print_summary:
        _print_filter_summary(
            graphs,
            kept_mask,
            meta_lookup=meta_lookup,
            label=f"filter_graphs_by_sphericity(max_sphericity={max_sphericity})",
        )

    scores = np.asarray(scores, dtype=float)
    kept, rejected = split_graphs_by_mask(graphs, kept_mask, inplace=inplace)

    if return_rejected and return_scores:
        return kept, rejected, scores
    if return_rejected:
        return kept, rejected
    if return_scores:
        return kept, scores
    return kept


# -----------------------------------------------------------------------------
# Marker-composition scoring and filtering
# -----------------------------------------------------------------------------


def graph_marker_fractions(graph):
    """Return marker-positive fractions for one graph as a 1-D numpy array."""
    if not hasattr(graph, "x") or graph.x is None:
        raise ValueError("graph has no node feature matrix g.x")

    x = graph.x.detach().cpu().numpy() if torch.is_tensor(graph.x) else np.asarray(graph.x)
    if x.ndim != 2:
        raise ValueError(f"g.x must be 2-D, got shape {x.shape}")
    if x.shape[0] == 0:
        return np.full(x.shape[1], np.nan, dtype=float)

    return np.asarray(x, dtype=float).mean(axis=0)


def estimate_marker_prevalence(graphs, *, weighted_by_cells=True, eps=1e-12):
    """Estimate global marker prevalence across a graph collection.

    ``weighted_by_cells=True`` computes the cell-level positive fraction over all
    graphs. ``False`` gives each organoid equal weight by averaging per-organoid
    fractions.
    """
    if len(graphs) == 0:
        raise ValueError("cannot estimate marker prevalence from an empty graph list")

    fractions = []
    weights = []

    for g in graphs:
        f = graph_marker_fractions(g)
        fractions.append(f)
        weights.append(int(g.x.shape[0]))

    fractions = np.vstack(fractions)

    if weighted_by_cells:
        weights = np.asarray(weights, dtype=float)
        prevalence = np.average(fractions, axis=0, weights=weights)
    else:
        prevalence = np.nanmean(fractions, axis=0)

    return np.clip(np.asarray(prevalence, dtype=float), eps, 1.0)


def marker_rarity_weights(
    prevalence,
    *,
    power=0.5,
    max_weight=20.0,
    normalize=True,
    eps=1e-12,
):
    """Convert marker prevalence into inverse-prevalence weights.

    ``power`` controls how aggressively rare markers are upweighted:

    - 0.0: all markers equal
    - 0.5: square-root inverse prevalence, usually a good default
    - 1.0: full inverse prevalence, more aggressive

    ``max_weight`` prevents extremely rare markers from dominating purely due to
    noise or annotation artifacts. With ``normalize=True``, the average weight is
    one, so score magnitudes remain easier to compare.
    """
    prevalence = np.asarray(prevalence, dtype=float)
    weights = 1.0 / np.power(np.clip(prevalence, eps, 1.0), float(power))

    if max_weight is not None:
        weights = np.minimum(weights, float(max_weight))
    if normalize:
        mean = np.nanmean(weights)
        if np.isfinite(mean) and mean > 0:
            weights = weights / mean

    return weights


def weighted_marker_presence_score(
    graph,
    *,
    marker_weights=None,
    prevalence=None,
    rarity_power=0.5,
    max_rarity_weight=20.0,
    half_saturation=0.01,
    eps=1e-12,
):
    """Score marker diversity while respecting marker-population imbalance.

    This is often more useful than plain Shannon entropy for your setting. For
    each marker, it computes a saturating presence term

        presence = fraction_positive / (fraction_positive + half_saturation)

    and then averages those presences with marker weights. Consequences:

    - a rare marker with only a small positive population can contribute strongly
    - an abundant marker does not keep increasing linearly once it is common
    - marker-specific biological importance can be supplied through
      ``marker_weights``

    If ``marker_weights`` is omitted but ``prevalence`` is supplied, inverse-
    prevalence rarity weights are used.
    """
    fractions = graph_marker_fractions(graph)

    if marker_weights is None:
        if prevalence is None:
            weights = np.ones_like(fractions, dtype=float)
        else:
            weights = marker_rarity_weights(
                prevalence,
                power=rarity_power,
                max_weight=max_rarity_weight,
                normalize=True,
                eps=eps,
            )
    else:
        weights = np.asarray(marker_weights, dtype=float)

    if weights.shape[0] != fractions.shape[0]:
        raise ValueError(
            f"marker_weights length {weights.shape[0]} != num markers {fractions.shape[0]}"
        )

    presence = fractions / (fractions + float(half_saturation) + eps)
    score = np.sum(weights * presence) / (np.sum(weights) + eps)
    return float(score)


def weighted_marker_entropy_score(
    graph,
    *,
    marker_weights=None,
    prevalence=None,
    rarity_power=0.5,
    max_rarity_weight=20.0,
    normalize=True,
    eps=1e-12,
):
    """Weighted Shannon entropy of marker composition.

    This is included for completeness, but for strongly imbalanced markers the
    saturating ``weighted_marker_presence_score`` is usually more intuitive.
    """
    fractions = graph_marker_fractions(graph)

    if marker_weights is None:
        if prevalence is None:
            weights = np.ones_like(fractions, dtype=float)
        else:
            weights = marker_rarity_weights(
                prevalence,
                power=rarity_power,
                max_weight=max_rarity_weight,
                normalize=True,
                eps=eps,
            )
    else:
        weights = np.asarray(marker_weights, dtype=float)

    weighted = weights * fractions
    p = weighted / (weighted.sum() + eps)
    h = -np.sum(p * np.log(p + eps))

    if normalize:
        active = int(np.sum(weighted > 0))
        if active > 1:
            h = h / np.log(active)
        else:
            h = 0.0

    return float(h)


def marker_count_score(graph, *, min_fraction=0.01, marker_weights=None):
    """Count represented markers, optionally weighted by marker importance."""
    fractions = graph_marker_fractions(graph)
    present = fractions >= float(min_fraction)

    if marker_weights is None:
        return float(np.sum(present))

    weights = np.asarray(marker_weights, dtype=float)
    if weights.shape[0] != fractions.shape[0]:
        raise ValueError(
            f"marker_weights length {weights.shape[0]} != num markers {fractions.shape[0]}"
        )
    return float(np.sum(weights[present]))


def marker_diversity_scores(
    graphs,
    *,
    method="weighted_presence",
    marker_weights=None,
    prevalence=None,
    reference_graphs=None,
    rarity_power=0.5,
    max_rarity_weight=20.0,
    half_saturation=0.01,
    min_fraction=0.01,
):
    """Compute one marker-diversity / informativeness score per graph.

    Parameters
    ----------
    method : {"weighted_presence", "weighted_entropy", "count"}
        ``weighted_presence`` is the recommended default for imbalanced binary
        markers.
    reference_graphs : list or None
        If supplied and ``prevalence`` is omitted, global marker prevalence is
        estimated from these graphs. This is useful for computing rarity weights
        from the full training corpus rather than only from spherical graphs.
    """
    if prevalence is None and reference_graphs is not None:
        prevalence = estimate_marker_prevalence(reference_graphs)

    scores = []
    for g in graphs:
        if method == "weighted_presence":
            score = weighted_marker_presence_score(
                g,
                marker_weights=marker_weights,
                prevalence=prevalence,
                rarity_power=rarity_power,
                max_rarity_weight=max_rarity_weight,
                half_saturation=half_saturation,
            )
        elif method == "weighted_entropy":
            score = weighted_marker_entropy_score(
                g,
                marker_weights=marker_weights,
                prevalence=prevalence,
                rarity_power=rarity_power,
                max_rarity_weight=max_rarity_weight,
            )
        elif method == "count":
            score = marker_count_score(
                g,
                min_fraction=min_fraction,
                marker_weights=marker_weights,
            )
        else:
            raise ValueError(
                "method must be one of {'weighted_presence', 'weighted_entropy', 'count'}"
            )
        scores.append(score)

    return np.asarray(scores, dtype=float)


def filter_graphs_by_marker_diversity(
    graphs,
    *,
    min_score=None,
    top_k=None,
    method="weighted_presence",
    marker_weights=None,
    prevalence=None,
    reference_graphs=None,
    rarity_power=0.5,
    max_rarity_weight=20.0,
    half_saturation=0.01,
    min_fraction=0.01,
    inplace=False,
    return_scores=False,
    return_rejected=False,
    print_summary=False,
):
    """Filter or sample graphs by marker-composition diversity.

    Use either ``min_score`` or ``top_k``. Set ``reference_graphs`` to the larger
    corpus when you want rarity weights to reflect the whole dataset.
    """
    if min_score is not None and top_k is not None:
        raise ValueError("Specify only one of min_score or top_k")

    scores = marker_diversity_scores(
        graphs,
        method=method,
        marker_weights=marker_weights,
        prevalence=prevalence,
        reference_graphs=reference_graphs,
        rarity_power=rarity_power,
        max_rarity_weight=max_rarity_weight,
        half_saturation=half_saturation,
        min_fraction=min_fraction,
    )

    if top_k is not None:
        top_k = int(top_k)
        if top_k < 0:
            raise ValueError("top_k must be non-negative")
        order = np.argsort(scores)[::-1]
        keep_indices = set(order[:top_k])
        kept_mask = np.asarray([i in keep_indices for i in range(len(graphs))], dtype=bool)

    elif min_score is not None:
        kept_mask = scores >= float(min_score)

    else:
        kept_mask = np.ones(len(graphs), dtype=bool)

    kept, rejected = split_graphs_by_mask(graphs, kept_mask, inplace=inplace)

    # --- summary (same style as other filters) ---
    if print_summary:
        _print_filter_summary(
            graphs,
            kept_mask,
            label=(
                f"filter_graphs_by_marker_diversity("
                f"min_score={min_score}, top_k={top_k})"
            ),
        )

    if return_rejected and return_scores:
        return kept, rejected, scores
    if return_rejected:
        return kept, rejected
    if return_scores:
        return kept, scores
    return kept