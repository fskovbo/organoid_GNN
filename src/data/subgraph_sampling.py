from collections import deque

import numpy as np
import torch

from src.graph.neighborhood import compute_hop_rings


def _subgraph_marker_coverage(subgraphs, k_hops, threshold=0.5):
    """
    Precompute center-marker and per-hop neighborhood-marker presence.

    Returns
    -------
    center_pos : (S, M) bool
        center_pos[s, y] = True if center marker y is positive in subgraph s.
    ring_pos : (S, H, M) bool
        ring_pos[s, h, m] = True if marker m appears in hop h+1 ring of subgraph s.
        (h index 0 corresponds to hop=1)
    """
    if len(subgraphs) == 0:
        raise ValueError("subgraphs is empty")

    M = int(subgraphs[0].x.shape[1])
    H = int(k_hops)
    S = len(subgraphs)

    center_pos = np.zeros((S, M), dtype=bool)
    ring_pos = np.zeros((S, H, M), dtype=bool)

    for si, g in enumerate(subgraphs):
        c = int(g.center_idx)
        x = g.x.detach().cpu().numpy()
        xb = x > threshold

        center_pos[si] = xb[c]

        rings = compute_hop_rings(g.edge_index, c, k_hops)

        for hi in range(H):
            nodes = rings[hi + 1]   # hop 1..k_hops
            if len(nodes) == 0:
                continue
            ring_pos[si, hi] = xb[nodes].any(axis=0)

    return center_pos, ring_pos


def _feature_lists_for_subgraphs(center_pos, ring_pos):
    """
    Convert boolean coverage arrays into compact per-subgraph feature lists.

    Returns
    -------
    center_feats : list[list[int]]
        center_feats[s] = list of center markers present in subgraph s
    pair_feats : list[list[tuple[int,int,int]]]
        pair_feats[s] = list of (hop_index, center_marker, neigh_marker) covered by subgraph s
    """
    S, M = center_pos.shape
    _, H, _ = ring_pos.shape

    center_feats = []
    pair_feats = []

    for s in range(S):
        ys = np.flatnonzero(center_pos[s]).tolist()
        center_feats.append(ys)

        pf = []
        if len(ys) > 0:
            for hi in range(H):
                ms = np.flatnonzero(ring_pos[s, hi]).tolist()
                for y in ys:
                    for m in ms:
                        pf.append((hi, y, m))
        pair_feats.append(pf)

    return center_feats, pair_feats


def _global_feature_frequencies(center_feats, pair_feats, M, H):
    center_freq = np.zeros(M, dtype=np.int64)
    pair_freq = np.zeros((H, M, M), dtype=np.int64)

    for ys in center_feats:
        for y in ys:
            center_freq[y] += 1

    for pf in pair_feats:
        for hi, y, m in pf:
            pair_freq[hi, y, m] += 1

    return center_freq, pair_freq


def center_specs_from_subgraphs(subgraphs):
    """
    Extract ordered list of (graph_idx, orig_center) from sampled ego-subgraphs.
    """
    specs = []
    for s in subgraphs:
        if not hasattr(s, "graph_idx"):
            raise ValueError("Subgraph is missing .graph_idx. Rebuild subgraphs with updated builder.")
        if not hasattr(s, "orig_center"):
            raise ValueError("Subgraph is missing .orig_center.")
        specs.append((int(s.graph_idx), int(s.orig_center)))
    return specs


def centers_per_graph_from_center_specs(center_specs, n_graphs):
    """
    Convert ordered center specs into per-graph center lists for convenience.
    """
    out = [[] for _ in range(n_graphs)]
    for gi, c in center_specs:
        out[int(gi)].append(int(c))
    return out


def _subgraph_group_ids(subgraphs):
    """Return one stable organoid/group identifier per ego-subgraph."""
    group_ids = []
    for i, subgraph in enumerate(subgraphs):
        if hasattr(subgraph, "graph_idx"):
            group_ids.append(("graph_idx", int(subgraph.graph_idx)))
        elif hasattr(subgraph, "organoid_str"):
            group_ids.append(("organoid_str", str(subgraph.organoid_str)))
        else:
            raise ValueError(
                f"Subgraph {i} has neither .graph_idx nor .organoid_str; "
                "organoid-weighted sampling requires one of these attributes."
            )
    return group_ids


def _balanced_group_sample(group_ids, sample_size, rng):
    """Sample nearly equal numbers from each represented group."""
    groups = {}
    for idx, group_id in enumerate(group_ids):
        groups.setdefault(group_id, []).append(idx)

    group_order = list(groups)
    rng.shuffle(group_order)
    for group_id in group_order:
        rng.shuffle(groups[group_id])

    selected = []
    offsets = {group_id: 0 for group_id in group_order}
    active = list(group_order)
    while active and len(selected) < sample_size:
        rng.shuffle(active)
        next_active = []
        for group_id in active:
            offset = offsets[group_id]
            members = groups[group_id]
            if offset < len(members) and len(selected) < sample_size:
                selected.append(int(members[offset]))
                offset += 1
                offsets[group_id] = offset
            if offset < len(members):
                next_active.append(group_id)
        active = next_active

    return np.asarray(selected, dtype=np.int64)


def _population_weights(selected_indices, group_ids, weighting):
    """Return normalized design weights aligned with selected_indices."""
    n_selected = len(selected_indices)
    if n_selected == 0:
        return np.zeros(0, dtype=np.float64)

    if weighting == "cell":
        return np.full(n_selected, 1.0 / n_selected, dtype=np.float64)

    selected_groups = [group_ids[int(idx)] for idx in selected_indices]
    counts = {}
    for group_id in selected_groups:
        counts[group_id] = counts.get(group_id, 0) + 1

    n_groups = len(counts)
    return np.asarray(
        [1.0 / (n_groups * counts[group_id]) for group_id in selected_groups],
        dtype=np.float64,
    )


def _resolve_marker_index(marker, marker_names):
    marker_names = list(marker_names)
    if isinstance(marker, str):
        if marker in marker_names:
            return marker_names.index(marker)
        lower = {str(name).lower(): i for i, name in enumerate(marker_names)}
        key = marker.lower()
        if key not in lower:
            raise ValueError(f"Marker {marker!r} not found in marker_names.")
        return lower[key]

    marker_idx = int(marker)
    if not (0 <= marker_idx < len(marker_names)):
        raise IndexError(
            f"Marker index {marker_idx} out of range for {len(marker_names)} markers."
        )
    return marker_idx


def sample_subgraphs_population(
    subgraphs,
    max_subgraphs,
    *,
    weighting="cell",
    seed=0,
    marker_names=None,
    k_hops=None,
    min_marker_count_per_hop=None,
    include_center=False,
    threshold=0.5,
    return_info=True,
):
    """Draw a representative sample and define marker-specific conditional samples.

    The representative draw contains at most ``max_subgraphs`` center cells.
    With ``weighting="cell"``, centers are sampled uniformly from the complete
    pool. With ``weighting="organoid"``, the budget is distributed as evenly as
    possible across organoids and the returned design weights give each
    represented organoid equal total weight.

    If ``min_marker_count_per_hop`` is provided, every marker receives a
    separate distance-specific conditional-effect sample. Each such sample starts
    with marker-positive centers from the representative draw and is topped up
    from the full pool until it reaches the requested minimum, as far as
    possible. Top-ups are not added to the returned representative sample.
    Their source indices and weights are reported separately in ``info``.

    Parameters
    ----------
    subgraphs : list[Data]
        Ego-subgraphs. Organoid weighting requires ``graph_idx`` or
        ``organoid_str`` on every subgraph.
    max_subgraphs : int or None
        Size of the representative population draw. ``None`` keeps all centers.
    weighting : {"cell", "organoid"}
        Population estimand. Cell weighting gives every center equal weight;
        organoid weighting gives every represented organoid equal total weight.
    min_marker_count_per_hop : int or None
        Minimum marker-positive centers requested for every marker at every
        hop. Set to None to disable marker-specific conditional samples.
    include_center : bool
        If True, also create marker-specific samples for distance zero, where
        marker presence is evaluated on the center cell itself.

    Returns
    -------
    selected_subgraphs : list[Data]
        The representative population draw only.
    info : dict, optional
        Population weights plus marker-specific sample indices, weights,
        availability, and top-up diagnostics. Marker sample source indices
        refer to the input ``subgraphs`` list and are intentionally separate
        from ``selected_subgraphs``.
    """
    if len(subgraphs) == 0:
        raise ValueError("subgraphs is empty")
    if weighting not in {"cell", "organoid"}:
        raise ValueError("weighting must be either 'cell' or 'organoid'")
    if max_subgraphs is not None and max_subgraphs <= 0:
        raise ValueError("max_subgraphs must be positive or None")

    rng = np.random.default_rng(seed)
    n_available = len(subgraphs)
    population_size = (
        n_available if max_subgraphs is None else min(int(max_subgraphs), n_available)
    )

    group_ids = None
    has_group_metadata = all(
        hasattr(subgraph, "graph_idx") or hasattr(subgraph, "organoid_str")
        for subgraph in subgraphs
    )
    if weighting == "organoid" or has_group_metadata:
        group_ids = _subgraph_group_ids(subgraphs)
    if weighting == "cell":
        population_source = rng.choice(
            n_available,
            size=population_size,
            replace=False,
        ).astype(np.int64)
    else:
        population_source = _balanced_group_sample(
            group_ids,
            population_size,
            rng,
        )

    population_weights = _population_weights(
        population_source,
        group_ids,
        weighting,
    )

    marker_source_by_hop = {}
    marker_topup_source_by_hop = {}
    marker_weights_by_hop = {}
    marker_available_by_hop = {}
    if min_marker_count_per_hop is not None:
        if marker_names is None or k_hops is None:
            raise ValueError(
                "marker_names and k_hops are required for marker-specific sampling."
            )
        if int(min_marker_count_per_hop) <= 0:
            raise ValueError("min_marker_count_per_hop must be positive")

        marker_names = list(marker_names)
        center_pos, ring_pos = _subgraph_marker_coverage(
            subgraphs,
            int(k_hops),
            threshold=threshold,
        )
        population_set = set(population_source.tolist())
        all_indices = np.arange(n_available, dtype=np.int64)

        full_group_counts = {}
        if weighting == "organoid":
            for group_id in group_ids:
                full_group_counts[group_id] = full_group_counts.get(group_id, 0) + 1

        for marker_idx, marker_name in enumerate(marker_names):
            marker_source_by_hop[marker_name] = {}
            marker_topup_source_by_hop[marker_name] = {}
            marker_weights_by_hop[marker_name] = {}
            marker_available_by_hop[marker_name] = {}

            hops = range(0 if include_center else 1, int(k_hops) + 1)
            for hop in hops:
                eligible_mask = (
                    center_pos[:, marker_idx]
                    if hop == 0
                    else ring_pos[:, hop - 1, marker_idx]
                )
                eligible = all_indices[eligible_mask]
                representative = population_source[eligible_mask[population_source]]
                marker_available_by_hop[marker_name][hop] = int(len(eligible))

                deficit = max(
                    0,
                    int(min_marker_count_per_hop) - len(representative),
                )
                candidates = np.asarray(
                    [idx for idx in eligible if int(idx) not in population_set],
                    dtype=np.int64,
                )
                n_topup = min(deficit, len(candidates))
                if n_topup:
                    if weighting == "cell":
                        topup = rng.choice(
                            candidates,
                            size=n_topup,
                            replace=False,
                        ).astype(np.int64)
                    else:
                        candidate_groups = [group_ids[int(idx)] for idx in candidates]
                        topup_local = _balanced_group_sample(
                            candidate_groups,
                            n_topup,
                            rng,
                        )
                        topup = candidates[topup_local]
                else:
                    topup = np.zeros(0, dtype=np.int64)

                marker_sample = np.concatenate([representative, topup]).astype(
                    np.int64,
                    copy=False,
                )
                if len(marker_sample) == 0:
                    marker_weights = np.zeros(0, dtype=np.float64)
                elif weighting == "cell":
                    marker_weights = np.full(
                        len(marker_sample),
                        1.0 / len(marker_sample),
                        dtype=np.float64,
                    )
                else:
                    eligible_group_counts = {}
                    sampled_group_counts = {}
                    for idx in eligible:
                        group_id = group_ids[int(idx)]
                        eligible_group_counts[group_id] = (
                            eligible_group_counts.get(group_id, 0) + 1
                        )
                    for idx in marker_sample:
                        group_id = group_ids[int(idx)]
                        sampled_group_counts[group_id] = (
                            sampled_group_counts.get(group_id, 0) + 1
                        )

                    marker_weights = np.asarray(
                        [
                            eligible_group_counts[group_ids[int(idx)]]
                            / full_group_counts[group_ids[int(idx)]]
                            / sampled_group_counts[group_ids[int(idx)]]
                            for idx in marker_sample
                        ],
                        dtype=np.float64,
                    )
                    marker_weights /= marker_weights.sum()

                marker_source_by_hop[marker_name][hop] = marker_sample
                marker_topup_source_by_hop[marker_name][hop] = topup
                marker_weights_by_hop[marker_name][hop] = marker_weights

    selected_subgraphs = [subgraphs[int(idx)] for idx in population_source]

    if not return_info:
        return selected_subgraphs

    selected_group_counts = {}
    if group_ids is not None:
        for idx in population_source:
            group_id = group_ids[int(idx)]
            selected_group_counts[str(group_id[1])] = (
                selected_group_counts.get(str(group_id[1]), 0) + 1
            )

    info = {
        "weighting": weighting,
        "n_available": int(n_available),
        "n_population": int(len(population_source)),
        "selected_indices": np.arange(len(population_source), dtype=np.int64),
        "selected_source_indices": population_source,
        "population_indices": np.arange(len(population_source), dtype=np.int64),
        "population_source_indices": population_source,
        "population_weights": population_weights,
        "sample_weights": population_weights.copy(),
        "population_group_counts": selected_group_counts,
        "min_marker_count_per_hop": min_marker_count_per_hop,
        "include_center": bool(include_center),
        "marker_sample_source_indices_by_hop": marker_source_by_hop,
        "marker_topup_source_indices_by_hop": marker_topup_source_by_hop,
        "marker_sample_weights_by_hop": marker_weights_by_hop,
        "marker_available_by_hop": marker_available_by_hop,
    }
    return selected_subgraphs, info


def _prepare_coverage_sampling(
    subgraphs,
    marker_names,
    k_hops,
    min_center_count,
    min_pair_count,
    threshold,
):
    """Precompute features, frequencies, targets, and empty coverage state."""
    n_markers = len(marker_names)
    n_hops = int(k_hops)
    center_pos, ring_pos = _subgraph_marker_coverage(
        subgraphs,
        k_hops,
        threshold=threshold,
    )
    center_feats, pair_feats = _feature_lists_for_subgraphs(center_pos, ring_pos)
    center_freq, pair_freq = _global_feature_frequencies(
        center_feats,
        pair_feats,
        n_markers,
        n_hops,
    )
    return {
        "n_markers": n_markers,
        "n_hops": n_hops,
        "n_subgraphs": len(subgraphs),
        "center_feats": center_feats,
        "pair_feats": pair_feats,
        "center_freq": center_freq,
        "pair_freq": pair_freq,
        "target_center": np.full(
            n_markers,
            int(min_center_count),
            dtype=np.int64,
        ),
        "target_pair": np.full(
            (n_hops, n_markers, n_markers),
            int(min_pair_count),
            dtype=np.int64,
        ),
        "covered_center": np.zeros(n_markers, dtype=np.int64),
        "covered_pair": np.zeros(
            (n_hops, n_markers, n_markers),
            dtype=np.int64,
        ),
    }


def _coverage_gain(idx, state, center_weight, pair_weight):
    """Return weighted unmet coverage gain for one candidate subgraph."""
    gain = 0.0
    for marker in state["center_feats"][idx]:
        if state["covered_center"][marker] < state["target_center"][marker]:
            gain += center_weight
    for hop_idx, center_marker, source_marker in state["pair_feats"][idx]:
        if (
            state["covered_pair"][hop_idx, center_marker, source_marker]
            < state["target_pair"][hop_idx, center_marker, source_marker]
        ):
            gain += pair_weight
    return gain


def _add_coverage_subgraph(idx, state):
    """Update center and pair coverage after selecting one subgraph."""
    for marker in state["center_feats"][idx]:
        state["covered_center"][marker] += 1
    for hop_idx, center_marker, source_marker in state["pair_feats"][idx]:
        state["covered_pair"][hop_idx, center_marker, source_marker] += 1


def _coverage_fill_weight(
    idx,
    state,
    fill_weight_center,
    fill_weight_pair,
):
    """Return rare-feature fill weight for one candidate subgraph."""
    weight = 1.0
    for marker in state["center_feats"][idx]:
        freq = state["center_freq"][marker]
        if freq > 0:
            weight += fill_weight_center / freq
    for hop_idx, center_marker, source_marker in state["pair_feats"][idx]:
        freq = state["pair_freq"][hop_idx, center_marker, source_marker]
        if freq > 0:
            weight += fill_weight_pair / freq
    return weight


def _coverage_diagnostics(selected, state, marker_names):
    """Build the common coverage diagnostics returned by coverage samplers."""
    unsatisfied_center = np.flatnonzero(
        state["covered_center"] < state["target_center"]
    )
    unsatisfied_pair = np.argwhere(
        state["covered_pair"] < state["target_pair"]
    )
    return {
        "selected_indices": np.asarray(selected, dtype=np.int64),
        "n_selected": int(len(selected)),
        "n_available": int(state["n_subgraphs"]),
        "center_target": state["target_center"],
        "center_covered": state["covered_center"],
        "pair_target": state["target_pair"],
        "pair_covered": state["covered_pair"],
        "center_freq_global": state["center_freq"],
        "pair_freq_global": state["pair_freq"],
        "unsatisfied_center_markers": [
            marker_names[int(i)] for i in unsatisfied_center
        ],
        "unsatisfied_pair_features": [
            {
                "hop": int(hop_idx + 1),
                "center_marker": marker_names[int(center_marker)],
                "neigh_marker": marker_names[int(source_marker)],
                "covered": int(
                    state["covered_pair"][
                        hop_idx,
                        center_marker,
                        source_marker,
                    ]
                ),
                "target": int(
                    state["target_pair"][
                        hop_idx,
                        center_marker,
                        source_marker,
                    ]
                ),
                "global_available": int(
                    state["pair_freq"][
                        hop_idx,
                        center_marker,
                        source_marker,
                    ]
                ),
            }
            for hop_idx, center_marker, source_marker in unsatisfied_pair
        ],
    }


def _single_source_cell_ids(subgraphs, k_hops, threshold=0.5):
    """Map each single-mode marker case to its selected physical source cell.

    Returns an object array with shape ``(subgraph, hop, marker)``. Entries are
    ``(group_kind, group_value, original_node_index)`` or ``None``.
    """
    if len(subgraphs) == 0:
        raise ValueError("subgraphs is empty")
    group_ids = _subgraph_group_ids(subgraphs)
    n_markers = int(subgraphs[0].x.shape[1])
    source_ids = np.empty(
        (len(subgraphs), int(k_hops), n_markers),
        dtype=object,
    )
    source_ids.fill(None)

    for subgraph_idx, subgraph in enumerate(subgraphs):
        if not hasattr(subgraph, "orig_nodes"):
            raise ValueError(
                "Reuse-aware sampling requires .orig_nodes on every subgraph."
            )
        center = int(subgraph.center_idx)
        rings = compute_hop_rings(
            subgraph.edge_index,
            center,
            int(k_hops),
        )
        x = subgraph.x.detach().cpu().numpy()
        original_nodes = subgraph.orig_nodes.detach().cpu().numpy()
        group_kind, group_value = group_ids[subgraph_idx]

        for hop_idx in range(int(k_hops)):
            nodes = np.asarray(rings[hop_idx + 1], dtype=np.int64)
            if len(nodes) == 0:
                continue
            for marker_idx in range(n_markers):
                positive = np.flatnonzero(x[nodes, marker_idx] > threshold)
                if len(positive) == 0:
                    continue
                local_node = int(nodes[int(positive[0])])
                source_ids[subgraph_idx, hop_idx, marker_idx] = (
                    group_kind,
                    group_value,
                    int(original_nodes[local_node]),
                )
    return source_ids


def _candidate_source_reuse_keys(idx, state, source_cell_ids, *, unmet_only):
    """Return unique `(hop, marker, physical cell)` keys for one candidate."""
    keys = set()
    for hop_idx, center_marker, source_marker in state["pair_feats"][idx]:
        if unmet_only and (
            state["covered_pair"][hop_idx, center_marker, source_marker]
            >= state["target_pair"][hop_idx, center_marker, source_marker]
        ):
            continue
        source_cell = source_cell_ids[idx, hop_idx, source_marker]
        if source_cell is not None:
            keys.add((hop_idx, source_marker, source_cell))
    return keys


def _update_source_reuse(idx, state, source_cell_ids, reuse_counts):
    """Record physical source-cell use for every single-mode case in a subgraph."""
    for key in _candidate_source_reuse_keys(
        idx,
        state,
        source_cell_ids,
        unmet_only=False,
    ):
        reuse_counts[key] = reuse_counts.get(key, 0) + 1


def _physical_cell_adjacency_from_subgraphs(subgraphs):
    """Reconstruct original-graph adjacency from ego-subgraph edge lists."""
    group_ids = _subgraph_group_ids(subgraphs)
    adjacency = {}
    for subgraph_idx, subgraph in enumerate(subgraphs):
        if not hasattr(subgraph, "orig_nodes"):
            raise ValueError(
                "Non-overlap sampling requires .orig_nodes on every subgraph."
            )
        group_id = group_ids[subgraph_idx]
        group_adjacency = adjacency.setdefault(group_id, {})
        original_nodes = subgraph.orig_nodes.detach().cpu().numpy().astype(
            int,
            copy=False,
        )
        edge_index = subgraph.edge_index.detach().cpu().numpy().astype(
            int,
            copy=False,
        )
        for src_local, dst_local in edge_index.T:
            src = int(original_nodes[src_local])
            dst = int(original_nodes[dst_local])
            group_adjacency.setdefault(src, set()).add(dst)
            group_adjacency.setdefault(dst, set()).add(src)
    return adjacency


def _physical_cell_adjacency_from_graphs(graphs):
    """Build physical-cell adjacency directly from full graph edge lists."""
    adjacency = {}
    for graph_idx, graph in enumerate(graphs):
        group_adjacency = adjacency.setdefault(("graph_idx", int(graph_idx)), {})
        edge_index = graph.edge_index.detach().cpu().numpy().astype(
            int,
            copy=False,
        )
        for src, dst in edge_index.T:
            src = int(src)
            dst = int(dst)
            group_adjacency.setdefault(src, set()).add(dst)
            group_adjacency.setdefault(dst, set()).add(src)
    return adjacency


def _physical_cells_within_hops(adjacency, source_cell, max_hops):
    """Return physical cells within graph distance <= max_hops from source_cell."""
    _, group_value, source_node = source_cell
    group_key = (source_cell[0], group_value)
    graph_adjacency = adjacency.get(group_key, {})
    source_node = int(source_node)
    max_hops = int(max_hops)

    visited = {source_node}
    queue = deque([(source_node, 0)])
    while queue:
        node, dist = queue.popleft()
        if dist >= max_hops:
            continue
        for neighbor in graph_adjacency.get(node, ()):
            neighbor = int(neighbor)
            if neighbor in visited:
                continue
            visited.add(neighbor)
            queue.append((neighbor, dist + 1))
    return {(source_cell[0], group_value, int(node)) for node in visited}


def _source_cell_from_case_row(
    row,
    *,
    graph_column,
    organoid_column,
    source_node_column,
):
    source_node = getattr(row, source_node_column)
    if source_node is None:
        return None
    try:
        if not np.isfinite(float(source_node)):
            return None
    except (TypeError, ValueError):
        return None

    graph_value = getattr(row, graph_column, None)
    if graph_value is not None:
        try:
            if np.isfinite(float(graph_value)):
                return ("graph_idx", int(graph_value), int(source_node))
        except (TypeError, ValueError):
            pass

    organoid_value = getattr(row, organoid_column, None)
    if organoid_value is not None:
        return ("organoid_str", str(organoid_value), int(source_node))
    return None


def flag_non_overlapping_source_cases(
    case_df,
    *,
    graphs=None,
    subgraphs=None,
    group_columns=("hop", "center_marker", "source_marker"),
    hop_column="hop",
    graph_column="graph_idx",
    organoid_column="organoid_str",
    source_node_column="orig_source_node",
    seed=0,
    shuffle=True,
):
    """Flag single-source perturbation cases passing a stratified non-overlap rule.

    The rule is enforced independently within each group in ``group_columns``.
    For the center-resolved ablation heatmaps, the intended grouping is
    ``(hop, center_marker, source_marker)``. Within one such group, a retained
    source cell at hop distance ``d`` blocks other source cells within graph
    distance ``d`` in the same original organoid graph.

    This operates at case level rather than subgraph level, so unrelated
    center-marker/source-marker/hop cases from the same ego-subgraph are not
    discarded.
    """
    if graphs is None and subgraphs is None:
        raise ValueError("Pass graphs or subgraphs to compute source-cell distances.")
    if source_node_column not in case_df.columns:
        raise ValueError(f"case_df is missing {source_node_column!r}.")
    for column in group_columns:
        if column not in case_df.columns:
            raise ValueError(f"case_df is missing group column {column!r}.")

    if graphs is not None:
        adjacency = _physical_cell_adjacency_from_graphs(graphs)
    else:
        adjacency = _physical_cell_adjacency_from_subgraphs(subgraphs)

    flagged = case_df.copy().reset_index(drop=True)
    passes = np.zeros(len(flagged), dtype=bool)
    rng = np.random.default_rng(seed)
    neighborhood_cache = {}

    grouped_positions = flagged.groupby(
        list(group_columns),
        dropna=False,
        sort=False,
    ).indices
    for positions in grouped_positions.values():
        positions = np.asarray(positions, dtype=int)
        if shuffle and len(positions) > 1:
            positions = rng.permutation(positions)
        blocked = set()
        for position in positions:
            row = flagged.iloc[int(position)]
            source_cell = _source_cell_from_case_row(
                row,
                graph_column=graph_column,
                organoid_column=organoid_column,
                source_node_column=source_node_column,
            )
            if source_cell is None:
                continue
            hop = int(getattr(row, hop_column))
            if source_cell in blocked:
                continue
            passes[int(position)] = True
            cache_key = (source_cell, hop)
            if cache_key not in neighborhood_cache:
                neighborhood_cache[cache_key] = _physical_cells_within_hops(
                    adjacency,
                    source_cell,
                    hop,
                )
            blocked.update(neighborhood_cache[cache_key])

    flagged["passes_non_overlap"] = passes
    flagged["non_overlap_excluded"] = ~passes
    return flagged


def _candidate_source_keys(idx, state, source_cell_ids, *, unmet_only=False):
    """Return unique ``(hop_idx, marker_idx, source_cell)`` keys for a candidate."""
    keys = set()
    for hop_idx, center_marker, source_marker in state["pair_feats"][idx]:
        if unmet_only and (
            state["covered_pair"][hop_idx, center_marker, source_marker]
            >= state["target_pair"][hop_idx, center_marker, source_marker]
        ):
            continue
        source_cell = source_cell_ids[idx, hop_idx, source_marker]
        if source_cell is not None:
            keys.add((hop_idx, source_marker, source_cell))
    return keys


def _is_nonoverlap_compatible(
    idx,
    state,
    source_cell_ids,
    blocked_cells,
    adjacency,
    neighborhood_cache,
):
    """Return True if all source cells introduced by candidate are unblocked."""
    keys = _candidate_source_keys(
        idx,
        state,
        source_cell_ids,
    )
    for hop_idx, _, source_cell in keys:
        if source_cell in blocked_cells.get(hop_idx, set()):
            return False

    # Also reject a subgraph if it would internally introduce two distinct
    # perturbation source cells that violate the same hop-specific exclusion.
    by_hop = {}
    for hop_idx, _, source_cell in keys:
        by_hop.setdefault(hop_idx, set()).add(source_cell)
    for hop_idx, source_cells in by_hop.items():
        source_cells = list(source_cells)
        radius = int(hop_idx + 1)
        for left_position, left_cell in enumerate(source_cells):
            cache_key = (left_cell, radius)
            if cache_key not in neighborhood_cache:
                neighborhood_cache[cache_key] = _physical_cells_within_hops(
                    adjacency,
                    left_cell,
                    radius,
                )
            blocked_by_left = neighborhood_cache[cache_key]
            for right_cell in source_cells[left_position + 1 :]:
                if right_cell != left_cell and right_cell in blocked_by_left:
                    return False
    return True


def _update_nonoverlap_blocks(
    idx,
    state,
    source_cell_ids,
    blocked_cells,
    adjacency,
    neighborhood_cache,
):
    """Block nearby physical source cells for each selected hop source."""
    for hop_idx, _, source_cell in _candidate_source_keys(
        idx,
        state,
        source_cell_ids,
    ):
        radius = int(hop_idx + 1)
        cache_key = (source_cell, radius)
        if cache_key not in neighborhood_cache:
            neighborhood_cache[cache_key] = _physical_cells_within_hops(
                adjacency,
                source_cell,
                radius,
            )
        blocked_cells.setdefault(hop_idx, set()).update(
            neighborhood_cache[cache_key]
        )


def _source_marker_priority(state):
    """Return source marker order from sparse to abundant."""
    source_frequency = state["pair_freq"].sum(axis=(0, 1))
    return np.argsort(source_frequency, kind="stable")


def sample_subgraphs_coverage(
    subgraphs,
    marker_names,
    k_hops,
    max_subgraphs,
    min_center_count=40,
    min_pair_count=12,
    center_weight=1.0,
    pair_weight=3.0,
    fill_weight_pair=2.0,
    fill_weight_center=1.0,
    seed=0,
    threshold=0.5,
    return_info=True,
):
    """
    Coverage-aware subsampling for perturbation analysis.

    Guarantees, as far as possible within the sample budget, minimum support for:
      1) each center marker
      2) each (hop, center marker, neighborhood marker) combination

    Then fills the remaining budget with weighted random sampling favoring rare features.

    Parameters
    ----------
    subgraphs : list[Data]
        Ego-subgraphs with .x, .edge_index, .center_idx
    marker_names : list[str]
        Marker names, length M
    k_hops : int
        Coverage is enforced across all hops 1..k_hops
    max_subgraphs : int or None
        Sample budget. If None or >= len(subgraphs), returns all subgraphs.
    min_center_count : int
        Target minimum number of subgraphs for each center marker.
    min_pair_count : int
        Target minimum number of subgraphs for each (hop, center marker, neighborhood marker).
    center_weight : float
        Greedy score weight for under-covered center markers.
    pair_weight : float
        Greedy score weight for under-covered pair features.
    fill_weight_pair : float
        Weight used in random fill for rare pair-features.
    fill_weight_center : float
        Weight used in random fill for rare center-features.
    seed : int
        RNG seed.
    threshold : float
        Marker positivity threshold.
    return_info : bool
        Whether to also return diagnostics.

    Returns
    -------
    subs_selected : list[Data]
    info : dict, optional
    """
    if max_subgraphs is None or max_subgraphs >= len(subgraphs):
        if return_info:
            return list(subgraphs), {
                "selected_indices": np.arange(len(subgraphs), dtype=np.int64),
                "note": "No subsampling applied",
            }
        return list(subgraphs)

    if max_subgraphs <= 0:
        raise ValueError("max_subgraphs must be positive or None")

    rng = np.random.default_rng(seed)
    state = _prepare_coverage_sampling(
        subgraphs,
        marker_names,
        k_hops,
        min_center_count,
        min_pair_count,
        threshold,
    )
    selected = []
    remaining = set(range(state["n_subgraphs"]))

    def add_subgraph(idx):
        selected.append(idx)
        remaining.remove(idx)
        _add_coverage_subgraph(idx, state)

    # ---------------------------
    # Phase A: greedy coverage
    # ---------------------------
    while len(selected) < max_subgraphs and len(remaining) > 0:
        remaining_list = list(remaining)
        scores = np.array(
            [
                _coverage_gain(i, state, center_weight, pair_weight)
                for i in remaining_list
            ],
            dtype=float,
        )

        best = scores.max()
        if best <= 0:
            break

        # random tie-break among best
        best_idx = np.flatnonzero(scores == best)
        chosen = remaining_list[int(rng.choice(best_idx))]
        add_subgraph(chosen)

    # ---------------------------
    # Phase B: weighted random fill
    # ---------------------------
    if len(selected) < max_subgraphs and len(remaining) > 0:
        remaining_list = list(remaining)
        weights = np.asarray(
            [
                _coverage_fill_weight(
                    idx,
                    state,
                    fill_weight_center,
                    fill_weight_pair,
                )
                for idx in remaining_list
            ],
            dtype=float,
        )

        n_fill = min(max_subgraphs - len(selected), len(remaining_list))

        if n_fill > 0:
            probs = weights / weights.sum()
            chosen_fill = rng.choice(remaining_list, size=n_fill, replace=False, p=probs)

            for idx in chosen_fill:
                add_subgraph(int(idx))

    selected = np.asarray(selected, dtype=np.int64)
    subs_selected = [subgraphs[int(i)] for i in selected]

    if not return_info:
        return subs_selected

    return subs_selected, _coverage_diagnostics(selected, state, marker_names)


def sample_subgraphs_coverage_reuse_aware(
    subgraphs,
    marker_names,
    k_hops,
    max_subgraphs,
    min_center_count=40,
    min_pair_count=12,
    center_weight=1.0,
    pair_weight=3.0,
    reuse_penalty=1.0,
    fill_weight_pair=2.0,
    fill_weight_center=1.0,
    fill_reuse_penalty=1.0,
    seed=0,
    threshold=0.5,
    return_info=True,
):
    """Coverage-aware sampling with a soft penalty for physical-cell reuse.

    This sampler targets the same center and
    ``(hop, center marker, source marker)`` coverage as
    :func:`sample_subgraphs_coverage`. During greedy coverage, candidate score
    is:

    ``coverage_gain - reuse_penalty * prior_source_cell_uses``.

    Reuse is tracked separately for each hop and source marker, using the exact
    physical cell that ``mode="single"`` perturbation would select. Coverage is
    never blocked by the penalty: if all useful candidates require reuse, the
    least-penalized candidate is still selected. The random fill phase also
    downweights candidates that reuse physical source cells.
    """
    if reuse_penalty < 0 or fill_reuse_penalty < 0:
        raise ValueError("reuse penalties must be non-negative")
    if not np.isclose(threshold, 0.5):
        raise ValueError(
            "Reuse-aware sampling currently requires threshold=0.5 to match "
            "the perturbation selector exactly."
        )
    if max_subgraphs is None or max_subgraphs >= len(subgraphs):
        if return_info:
            return list(subgraphs), {
                "selected_indices": np.arange(len(subgraphs), dtype=np.int64),
                "note": "No subsampling applied",
            }
        return list(subgraphs)
    if max_subgraphs <= 0:
        raise ValueError("max_subgraphs must be positive or None")

    rng = np.random.default_rng(seed)
    state = _prepare_coverage_sampling(
        subgraphs,
        marker_names,
        k_hops,
        min_center_count,
        min_pair_count,
        threshold,
    )
    source_cell_ids = _single_source_cell_ids(
        subgraphs,
        k_hops,
        threshold=threshold,
    )
    selected = []
    remaining = set(range(state["n_subgraphs"]))
    reuse_counts = {}

    def add_subgraph(idx):
        selected.append(idx)
        remaining.remove(idx)
        _add_coverage_subgraph(idx, state)
        _update_source_reuse(idx, state, source_cell_ids, reuse_counts)

    while len(selected) < max_subgraphs and remaining:
        remaining_list = list(remaining)
        coverage_gains = np.asarray(
            [
                _coverage_gain(idx, state, center_weight, pair_weight)
                for idx in remaining_list
            ],
            dtype=float,
        )
        if coverage_gains.max() <= 0:
            break

        scores = coverage_gains.copy()
        for position, idx in enumerate(remaining_list):
            if coverage_gains[position] <= 0:
                scores[position] = -np.inf
                continue
            reuse_keys = _candidate_source_reuse_keys(
                idx,
                state,
                source_cell_ids,
                unmet_only=True,
            )
            reuse_cost = sum(reuse_counts.get(key, 0) for key in reuse_keys)
            scores[position] -= reuse_penalty * reuse_cost

        best = np.nanmax(scores)
        best_positions = np.flatnonzero(scores == best)
        chosen = remaining_list[int(rng.choice(best_positions))]
        add_subgraph(chosen)

    while len(selected) < max_subgraphs and remaining:
        remaining_list = list(remaining)
        weights = np.empty(len(remaining_list), dtype=float)
        for position, idx in enumerate(remaining_list):
            weight = _coverage_fill_weight(
                idx,
                state,
                fill_weight_center,
                fill_weight_pair,
            )
            reuse_keys = _candidate_source_reuse_keys(
                idx,
                state,
                source_cell_ids,
                unmet_only=False,
            )
            reuse_cost = sum(reuse_counts.get(key, 0) for key in reuse_keys)
            weights[position] = weight / (
                1.0 + fill_reuse_penalty * reuse_cost
            )
        probabilities = weights / weights.sum()
        chosen = int(rng.choice(remaining_list, p=probabilities))
        add_subgraph(chosen)

    selected = np.asarray(selected, dtype=np.int64)
    selected_subgraphs = [subgraphs[int(idx)] for idx in selected]
    if not return_info:
        return selected_subgraphs

    info = _coverage_diagnostics(selected, state, marker_names)
    reuse_values = np.asarray(list(reuse_counts.values()), dtype=np.int64)
    info.update(
        {
            "reuse_penalty": float(reuse_penalty),
            "fill_reuse_penalty": float(fill_reuse_penalty),
            "source_cell_reuse_counts": reuse_counts,
            "n_unique_source_cell_hop_markers": int(len(reuse_counts)),
            "max_source_cell_reuse": (
                int(reuse_values.max()) if len(reuse_values) else 0
            ),
        }
    )
    return selected_subgraphs, info


def sample_subgraphs_coverage_non_overlapping_sources(
    subgraphs,
    marker_names,
    k_hops,
    max_subgraphs,
    min_center_count=40,
    min_pair_count=12,
    center_weight=1.0,
    pair_weight=3.0,
    fill_weight_pair=2.0,
    fill_weight_center=1.0,
    seed=0,
    threshold=0.5,
    return_info=True,
):
    """Coverage-aware sampling with hard non-overlap of perturbation source cells.

    This diagnostic sampler is intended for ``mode="single"`` ablation. For a
    selected perturbation source cell at hop distance ``d``, no other selected
    perturbation source cell at the same hop may lie within graph distance
    ``d`` in the original organoid graph. The sampler first satisfies center
    marker coverage, then tries to satisfy sparse source markers before more
    abundant source markers, while retaining the same center/pair coverage diagnostics as
    :func:`sample_subgraphs_coverage`.
    """
    if not np.isclose(threshold, 0.5):
        raise ValueError(
            "Non-overlap sampling currently requires threshold=0.5 to match "
            "the perturbation selector exactly."
        )
    if max_subgraphs is None or max_subgraphs >= len(subgraphs):
        if return_info:
            return list(subgraphs), {
                "selected_indices": np.arange(len(subgraphs), dtype=np.int64),
                "note": "No subsampling applied",
            }
        return list(subgraphs)
    if max_subgraphs <= 0:
        raise ValueError("max_subgraphs must be positive or None")

    rng = np.random.default_rng(seed)
    state = _prepare_coverage_sampling(
        subgraphs,
        marker_names,
        k_hops,
        min_center_count,
        min_pair_count,
        threshold,
    )
    source_cell_ids = _single_source_cell_ids(
        subgraphs,
        k_hops,
        threshold=threshold,
    )
    adjacency = _physical_cell_adjacency_from_subgraphs(subgraphs)

    selected = []
    remaining = set(range(state["n_subgraphs"]))
    blocked_cells = {}
    neighborhood_cache = {}

    def compatible(idx):
        return _is_nonoverlap_compatible(
            idx,
            state,
            source_cell_ids,
            blocked_cells,
            adjacency,
            neighborhood_cache,
        )

    def add_subgraph(idx):
        selected.append(idx)
        remaining.remove(idx)
        _add_coverage_subgraph(idx, state)
        _update_nonoverlap_blocks(
            idx,
            state,
            source_cell_ids,
            blocked_cells,
            adjacency,
            neighborhood_cache,
        )

    # Phase A: cover center markers first so rare centers are not lost as a
    # side effect of perturbation-source blocking.
    for center_marker in np.argsort(state["center_freq"], kind="stable"):
        if state["center_freq"][center_marker] <= 0:
            continue
        while (
            len(selected) < max_subgraphs
            and state["covered_center"][center_marker]
            < state["target_center"][center_marker]
            and remaining
        ):
            candidates = [
                idx
                for idx in remaining
                if center_marker in state["center_feats"][idx] and compatible(idx)
            ]
            if not candidates:
                break
            scores = np.asarray(
                [
                    _coverage_gain(
                        idx,
                        state,
                        center_weight,
                        pair_weight,
                    )
                    for idx in candidates
                ],
                dtype=float,
            )
            best = scores.max()
            best_positions = np.flatnonzero(scores == best)
            chosen = candidates[int(rng.choice(best_positions))]
            add_subgraph(chosen)

    # Phase B: satisfy sparse source markers before abundant source markers.
    for source_marker in _source_marker_priority(state):
        for hop_idx in range(state["n_hops"]):
            feature_order = np.argsort(
                state["pair_freq"][hop_idx, :, source_marker],
                kind="stable",
            )
            for center_marker in feature_order:
                if state["pair_freq"][hop_idx, center_marker, source_marker] <= 0:
                    continue
                feature = (hop_idx, int(center_marker), int(source_marker))
                while (
                    len(selected) < max_subgraphs
                    and state["covered_pair"][feature] < state["target_pair"][feature]
                    and remaining
                ):
                    candidates = [
                        idx
                        for idx in remaining
                        if feature in state["pair_feats"][idx] and compatible(idx)
                    ]
                    if not candidates:
                        break
                    scores = np.asarray(
                        [
                            _coverage_gain(
                                idx,
                                state,
                                center_weight,
                                pair_weight,
                            )
                            for idx in candidates
                        ],
                        dtype=float,
                    )
                    best = scores.max()
                    best_positions = np.flatnonzero(scores == best)
                    chosen = candidates[int(rng.choice(best_positions))]
                    add_subgraph(chosen)

    # Phase C: cover any remaining center markers if possible without violating
    # non-overlap. This can still help if pair coverage selected new compatible
    # candidates.
    while len(selected) < max_subgraphs and remaining:
        candidates = [idx for idx in remaining if compatible(idx)]
        if not candidates:
            break
        scores = np.asarray(
            [
                sum(
                    state["covered_center"][marker] < state["target_center"][marker]
                    for marker in state["center_feats"][idx]
                )
                for idx in candidates
            ],
            dtype=float,
        )
        if scores.max() <= 0:
            break
        best_positions = np.flatnonzero(scores == scores.max())
        chosen = candidates[int(rng.choice(best_positions))]
        add_subgraph(chosen)

    # Phase D: fill with compatible rare-feature candidates.
    while len(selected) < max_subgraphs and remaining:
        candidates = [idx for idx in remaining if compatible(idx)]
        if not candidates:
            break
        weights = np.asarray(
            [
                _coverage_fill_weight(
                    idx,
                    state,
                    fill_weight_center,
                    fill_weight_pair,
                )
                for idx in candidates
            ],
            dtype=float,
        )
        probabilities = weights / weights.sum()
        chosen = int(rng.choice(candidates, p=probabilities))
        add_subgraph(chosen)

    selected = np.asarray(selected, dtype=np.int64)
    selected_subgraphs = [subgraphs[int(idx)] for idx in selected]
    if not return_info:
        return selected_subgraphs

    info = _coverage_diagnostics(selected, state, marker_names)
    info.update(
        {
            "sampler": "non_overlapping",
            "n_blocked_hop_cells": {
                f"hop_{hop_idx + 1}": len(cells)
                for hop_idx, cells in blocked_cells.items()
            },
        }
    )
    return selected_subgraphs, info



def sample_subgraphs_by_center_marker(
    subgraphs,
    marker_names,
    max_subgraphs,
    min_center_count=40,
    seed=0,
    threshold=0.5,
    fill_weight=1.0,
    return_info=True,
):
    """
    Subsample ego-subgraphs while ensuring reasonable coverage of center markers.

    This sampler is intended for analyses where the main quantity of interest is
    the marker identity of the center node, rather than neighborhood marker
    composition.

    Strategy
    --------
    1) Greedily select subgraphs until each center marker reaches at least
       `min_center_count`, as far as possible within the available pool.
    2) Fill the remaining budget with weighted random sampling that favors
       subgraphs whose center markers are globally rare.

    Parameters
    ----------
    subgraphs : list[Data]
        Ego-subgraphs with attributes:
        - x : node features
        - center_idx : center node index within the subgraph
    marker_names : list[str]
        Names of marker channels.
    max_subgraphs : int or None
        Number of subgraphs to keep. If None or >= len(subgraphs), returns all.
    min_center_count : int
        Target minimum number of samples for each center marker.
    seed : int
        Random seed.
    threshold : float
        Marker positivity threshold for the center node.
    fill_weight : float
        Strength of rare-marker weighting in the random fill phase.
    return_info : bool
        Whether to also return diagnostics.

    Returns
    -------
    subs_selected : list[Data]
        Selected subgraphs.
    info : dict, optional
        Coverage diagnostics.
    """
    if len(subgraphs) == 0:
        raise ValueError("subgraphs is empty")

    if max_subgraphs is None or max_subgraphs >= len(subgraphs):
        if return_info:
            # still compute center coverage for inspection
            M = int(subgraphs[0].x.shape[1])
            center_pos = np.zeros((len(subgraphs), M), dtype=bool)
            for si, g in enumerate(subgraphs):
                c = int(g.center_idx)
                center_pos[si] = (g.x[c].detach().cpu().numpy() > threshold)

            covered = center_pos.sum(axis=0).astype(np.int64)
            info = {
                "selected_indices": np.arange(len(subgraphs), dtype=np.int64),
                "n_selected": int(len(subgraphs)),
                "n_available": int(len(subgraphs)),
                "center_target": np.full(M, int(min_center_count), dtype=np.int64),
                "center_covered": covered,
                "center_freq_global": covered.copy(),
                "unsatisfied_center_markers": [
                    marker_names[i] for i in np.flatnonzero(covered < min_center_count)
                ],
                "note": "No subsampling applied",
            }
            return list(subgraphs), info
        return list(subgraphs)

    if max_subgraphs <= 0:
        raise ValueError("max_subgraphs must be positive or None")

    rng = np.random.default_rng(seed)
    M = int(subgraphs[0].x.shape[1])
    S = len(subgraphs)

    # center marker presence for each subgraph
    center_pos = np.zeros((S, M), dtype=bool)
    for si, g in enumerate(subgraphs):
        c = int(g.center_idx)
        center_pos[si] = (g.x[c].detach().cpu().numpy() > threshold)

    center_freq_global = center_pos.sum(axis=0).astype(np.int64)
    target_center = np.full(M, int(min_center_count), dtype=np.int64)
    covered_center = np.zeros(M, dtype=np.int64)

    selected = []
    remaining = set(range(S))

    def score_subgraph(idx):
        score = 0.0
        ys = np.flatnonzero(center_pos[idx])
        for y in ys:
            deficit = max(0, target_center[y] - covered_center[y])
            if deficit > 0:
                score += 1.0
        return score

    def add_subgraph(idx):
        selected.append(idx)
        remaining.remove(idx)
        ys = np.flatnonzero(center_pos[idx])
        for y in ys:
            covered_center[y] += 1

    # Phase A: greedy coverage for center markers
    while len(selected) < max_subgraphs and len(remaining) > 0:
        remaining_list = list(remaining)
        scores = np.array([score_subgraph(i) for i in remaining_list], dtype=float)

        best = scores.max()
        if best <= 0:
            break

        best_idx = np.flatnonzero(scores == best)
        chosen = remaining_list[int(rng.choice(best_idx))]
        add_subgraph(chosen)

    # Phase B: fill remaining budget with weighted random sampling favoring rare center markers
    if len(selected) < max_subgraphs and len(remaining) > 0:
        remaining_list = list(remaining)
        weights = np.zeros(len(remaining_list), dtype=float)

        for j, idx in enumerate(remaining_list):
            ys = np.flatnonzero(center_pos[idx])
            w = 1.0
            for y in ys:
                freq = center_freq_global[y]
                if freq > 0:
                    w += fill_weight / freq
            weights[j] = w

        n_fill = min(max_subgraphs - len(selected), len(remaining_list))
        if n_fill > 0:
            probs = weights / weights.sum()
            chosen_fill = rng.choice(remaining_list, size=n_fill, replace=False, p=probs)
            for idx in chosen_fill:
                add_subgraph(int(idx))

    selected = np.array(selected, dtype=np.int64)
    subs_selected = [subgraphs[int(i)] for i in selected]

    if not return_info:
        return subs_selected

    unsatisfied_center = np.flatnonzero(covered_center < target_center)

    info = {
        "selected_indices": selected,
        "n_selected": int(len(selected)),
        "n_available": int(S),
        "center_target": target_center,
        "center_covered": covered_center,
        "center_freq_global": center_freq_global,
        "unsatisfied_center_markers": [marker_names[i] for i in unsatisfied_center],
    }

    return subs_selected, info


def print_sampling_summary(sample_info, marker_names):
    print(f"Selected {sample_info['n_selected']} / {sample_info['n_available']} subgraphs")
    print()

    print("Center coverage:")
    for m, tgt, cov, avail in zip(
        marker_names,
        sample_info["center_target"],
        sample_info["center_covered"],
        sample_info["center_freq_global"],
    ):
        status = "OK" if cov >= tgt else "LOW"
        print(
            f"{m:12s}  covered={int(cov):4d}  "
            f"target={int(tgt):4d}  available={int(avail):4d}  {status}"
        )

    print()
    print(f"Unsatisfied center markers: {sample_info['unsatisfied_center_markers']}")
    print(f"Unsatisfied pair features: {len(sample_info['unsatisfied_pair_features'])}")
