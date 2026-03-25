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
    M = len(marker_names)
    H = int(k_hops)
    S = len(subgraphs)

    center_pos, ring_pos = _subgraph_marker_coverage(subgraphs, k_hops, threshold=threshold)
    center_feats, pair_feats = _feature_lists_for_subgraphs(center_pos, ring_pos)

    # Targets
    target_center = np.full(M, int(min_center_count), dtype=np.int64)
    target_pair = np.full((H, M, M), int(min_pair_count), dtype=np.int64)

    # Tracking coverage in selected set
    covered_center = np.zeros(M, dtype=np.int64)
    covered_pair = np.zeros((H, M, M), dtype=np.int64)

    selected = []
    remaining = set(range(S))

    def score_subgraph(idx):
        score = 0.0

        for y in center_feats[idx]:
            deficit = max(0, target_center[y] - covered_center[y])
            if deficit > 0:
                score += center_weight

        for hi, y, m in pair_feats[idx]:
            deficit = max(0, target_pair[hi, y, m] - covered_pair[hi, y, m])
            if deficit > 0:
                score += pair_weight

        return score

    def add_subgraph(idx):
        selected.append(idx)
        remaining.remove(idx)

        for y in center_feats[idx]:
            covered_center[y] += 1

        for hi, y, m in pair_feats[idx]:
            covered_pair[hi, y, m] += 1

    # ---------------------------
    # Phase A: greedy coverage
    # ---------------------------
    while len(selected) < max_subgraphs and len(remaining) > 0:
        remaining_list = list(remaining)
        scores = np.array([score_subgraph(i) for i in remaining_list], dtype=float)

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
        center_freq_global, pair_freq_global = _global_feature_frequencies(center_feats, pair_feats, M, H)

        remaining_list = list(remaining)
        weights = np.zeros(len(remaining_list), dtype=float)

        for j, idx in enumerate(remaining_list):
            w = 1.0

            # favor rare center markers
            for y in center_feats[idx]:
                freq = center_freq_global[y]
                if freq > 0:
                    w += fill_weight_center / freq

            # favor rare pair features
            for hi, y, m in pair_feats[idx]:
                freq = pair_freq_global[hi, y, m]
                if freq > 0:
                    w += fill_weight_pair / freq

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

    # Diagnostics
    center_freq_global, pair_freq_global = _global_feature_frequencies(center_feats, pair_feats, M, H)

    unsatisfied_center = np.argwhere(covered_center < target_center).ravel().tolist()
    unsatisfied_pair = np.argwhere(covered_pair < target_pair)

    info = {
        "selected_indices": selected,
        "n_selected": int(len(selected)),
        "n_available": int(S),
        "center_target": target_center,
        "center_covered": covered_center,
        "pair_target": target_pair,
        "pair_covered": covered_pair,
        "center_freq_global": center_freq_global,
        "pair_freq_global": pair_freq_global,
        "unsatisfied_center_markers": [marker_names[i] for i in unsatisfied_center],
        "unsatisfied_pair_features": [
            {
                "hop": int(hi + 1),
                "center_marker": marker_names[int(y)],
                "neigh_marker": marker_names[int(m)],
                "covered": int(covered_pair[hi, y, m]),
                "target": int(target_pair[hi, y, m]),
                "global_available": int(pair_freq_global[hi, y, m]),
            }
            for hi, y, m in unsatisfied_pair
        ],
    }

    return subs_selected, info



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