import copy
import numpy as np
import torch
from torch_geometric.loader import DataLoader
from src.graph.neighborhood import compute_hop_rings


# -----------------------------
# Helper: predict (mu, logvar) for center node of each subgraph in a list
# -----------------------------
@torch.no_grad()
def predict_subgraph_center_distribution(
    subgraphs,
    model,
    device=None,
    batch_size=64,
    num_workers=0,
    pin_memory=True,
):
    """
    Predict mean and log-variance at the center node of each subgraph.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = model.to(device).eval()
    loader = DataLoader(
        subgraphs,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    mu_centers = []
    lv_centers = []

    for batch in loader:
        batch = batch.to(device, non_blocking=True)

        try:
            (mu, logvar), _ = model(batch.x, batch.edge_index, data=batch)
        except TypeError:
            (mu, logvar), _ = model(batch.x, batch.edge_index)

        mu = mu.view(-1)
        logvar = logvar.view(-1)

        ptr = batch.ptr
        centers = []
        for i in range(batch.num_graphs):
            c = int(batch.center_idx[i].item()) if torch.is_tensor(batch.center_idx[i]) else int(batch.center_idx[i])
            centers.append(ptr[i].item() + c)
        centers = torch.tensor(centers, device=device, dtype=torch.long)

        mu_centers.append(mu[centers].detach().cpu().numpy())
        lv_centers.append(logvar[centers].detach().cpu().numpy())

    mu_c = np.concatenate(mu_centers, axis=0).astype(np.float64)
    lv_c = np.concatenate(lv_centers, axis=0).astype(np.float64)
    return mu_c, lv_c


# -----------------------------
# Core: build perturbed copies and aggregate delta(mu) and delta(logvar)
# -----------------------------
def compute_perturbation_influence_maps(
    subgraphs,               # list[Data]: ego-subgraphs; each must have .center_idx and x[:, M] binary markers
    model,                   # torch.nn.Module: trained GNN; forward(x, edge_index) -> (mu, logvar) for all nodes
    marker_names,            # list[str]: length M; names of marker columns in x
    k_hops,                  # int: analyze hop rings 1..k_hops around the center node
    mode="all",              # "all" or "single"
    max_subgraphs=200,       # int|None: subsample subgraphs for speed (None = use all)
    device=None,             # str|None: "cuda" / "cpu"; if None chooses automatically
    batch_size=64,           # int: DataLoader batch size for inference on perturbed graphs
):
    """
    As before, but:
      - always uses "zero" perturbation
      - supports mode="all" and mode="single"
      - skips cases with n_pert==0 (no positives to remove)
      - center-resolved outputs are normalized per *case* (not per-positive) to stabilize rare markers,
        while total outputs remain per-positive for minimal change.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if mode not in ("all", "single"):
        raise ValueError("mode must be 'all' or 'single'")

    # Subsample for speed
    if max_subgraphs is not None and len(subgraphs) > max_subgraphs:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(subgraphs), size=max_subgraphs, replace=False)
        subs = [subgraphs[int(i)] for i in idx]
    else:
        subs = list(subgraphs)

    M = len(marker_names)
    hops = list(range(1, k_hops + 1))
    H = len(hops)
    S = len(subs)

    # Baseline predictions for each subgraph center
    base_mu, base_lv = predict_subgraph_center_distribution(subs, model, device=device, batch_size=batch_size)

    # Center marker matrix for conditioning: (S, M)
    center_X = np.zeros((S, M), dtype=np.int8)
    for si, g in enumerate(subs):
        c = int(g.center_idx)
        center_X[si, :] = g.x[c].detach().cpu().numpy().astype(np.int8)

    # Build all perturbed graphs in one big list for efficient batching
    perturbed = []
    meta = []  # (subgraph_index, hop_index_in_hops_list, marker_index, n_pert)

    for si, g in enumerate(subs):
        c = int(g.center_idx)
        rings = compute_hop_rings(g.edge_index, c, k_hops)

        for hi, hop in enumerate(hops):
            nodes = rings[hop]
            if len(nodes) == 0:
                continue

            for mi in range(M):
                # Identify positives for marker mi in this ring
                x_ring = g.x[nodes, mi]
                pos_mask = (x_ring > 0.5)
                n_pos = int(pos_mask.sum().item())

                # IMPORTANT: skip if there is nothing to perturb
                if n_pos == 0:
                    continue

                gp = copy.deepcopy(g)
                x = gp.x

                if mode == "all":
                    # zero all positives
                    # NOTE: use tensor indexing carefully to avoid creating a copy
                    ring_nodes_t = torch.as_tensor(nodes, device=x.device, dtype=torch.long)
                    # indices in ring_nodes_t where pos_mask is true
                    pos_idx = torch.nonzero(pos_mask, as_tuple=False).view(-1)
                    x[ring_nodes_t[pos_idx], mi] = 0
                    n_pert = n_pos

                else:  # mode == "single"
                    # zero exactly one positive (deterministic: first positive)
                    pos_idx_in_ring = int(torch.nonzero(pos_mask, as_tuple=False)[0].item())
                    node_to_zero = int(nodes[pos_idx_in_ring])
                    x[node_to_zero, mi] = 0
                    n_pert = 1

                gp.x = x
                perturbed.append(gp)
                meta.append((si, hi, mi, int(n_pert)))

    # Predictions on perturbed graphs
    pert_mu, pert_lv = predict_subgraph_center_distribution(perturbed, model, device=device, batch_size=batch_size)

    # Aggregate deltas by (hop, marker)
    delta_mu_total = np.zeros((H, M), dtype=np.float64)
    delta_mu_abs_total = np.zeros((H, M), dtype=np.float64)
    delta_lv_total = np.zeros((H, M), dtype=np.float64)
    delta_lv_abs_total = np.zeros((H, M), dtype=np.float64)

    delta_mu_cmarker = np.zeros((H, M, M), dtype=np.float64)
    delta_mu_abs_cmarker = np.zeros((H, M, M), dtype=np.float64)
    delta_lv_cmarker = np.zeros((H, M, M), dtype=np.float64)
    delta_lv_abs_cmarker = np.zeros((H, M, M), dtype=np.float64)

    # Denominators you already had (per-positive)
    counts_total = np.zeros((H, M), dtype=np.int64)
    counts_cmarker = np.zeros((H, M, M), dtype=np.int64)

    # NEW denominators for stabilized center-resolved stats (per-case)
    cases_cmarker = np.zeros((H, M, M), dtype=np.int64)

    for j, (si, hi, mi, n_pert) in enumerate(meta):
        dmu = pert_mu[j] - base_mu[si]
        dlv = pert_lv[j] - base_lv[si]

        # Total: keep your original “per perturbed positive”
        delta_mu_total[hi, mi] += dmu
        delta_mu_abs_total[hi, mi] += abs(dmu)
        delta_lv_total[hi, mi] += dlv
        delta_lv_abs_total[hi, mi] += abs(dlv)
        counts_total[hi, mi] += n_pert

        # Center-conditioned: accumulate numerator once per qualifying case
        ys = np.where(center_X[si, :] == 1)[0]
        for y in ys:
            delta_mu_cmarker[hi, y, mi] += dmu
            delta_mu_abs_cmarker[hi, y, mi] += abs(dmu)
            delta_lv_cmarker[hi, y, mi] += dlv
            delta_lv_abs_cmarker[hi, y, mi] += abs(dlv)

            # Keep per-positive counts (for debugging / optional later)
            counts_cmarker[hi, y, mi] += n_pert
            # NEW: per-case count for stable mean
            cases_cmarker[hi, y, mi] += 1

    # Normalize totals (unchanged)
    mask_t = counts_total > 0
    delta_mu_total[mask_t] /= counts_total[mask_t]
    delta_mu_abs_total[mask_t] /= counts_total[mask_t]
    delta_lv_total[mask_t] /= counts_total[mask_t]
    delta_lv_abs_total[mask_t] /= counts_total[mask_t]

    # Normalize center-resolved by CASES (stabilized)
    mask_cc = cases_cmarker > 0
    delta_mu_cmarker[mask_cc] /= cases_cmarker[mask_cc]
    delta_mu_abs_cmarker[mask_cc] /= cases_cmarker[mask_cc]
    delta_lv_cmarker[mask_cc] /= cases_cmarker[mask_cc]
    delta_lv_abs_cmarker[mask_cc] /= cases_cmarker[mask_cc]

    return {
        "delta_mu_total": delta_mu_total,
        "delta_mu_abs_total": delta_mu_abs_total,
        "delta_lv_total": delta_lv_total,
        "delta_lv_abs_total": delta_lv_abs_total,
        "counts_total": counts_total,
        "delta_mu_cmarker": delta_mu_cmarker,
        "delta_mu_abs_cmarker": delta_mu_abs_cmarker,
        "delta_lv_cmarker": delta_lv_cmarker,
        "delta_lv_abs_cmarker": delta_lv_abs_cmarker,
        "counts_cmarker": counts_cmarker,
        # Optional but useful (does not affect your plotting)
        "cases_cmarker": cases_cmarker,
        "hops": hops,
        "marker_names": marker_names,
    }