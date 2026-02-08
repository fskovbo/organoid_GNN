import copy
import numpy as np
import torch
from torch_geometric.loader import DataLoader



# -----------------------------
# Helper: predict (mu, logvar) for center node of each subgraph in a list
# -----------------------------
@torch.no_grad()
def predict_center_mu_logvar(subgraphs, model, device=None, batch_size=64, num_workers=0, pin_memory=True):
    """
    Inputs:
      subgraphs : list of torch_geometric.data.Data
                 each must have .x, .edge_index, and .center_idx (int)
      model     : GNN model; forward returns (mu, logvar) with shape (num_nodes, 1) or (num_nodes,)
      device    : 'cuda'/'cpu' (optional)

    Outputs:
      mu_c      : np.ndarray, shape (B,), predicted mean at center node for each subgraph
      lv_c      : np.ndarray, shape (B,), predicted log-variance at center node for each subgraph
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = model.to(device).eval()
    loader = DataLoader(subgraphs, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=pin_memory)

    mu_centers = []
    lv_centers = []

    for batch in loader:
        batch = batch.to(device, non_blocking=True)

        (mu, logvar), _ = model(batch.x, batch.edge_index)

        # Ensure shape (num_nodes,)
        mu = mu.view(-1)
        logvar = logvar.view(-1)

        # DataLoader concatenates graphs; need center indices with graph offsets
        # PyG provides batch.batch (node->graph id) and batch.ptr (graph start pointers)
        ptr = batch.ptr  # shape (num_graphs+1,)
        # For each graph i, center node in the batch is ptr[i] + center_idx_of_graph_i
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
# Helper: compute hop rings (nodes at exact hop distance) in a subgraph
# -----------------------------
def hop_rings(edge_index, center_idx, max_hops):
    """
    Inputs:
      edge_index  : torch.LongTensor shape (2, E)
      center_idx  : int, center node index in this subgraph
      max_hops    : int

    Outputs:
      rings       : list of lists; rings[h] = list of node indices at exact hop distance h
                   includes rings[0] = [center_idx]
    """
    src = edge_index[0].tolist()
    dst = edge_index[1].tolist()

    # Treat as undirected for "neighborhood" (common in tissue adjacency)
    adj = {}
    for u, v in zip(src, dst):
        adj.setdefault(u, set()).add(v)
        adj.setdefault(v, set()).add(u)

    visited = set([center_idx])
    frontier = set([center_idx])
    rings = [[center_idx]]

    for h in range(1, max_hops + 1):
        nxt = set()
        for u in frontier:
            for v in adj.get(u, ()):
                if v not in visited:
                    nxt.add(v)
        rings.append(sorted(nxt))
        visited |= nxt
        frontier = nxt
        if len(frontier) == 0:
            # pad remaining hops with empty rings
            for _ in range(h + 1, max_hops + 1):
                rings.append([])
            break

    return rings


# -----------------------------
# Core: build perturbed copies and aggregate delta(mu) and delta(logvar)
# -----------------------------
def perturbation_influence_maps(
    subgraphs,               # list[Data]: ego-subgraphs; each must have .center_idx and x[:, M] binary markers
    model,                   # torch.nn.Module: trained GNN; forward(x, edge_index) -> (mu, logvar) for all nodes
    marker_names,            # list[str]: length M; names of marker columns in x
    k_hops,                  # int: analyze hop rings 1..k_hops around the center node
    perturb="zero",          # str: "zero" sets marker->0 on selected nodes; "flip" sets marker->1-x
    max_subgraphs=200,       # int|None: subsample subgraphs for speed (None = use all)
    device=None,             # str|None: "cuda" / "cpu"; if None chooses automatically
    batch_size=64,           # int: DataLoader batch size for inference on perturbed graphs
):
    """
    Performs node-centered perturbation attribution on ego-subgraphs to quantify how marker composition
    in each hop-ring affects the *center node's* predicted curvature distribution.

    For each subgraph and each hop h=1..k_hops:
      - selects the nodes exactly h hops from the center node
      - for each marker X (feature column), perturbs that marker on those nodes (zero or flip)
      - runs the model and measures the change in the center node prediction:
            Δμ  = μ_perturbed - μ_base
            Δlv = logvar_perturbed - logvar_base
      - accumulates these deltas as (a) overall averages across all centers (total),
        and (b) averages conditioned on the *center node’s* marker(s) Y.

    Normalization:
      - your current implementation normalizes by the number of perturbed positives (n_pert) per case,
        so effects are reported “per perturbed positive” rather than “per subgraph”.

    Returns
    -------
    result : dict with keys

      Total (center type NOT resolved):
        - "delta_mu_total"        : (H, M) mean Δμ per perturbed positive
        - "delta_mu_abs_total"    : (H, M) mean |Δμ| per perturbed positive
        - "delta_lv_total"        : (H, M) mean Δlogvar per perturbed positive
        - "delta_lv_abs_total"    : (H, M) mean |Δlogvar| per perturbed positive
        - "counts_total"          : (H, M) total # perturbed positives accumulated (denominator)

      Center-type resolved (center marker Y resolved):
        - "delta_mu_cmarker"      : (H, M_center, M_pert) mean Δμ for centers with marker Y
        - "delta_mu_abs_cmarker"  : (H, M_center, M_pert) mean |Δμ| for centers with marker Y
        - "delta_lv_cmarker"      : (H, M_center, M_pert) mean Δlogvar for centers with marker Y
        - "delta_lv_abs_cmarker"  : (H, M_center, M_pert) mean |Δlogvar| for centers with marker Y
        - "counts_cmarker"        : (H, M_center, M_pert) total # perturbed positives accumulated

      Metadata:
        - "hops"                  : list[int] hop indices used (length H)
        - "marker_names"          : list[str] marker names (length M)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

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
    S = len(subgraphs)

    # Baseline predictions for each subgraph center
    base_mu, base_lv = predict_center_mu_logvar(subs, model, device=device, batch_size=batch_size)

    # Center marker matrix for conditioning: (S, M)
    center_X = np.zeros((S, M), dtype=np.int8)
    for si, g in enumerate(subgraphs):
        c = int(g.center_idx)
        center_X[si, :] = g.x[c].detach().cpu().numpy().astype(np.int8)

    # Build all perturbed graphs in one big list for efficient batching
    perturbed = []
    meta = []  # (subgraph_index, hop_index_in_hops_list, marker_index)

    for si, g in enumerate(subs):
        c = int(g.center_idx)
        rings = hop_rings(g.edge_index, c, k_hops)

        for hi, hop in enumerate(hops):
            nodes = rings[hop]
            if len(nodes) == 0:
                continue

            for mi in range(M):
                gp = copy.deepcopy(g)
                x = gp.x

                if perturb == "zero":
                    n_pert = (x[nodes, mi] > 0.5).sum().item() # only perturbs positives
                    x[nodes, mi] = 0
                elif perturb == "flip":
                    n_pert = len(nodes) # perturbs all
                    x[nodes, mi] = 1 - x[nodes, mi]
                else:
                    raise ValueError("perturb must be 'zero' or 'flip'")

                gp.x = x
                perturbed.append(gp)
                meta.append((si, hi, mi, int(n_pert)))

       
    # Predictions on perturbed graphs
    pert_mu, pert_lv = predict_center_mu_logvar(perturbed, model, device=device, batch_size=batch_size)

    # Aggregate deltas by (hop, marker)
    delta_mu_total = np.zeros((H, M), dtype=np.float64)
    delta_mu_abs_total = np.zeros((H, M), dtype=np.float64)
    delta_lv_total = np.zeros((H, M), dtype=np.float64)
    delta_lv_abs_total = np.zeros((H, M), dtype=np.float64)
    
    delta_mu_cmarker = np.zeros((H, M, M), dtype=np.float64)
    delta_mu_abs_cmarker = np.zeros((H, M, M), dtype=np.float64)
    delta_lv_cmarker = np.zeros((H, M, M), dtype=np.float64)
    delta_lv_abs_cmarker = np.zeros((H, M, M), dtype=np.float64)
    
    counts_total = np.zeros((H, M), dtype=np.int64)
    counts_cmarker = np.zeros((H, M, M), dtype=np.int64)
    
    for j, (si, hi, mi, n_pert) in enumerate(meta):
        dmu = pert_mu[j] - base_mu[si]
        dlv = pert_lv[j] - base_lv[si]

        delta_mu_total[hi, mi] += dmu
        delta_mu_abs_total[hi, mi] += abs(dmu)
        delta_lv_total[hi, mi] += dlv
        delta_lv_abs_total[hi, mi] += abs(dlv)        
        counts_total[hi, mi] += n_pert # normalize by how many cells perturbed

         # Condition on center marker(s) Y
         # Total would be the sum over center marker IF markers were exclusive
        ys = np.where(center_X[si, :] == 1)[0]
        for y in ys:
            delta_mu_cmarker[hi, y, mi] += dmu
            delta_mu_abs_cmarker[hi, y, mi] += abs(dmu)
            delta_lv_cmarker[hi, y, mi] += dlv
            delta_lv_abs_cmarker[hi, y, mi] += abs(dlv)
            counts_cmarker[hi, y, mi] += n_pert


    # Normalize
    mask_t = counts_total > 0
    delta_mu_total[mask_t] /= counts_total[mask_t]
    delta_mu_abs_total[mask_t] /= counts_total[mask_t]
    delta_lv_total[mask_t] /= counts_total[mask_t]
    delta_lv_abs_total[mask_t] /= counts_total[mask_t]

    mask_c = counts_cmarker > 0
    delta_mu_cmarker[mask_c] /= counts_cmarker[mask_c]
    delta_mu_abs_cmarker[mask_c] /= counts_cmarker[mask_c]
    delta_lv_cmarker[mask_c] /= counts_cmarker[mask_c]
    delta_lv_abs_cmarker[mask_c] /= counts_cmarker[mask_c]

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
        "hops": hops,
        "marker_names": marker_names,
    }


# -----------------------------
# Quick plotting helpers
# -----------------------------
def plot_influence_heatmap(mat, hops, marker_names, title):
    """
    Inputs:
      mat          : np.ndarray (H, M)
      hops         : list[int]
      marker_names : list[str]
      title        : str

    Output:
      matplotlib heatmap
    """
    import matplotlib.pyplot as plt

    plt.figure(figsize=(0.7 * len(marker_names) + 3, 0.7 * len(hops) + 2))
    plt.imshow(mat, aspect="auto")
    plt.colorbar()
    plt.yticks(range(len(hops)), [f"hop {h}" for h in hops])
    plt.xticks(range(len(marker_names)), marker_names, rotation=60, ha="right")
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_influence_center_resolved(
    mat,
    hops,
    marker_names,
    title,
    center_zero=False,
    sort_center=True,
    cmap="viridis",        # <-- NEW
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
    """
    import numpy as np
    import matplotlib.pyplot as plt

    mat = np.asarray(mat)
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
    center_names = [marker_names[i] for i in order]

    fig, axes = plt.subplots(
        1, H,
        figsize=(0.50 * len(marker_names) * H + 2.5,
                 0.35 * len(marker_names) + 2),
        squeeze=False
    )
    axes = axes[0]

    for hi, hop in enumerate(hops):
        ax = axes[hi]
        data = mat_s[hi]

        if center_zero:
            vmax = np.nanmax(np.abs(data))
            vmin = -vmax
            im = ax.imshow(
                data,
                aspect="auto",
                vmin=vmin,
                vmax=vmax,
                cmap=cmap,      # <-- USED HERE
            )
        else:
            im = ax.imshow(
                data,
                aspect="auto",
                cmap=cmap,      # <-- AND HERE
            )

        ax.set_xlabel("perturbation marker X", fontsize=13)
        ax.set_ylabel("center marker Y", fontsize=13)

        ax.set_xticks(range(len(marker_names)))
        ax.set_xticklabels(marker_names, rotation=60, ha="right", fontsize=11)
        ax.set_yticks(range(len(center_names)))
        ax.set_yticklabels(center_names, fontsize=11)

        ax.set_title(f"hop {hop}", fontsize=14)

        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=11)

    fig.suptitle(title, fontsize=16)
    fig.tight_layout()
    plt.show()
