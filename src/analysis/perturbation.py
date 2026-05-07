import copy

import numpy as np
import torch
from torch_geometric.loader import DataLoader

from src.graph.neighborhood import compute_hop_rings


def _select_target_output(arr, target_index=0, name="output"):
    """Return a 1-D per-node tensor for one selected target."""
    if arr.ndim == 1:
        if target_index not in (None, 0):
            raise IndexError(
                f"{name} is 1-D, so only target_index=0 is valid; got {target_index}."
            )
        return arr.contiguous()

    if arr.ndim == 2:
        if arr.shape[1] == 1:
            if target_index not in (None, 0):
                raise IndexError(
                    f"{name} has one target, so only target_index=0 is valid; got {target_index}."
                )
            return arr[:, 0].contiguous()
        if target_index is None:
            raise ValueError(
                f"{name} has shape {tuple(arr.shape)}. Pass target_index to select one target."
            )
        if not (0 <= int(target_index) < arr.shape[1]):
            raise IndexError(
                f"target_index={target_index} out of range for {name} with shape {tuple(arr.shape)}."
            )
        return arr[:, int(target_index)].contiguous()

    raise ValueError(
        f"Expected {name} to be 1-D or 2-D after covariance conversion, got {tuple(arr.shape)}"
    )


@torch.no_grad()
def predict_subgraph_center_distribution(
    subgraphs,
    model,
    device=None,
    batch_size=64,
    num_workers=0,
    pin_memory=True,
    target_index=0,
):
    """Predict mean and log-variance for one selected target at each subgraph center."""
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

        if logvar.ndim == 3:
            cov = logvar @ logvar.transpose(-1, -2)
            logvar = torch.log(torch.diagonal(cov, dim1=-2, dim2=-1).clamp_min(1e-12))

        mu = _select_target_output(mu, target_index=target_index, name="mu")
        logvar = _select_target_output(logvar, target_index=target_index, name="logvar")

        ptr = batch.ptr
        centers = []
        for i in range(batch.num_graphs):
            c = (
                int(batch.center_idx[i].item())
                if torch.is_tensor(batch.center_idx[i])
                else int(batch.center_idx[i])
            )
            centers.append(ptr[i].item() + c)
        centers = torch.tensor(centers, device=device, dtype=torch.long)

        mu_centers.append(mu[centers].detach().cpu().numpy())
        lv_centers.append(logvar[centers].detach().cpu().numpy())

    mu_c = np.concatenate(mu_centers, axis=0).astype(np.float64)
    lv_c = np.concatenate(lv_centers, axis=0).astype(np.float64)
    return mu_c, lv_c


def _select_center_target_value(y_center, target_index=0):
    arr = y_center.detach().cpu().numpy() if torch.is_tensor(y_center) else np.asarray(y_center)
    arr = np.asarray(arr)
    if arr.ndim == 0:
        if target_index not in (None, 0):
            raise IndexError(f"Scalar target only supports target_index=0; got {target_index}.")
        return float(arr)
    if arr.ndim == 1:
        if arr.shape[0] == 1:
            if target_index not in (None, 0):
                raise IndexError(f"Single target only supports target_index=0; got {target_index}.")
            return float(arr[0])
        if target_index is None:
            raise ValueError("Multi-target y requires target_index.")
        return float(arr[int(target_index)])
    raise ValueError(f"Expected scalar or 1-D center target, got shape {arr.shape}.")


def _subgraph_center_targets(subgraphs, target_index=0):
    y_center = []
    for g in subgraphs:
        c = int(g.center_idx)
        y_center.append(_select_center_target_value(g.y[c], target_index=target_index))
    return np.asarray(y_center, dtype=np.float64)


def _subsample_subgraphs(subgraphs, max_subgraphs):
    if max_subgraphs is not None and len(subgraphs) > max_subgraphs:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(subgraphs), size=max_subgraphs, replace=False)
        return [subgraphs[int(i)] for i in idx]
    return list(subgraphs)


def _center_marker_matrix(subgraphs, n_markers):
    center_x = np.zeros((len(subgraphs), n_markers), dtype=np.int8)
    for si, g in enumerate(subgraphs):
        c = int(g.center_idx)
        center_x[si, :] = g.x[c].detach().cpu().numpy().astype(np.int8)
    return center_x


def _validate_mode(mode):
    if mode not in ("all", "single"):
        raise ValueError("mode must be 'all' or 'single'")


def _validate_normalization(value, name):
    if value not in ("cases", "perturbed_cells"):
        raise ValueError(f"{name} must be 'cases' or 'perturbed_cells'; got {value!r}.")


def _resolve_normalization(normalize_by, normalize_total_by, normalize_center_by):
    if normalize_by is not None:
        normalize_total_by = normalize_by
        normalize_center_by = normalize_by

    _validate_normalization(normalize_total_by, "normalize_total_by")
    _validate_normalization(normalize_center_by, "normalize_center_by")
    return normalize_total_by, normalize_center_by


def _build_marker_perturbations(subgraphs, n_markers, k_hops, hops, mode):
    perturbed = []
    meta = []  # (subgraph_index, hop_index, marker_index, n_perturbed)

    for si, g in enumerate(subgraphs):
        c = int(g.center_idx)
        rings = compute_hop_rings(g.edge_index, c, k_hops)

        for hi, hop in enumerate(hops):
            nodes = rings[hop]
            if len(nodes) == 0:
                continue

            for mi in range(n_markers):
                x_ring = g.x[nodes, mi]
                pos_mask = x_ring > 0.5
                n_pos = int(pos_mask.sum().item())
                if n_pos == 0:
                    continue

                gp = copy.deepcopy(g)
                x = gp.x

                if mode == "all":
                    ring_nodes_t = torch.as_tensor(nodes, device=x.device, dtype=torch.long)
                    pos_idx = torch.nonzero(pos_mask, as_tuple=False).view(-1)
                    x[ring_nodes_t[pos_idx], mi] = 0
                    n_pert = n_pos
                else:
                    pos_idx_in_ring = int(torch.nonzero(pos_mask, as_tuple=False)[0].item())
                    node_to_zero = int(nodes[pos_idx_in_ring])
                    x[node_to_zero, mi] = 0
                    n_pert = 1

                gp.x = x
                perturbed.append(gp)
                meta.append((si, hi, mi, int(n_pert)))

    return perturbed, meta


def _init_aggregate_arrays(n_hops, n_markers):
    return (
        np.zeros((n_hops, n_markers), dtype=np.float64),
        np.zeros((n_hops, n_markers), dtype=np.float64),
        np.zeros((n_hops, n_markers, n_markers), dtype=np.float64),
        np.zeros((n_hops, n_markers, n_markers), dtype=np.float64),
        np.zeros((n_hops, n_markers), dtype=np.int64),
        np.zeros((n_hops, n_markers), dtype=np.int64),
        np.zeros((n_hops, n_markers, n_markers), dtype=np.int64),
        np.zeros((n_hops, n_markers, n_markers), dtype=np.int64),
    )


def _normalize_aggregates(
    total,
    total_abs,
    cmarker,
    cmarker_abs,
    counts_total,
    cases_total,
    counts_cmarker,
    cases_cmarker,
    *,
    normalize_total_by,
    normalize_center_by,
):
    denom_total = counts_total if normalize_total_by == "perturbed_cells" else cases_total
    mask_total = denom_total > 0
    total[mask_total] /= denom_total[mask_total]
    total_abs[mask_total] /= denom_total[mask_total]

    denom_cmarker = counts_cmarker if normalize_center_by == "perturbed_cells" else cases_cmarker
    mask_cmarker = denom_cmarker > 0
    cmarker[mask_cmarker] /= denom_cmarker[mask_cmarker]
    cmarker_abs[mask_cmarker] /= denom_cmarker[mask_cmarker]


def _accumulate_scalar_effects(
    effects,
    meta,
    center_x,
    n_hops,
    n_markers,
    *,
    normalize_total_by,
    normalize_center_by,
):
    (
        total,
        total_abs,
        cmarker,
        cmarker_abs,
        counts_total,
        cases_total,
        counts_cmarker,
        cases_cmarker,
    ) = _init_aggregate_arrays(n_hops, n_markers)

    for effect, (si, hi, mi, n_pert) in zip(effects, meta):
        total[hi, mi] += effect
        total_abs[hi, mi] += abs(effect)
        counts_total[hi, mi] += n_pert
        cases_total[hi, mi] += 1

        center_markers = np.where(center_x[si, :] == 1)[0]
        for y_marker in center_markers:
            cmarker[hi, y_marker, mi] += effect
            cmarker_abs[hi, y_marker, mi] += abs(effect)
            counts_cmarker[hi, y_marker, mi] += n_pert
            cases_cmarker[hi, y_marker, mi] += 1

    _normalize_aggregates(
        total,
        total_abs,
        cmarker,
        cmarker_abs,
        counts_total,
        cases_total,
        counts_cmarker,
        cases_cmarker,
        normalize_total_by=normalize_total_by,
        normalize_center_by=normalize_center_by,
    )

    return {
        "total": total,
        "total_abs": total_abs,
        "cmarker": cmarker,
        "cmarker_abs": cmarker_abs,
        "counts_total": counts_total,
        "cases_total": cases_total,
        "counts_cmarker": counts_cmarker,
        "cases_cmarker": cases_cmarker,
    }


def _prepare_perturbation_run(
    subgraphs,
    marker_names,
    k_hops,
    mode,
    max_subgraphs,
):
    _validate_mode(mode)
    subs = _subsample_subgraphs(subgraphs, max_subgraphs)
    n_markers = len(marker_names)
    hops = list(range(1, k_hops + 1))
    center_x = _center_marker_matrix(subs, n_markers)
    perturbed, meta = _build_marker_perturbations(subs, n_markers, k_hops, hops, mode)
    return subs, n_markers, hops, center_x, perturbed, meta


def compute_perturbation_influence_maps(
    subgraphs,
    model,
    marker_names,
    k_hops,
    mode="all",
    max_subgraphs=200,
    device=None,
    batch_size=64,
    target_index=0,
    normalize_by=None,
    normalize_total_by="perturbed_cells",
    normalize_center_by="perturbed_cells",
):
    """Measure how marker removal changes center-cell predicted mean/log-variance.

    Normalization can be controlled independently for total and center-resolved
    maps. Use ``normalize_by`` to set both to either ``"cases"`` or
    ``"perturbed_cells"``.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    normalize_total_by, normalize_center_by = _resolve_normalization(
        normalize_by,
        normalize_total_by,
        normalize_center_by,
    )

    subs, n_markers, hops, center_x, perturbed, meta = _prepare_perturbation_run(
        subgraphs,
        marker_names,
        k_hops,
        mode,
        max_subgraphs,
    )

    base_mu, base_lv = predict_subgraph_center_distribution(
        subs,
        model,
        device=device,
        batch_size=batch_size,
        target_index=target_index,
    )

    if len(perturbed) == 0:
        pert_mu = np.zeros((0,), dtype=np.float64)
        pert_lv = np.zeros((0,), dtype=np.float64)
    else:
        pert_mu, pert_lv = predict_subgraph_center_distribution(
            perturbed,
            model,
            device=device,
            batch_size=batch_size,
            target_index=target_index,
        )

    n_hops = len(hops)
    mu_effects = [pert_mu[j] - base_mu[si] for j, (si, _, _, _) in enumerate(meta)]
    lv_effects = [pert_lv[j] - base_lv[si] for j, (si, _, _, _) in enumerate(meta)]

    mu_ag = _accumulate_scalar_effects(
        mu_effects,
        meta,
        center_x,
        n_hops,
        n_markers,
        normalize_total_by=normalize_total_by,
        normalize_center_by=normalize_center_by,
    )
    lv_ag = _accumulate_scalar_effects(
        lv_effects,
        meta,
        center_x,
        n_hops,
        n_markers,
        normalize_total_by=normalize_total_by,
        normalize_center_by=normalize_center_by,
    )

    return {
        "delta_mu_total": mu_ag["total"],
        "delta_mu_abs_total": mu_ag["total_abs"],
        "delta_lv_total": lv_ag["total"],
        "delta_lv_abs_total": lv_ag["total_abs"],
        "counts_total": mu_ag["counts_total"],
        "cases_total": mu_ag["cases_total"],
        "delta_mu_cmarker": mu_ag["cmarker"],
        "delta_mu_abs_cmarker": mu_ag["cmarker_abs"],
        "delta_lv_cmarker": lv_ag["cmarker"],
        "delta_lv_abs_cmarker": lv_ag["cmarker_abs"],
        "counts_cmarker": mu_ag["counts_cmarker"],
        "cases_cmarker": mu_ag["cases_cmarker"],
        "hops": hops,
        "marker_names": marker_names,
        "target_index": target_index,
        "normalize_total_by": normalize_total_by,
        "normalize_center_by": normalize_center_by,
    }


def compute_perturbation_mse_influence_maps(
    subgraphs,
    model,
    marker_names,
    k_hops,
    mode="all",
    max_subgraphs=200,
    device=None,
    batch_size=64,
    target_index=0,
    target_transform=None,
    normalize_by=None,
    normalize_total_by="perturbed_cells",
    normalize_center_by="perturbed_cells",
):
    """Measure how marker removal changes center-cell squared prediction error.

    Positive delta MSE means the perturbation worsened prediction accuracy;
    negative delta MSE means it improved prediction accuracy. If
    ``target_transform`` is provided, MSE is computed in the original target
    scale.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    normalize_total_by, normalize_center_by = _resolve_normalization(
        normalize_by,
        normalize_total_by,
        normalize_center_by,
    )

    subs, n_markers, hops, center_x, perturbed, meta = _prepare_perturbation_run(
        subgraphs,
        marker_names,
        k_hops,
        mode,
        max_subgraphs,
    )

    y_center = _subgraph_center_targets(subs, target_index=target_index)
    base_mu, _ = predict_subgraph_center_distribution(
        subs,
        model,
        device=device,
        batch_size=batch_size,
        target_index=target_index,
    )

    if target_transform is not None:
        y_center, base_mu, _ = target_transform.inverse_distribution(y_center, base_mu, log_var=None)
        y_center = np.asarray(y_center, dtype=np.float64)
        base_mu = np.asarray(base_mu, dtype=np.float64)

    base_sqerr = np.square(base_mu - y_center)

    if len(perturbed) == 0:
        pert_mu = np.zeros((0,), dtype=np.float64)
    else:
        pert_mu, _ = predict_subgraph_center_distribution(
            perturbed,
            model,
            device=device,
            batch_size=batch_size,
            target_index=target_index,
        )
        if target_transform is not None:
            _, pert_mu, _ = target_transform.inverse_distribution(None, pert_mu, log_var=None)
            pert_mu = np.asarray(pert_mu, dtype=np.float64)

    n_hops = len(hops)
    mse_effects = [
        float((pert_mu[j] - y_center[si]) ** 2 - base_sqerr[si])
        for j, (si, _, _, _) in enumerate(meta)
    ]
    ag = _accumulate_scalar_effects(
        mse_effects,
        meta,
        center_x,
        n_hops,
        n_markers,
        normalize_total_by=normalize_total_by,
        normalize_center_by=normalize_center_by,
    )

    return {
        "delta_mse_total": ag["total"],
        "delta_mse_abs_total": ag["total_abs"],
        "delta_mse_cmarker": ag["cmarker"],
        "delta_mse_abs_cmarker": ag["cmarker_abs"],
        "counts_total": ag["counts_total"],
        "cases_total": ag["cases_total"],
        "counts_cmarker": ag["counts_cmarker"],
        "cases_cmarker": ag["cases_cmarker"],
        "base_sqerr_center": base_sqerr,
        "hops": hops,
        "marker_names": marker_names,
        "target_index": target_index,
        "mse_scale": "original" if target_transform is not None else "transformed",
        "normalize_total_by": normalize_total_by,
        "normalize_center_by": normalize_center_by,
    }
