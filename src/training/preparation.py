"""Reusable fold preparation; experiment grids and training loops stay in notebooks."""
import copy
import numpy as np
import torch
from torch_geometric.data import Data
from src.data.target_transforms import AsinhStandardizeTransform, GlobalBaselineResidualTransform, standardize_graph_global_features


def global_values(graph, names):
    """Build named log globals in the recorded order, validating only requested inputs."""
    md = getattr(graph, 'meta', {})
    values = {'log_num_cells': float(np.log(len(graph.x)))}
    for name in names:
        if name == 'log_num_cells':
            continue
        area, volume = md.get('total_surface_area'), md.get('total_volume')
        raw = {'log_surface_area': area, 'log_volume': volume,
               'log_volume_over_area': (volume / area if volume is not None and area else None)}[name]
        if raw is None or not np.isfinite(raw) or raw <= 0:
            raise ValueError(f'Invalid {name} for {graph.organoid_str}')
        values[name] = float(np.log(raw))
    return [values[name] for name in names]


def prepare_fold(graphs, split, *, global_features, residualize=False,
                 baseline_features=('log_num_cells', 'log_surface_area', 'log_volume', 'log_volume_over_area'),
                 baseline_kwargs=None):
    """Always fit a global baseline; optionally subtract it before target scaling.

    Returns compact model inputs and fitted objects. Raw metadata/geometry stay
    in the saved cohort, not in PyG batches. Baseline offsets are saved per graph
    so predictions can be reconstructed without passing geometry to the head.
    All fitted preprocessing and baseline optimization use training organoids only.
    Baseline predictions are always retained; reconstruction offsets are zero
    unless residualization was requested.
    """
    if not baseline_features:
        raise ValueError('Choose at least one baseline feature; the baseline is always trained.')
    names = list(dict.fromkeys([*global_features, *baseline_features]))
    groups = {}
    for role in ('train', 'val'):
        groups[role] = []
        for i in split[role + '_indices']:
            g = graphs[i]
            d = Data(x=g.x.clone(), y=g.y.clone(), edge_index=g.edge_index.clone(),
                     organoid_str=str(g.organoid_str), full_num_cells=float(len(g.x)))
            if names:
                d.global_feat = torch.tensor([global_values(g, names)], dtype=torch.float32)
            groups[role].append(d)
    if not groups['train'] or not groups['val']:
        raise ValueError('Both training and validation must be nonempty')
    center, scale = (standardize_graph_global_features(groups['train'], groups['val'])
                     if names else (np.array([]), np.array([])))
    baseline_groups = copy.deepcopy(groups)
    cols = [names.index(n) for n in baseline_features]
    for group in baseline_groups.values():
        for g in group:
            g.global_feat = g.global_feat[:, cols].clone()
    baseline = GlobalBaselineResidualTransform(**(baseline_kwargs or {})).fit(baseline_groups['train'])
    offsets, baseline_predictions, baseline_validation_mse = {}, {}, []
    for role, group in baseline_groups.items():
        predictions = np.asarray(baseline._predict_baseline(group)).reshape(-1)
        if residualize:
            baseline.transform_graphs(group, in_place=True)
        start = 0
        for g, b in zip(groups[role], group):
            stop = start + len(g.x)
            prediction = predictions[start:stop].copy()
            baseline_predictions[g.organoid_str] = prediction
            if role == 'val':
                truth = g.y.detach().cpu().numpy().reshape(-1)
                baseline_validation_mse.append(dict(organoid_str=g.organoid_str, n_cells=len(g.x),
                    mse=float(np.mean((truth-prediction)**2))))
            # Offsets describe target residualization, not whether a baseline exists.
            offsets[g.organoid_str] = ((g.y - b.y).cpu().numpy() if residualize
                                      else np.zeros(len(g.x)))
            if residualize:
                g.y = b.y.clone()
            start = stop
    baseline.model.cpu()
    baseline.prediction_cache.clear()
    transform = AsinhStandardizeTransform(robust=True).fit(groups['train'])
    cols = [names.index(n) for n in global_features]
    for group in groups.values():
        transform.transform_graphs(group, in_place=True)
        for g in group:
            if cols:
                g.global_feat = g.global_feat[:, cols].clone()
            elif 'global_feat' in g:
                del g.global_feat
    return dict(groups=groups, transform=transform, baseline=baseline, baseline_offsets=offsets,
                baseline_predictions=baseline_predictions, baseline_validation_mse=baseline_validation_mse,
                baseline_features=list(baseline_features), residualized=bool(residualize),
                global_features=list(global_features), all_global_features=names,
                global_center=np.asarray(center), global_scale=np.asarray(scale))


def select_marker_inputs(groups, marker_names, selected):
    """Select marker columns, preserving targets, graph order and input objects."""
    columns = [marker_names.index(n) for n in selected]
    result = copy.deepcopy(groups)
    for group in result.values():
        for g in group:
            g.x = g.x[:, columns].clone()
            for cached in ('x_ring', 'ring_sizes'):
                if cached in g:
                    del g[cached]
    return result
