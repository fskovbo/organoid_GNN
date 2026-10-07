"""Reusable fold preparation; experiment grids and training loops stay in notebooks."""
import copy
import numpy as np
import torch
from torch_geometric.data import Data
from src.data.target_transforms import AsinhStandardizeTransform, IdentityTransform, GlobalBaselineResidualTransform, standardize_graph_global_features


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
                 baseline_kwargs=None, target_scaling='asinh'):
    """Always fit a global baseline; optionally subtract it before target scaling.

    Returns compact model inputs and fitted objects. Raw metadata/geometry stay
    in the saved cohort, not in PyG batches. Baseline offsets are saved per graph
    so predictions can be reconstructed without passing geometry to the head.
    All fitted preprocessing and baseline optimization use training organoids only.
    Baseline predictions are always retained; reconstruction offsets are zero
    unless residualization was requested.
    ``target_scaling='identity'`` retains double-precision physical residuals
    for additive models; the default asinh path is unchanged.
    """
    if target_scaling not in ('asinh', 'identity'):
        raise ValueError('target_scaling must be asinh or identity.')
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
                    mse=float(np.mean((truth-prediction)**2)),
                    mae=float(np.mean(np.abs(truth-prediction)))))
            # Offsets describe target residualization, not whether a baseline exists.
            if target_scaling == 'identity':
                offsets[g.organoid_str] = prediction.copy() if residualize else np.zeros(len(g.x))
                g.y = g.y.to(torch.float64) - torch.as_tensor(
                    offsets[g.organoid_str], dtype=torch.float64).reshape(g.y.shape)
            else:
                offsets[g.organoid_str] = ((g.y - b.y).cpu().numpy() if residualize
                                          else np.zeros(len(g.x)))
                if residualize:
                    g.y = b.y.clone()
            start = stop
    baseline.model.cpu()
    baseline.prediction_cache.clear()
    transform = (AsinhStandardizeTransform(robust=True) if target_scaling == 'asinh'
                 else IdentityTransform()).fit(groups['train'])
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


def physical_fate_fold(source):
    """Restore an exclusive observed fold to physical (possibly residual) targets.

    Reuses the saved membership, target cleanup and baseline without fitting.
    Removes only all-zero missingness channels. Source inputs are not mutated.
    A spatially constant saved baseline is required so residual-space coupling
    is equivalent to physical-curvature coupling plus the same baseline.
    """
    from src.data.neighborhood_counts import fate_identities
    markers=list(source['marker_names'])
    if source.get('baseline') is None or source.get('baseline_predictions') is None:
        raise ValueError('A fitted saved baseline and predictions are required.')
    result=copy.deepcopy(source)
    for group in result['groups'].values():
        for g in group:
            if g.x.shape[1]<len(markers) or torch.any(g.x[:,len(markers):]!=0):
                raise ValueError('Expected all named fates and no masked source cells.')
            g.x=g.x[:,:len(markers)].clone()
            fate_identities(g.x)
            oid=str(g.organoid_str)
            old_offset=np.asarray(source['baseline_offsets'][oid]).reshape(-1)
            physical=np.asarray(source['transform'].inverse(g.y.cpu().numpy()),dtype=float).reshape(-1)+old_offset
            prediction=np.asarray(source['baseline_predictions'][oid],dtype=float).reshape(-1)
            if len(old_offset)!=len(g.x) or len(prediction)!=len(g.x):
                raise ValueError('Saved baseline arrays must match the organoid cells.')
            if not np.isfinite(physical).all() or not np.isfinite(prediction).all():
                raise ValueError('Finite restored targets and baseline predictions required.')
            if not np.allclose(prediction,prediction[0],rtol=1e-6,atol=1e-12):
                raise ValueError('Coupling requires a constant baseline prediction per organoid.')
            constant=float(prediction.mean()) if source['residualized'] else 0.
            # Historical offsets were computed as y - (y - baseline) in float32.
            # Preserve restored targets while removing only that subtraction noise.
            tolerance=8*np.finfo(np.float32).eps*max(np.max(np.abs(physical)),abs(constant),1e-12)
            if not np.allclose(old_offset,constant,rtol=0,atol=tolerance+1e-12):
                raise ValueError('Saved offsets disagree with the constant baseline beyond rounding error.')
            offset=np.full(len(g.x),constant,dtype=float)
            result['baseline_offsets'][oid]=offset
            g.y=torch.tensor(physical-offset,dtype=torch.float64)
            g.full_num_cells=float(len(g.x))
            for name in ('global_feat','x_ring','ring_sizes'):
                if name in g:del g[name]
    result['transform']=IdentityTransform().fit(result['groups']['train'])
    result['source_global_features']=source.get('global_features',[])
    result['global_features']=[]
    if 'inner_membership' in result:result['source_inner_membership']=result.pop('inner_membership')
    result['input_encoding']='exclusive observed fates; Unassigned is the all-zero row'
    return result


def combine_target_folds(folds,target_indices):
    """Join separately fitted scalar targets on identical ordered graphs.

    Baselines/transforms remain available by target; reconstructed y is always
    physical residual curvature, with no nonlinear target transform.
    """
    target_indices=list(target_indices);selected=[folds[t] for t in target_indices]
    result=copy.deepcopy(selected[0]);result['target_indices']=target_indices
    result['baselines_by_target']={t:f['baseline'] for t,f in zip(target_indices,selected)}
    result['preprocessing_by_target']={t:{k:v for k,v in f.items() if k not in ('groups','baseline')} for t,f in zip(target_indices,selected)}
    result['baseline_offsets']={};result['baseline_predictions']={}
    for role in ('train','val'):
        for index,g in enumerate(result['groups'][role]):
            oid=str(g.organoid_str);targets=[];offsets=[];pred=[]
            for f in selected:
                other=f['groups'][role][index]
                if str(other.organoid_str)!=oid or not torch.equal(g.x,other.x) or not torch.equal(g.edge_index,other.edge_index):raise ValueError('Targets have different graph membership/order.')
                targets.append(other.y.reshape(-1));offsets.append(f['baseline_offsets'][oid]);pred.append(f['baseline_predictions'][oid])
            g.y=torch.stack(targets,dim=1)
            result['baseline_offsets'][oid]=np.column_stack(offsets)
            result['baseline_predictions'][oid]=np.column_stack(pred)
    result['transform']=IdentityTransform().fit(result['groups']['train'])
    return result


def load_matching_fold(reference_run, settings, graphs, membership, fold):
    """Reuse fitted fold preprocessing only after checking cohort/settings equality.

    A variant may change its marker panel, architecture or optimizer, but cannot
    silently change the cohort, target cleanup, globals or baseline definition.
    Returned groups still contain the complete observed fate panel.
    """
    import json
    from pathlib import Path
    from src.artifacts.runs import AnalysisRun
    run = AnalysisRun(Path(reference_run))
    keys = ('dataset', 'target_indices', 'timepoints', 'blacklist', 'sphericity_max',
            'complexity_min', 'interpolate_outliers', 'outlier_quantiles',
            'exclusive_markers', 'global_features', 'residualize', 'baseline_features',
            'baseline_hidden_dim', 'baseline_max_epochs', 'baseline_patience',
            'split_seed', 'n_folds', 'val_fraction')
    differences = {k: (settings.get(k), run.settings.get(k)) for k in keys
                   if settings.get(k) != run.settings.get(k)}
    if differences:
        raise ValueError(f'Reference preprocessing settings differ: {differences}')
    if json.loads((run.directory/'splits.json').read_text()) != membership:
        raise ValueError('Reference train/validation membership differs.')
    result = run.fold_inputs(fold)
    raw = result['raw_graphs']
    if set(raw) != {str(g.organoid_str) for g in graphs}:
        raise ValueError('Reference cohort differs.')
    for g in graphs:
        other = raw[str(g.organoid_str)]
        if any(not torch.equal(getattr(g, key), getattr(other, key)) for key in ('x', 'y', 'edge_index')):
            raise ValueError(f'Reference fate/target/edge mismatch: {g.organoid_str}')
    # Return the same schema as prepare_fold, without the raw cohort or model metadata.
    keep = ('groups', 'transform', 'baseline', 'baseline_offsets', 'baseline_predictions',
            'baseline_validation_mse', 'baseline_features', 'residualized',
            'global_features', 'all_global_features', 'global_center', 'global_scale')
    return {key: result[key] for key in keep}
