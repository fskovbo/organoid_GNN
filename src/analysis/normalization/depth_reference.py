"""Normalize paired effects by a held-out, center-only curvature predictor.

Reference predictions use their own saved input scaling and inverse target
transform. Saved residualization offsets restore physical curvature exactly;
their observed-organoid values remain fixed during counterfactual N sweeps.
"""
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from src.analysis.interventions.perturbation import predict_subgraph_center_distribution
from src.analysis.interventions.size_sweeps import _size_override


def validate_depth_reference(selection, reference):
    """Require paired folds and identical observed fates, without geometry inputs."""
    if getattr(reference['model'], 'num_layers', None) != 0:
        raise ValueError('The normalization reference must have depth 0.')
    if reference['global_features'] != ['log_num_cells']:
        raise ValueError('A center-fate/N reference must receive only log_num_cells as its global input.')
    if list(reference['marker_names']) != list(selection['marker_names']):
        raise ValueError('Reference marker names/order differ from the ablation model.')
    k = len(reference['marker_names'])
    for role in ('train','val'):
        original = {str(g.organoid_str):g for g in selection['groups'][role]}
        restored = {str(g.organoid_str):g for g in reference['groups'][role]}
        if original.keys() != restored.keys():
            raise ValueError(f'Reference {role} organoids must match the ablation fold exactly.')
        for org,g in restored.items():
            other = original[org]
            if g.x.shape[1] not in (k,k+1) or other.x.shape[1] not in (k,k+1):
                raise ValueError('Unrecognized reference fate encoding.')
            if not torch.equal(g.x[:,:k].cpu(),other.x[:,:k].cpu()) or not torch.equal(g.edge_index.cpu(),other.edge_index.cpu()):
                raise ValueError(f'Reference fates or node ordering differ for {org}.')
            if (g.x.shape[1] == k+1 and torch.any(g.x[:,-1] != 0)) or (other.x.shape[1] == k+1 and torch.any(other.x[:,-1] != 0)):
                raise ValueError('Reference evaluation requires fully observed center fates.')
    if 'baseline_offsets' not in reference:
        raise ValueError('Saved residualization offsets are required, including zeros for nonresidualized models.')


def _offset(reference, org, node):
    values = np.asarray(reference['baseline_offsets'][org]).reshape(-1)
    return float(values[0] if len(values)==1 else values[node])


def predict_depth_reference(reference, requests, *, role='val', device='cpu', batch_size=512):
    """Predict unique (organoid, center, supplied N) requests without neighbors.

    Requests include evaluated_n. It is also explicitly supplied for observed
    cases, using the reference fold's own log-N scaling. The returned physical
    mean is the inverse-transformed predicted mean plus its saved offset.
    """
    if getattr(reference['model'],'num_layers',None) != 0:
        raise ValueError('A depth-0 model is required.')
    keys = ['organoid_str','orig_center','evaluated_n']
    result = requests[keys].drop_duplicates().reset_index(drop=True).copy()
    graphs = {str(g.organoid_str):g for g in reference['groups'][role]}
    j = reference['all_global_features'].index('log_num_cells')
    column = reference['global_features'].index('log_num_cells')
    singletons, offsets = [], []
    for row in result.itertuples():
        graph = graphs[row.organoid_str]
        node = int(row.orig_center)
        if not 0 <= node < len(graph.x):
            raise ValueError('Reference node index is outside the saved organoid.')
        singleton = Data(x=graph.x[node:node+1].clone(),edge_index=torch.empty((2,0),dtype=torch.long),
            center_idx=0,global_feat=graph.global_feat.clone())
        singleton = _size_override(singleton,float(row.evaluated_n),
            reference['global_center'][j],reference['global_scale'][j],column)
        singletons.append(singleton)
        offsets.append(_offset(reference,row.organoid_str,node))
    if not singletons:
        for column in ('depth0_residual','depth0_baseline','depth0_curvature'):
            result[column] = pd.Series(dtype=float)
        return result
    z,_ = predict_subgraph_center_distribution(singletons,reference['model'],device=device,batch_size=batch_size,
                                               pin_memory=str(device).startswith('cuda'))
    result['depth0_residual'] = np.asarray(reference['transform'].inverse(z)).reshape(-1)
    result['depth0_baseline'] = offsets
    result['depth0_curvature'] = result.depth0_residual + result.depth0_baseline
    return result


def calibrate_depth_reference(reference, *, fraction=.05, min_abs=0., device='cpu', batch_size=512):
    """Near-zero cutoff from train-only predicted |K0|, with equal organoid weight.

    For each organoid, average absolute reference predictions over its observed
    cells; use the median of those organoid means as the reference magnitude.
    Predict each unique center fate once per organoid (depth 0 has no neighbors).
    Exclude small denominators rather than replacing them with an epsilon.
    """
    if not np.isfinite(fraction) or not 0 < fraction < 1 or not np.isfinite(min_abs) or min_abs < 0:
        raise ValueError('Use 0 < fraction < 1 and a nonnegative absolute cutoff.')
    requests = []
    for graph in reference['groups']['train']:
        _,indices,counts = np.unique(graph.x.cpu().numpy(),axis=0,return_index=True,return_counts=True)
        offsets = np.asarray(reference['baseline_offsets'][str(graph.organoid_str)]).reshape(-1)
        if len(offsets)>1 and not np.allclose(offsets,offsets[0],rtol=0,atol=1e-6):
            raise ValueError('Expected an organoid-constant residualization baseline.')
        requests.extend(dict(organoid_str=str(graph.organoid_str),orig_center=int(node),
            evaluated_n=len(graph.x),n_cells=int(count)) for node,count in zip(indices,counts))
    requests = pd.DataFrame(requests)
    prediction = predict_depth_reference(reference,requests,role='train',device=device,batch_size=batch_size)
    prediction = prediction.merge(requests,on=['organoid_str','orig_center','evaluated_n'],validate='one_to_one')
    if not np.isfinite(prediction.depth0_curvature).all():
        raise ValueError('Nonfinite training reference predictions.')
    weighted = prediction.assign(magnitude=prediction.depth0_curvature.abs()*prediction.n_cells)
    organoids = weighted.groupby('organoid_str').agg(magnitude=('magnitude','sum'),cells=('n_cells','sum'))
    typical = float((organoids.magnitude/organoids.cells).median())
    threshold = max(float(min_abs),fraction*typical)
    if not threshold > 0:
        raise ValueError('Depth-0 reference magnitude is zero; cannot define a relative effect.')
    return dict(threshold=threshold,fraction=fraction,min_abs=min_abs,
        typical_train_abs_curvature=typical,n_train_organoids=len(organoids))


def add_depth_reference(frame, reference, *, calibration, device='cpu', batch_size=512):
    """Append signed reference values and ΔK/|K0|; keep exclusions explicit."""
    predictions = predict_depth_reference(reference,frame,device=device,batch_size=batch_size)
    result = frame.merge(predictions,on=['organoid_str','orig_center','evaluated_n'],how='left',validate='many_to_one')
    magnitude = result.depth0_curvature.abs()
    valid = np.isfinite(magnitude) & (magnitude >= calibration['threshold'])
    result['depth0_reference_threshold'] = calibration['threshold']
    result['depth0_valid'] = valid
    result['depth0_exclusion'] = np.where(~np.isfinite(magnitude),'nonfinite',np.where(valid,'','near_zero'))
    result['delta_depth0_relative'] = result.delta_mu / magnitude.where(valid)
    return result
