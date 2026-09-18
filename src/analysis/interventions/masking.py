"""Missing-fate FiLM training and out-of-fold prediction benchmarks.

A final input flag distinguishes observed all-negative cells from hidden fate.
Masks are Bernoulli draws independent of fate, N, targets and geometry. The
original seven-channel data and target preprocessing are never modified.
"""
import copy
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import pickle
import shutil

import numpy as np
import pandas as pd
from scipy.stats import norm
import torch
from torch import nn
from torch_geometric.loader import DataLoader

from src.analysis.size_conditioning.cohort_inputs import make_model, seed_all, load_cohort, fold_graphs
from src.analysis.interventions.replacement import identity_codes, make_cases, _write_csv, UNASSIGNED
from src.analysis.interventions.size_sweeps import _size_override
from src.analysis.interventions.perturbation import predict_subgraph_center_distribution
from src.data.metadata import load_marker_names_from_dir
from src.data.subgraphs import build_ego_subgraphs_for_graph
from src.graph.neighborhood import compute_hop_rings
from src.inference.predict import predict_targets
from src.training.losses import CompositeLoss, make_base_loss, WeightedLossTerm, edge_loss_term


def rate_tag(rate):
    return f'p{float(rate):g}'


def checkpoint_path(out, fold, seed, rate):
    return Path(out) / 'checkpoints' / f'fold_{fold}_seed_{seed}_{rate_tag(rate)}.pt'


def load_mask_checkpoint(out, fold, seed, rate, settings, markers):
    model = make_mask_model(settings, markers)
    checkpoint = torch.load(checkpoint_path(out, fold, seed, rate), map_location='cpu', weights_only=True)
    if (checkpoint['fold'], checkpoint['seed'], checkpoint['rate']) != (fold, seed, rate):
        raise ValueError('Checkpoint metadata does not match the requested model.')
    model.load_state_dict(checkpoint['model_state'])
    return model


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def checked_settings(path, config):
    path = Path(path)
    # Normalize tuples and numpy-free JSON types before comparing on resume.
    config = json.loads(json.dumps(config))
    if path.exists() and json.loads(path.read_text()) != config:
        raise ValueError(f'Settings or dependencies changed: use a new output directory ({path}).')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=2))


def score_arrays(truth_z, mu_z, lv_z, transform):
    truth_z, mu_z, lv_z = [np.asarray(a).reshape(-1) for a in (truth_z, mu_z, lv_z)]
    if not all(np.isfinite(a).all() for a in (truth_z, mu_z, lv_z)):
        raise FloatingPointError('Non-finite benchmark predictions or targets.')
    truth = np.asarray(transform.inverse(truth_z)).reshape(-1)
    mu = np.asarray(transform.inverse(mu_z)).reshape(-1)
    error_z = mu_z - truth_z
    standardized = error_z / np.exp(.5 * lv_z)
    values = dict(mse_z=error_z**2, mse_mu=(mu-truth)**2, bias_z=error_z,
                  bias_mu=mu-truth, nll_z=.5*(np.log(2*np.pi)+lv_z+standardized**2),
                  standardized_error=standardized, standardized_error2=standardized**2)
    for level in (.5, .8, .95):
        values[f'coverage_{int(level*100)}'] = (np.abs(standardized) <= norm.ppf((1+level)/2)).astype(float)
    return values


def benchmark_intact(graphs, model, transform, markers, *, batch_size=128, device='cpu'):
    y, mu, lv, _ = predict_targets(graphs, ObservedFateAdapter(model), device=device,
        batch_size=batch_size, return_log_var=True, pin_memory=str(device).startswith('cuda'))
    scores = score_arrays(y, mu, lv, transform)
    rows = []; offset = 0; names = list(markers)+[UNASSIGNED]
    for g in graphs:
        labels = identity_codes(g.x); n = len(labels)
        for ident in [None, *range(len(names))]:
            mask = np.ones(n, dtype=bool) if ident is None else labels == ident
            if not mask.any():
                continue
            rows.append(dict(organoid_str=g.organoid_str, observed_n=n,
                center_marker='All' if ident is None else names[ident], n_nodes=int(mask.sum()),
                **{key: float(value[offset:offset+n][mask].mean()) for key,value in scores.items()}))
        offset += n
    return pd.DataFrame(rows)


def evaluate_single_mask(subgraphs, cases, model, transform, *, size_center, size_scale,
                         count=None, batch_size=128, device='cpu', include_zero=True):
    """Paired intact/missing (and zero) predictions of the SAME trained model."""
    if cases.empty:
        raise ValueError('No mask cases.')
    raw = [_size_override(g, count, size_center, size_scale) for g in subgraphs]
    base_z, base_lv = predict_subgraph_center_distribution(raw, ObservedFateAdapter(model),
                                                           device=device, batch_size=batch_size)
    idx = cases.subgraph_index.to_numpy(int)
    result = cases.copy()
    result['analysis'] = 'observed' if count is None else 'sweep'
    result['evaluated_n'] = result.observed_n if count is None else float(count)
    result['intact_z'], result['intact_lv'] = base_z[idx], base_lv[idx]
    result['intact_mu'] = np.asarray(transform.inverse(base_z[idx])).reshape(-1)
    for edit in (['mask', 'zero'] if include_zero else ['mask']):
        chunks_z, chunks_lv = [], []
        records = cases.to_dict('records')
        for start in range(0, len(records), batch_size):
            changed = []
            for case in records[start:start+batch_size]:
                original = raw[case['subgraph_index']]
                source = case['source_node']
                if source == int(original.center_idx):
                    raise ValueError('Keep recipient identity observed.')
                if identity_codes(original.x[source:source+1])[0] != case['source_identity']:
                    raise ValueError('Case/source identity mismatch.')
                g = copy.copy(original)
                if edit == 'mask':
                    mask = torch.zeros(len(g.x), dtype=torch.bool); mask[source] = True
                    g.x = encode_fates(original.x, mask)
                else:
                    g.x = encode_fates(original.x); g.x[source, :-1] = 0
                changed.append(g)
            z, lv = predict_subgraph_center_distribution(changed, model, device=device, batch_size=batch_size)
            chunks_z.append(z); chunks_lv.append(lv)
        z, lv = np.concatenate(chunks_z), np.concatenate(chunks_lv)
        result[f'{edit}_z'], result[f'{edit}_lv'] = z, lv
        result[f'{edit}_mu'] = np.asarray(transform.inverse(z)).reshape(-1)
        result[f'{edit}_delta_z'] = z - result.intact_z
        result[f'{edit}_delta_mu'] = result[f'{edit}_mu'] - result.intact_mu
    if count is None:
        truth = np.array([float(raw[c['subgraph_index']].y[int(raw[c['subgraph_index']].center_idx)].reshape(-1)[0])
                          for c in cases.to_dict('records')])
        result['truth_z'] = truth
        for state in ['intact', 'mask']:
            for key, value in score_arrays(truth, result[f'{state}_z'], result[f'{state}_lv'], transform).items():
                result[f'{state}_{key}'] = value
        result['delta_mse_z'] = result.mask_mse_z - result.intact_mse_z
        result['delta_mse_mu'] = result.mask_mse_mu - result.intact_mse_mu
    return result


def make_quality_cases(graphs, markers, *, centers_per_organoid=16, seed=1701, receptive_hops=2):
    """Fate-independent center/source sampling for marginal error and calibration.

    Stratified identity cases are useful for pair effects, but would change the
    hidden-identity prior if pooled unweighted for predictive calibration.
    """
    if centers_per_organoid < 1:
        raise ValueError('Need at least one quality center per organoid.')
    if not isinstance(receptive_hops, int) or receptive_hops < 2:
        raise ValueError('Receptive field must include both sampled source hops.')
    rng=np.random.default_rng(seed); names=list(markers)+[UNASSIGNED]
    subs=[]; rows=[]
    for graph in graphs:
        labels=identity_codes(graph.x)
        for sub in build_ego_subgraphs_for_graph(graph,num_hops=receptive_hops,
                max_centers=centers_per_organoid,rng=rng):
            si=len(subs);subs.append(sub);center=int(sub.orig_center)
            rings=compute_hop_rings(sub.edge_index,int(sub.center_idx),2)
            for hop in (1,2):
                if not rings[hop]: continue
                source=int(rng.choice(rings[hop]));original=int(sub.orig_nodes[source])
                rows.append(dict(case_id=len(rows),subgraph_index=si,organoid_str=graph.organoid_str,
                    orig_center=center,orig_source_node=original,source_node=source,
                    center_identity=int(labels[center]),center_marker=names[labels[center]],
                    source_identity=int(labels[original]),source_marker_name=names[labels[original]],
                    hop=hop,observed_n=len(graph.x)))
    return subs,pd.DataFrame(rows)


def load_mask_benchmarks(output_dir):
    out=Path(output_dir)
    if not (out/'benchmarks/complete.json').exists():
        raise RuntimeError('Complete the masking benchmarks before loading results.')
    cfg=json.loads((out/'settings.json').read_text())
    intact,masked,pairs=[],[],[]
    for f in cfg['folds']:
        for s in cfg['seeds']:
            for p in cfg['training']['rates']:
                stem=out/'benchmarks'/f'fold_{f}_seed_{s}_{rate_tag(p)}'
                intact.append(pd.read_csv(str(stem)+'_intact.csv.gz'))
                if p>0:
                    masked.append(pd.read_csv(str(stem)+'_masked.csv.gz'))
                    pairs.append(pd.read_csv(str(stem)+'_pair_effects.csv.gz'))
    intact=pd.concat(intact,ignore_index=True)
    masked=pd.concat(masked,ignore_index=True) if masked else pd.DataFrame()
    pairs=pd.concat(pairs,ignore_index=True) if pairs else pd.DataFrame()
    return dict(intact=intact,masked=masked,pair_effects=pairs,config=cfg)

from src.data.fate_masking import encode_fates, random_mask, ObservedFateAdapter, _forward_with_mask

from src.training.masking import MaskTrainingConfig, inner_split, _loss, intact_validation_mse, train_mask_model

from src.models.fate_masking import make_mask_model
