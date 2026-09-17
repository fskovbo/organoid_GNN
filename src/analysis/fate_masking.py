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

from src.analysis.exclusive_size_ablation import make_model, seed_all, load_cohort, fold_graphs
from src.analysis.replacement_ablation import identity_codes, make_cases, _write_csv, UNASSIGNED
from src.analysis.pseudotime import _size_override
from src.analysis.perturbation import predict_subgraph_center_distribution
from src.data.metadata import load_marker_names_from_dir
from src.data.subgraphs import build_ego_subgraphs_for_graph
from src.graph.neighborhood import compute_hop_rings
from src.inference.predict import predict_targets
from src.training.losses import CompositeLoss, make_base_loss, WeightedLossTerm, edge_loss_term


@dataclass(frozen=True)
class MaskTrainingConfig:
    rates: tuple = (0., .01, .02, .05)
    masked_loss_weight: float = .5
    inner_val_fraction: float = .15
    inner_split_seed: int = 8123
    max_epochs: int = 500
    patience: int = 30
    batch_size: int = 64
    num_workers: int = 0

    def __post_init__(self):
        if not self.rates or len(set(self.rates)) != len(self.rates) or any(not 0 <= p < 1 for p in self.rates):
            raise ValueError('Masking rates must be distinct and in [0, 1).')
        if 0. not in self.rates:
            raise ValueError('Include rate 0 as the matched unmasked control.')
        if not any(p > 0 for p in self.rates):
            raise ValueError('Include at least one positive masking rate.')
        if not 0 < self.masked_loss_weight < 1 or not 0 < self.inner_val_fraction < .5:
            raise ValueError('Invalid loss weight or inner validation fraction.')
        if min(self.max_epochs, self.patience, self.batch_size) < 1:
            raise ValueError('Epochs, patience and batch size must be positive.')


def rate_tag(rate):
    return f'p{float(rate):g}'


def encode_fates(x, mask=None):
    """Return a new [fates, missing] matrix; erase ALL fate bits on masked nodes."""
    if x.ndim != 2:
        raise ValueError('Expected a node-by-marker matrix.')
    if mask is None:
        mask = torch.zeros(len(x), dtype=torch.bool, device=x.device)
    else:
        mask = torch.as_tensor(mask, dtype=torch.bool, device=x.device)
    if mask.shape != (len(x),):
        raise ValueError('Mask must contain one boolean per node.')
    result = torch.cat([x.clone(), mask[:, None].to(x.dtype)], dim=1)
    result[mask, :-1] = 0
    return result


def random_mask(n_nodes, rate, generator):
    """CPU RNG independent of model/dropout RNG; rates share nested random draws."""
    if not 0 <= rate < 1:
        raise ValueError('Masking probability must be in [0, 1).')
    return torch.rand(n_nodes, generator=generator) < rate


class ObservedFateAdapter(nn.Module):
    """Use an eight-input network with existing seven-input inference utilities."""
    def __init__(self, network):
        super().__init__()
        self.network = network

    def forward(self, x, edge_index, data=None):
        encoded = encode_fates(x)
        view = copy.copy(data) if data is not None else None
        if view is not None:
            view.x = encoded
        return self.network(encoded, edge_index, data=view)


def _forward_with_mask(model, batch, mask=None):
    # Erase fate in both the explicit x argument and data.x. No original source
    # identity remains accessible to the network through the batch container.
    view = copy.copy(batch)
    view.x = encode_fates(batch.x, mask)
    return model(view.x, view.edge_index, data=view)


def make_mask_model(settings, markers, seed=None):
    """Same base initialization at each rate; new mask columns start at zero."""
    model = make_model(settings, list(markers) + ['fate_missing'])
    if seed is None:
        return model
    original = make_model(settings, markers, seed=seed)
    state = model.state_dict()
    for key, value in original.state_dict().items():
        if state[key].shape == value.shape:
            state[key] = value.clone()
        elif key in ('convs.0.nn.0.weight', 'input_proj.weight'):
            if state[key].shape != (value.shape[0], value.shape[1] + 1):
                raise ValueError(f'Unexpected mask-input shape for {key}')
            state[key].zero_(); state[key][:, :-1] = value
        else:
            raise ValueError(f'Unexpected initialization mismatch: {key}')
    model.load_state_dict(state)
    return model


def inner_split(n_graphs, fraction, seed):
    if n_graphs < 3:
        raise ValueError('Need at least three outer-training organoids.')
    order = np.random.default_rng(seed).permutation(n_graphs)
    n = min(n_graphs - 1, max(1, int(np.ceil(n_graphs * fraction))))
    return order[n:].tolist(), order[:n].tolist()


def _loss(settings):
    return CompositeLoss(make_base_loss(), [WeightedLossTerm(
        name='edge', fn=edge_loss_term, weight=settings['EDGE_LOSS_WEIGHT'],
        params=settings['EDGE_LOSS_PARAMS'])])


@torch.no_grad()
def intact_validation_mse(model, loader, device):
    model.eval()
    total, n = 0., 0
    for batch in loader:
        batch = batch.to(device)
        (mu, _), _ = _forward_with_mask(model, batch)
        errors = (mu.reshape(-1) - batch.y.reshape(-1)).square()
        total += float(errors.sum()); n += len(errors)
    return total / n


def train_mask_model(model, train_graphs, early_graphs, settings, config, *, rate, seed, device):
    """Matched optimizer steps/BN passes even for rate 0; select on intact inner MSE.

    Backpropagate the two weighted passes separately before one optimizer step,
    bounding activation memory. Validation is always eval-mode and unmasked.
    """
    seed_all(seed)
    loader_rng = torch.Generator().manual_seed(seed + 1000)
    mask_rng = torch.Generator().manual_seed(seed + 2000)
    loader = DataLoader(train_graphs, batch_size=config.batch_size, shuffle=True,
        generator=loader_rng, num_workers=config.num_workers, pin_memory=str(device).startswith('cuda'))
    early = DataLoader(early_graphs, batch_size=config.batch_size, shuffle=False, num_workers=0)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings['LR'], weight_decay=settings['WEIGHT_DECAY'])
    loss_fn = _loss(settings)
    best, best_state, remaining, history = np.inf, None, config.patience, []
    for epoch in range(1, config.max_epochs + 1):
        model.train(); total_loss = 0.; total_nodes = 0; masked_nodes = 0
        for batch in loader:
            batch = batch.to(device)
            # All targets remain unchanged; flatten a single target to avoid
            # accidental broadcasting in the repository's auxiliary edge loss.
            batch.y = batch.y.reshape(-1)
            mask = random_mask(len(batch.x), rate, mask_rng).to(batch.x.device)
            optimizer.zero_grad(set_to_none=True)
            batch_loss = 0.
            for selected, weight in [(None, 1 - config.masked_loss_weight), (mask, config.masked_loss_weight)]:
                (mu, lv), _ = _forward_with_mask(model, batch, selected)
                loss = loss_fn(mu=mu, log_scale2=lv, batch=batch)
                if not torch.isfinite(loss):
                    raise FloatingPointError('Non-finite masking training loss.')
                (weight * loss).backward()
                batch_loss += weight * float(loss.detach())
            nn.utils.clip_grad_norm_(model.parameters(), 2.)
            optimizer.step()
            total_loss += batch_loss * len(batch.x); total_nodes += len(batch.x)
            masked_nodes += int(mask.sum())
        score = intact_validation_mse(model, early, device)
        if not np.isfinite(score):
            raise FloatingPointError('Non-finite early-stopping score.')
        history.append(dict(epoch=epoch, train_loss=total_loss/total_nodes, inner_intact_mse_z=score,
                            realized_mask_rate=masked_nodes/total_nodes))
        print(f'{rate_tag(rate)} seed={seed} epoch={epoch}: loss={history[-1]["train_loss"]:.5g}, '
              f'inner intact MSE={score:.5g}', flush=True)
        if score < best - 1e-7:
            best = score; remaining = config.patience
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            remaining -= 1
            if remaining <= 0:
                break
    model.load_state_dict(best_state)
    return model, pd.DataFrame(history)


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


def train_masking_experiment(root, reference_run, output_dir, *, config=MaskTrainingConfig(),
                             folds=None, seeds=None, device=None):
    root, reference, out = map(lambda p: Path(p).resolve(), (root, reference_run, output_dir))
    settings = json.loads((reference / 'settings.json').read_text())
    if settings.get('FEATURE_ENCODING') != 'exclusive_ordered' or settings['NUM_LAYERS'] != 2:
        raise ValueError('Expected depth-two exclusive models.')
    if settings['MODEL_GLOBAL_FEATURES']['gin_film_size'] != ['log_num_cells']:
        raise ValueError('Expected log(N) only, without geometry in the head.')
    splits = json.loads((reference / 'splits.json').read_text())
    folds = list(folds if folds is not None else [s['fold'] for s in splits])
    seeds = list(seeds if seeds is not None else settings['MODEL_SEEDS'])
    if (not folds or not seeds or len(set(folds)) != len(folds) or len(set(seeds)) != len(seeds)
            or not set(folds) <= {s['fold'] for s in splits}):
        raise ValueError('Invalid fold/seed selection.')
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = root / 'training_data' / settings['DATASET_NAME']
    markers = load_marker_names_from_dir(str(dataset))
    artifacts = ['settings.json', 'splits.json', 'tables/cohort.csv', 'geometric_normalization/references.csv']
    artifacts += [f'checkpoints/fold_{fold}_preprocessing.pkl' for fold in folds]
    dependencies = {name: _sha(reference / name) for name in artifacts}
    data_stats = [(p.name, p.stat().st_size, p.stat().st_mtime_ns) for p in sorted(dataset.iterdir())]
    checked_settings(out / 'settings.json', dict(reference_run=str(reference), markers=markers, folds=folds,
        seeds=seeds, training=asdict(config), model_settings=settings, dependencies=dependencies,
        data_stat_sha256=hashlib.sha256(json.dumps(data_stats).encode()).hexdigest(), code_sha256=_sha(__file__)))
    for folder in ('checkpoints', 'history', 'benchmarks', 'figures'):
        (out / folder).mkdir(exist_ok=True)
    for name in artifacts:
        target = out / 'reference' / name
        target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(reference / name, target)
    graphs = load_cohort(dataset, settings, reference)
    for g in graphs:
        identity_codes(g.x)
    for split in splits:
        fold = split['fold']
        if fold not in folds:
            continue
        with (reference / f'checkpoints/fold_{fold}_preprocessing.pkl').open('rb') as f:
            pre = pickle.load(f)
        groups = fold_graphs(graphs, split, pre, settings)
        fit_ids, early_ids = inner_split(len(groups['train']), config.inner_val_fraction, config.inner_split_seed + fold)
        fit = [groups['train'][i] for i in fit_ids]; early = [groups['train'][i] for i in early_ids]
        membership = dict(fit=[g.organoid_str for g in fit], early_stopping=[g.organoid_str for g in early],
                          benchmark=[g.organoid_str for g in groups['val']])
        assert not (set(membership['fit']) & set(membership['early_stopping']))
        assert not ((set(membership['fit']) | set(membership['early_stopping'])) & set(membership['benchmark']))
        (out / f'fold_{fold}_membership.json').write_text(json.dumps(membership, indent=2))
        for rate in config.rates:
            for seed in seeds:
                path = checkpoint_path(out, fold, seed, rate)
                if path.exists():
                    print(f'Reuse completed checkpoint: {path.name}', flush=True); continue
                print(f'Train fold={fold}, seed={seed}, masking rate={rate:g}', flush=True)
                model = make_mask_model(settings, markers, seed=seed)
                model, history = train_mask_model(model, fit, early, settings, config,
                                                  rate=rate, seed=seed, device=device)
                payload = dict(model_state={k:v.detach().cpu() for k,v in model.state_dict().items()},
                               history=history.to_dict('records'), rate=float(rate), fold=fold, seed=seed)
                temp = path.with_suffix('.tmp'); torch.save(payload, temp); temp.replace(path)
                history.to_csv(out / 'history' / f'{path.stem}.csv', index=False)
                del model
    (out / 'training_complete.json').write_text(json.dumps(dict(folds=folds, seeds=seeds, rates=list(config.rates))))
    return out


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


def make_quality_cases(graphs, markers, *, centers_per_organoid=16, seed=1701):
    """Fate-independent center/source sampling for marginal error and calibration.

    Stratified identity cases are useful for pair effects, but would change the
    hidden-identity prior if pooled unweighted for predictive calibration.
    """
    if centers_per_organoid < 1:
        raise ValueError('Need at least one quality center per organoid.')
    rng=np.random.default_rng(seed); names=list(markers)+[UNASSIGNED]
    subs=[]; rows=[]
    for graph in graphs:
        labels=identity_codes(graph.x)
        for sub in build_ego_subgraphs_for_graph(graph,num_hops=2,
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


def benchmark_masking_experiment(root, output_dir, *, centers_per_identity=2, sampling_seed=1701,
                                 quality_centers_per_organoid=16, batch_size=128, device=None):
    """Score full intact validation graphs and exactly-one-neighbor masking cases."""
    root, out = Path(root).resolve(), Path(output_dir).resolve()
    if not (out / 'training_complete.json').exists():
        raise RuntimeError('Complete model training before the benchmark.')
    cfg = json.loads((out / 'settings.json').read_text()); settings=cfg['model_settings']; markers=cfg['markers']
    reference = out / 'reference'; device=device or ('cuda' if torch.cuda.is_available() else 'cpu')
    paths = [checkpoint_path(out,f,s,p) for f in cfg['folds'] for s in cfg['seeds'] for p in cfg['training']['rates']]
    checked_settings(out / 'benchmarks/settings.json', dict(centers_per_identity=centers_per_identity,
        quality_centers_per_organoid=quality_centers_per_organoid,
        sampling_seed=sampling_seed, batch_size=batch_size, code_sha256=_sha(__file__),
        training_settings_sha256=_sha(out/'settings.json'), checkpoints={p.name:_sha(p) for p in paths}))
    graphs=load_cohort(root/'training_data'/settings['DATASET_NAME'], settings, reference)
    refs=pd.read_csv(reference/'geometric_normalization/references.csv').set_index('fold')
    for split in json.loads((reference/'splits.json').read_text()):
        fold=split['fold']
        if fold not in cfg['folds']: continue
        with (reference/f'checkpoints/fold_{fold}_preprocessing.pkl').open('rb') as f: pre=pickle.load(f)
        val=fold_graphs(graphs,split,pre,settings)['val']
        subs,cases,_=make_cases(val,markers,centers_per_identity=centers_per_identity,
                                sweep_centers_per_identity=1,seed=sampling_seed+fold)
        _write_csv(cases,out/'benchmarks'/f'fold_{fold}_cases.csv.gz')
        quality_subs,quality_cases=make_quality_cases(val,markers,
            centers_per_organoid=quality_centers_per_organoid,seed=sampling_seed+10000+fold)
        _write_csv(quality_cases,out/'benchmarks'/f'fold_{fold}_quality_cases.csv.gz')
        for rate in cfg['training']['rates']:
            for seed in cfg['seeds']:
                stem=f'fold_{fold}_seed_{seed}_{rate_tag(rate)}'
                intact_path=out/'benchmarks'/f'{stem}_intact.csv.gz'
                mask_path=out/'benchmarks'/f'{stem}_masked.csv.gz'
                pair_path=out/'benchmarks'/f'{stem}_pair_effects.csv.gz'
                if intact_path.exists() and (rate==0 or (mask_path.exists() and pair_path.exists())): continue
                model=load_mask_checkpoint(out,fold,seed,rate,settings,markers)
                metadata=dict(fold=fold,seed=seed,mask_rate=rate)
                if not intact_path.exists():
                    intact=benchmark_intact(val,model,pre['residual_transform'],markers,batch_size=batch_size,device=device)
                    _write_csv(intact.assign(**metadata),intact_path)
                # An untrained missingness flag is not a valid masked baseline.
                if rate>0:
                    for selected_subs,selected_cases,path in [(quality_subs,quality_cases,mask_path),(subs,cases,pair_path)]:
                        if path.exists(): continue
                        frame=evaluate_single_mask(selected_subs,selected_cases,model,pre['residual_transform'],
                            size_center=pre['size_center'],size_scale=pre['size_scale'],batch_size=batch_size,device=device)
                        factor=np.exp(refs.loc[fold,'alpha']+refs.loc[fold,'beta']*np.log(frame.observed_n))/(4*np.pi)
                        frame['mask_delta_relative']=frame.mask_delta_mu*factor
                        frame['zero_delta_relative']=frame.zero_delta_mu*factor
                        _write_csv(frame.assign(**metadata),path)
                print(f'Benchmarked {stem}',flush=True)
    (out/'benchmarks/complete.json').write_text('{}')
    return out/'benchmarks'


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
