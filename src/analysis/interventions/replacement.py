"""Train-fold, context-matched fate replacements for exclusive FiLM checkpoints.

Unassigned is an observed identity. A replacement changes the entire source
vector, never the center, topology, or graph size. These are predictor contrasts,
not physical cell conversions. No target, prediction or crypt label enters matching.
"""
from dataclasses import asdict, dataclass
import copy
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import sparse
from sklearn.neighbors import NearestNeighbors

from src.analysis.interventions.perturbation import predict_subgraph_center_distribution
from src.analysis.interventions.size_sweeps import _size_override, summarize_by_organoid
from src.data.subgraphs import build_ego_subgraphs_for_graph
from src.graph.neighborhood import compute_hop_rings

UNASSIGNED = 'Unassigned'
METHODS = ('zero', 'replacement_fixed', 'replacement_size_dependent')


@dataclass(frozen=True)
class MatchConfig:
    # Distances: one unit = degree factor two, 25 percentage points in each
    # neighbor fraction, or an N factor two. Hard calipers apply after kNN.
    neighbors: int = 256
    max_size_ratio: float = 2.0
    max_degree_ratio: float = 2.0
    max_composition_l1: float = 1.0
    min_cells: int = 20
    min_organoids: int = 5
    min_identity_cells: int = 5
    min_identity_organoids: int = 3
    max_donors_per_organoid: int = 256

    def __post_init__(self):
        for key in ('neighbors', 'min_cells', 'min_organoids', 'min_identity_cells',
                    'min_identity_organoids', 'max_donors_per_organoid'):
            if getattr(self, key) < 1:
                raise ValueError(f'{key} must be positive')
        if self.max_size_ratio <= 1 or self.max_degree_ratio <= 1 or self.max_composition_l1 <= 0:
            raise ValueError('Matching calipers must be positive (ratios > 1).')


def identity_codes(x):
    x = np.asarray(x.detach().cpu() if torch.is_tensor(x) else x)
    if x.ndim != 2 or not np.isin(x, [0, 1]).all() or (x.sum(1) > 1).any():
        raise ValueError('Replacement analysis requires binary exclusive fate vectors.')
    return np.where(x.sum(1) == 0, x.shape[1], x.argmax(1))


def describe_graph(graph):
    """Full-graph source context; never truncate descriptors to a center ego.

    reach_h is a bit mask of identities at exact hop h. The candidate's own
    identity is excluded from neighbor composition and all exact-hop masks.
    """
    labels = identity_codes(graph.x)
    n, k = len(labels), graph.x.shape[1] + 1
    edge = graph.edge_index.detach().cpu().numpy()
    a = sparse.csr_matrix((np.ones(edge.shape[1]), (edge[0], edge[1])), shape=(n, n))
    a = a.maximum(a.T)
    a.setdiag(0); a.eliminate_zeros(); a.data[:] = 1
    two = a @ a
    two.data[:] = 1
    two = two - two.multiply(a)
    two.setdiag(0); two.eliminate_zeros()
    onehot = np.eye(k)[labels]
    degree = np.asarray(a.sum(axis=1)).ravel()
    fractions = (a @ onehot) / np.maximum(degree[:, None], 1)
    frame = pd.DataFrame(dict(organoid_str=str(graph.organoid_str), node=np.arange(n),
                              identity=labels, n_cells=n, degree=degree))
    for j in range(k):
        frame[f'fraction_{j}'] = fractions[:, j]
    for hop, adjacency in [(1, a), (2, two)]:
        frame[f'reach_{hop}'] = ((adjacency @ onehot > 0) * (1 << np.arange(k))).sum(1).astype(int)
    return frame


class ReplacementMatcher:
    """Coarse nearest-neighbor matching, with explicit support and no fallback.

    Donors must have a different identity and the recipient identity at the
    same exact hop. Matching is descriptive, not a causal adjustment. Within
    selected donors, each organoid has equal total weight; rare unsupported
    identities are removed before the remaining weights are renormalized.
    """
    def __init__(self, donors, n_identities, config=MatchConfig(), *, validation_organoids=()):
        self.donors = donors.reset_index(drop=True).copy()
        self.k, self.config = n_identities, config
        if set(self.donors.organoid_str) & set(validation_organoids):
            raise ValueError('Validation organoids must not enter the donor pool.')
        self.fractions = [f'fraction_{j}' for j in range(self.k)]
        self.cache = {}

    def _features(self, frame, n):
        return np.column_stack([np.log1p(frame.degree) / np.log(2),
                                frame[self.fractions].to_numpy() / .25,
                                np.log(np.broadcast_to(n, len(frame))) / np.log(2)])

    def query(self, context, *, center_identity, source_identity, hop, count):
        if not np.isfinite(count) or count <= 0:
            raise ValueError('Matching N must be positive and finite.')
        key = (int(center_identity), int(source_identity), int(hop))
        if key not in self.cache:
            eligible_indices = np.flatnonzero((self.donors.identity != source_identity) &
                ((self.donors[f'reach_{hop}'].to_numpy() & (1 << int(center_identity))) != 0))
            eligible = self.donors.iloc[eligible_indices]
            index = None
            if len(eligible):
                index = NearestNeighbors(n_neighbors=min(self.config.neighbors, len(eligible)),
                                        algorithm='kd_tree', n_jobs=1).fit(
                                            self._features(eligible, eligible.n_cells))
            # Keep only row indices and numeric search trees, not a duplicate
            # donor DataFrame for every center/source/hop combination.
            self.cache[key] = eligible_indices, index
        eligible_indices, index = self.cache[key]
        weights = np.full(self.k, np.nan)
        audit = dict(supported=False, donor_cells=0, donor_organoids=0,
                     candidate_cells=0, candidate_organoids=0,
                     max_distance=np.nan, reason='no eligible donors')
        counts = np.zeros(self.k, dtype=int); orgcounts = counts.copy()
        if index is None:
            return weights, audit, counts, orgcounts
        q = pd.DataFrame([context])
        distance, positions = index.kneighbors(self._features(q, count))
        selected = self.donors.iloc[eligible_indices[positions[0]]].copy()
        selected['distance'] = distance[0]
        cfg = self.config
        n_ratio = np.maximum(selected.n_cells / count, count / selected.n_cells)
        degree_ratio = np.maximum((selected.degree + 1) / (context['degree'] + 1),
                                  (context['degree'] + 1) / (selected.degree + 1))
        l1 = np.abs(selected[self.fractions].to_numpy() - q[self.fractions].to_numpy()).sum(1)
        selected = selected[(n_ratio <= cfg.max_size_ratio) &
                            (degree_ratio <= cfg.max_degree_ratio) & (l1 <= cfg.max_composition_l1)]
        audit.update(candidate_cells=len(selected), candidate_organoids=selected.organoid_str.nunique())
        for identity, group in selected.groupby('identity'):
            counts[identity] = len(group); orgcounts[identity] = group.organoid_str.nunique()
        keep = (counts >= cfg.min_identity_cells) & (orgcounts >= cfg.min_identity_organoids)
        selected = selected[selected.identity.isin(np.flatnonzero(keep))]
        audit.update(donor_cells=len(selected), donor_organoids=selected.organoid_str.nunique(),
                     reason='insufficient donor support')
        if len(selected) < cfg.min_cells or selected.organoid_str.nunique() < cfg.min_organoids:
            return weights, audit, counts, orgcounts
        per_cell = 1 / selected.groupby('organoid_str').organoid_str.transform('size')
        sums = per_cell.groupby(selected.identity).sum()
        weights = np.zeros(self.k)
        weights[sums.index.to_numpy(int)] = sums.to_numpy() / sums.sum()
        audit.update(supported=True, reason='supported', max_distance=float(selected.distance.max()))
        return weights, audit, counts, orgcounts


def make_cases(graphs, marker_names, *, hops=(1, 2), centers_per_identity=2,
               sweep_centers_per_identity=1, seed=1701, receptive_hops=2):
    """Sample centers within each identity, then one source/identity/exact ring.

    Sweep centers are a fixed subset of static centers. Every method and N
    uses the same source choice. Cases include unassigned centers and sources.
    """
    if not hops or not set(hops) <= {1, 2}:
        raise ValueError('Supported exact hops are 1 and 2.')
    if not isinstance(receptive_hops, int) or receptive_hops < max(hops):
        raise ValueError('Receptive field must include every sampled source hop.')
    if not 1 <= sweep_centers_per_identity <= centers_per_identity:
        raise ValueError('Require 1 <= sweep centers <= static centers per identity.')
    names = list(marker_names) + [UNASSIGNED]
    rng = np.random.default_rng(seed)
    subgraphs, cases, contexts = [], [], []
    for graph in graphs:
        desc = describe_graph(graph)
        labels = desc.identity.to_numpy()
        centers, sweep = [], set()
        for ident in range(len(names)):
            available = np.flatnonzero(labels == ident)
            chosen = rng.choice(available, min(centers_per_identity, len(available)), replace=False)
            centers.extend(chosen.tolist()); sweep.update(chosen[:sweep_centers_per_identity].tolist())
        for sub in build_ego_subgraphs_for_graph(graph, num_hops=receptive_hops, centers=centers):
            si = len(subgraphs); subgraphs.append(sub)
            center = int(sub.orig_center)
            local_labels = labels[sub.orig_nodes.numpy()]
            rings = compute_hop_rings(sub.edge_index, int(sub.center_idx), max(hops))
            for hop in hops:
                nodes = np.asarray(rings[hop], dtype=int)
                for ident, name in enumerate(names):
                    sources = nodes[local_labels[nodes] == ident]
                    if not len(sources):
                        continue
                    source = int(rng.choice(sources)); original = int(sub.orig_nodes[source])
                    case = dict(case_id=len(cases), subgraph_index=si, organoid_str=str(graph.organoid_str),
                        orig_center=center, source_node=source, orig_source_node=original,
                        center_identity=int(labels[center]), center_marker=names[labels[center]],
                        source_identity=ident, source_marker_name=name, hop=hop,
                        observed_n=len(graph.x), sweep=center in sweep)
                    cases.append(case); contexts.append(desc.iloc[original].to_dict())
    return subgraphs, pd.DataFrame(cases), contexts


def replacement_weights(cases, contexts, matcher, *, count=None):
    """count=None means use each source's observed N (frozen sweep reference)."""
    weights, audits = [], []
    for case, context in zip(cases.to_dict('records'), contexts):
        n = case['observed_n'] if count is None else count
        p, audit, cells, orgs = matcher.query(context, center_identity=case['center_identity'],
            source_identity=case['source_identity'], hop=case['hop'], count=n)
        weights.append(p)
        audits.append(dict(case_id=case['case_id'], reference_n=n, **audit,
            **{f'identity_cells_{j}': int(cells[j]) for j in range(matcher.k)},
            **{f'identity_organoids_{j}': int(orgs[j]) for j in range(matcher.k)}))
    return np.asarray(weights), pd.DataFrame(audits)


def evaluate_replacements(subgraphs, cases, model, transform, *, size_center, size_scale,
                          count=None, batch_size=128, device='cpu'):
    """Enumerate discrete fates, inverting the target transform BEFORE averaging.

    Returns [case, identity] predictions. The original-identity entry is the
    intact prediction; unassigned -> unassigned is exactly a no-op.
    """
    if cases.empty:
        raise ValueError('No replacement cases to evaluate.')
    if batch_size < 1 or size_scale <= 0:
        raise ValueError('Batch size and size scale must be positive.')
    graphs = [_size_override(g, count, size_center, size_scale) for g in subgraphs]
    k = graphs[0].x.shape[1] + 1
    base_z, base_lv = predict_subgraph_center_distribution(graphs, model, batch_size=batch_size, device=device)
    _, base_mu, _ = transform.inverse_distribution(None, base_z, log_var=base_lv)
    indices = cases.subgraph_index.to_numpy(int)
    z = np.repeat(base_z[indices, None], k, axis=1)
    mu = np.repeat(np.asarray(base_mu).reshape(-1)[indices, None], k, axis=1)
    pending, slots = [], []

    def flush():
        if not pending:
            return
        pz, plv = predict_subgraph_center_distribution(pending, model, batch_size=batch_size, device=device)
        _, pmu, _ = transform.inverse_distribution(None, pz, log_var=plv)
        for (row, identity), zv, mv in zip(slots, pz, np.asarray(pmu).reshape(-1)):
            z[row, identity] = zv; mu[row, identity] = mv
        pending.clear(); slots.clear()

    for row, case in enumerate(cases.to_dict('records')):
        original = graphs[case['subgraph_index']]
        if int(identity_codes(original.x[case['source_node']:case['source_node'] + 1])[0]) != case['source_identity']:
            raise ValueError('Source identity differs from manifest.')
        if case['source_node'] == int(original.center_idx):
            raise ValueError('Replacement must leave the recipient center unchanged.')
        for identity in range(k):
            if identity == case['source_identity']:
                continue
            g = copy.copy(original); g.x = original.x.clone()
            g.x[case['source_node']] = 0
            if identity < k - 1:
                g.x[case['source_node'], identity] = 1
            pending.append(g); slots.append((row, identity))
            if len(pending) >= batch_size:
                flush()
    flush()
    return dict(mu=mu, z=z, base_mu=np.asarray(base_mu).reshape(-1)[indices], base_z=base_z[indices])


def contrast_table(cases, predictions, fixed_weights, dependent_weights, *, count, fold, seed, reference):
    """All three methods on identical cases; unsupported mixtures remain NaN."""
    if not (fixed_weights.shape == dependent_weights.shape == predictions['mu'].shape):
        raise ValueError('Prediction and weight shapes disagree.')
    result = cases.copy()
    result['fold'], result['seed'] = fold, seed
    result['analysis'] = 'observed' if count is None else 'sweep'
    result['evaluated_n'] = result.observed_n if count is None else count
    result['size_fold_change'] = np.maximum(result.evaluated_n / result.observed_n,
                                            result.observed_n / result.evaluated_n)
    factor = np.exp(reference['alpha'] + reference['beta'] * np.log(result.evaluated_n)) / (4 * np.pi)
    delta = predictions['mu'] - predictions['base_mu'][:, None]
    dz = predictions['z'] - predictions['base_z'][:, None]
    result['base_mu'] = predictions['base_mu']; result['base_z'] = predictions['base_z']
    result['fixed_supported'] = np.isfinite(fixed_weights).all(1)
    result['dependent_supported'] = np.isfinite(dependent_weights).all(1)
    # Local support of fixed alternatives can change even though fixed weights
    # themselves must not. Keep this separate from whole-case mixture support.
    result['fixed_mass_supported_at_N'] = np.where(
        result.fixed_supported & result.dependent_supported,
        (fixed_weights * (dependent_weights > 0)).sum(1), np.nan)
    for j in range(delta.shape[1]):
        result[f'delta_identity_{j}'] = delta[:, j]
        result[f'delta_z_identity_{j}'] = dz[:, j]
        result[f'p_fixed_{j}'] = fixed_weights[:, j]
        result[f'p_dependent_{j}'] = dependent_weights[:, j]
    result['normalization_factor'] = factor
    for method, weights in [('zero', None), ('replacement_fixed', fixed_weights),
                            ('replacement_size_dependent', dependent_weights)]:
        result[f'{method}_delta_mu'] = delta[:, -1] if weights is None else (weights * delta).sum(1)
        result[f'{method}_delta_z'] = dz[:, -1] if weights is None else (weights * dz).sum(1)
        result[f'{method}_delta_relative'] = result[f'{method}_delta_mu'] * factor
    return result


def _write_csv(frame, path):
    temp = path.with_name(path.name + '.tmp')
    frame.to_csv(temp, index=False, compression='gzip' if str(path).endswith('.gz') else None)
    temp.replace(path)


def load_comparison(output_dir, *, bootstrap_samples=500, seed=42):
    """Paired, constant-support summaries; average seeds/cases within organoids.

    Sweeps require fixed AND N-dependent support at EVERY supplied N, avoiding
    changing case membership along a curve. Static uses common support at each
    case's observed N. All-cases zero results are saved separately as a diagnostic.
    """
    out = Path(output_dir)
    if not (out / 'complete.json').exists():
        raise RuntimeError('Inference is incomplete; run or resume it before plotting.')
    config = json.loads((out / 'settings.json').read_text())
    if 'folds' in config and 'seeds' in config:
        paths = [out / 'tables' / f'fold_{fold}_seed_{model_seed}_{label}.csv.gz'
                 for fold in config['folds'] for model_seed in config['seeds']
                 for label in ['observed', *[f'N{n:g}' for n in config['counts']]]]
        missing = [str(p) for p in paths if not p.is_file()]
        if missing:
            raise FileNotFoundError(f'Missing inference shards: {missing[:3]}')
    else:
        paths = sorted((out / 'tables').glob('*.csv.gz'))
    frame = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    if frame.duplicated(['fold', 'seed', 'case_id', 'analysis', 'evaluated_n']).any():
        raise ValueError('Duplicate inference cases.')
    keys = ['fold', 'case_id']
    frame['common_support'] = frame.fixed_supported & frame.dependent_supported
    swept = frame[frame.analysis == 'sweep']
    stable = swept.groupby(keys).agg(ok=('common_support', 'all'), n=('evaluated_n', 'nunique'))
    stable['ok'] &= stable.n == len(config['counts'])
    mask = pd.MultiIndex.from_frame(frame[keys]).map(stable.ok).fillna(False).to_numpy(bool)
    frame.loc[frame.analysis == 'sweep', 'common_support'] &= mask[frame.analysis == 'sweep']
    frame['coordinate'] = frame.evaluated_n
    # Fixed descriptive bins; static x positions are actual median observed counts.
    bins = [0, 150, 250, 350, 500, 800, np.inf]
    observed = frame.analysis == 'observed'
    frame.loc[observed, 'coordinate'] = pd.cut(frame.loc[observed, 'observed_n'], bins,
                                              right=False, labels=False).astype(float)
    support = frame.drop_duplicates(['fold', 'case_id', 'analysis', 'evaluated_n']).groupby(
        ['analysis', 'center_marker', 'source_marker_name', 'hop', 'coordinate'], observed=True).agg(
            n_cases=('case_id', 'size'), n_supported=('common_support', 'sum'),
            n_organoids=('organoid_str', 'nunique'),
            median_size_fold_change=('size_fold_change', 'median')).reset_index()
    support['fraction_supported'] = support.n_supported / support.n_cases
    selected = frame[frame.common_support].copy()
    groups = ['analysis', 'center_marker', 'source_marker_name', 'hop', 'coordinate']
    pieces = []
    for method in METHODS:
        for metric in ('delta_mu', 'delta_relative', 'delta_z'):
            part = selected[groups + ['fold', 'case_id', 'seed', 'organoid_str', 'evaluated_n']].copy()
            part['effect'] = selected[f'{method}_{metric}'].to_numpy()
            summary = summarize_by_organoid(part, groups, 'effect', bootstrap_samples=bootstrap_samples, seed=seed)
            if not summary.empty:
                positions = part.groupby(groups, observed=True).evaluated_n.median().rename('n').reset_index()
                summary = summary.merge(positions, on=groups, validate='one_to_one')
                summary['method'], summary['metric'] = method, metric
                pieces.append(summary)
    summary = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()
    _write_csv(support, out / 'support_summary.csv')
    _write_csv(summary, out / 'comparison_summary.csv')
    return dict(cases=frame, paired_cases=selected, summary=summary, support=support, config=config)
