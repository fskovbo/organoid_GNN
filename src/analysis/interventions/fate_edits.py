"""Shared single-source fate interventions on restored validation graphs.

Sampling, inference and summaries are independent of notebook orchestration.
Zeroing removes one fate bit; masking hides the entire source identity using a
trained flag; replacement averages predictions over supported discrete fates.
"""
import copy
import hashlib
import json
import numpy as np
import pandas as pd
import torch
from src.data.subgraphs import build_ego_subgraphs_for_graph
from src.graph.neighborhood import compute_hop_rings
from src.data.subgraph_sampling import sample_subgraphs_coverage, flag_non_overlapping_source_cases
from src.data.fate_masking import encode_fates, ObservedFateAdapter
from src.analysis.interventions.perturbation import predict_subgraph_center_distribution
from src.analysis.interventions.size_sweeps import _size_override, summarize_by_organoid
from src.analysis.interventions.replacement import identity_codes, describe_graph, ReplacementMatcher, MatchConfig

METHODS = ('marker_zeroing', 'masking', 'replacement')
CASE_KEYS = ['fold', 'graph_signature', 'organoid_str', 'orig_center', 'orig_source_node', 'hop', 'source_marker_name', 'center_marker']


def fate_graphs(selection, role='val'):
    """Copy saved graph inputs, stripping only an observed missingness channel."""
    k = len(selection['marker_names'])
    result = []
    for original in selection['groups'][role]:
        graph = copy.copy(original)
        if original.x.shape[1] not in (k, k + 1):
            raise ValueError('Unexpected number of saved fate channels.')
        if original.x.shape[1] == k + 1 and torch.any(original.x[:, -1] != 0):
            raise ValueError('Validation and donor inputs must have observed fate.')
        graph.x = original.x[:, :k].clone()
        graph.full_num_cells = len(graph.x)
        result.append(graph)
    return result


def sample_fate_contexts(graphs, markers, depth, *, scheme='uniform', centers=5, seed=1701,
                         max_subgraphs=2000, min_center_count=50, min_pair_count=25,
                         apply_non_overlap=False,
                         non_overlap_group_columns=('hop', 'center_marker', 'source_marker'),
                         return_info=False):
    """Sample single-source cases, optionally using the historical coverage design.

    ``coverage`` uses the original greedy center/pair minimum sampler, followed
    by rare-feature weighted filling up to a per-fold budget. As in the old
    single-source intervention, it chooses the first positive source in each
    exact ring. Uniform/stratified sampling retains random source selection.
    Unassigned is an additional sampling identity, never a model input channel.
    Optional non-overlap blocks sources within distance <= hop independently
    per recipient/source/hop bin, before inference and independently of N.
    Targets are best-effort: unavailable pairs, budget and non-overlap can
    prevent reaching them. Returned diagnostics distinguish these stages.
    """
    if not isinstance(depth, int) or depth < 1:
        raise ValueError('Neighbor ablation requires model depth >= 1; depth 0 has no neighbor interactions.')
    if scheme not in ('coverage', 'uniform', 'stratified') or centers < 1:
        raise ValueError('Choose coverage/uniform/stratified sampling and a positive recipient quota.')
    if min_center_count < 0 or min_pair_count < 0:
        raise ValueError('Coverage targets must be nonnegative.')
    if max_subgraphs is not None and max_subgraphs < 1:
        raise ValueError('Coverage budget must be positive or None.')
    graphs = sorted(graphs, key=lambda g: str(g.organoid_str))
    rng = np.random.default_rng(seed)
    subs, signatures = [], {}
    names = [*markers, 'Unassigned']
    for gi, graph in enumerate(graphs):
        x = graph.x.detach().cpu().numpy()
        if x.shape[1] != len(markers) or not np.isin(x, [0, 1]).all():
            raise ValueError('Sampling requires binary fate channels in saved marker order.')
        signatures[str(graph.organoid_str)] = hashlib.sha256(x.tobytes() + graph.edge_index.cpu().numpy().tobytes()).hexdigest()
        identities = [np.flatnonzero(x[:, j] > .5) for j in range(len(markers))]
        identities.append(np.flatnonzero(x.sum(1) == 0))
        if scheme == 'coverage':
            chosen = range(len(x))
        elif scheme == 'uniform':
            chosen = rng.choice(len(x), min(centers, len(x)), replace=False)
        else:
            chosen = sorted({int(i) for ids in identities for i in rng.choice(ids, min(centers, len(ids)), replace=False)})
        subs.extend(build_ego_subgraphs_for_graph(graph, num_hops=depth, centers=sorted(chosen), graph_idx=gi))
    if not subs:
        raise ValueError('No validation recipients available for sampling.')
    coverage = None
    if scheme == 'coverage':
        proxies = []
        for sub in subs:
            proxy = copy.copy(sub)
            proxy.x = torch.cat([sub.x, (sub.x.sum(1) == 0).to(sub.x.dtype)[:, None]], dim=1)
            proxies.append(proxy)
        _, coverage = sample_subgraphs_coverage(proxies, names, depth, max_subgraphs,
            min_center_count=min_center_count, min_pair_count=min_pair_count, seed=seed)
        subs = [subs[int(i)] for i in coverage['selected_indices']]
        del proxies
    rows = []
    for si, sub in enumerate(subs):
        graph = graphs[int(sub.graph_idx)]
        x = graph.x.detach().cpu().numpy()
        center = int(sub.orig_center)
        center_names = [m for j, m in enumerate(markers) if x[center, j] > .5] or ['Unassigned']
        rings = compute_hop_rings(sub.edge_index, int(sub.center_idx), depth)
        for hop in range(1, depth + 1):
            nodes = np.asarray(rings[hop], dtype=int)
            if not len(nodes):
                continue
            sx = sub.x[nodes].cpu().numpy()
            for marker, name in enumerate(names):
                available = nodes[sx[:, marker] > .5] if marker < len(markers) else nodes[sx.sum(1) == 0]
                if not len(available):
                    continue
                source = int(available[0] if scheme == 'coverage' else rng.choice(available))
                original = int(sub.orig_nodes[source])
                key = [str(graph.organoid_str), center, original, hop, name]
                rows.append(dict(case_id=hashlib.sha256(json.dumps(key).encode()).hexdigest()[:24],
                    subgraph_index=si, graph_idx=int(sub.graph_idx), organoid_str=str(graph.organoid_str),
                    orig_center=center, source_node=source, orig_source_node=original, source_marker=marker,
                    source_marker_name=name, center_marker_names=center_names, hop=hop, observed_n=len(x),
                    graph_signature=signatures[str(graph.organoid_str)], eligible_sources=len(available)))
    cases = pd.DataFrame(rows)
    pair_columns = ['hop', 'center_marker', 'source_marker_name']
    if not cases.empty:
        pairs = expand_recipients(cases)
        if apply_non_overlap:
            pairs = flag_non_overlapping_source_cases(pairs, graphs=graphs,
                group_columns=non_overlap_group_columns, seed=seed, shuffle=True)
        else:
            pairs['passes_non_overlap'] = True
        retained = pairs[pairs.passes_non_overlap].groupby('case_id').center_marker.agg(list)
        # Coexpressing recipients can pass in one marker bin and fail in another.
        # Keep their graph features intact; restrict only the summary memberships.
        cases = cases[cases.case_id.isin(retained.index)].copy()
        cases['center_marker_names'] = cases.case_id.map(retained)
        before = pairs.groupby(pair_columns).size()
        after = pairs[pairs.passes_non_overlap].groupby(pair_columns).size()
    else:
        pairs = pd.DataFrame()
        before = after = pd.Series(dtype=int)
    if not return_info:
        return subs, cases.reset_index(drop=True)
    pair_rows = []
    for hi in range(depth):
        for ci, center in enumerate(names):
            for mi, source in enumerate(names):
                key = (hi + 1, center, source)
                selected, kept = int(before.get(key, 0)), int(after.get(key, 0))
                pair_rows.append(dict(hop=hi+1, center_marker=center, source_marker_name=source,
                    global_available=int(coverage['pair_freq_global'][hi, ci, mi]) if coverage else np.nan,
                    target_count=min_pair_count, coverage_selected_count=selected, selected_count=kept,
                    non_overlap_excluded_count=selected-kept, target_met=kept >= min_pair_count))
    center_rows = []
    for ci, name in enumerate(names):
        count = sum(bool(sub.x[int(sub.center_idx), ci] > .5) if ci < len(markers)
                    else bool(sub.x[int(sub.center_idx)].sum() == 0) for sub in subs)
        center_rows.append(dict(marker=name, target_count=min_center_count, selected_count=count,
            global_available=int(coverage['center_freq_global'][ci]) if coverage else np.nan,
            target_met=count >= min_center_count))
    info = dict(center_coverage=pd.DataFrame(center_rows), pair_coverage=pd.DataFrame(pair_rows),
                case_non_overlap=pairs, n_selected=len(subs))
    return subs, cases.reset_index(drop=True), info


def build_replacement_reference(selection, cases, depth, *, config=MatchConfig(), seed=1701):
    """Fit a donor index on outer TRAINING organoids only, with exact-hop context."""
    rng = np.random.default_rng(seed)
    donors, lookup = [], {}
    groups = {role: fate_graphs(selection, role) for role in ('train','val')}
    for graph in sorted(groups['train'], key=lambda g: str(g.organoid_str)):
        desc = describe_graph(graph, max_hops=depth)
        donors.append(desc.iloc[rng.choice(len(desc), min(len(desc), config.max_donors_per_organoid), replace=False)])
    for graph in groups['val']:
        lookup[str(graph.organoid_str)] = describe_graph(graph, max_hops=depth)
    matcher = ReplacementMatcher(pd.concat(donors, ignore_index=True), len(selection['marker_names'])+1,
        config, validation_organoids=list(lookup))
    contexts = [lookup[c.organoid_str].iloc[int(c.orig_source_node)].to_dict() for c in cases.itertuples()]
    return matcher, contexts


def replacement_distribution(cases, contexts, matcher, markers, *, count=None):
    weights, audit = [], []
    for row, context in zip(cases.itertuples(), contexts):
        names = row.center_marker_names
        if len(names) != 1:
            raise ValueError('Replacement requires exclusive recipient identities.')
        center_identity = (list(markers)+['Unassigned']).index(names[0])
        p, info, cells, organoids = matcher.query(context, center_identity=center_identity,
            source_identity=row.source_marker, hop=row.hop, count=row.observed_n if count is None else count)
        weights.append(p)
        audit.append(dict(case_id=row.case_id, **info,
            **{f'donor_cells_{i}':int(v) for i,v in enumerate(cells)},
            **{f'donor_organoids_{i}':int(v) for i,v in enumerate(organoids)}))
    return np.asarray(weights), pd.DataFrame(audit)


def evaluate_fate_edit(selection, subgraphs, cases, method, *, count=None, weights=None,
                       device='cpu', batch_size=128):
    """One method on a fixed manifest; graph and recipient identity stay intact.

    Invert each discrete prediction before averaging replacement alternatives.
    A masking-trained checkpoint also supports zeroing/replacement with flag=0,
    enabling all three methods to be compared in exactly the same model.
    """
    if method not in METHODS or cases.empty or batch_size < 1:
        raise ValueError('Choose a supported method, nonempty cases and a positive batch size.')
    model, markers = selection['model'], selection['marker_names']
    trained_flag = selection['groups']['val'][0].x.shape[1] == len(markers) + 1
    rate = selection.get('record', {}).get('rate')
    if method == 'masking' and (not trained_flag or rate is None or not float(rate) > 0):
        raise ValueError('Masking requires a checkpoint trained with a positive masking rate.')
    observed_model = ObservedFateAdapter(model) if trained_flag else model
    size_center, size_scale, column = 0., 1., 0
    if count is not None:
        if 'log_num_cells' not in selection['global_features']:
            raise ValueError('N sweeps require a saved log_num_cells input.')
        column = selection['global_features'].index('log_num_cells')
        j = selection['all_global_features'].index('log_num_cells')
        size_center, size_scale = selection['global_center'][j], selection['global_scale'][j]
    graphs = [_size_override(g, count, size_center, size_scale, column) for g in subgraphs]
    z, _ = predict_subgraph_center_distribution(graphs, observed_model, batch_size=batch_size, device=device)
    transform = selection['transform']
    base_z = z[cases.subgraph_index.to_numpy(int)]
    base_mu = np.asarray(transform.inverse(base_z)).reshape(-1)
    result = cases.copy()
    result['analysis'] = 'observed' if count is None else 'sweep'
    result['evaluated_n'] = result.observed_n if count is None else float(count)
    result['method'] = method
    result['base_z'], result['base_mu'] = base_z, base_mu
    if method == 'replacement':
        for graph in graphs:
            identity_codes(graph.x)
        weights = np.asarray(weights)
        if weights.shape != (len(cases), len(markers)+1):
            raise ValueError('Replacement weights must match cases and identities.')
        supported = np.isfinite(weights).all(1)
        if np.any(weights[supported] < 0) or not np.allclose(weights[supported].sum(1),1):
            raise ValueError('Supported replacement weights must form probability distributions.')
        if np.any(weights[np.flatnonzero(supported),cases.source_marker.to_numpy(int)[supported]] != 0):
            raise ValueError('Replacement must exclude the original source identity.')
    else:
        supported = np.ones(len(cases), dtype=bool)
    totals_mu, totals_z = np.zeros(len(cases)), np.zeros(len(cases))
    pending, slots = [], []
    def flush():
        if not pending:
            return
        new_z, _ = predict_subgraph_center_distribution(pending, model if method=='masking' else observed_model,
                                                       batch_size=batch_size, device=device)
        new_mu = np.asarray(transform.inverse(new_z)).reshape(-1)
        for (row, weight), mz, mm in zip(slots, new_z, new_mu):
            totals_z[row] += weight * mz
            totals_mu[row] += weight * mm
        pending.clear(); slots.clear()
    for row, case in enumerate(cases.itertuples()):
        if not supported[row]:
            continue
        original = graphs[case.subgraph_index]
        if case.source_node == int(original.center_idx):
            raise ValueError('Do not ablate the recipient center.')
        alternatives = [(i,float(p)) for i,p in enumerate(weights[row]) if p>0] if method=='replacement' else [(None,1.)]
        for identity, probability in alternatives:
            edited = copy.copy(original)
            edited.x = original.x.clone()
            if method == 'masking':
                mask = torch.zeros(len(edited.x), dtype=torch.bool); mask[case.source_node] = True
                edited.x = encode_fates(edited.x, mask)
            elif method == 'replacement':
                edited.x[case.source_node] = 0
                if identity < len(markers):
                    edited.x[case.source_node,identity] = 1
            elif case.source_marker < len(markers):
                edited.x[case.source_node,case.source_marker] = 0
            pending.append(edited);slots.append((row,probability))
            if len(pending)>=batch_size:
                flush()
    flush()
    result['supported'] = supported
    result['delta_mu'] = np.where(supported, totals_mu-base_mu, np.nan)
    result['delta_z'] = np.where(supported, totals_z-base_z, np.nan)
    if method=='replacement':
        for j in range(len(markers)+1):
            result[f'p_{j}'] = weights[:,j]
    return result


def add_geometric_scale(frame, selection):
    """Normalize by train-only log(area) versus log(N) regression for this fold."""
    raw = selection['raw_graphs']
    x, y = [], []
    for g in selection['groups']['train']:
        meta = raw[g.organoid_str].meta
        area = float(np.asarray(meta['total_surface_area']).reshape(-1)[0])
        if not np.isfinite(area) or area<=0:
            raise ValueError('Geometric normalization needs positive training surface areas.')
        x.append(np.log(len(g.x)));y.append(np.log(area))
    if len(x)<3 or np.ptp(x)==0:
        raise ValueError('Area scaling requires at least three training organoids with varying N.')
    alpha,beta = np.linalg.lstsq(np.column_stack([np.ones(len(x)),x]),y,rcond=None)[0]
    result = frame.copy()
    result['normalization_factor'] = np.exp(alpha+beta*np.log(result.evaluated_n))/(4*np.pi)
    result['delta_relative'] = result.delta_mu*result.normalization_factor
    return result, dict(alpha=float(alpha), beta=float(beta), n_train=len(x))


def expand_recipients(frame):
    result = frame.copy()
    result['center_marker'] = result.center_marker_names.map(lambda x: json.loads(x) if isinstance(x,str) else x)
    return result.explode('center_marker',ignore_index=True)


def supported_effects(frame):
    """Keep a fixed supported source cohort across every supplied N per checkpoint."""
    result = frame.copy()
    ok = result.supported.astype(bool) & np.isfinite(result.delta_mu)
    swept = result.analysis.eq('sweep')
    if swept.any():
        keys = ['model_key','case_id']
        stable = result[swept].assign(ok=ok[swept]).groupby(keys).ok.all()
        ok.loc[swept] &= pd.MultiIndex.from_frame(result.loc[swept,keys]).map(stable).to_numpy(bool)
    return result[ok].copy()


def summarize_fate_effects(frame, *, view='total', metric='delta_mu', size_bins=(0,150,250,350,500,800,np.inf), draws=500, seed=42):
    """Equal-organoid summaries; cases and repeated model seeds average within organoid."""
    selected = expand_recipients(supported_effects(frame)) if 'center_marker' not in frame else supported_effects(frame)
    if view=='total':
        selected = selected[selected.analysis=='observed']
        groups = ['center_marker','source_marker_name','hop']
    elif view=='size':
        selected['coordinate'] = selected.evaluated_n
        observed = selected.analysis=='observed'
        selected.loc[observed,'coordinate'] = pd.cut(selected.loc[observed,'observed_n'],size_bins,labels=False,right=False).astype(float)
        groups = ['analysis','center_marker','source_marker_name','hop','coordinate']
    else:
        raise ValueError('view must be total or size.')
    summary = summarize_by_organoid(selected,groups,metric,bootstrap_samples=draws,seed=seed)
    if not summary.empty:
        finite = selected[np.isfinite(selected[metric])]
        unique = finite.drop_duplicates(groups + ['organoid_str', 'orig_center', 'orig_source_node'])
        support = unique.groupby(groups, observed=True).size().rename('n_cases').reset_index()
        summary = summary.merge(support, on=groups, validate='one_to_one')
    else:
        summary['n_cases'] = pd.Series(dtype=int)
    if view=='size' and not summary.empty:
        positions = selected.groupby(groups,observed=True).evaluated_n.median().rename('N').reset_index()
        summary = summary.merge(positions,on=groups,validate='one_to_one')
    return summary


def count_fate_cases(frame, *, metric='delta_mu', size_bins=None):
    """Retained pair support by observed/sweep cohort, without seed/N replication.

    Sweeps use the same all-N support restriction as effect summaries. Observed
    counts pool the configured size bins, when supplied; these are not counts
    per individual bin. Include pairs below plot thresholds for diagnosis.
    """
    selected = supported_effects(frame)
    if 'center_marker' not in selected:
        selected = expand_recipients(selected)
    selected = selected[np.isfinite(selected[metric])].copy()
    if size_bins is not None:
        observed = selected.analysis.eq('observed')
        in_range = pd.cut(selected.observed_n, size_bins, right=False).notna()
        selected = selected[~observed | in_range]
    groups = ['analysis', 'hop', 'center_marker', 'source_marker_name']
    unique = selected.drop_duplicates(groups + ['organoid_str', 'orig_center', 'orig_source_node'])
    return unique.groupby(groups, observed=True).agg(
        n_cases=('case_id', 'size'), n_organoids=('organoid_str', 'nunique')).reset_index()


def align_method_cases(tables):
    """Intersect physical source cases and N coordinates across saved methods.

    Do not pool different seeds as independent replicates. Different checkpoints
    remain explicitly labelled; callers report that as a model+method comparison.
    """
    expanded = {label:expand_recipients(supported_effects(table)) for label,table in tables.items()}
    keys = CASE_KEYS + ['analysis','evaluated_n']
    common = None
    for table in expanded.values():
        idx = pd.MultiIndex.from_frame(table[keys].drop_duplicates())
        common = idx if common is None else common.intersection(idx)
    if common is None or common.empty:
        raise ValueError('No common supported source cases. Use matching cohort/sampling settings.')
    aligned = {label:table[pd.MultiIndex.from_frame(table[keys]).isin(common)].copy() for label,table in expanded.items()}
    # Intersection must itself remain constant across all supplied sweep counts.
    sweep_counts = sorted({float(n) for table in aligned.values() for n in table.loc[table.analysis=='sweep','evaluated_n']})
    if sweep_counts:
        for label,table in aligned.items():
            swept = table.analysis=='sweep'
            coverage = table[swept].groupby(CASE_KEYS).evaluated_n.nunique()
            complete = coverage[coverage==len(sweep_counts)].index
            keep = ~swept | pd.MultiIndex.from_frame(table[CASE_KEYS]).isin(complete)
            aligned[label] = table[keep].copy()
    return aligned


def load_ablation_result(directory):
    """Load a completed, explicitly selected result bundle without inference."""
    from pathlib import Path
    directory = Path(directory)
    if not (directory/'complete.json').exists():
        raise FileNotFoundError(f'Complete ablation inference first: {directory}')
    config = json.loads((directory/'settings.json').read_text())
    paths = [directory/'cases'/f'{key}.csv.gz' for key in config['model_keys']]
    frames = [pd.read_csv(path) for path in paths]
    frame = pd.concat(frames,ignore_index=True)
    if frame.duplicated(['model_key','case_id','analysis','evaluated_n']).any():
        raise ValueError('Duplicate source cases in saved ablation results.')
    return frame,config
