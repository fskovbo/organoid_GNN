"""Organoid-paired summaries and matched-center common-scaling diagnostics."""
import itertools
import json
import warnings
from pathlib import Path
import numpy as np
import pandas as pd

EFFECT_METRICS = ['delta_z', 'delta_raw', 'delta_relative', 'delta_fixed_reference',
                  'h1_cosine', 'h2_cosine', 'h1_ratio', 'h2_ratio']
INTERACTION_METRICS = ['interaction_z', 'interaction_raw', 'interaction_relative',
                       'h1_interaction_ratio', 'h2_interaction_ratio']


def cluster_summary(frame, keys, metrics, *, draws=1000, seed=42):
    """Average within organoid first; resample whole organoids jointly across metrics."""
    org = frame.groupby([*keys, 'organoid_str'], observed=True)[metrics].mean().reset_index()
    rng = np.random.default_rng(seed)
    records = []
    for values, group in org.groupby(keys, observed=True, sort=True):
        values = values if isinstance(values, tuple) else (values,)
        x = group[metrics].to_numpy()
        indices = rng.integers(len(x), size=(draws, len(x)))
        for j, metric in enumerate(metrics):
            column = x[:, j]
            finite = np.isfinite(column)
            if not finite.any():
                continue
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                bootstrap = np.nanmean(column[indices], axis=1)
            low, high = np.nanquantile(bootstrap, [.025, .975]) if finite.sum() > 1 else (np.nan, np.nan)
            records.append(dict(zip(keys, values), metric=metric, mean=float(np.nanmean(column)),
                                ci_low=low, ci_high=high, n_organoids=int(finite.sum())))
    return pd.DataFrame(records)


def scale_statistics(curves, ref_index):
    """Common positive gain and signed deviations for two-marker response vectors.

    Curves have shape (..., N, 2). Departures use the reference vector's norm,
    avoiding division by individual marker effects. Near-zero vectors are undefined.
    """
    reference = curves[..., ref_index:ref_index+1, :]
    refnorm2 = np.sum(reference**2, axis=-1, keepdims=True)
    gain = np.maximum(0, np.sum(curves * reference, axis=-1, keepdims=True) /
                      np.where(refnorm2 > 1e-16, refnorm2, np.nan))
    remainder = curves - gain * reference
    deviation = remainder / np.sqrt(np.where(refnorm2 > 1e-16, refnorm2, np.nan))
    length = np.linalg.norm(curves, axis=-1)
    mismatch = np.linalg.norm(remainder, axis=-1) / np.where(length > 1e-8, length, np.nan)
    return gain[..., 0], deviation, mismatch


def matched_scaling(effects, counts, reference_n, *, draws=1000, seed=42, minimum=10):
    """Use the same centers containing both markers, across every size and seed.

    Each pair has its own explicitly reported common-support cohort. Repeat the
    full scale fit inside each paired organoid bootstrap. Intervals are descriptive,
    not calibrated tests of the positive mismatch statistic against zero.
    """
    rng = np.random.default_rng(seed)
    rows, seed_rows, support = [], [], []
    ref_index = counts.index(reference_n)
    for route in ['full', 'film_only', 'head_only']:
        for hop in [1, 2]:
            data = effects[(effects.route == route) & (effects.hop == hop)]
            for metric in ['delta_relative', 'delta_z']:
                wide = data.pivot(index=['fold', 'seed', 'organoid_str', 'orig_center', 'n'],
                                  columns='source_marker_name', values=metric)
                for a, b in itertools.combinations(sorted(wide.columns), 2):
                    pair = wide[[a, b]].dropna().reset_index()
                    centers = pair[['fold', 'organoid_str', 'orig_center']].drop_duplicates()
                    org = pair.groupby(['organoid_str', 'n'])[[a, b]].mean()
                    organs = sorted(pair.organoid_str.unique())
                    support.append(dict(route=route, hop=hop, metric=metric, marker_a=a, marker_b=b,
                                        n_centers=len(centers), n_organoids=len(organs)))
                    if len(organs) < minimum:
                        continue
                    idx = pd.MultiIndex.from_product([organs, counts], names=['organoid_str', 'n'])
                    cube = org.reindex(idx).to_numpy().reshape(len(organs), len(counts), 2)
                    if not np.isfinite(cube).all():
                        raise ValueError('Common-support cohort changed with N.')
                    mean = cube.mean(axis=0)
                    gain, deviation, mismatch = scale_statistics(mean, ref_index)
                    boot_means = cube[rng.integers(len(cube), size=(draws, len(cube)))].mean(axis=1)
                    bg, bd, bm = scale_statistics(boot_means, ref_index)
                    for j, n in enumerate(counts):
                        record = dict(route=route, hop=hop, metric=metric, marker_a=a, marker_b=b, n=n,
                                      n_organoids=len(organs), n_centers=len(centers))
                        for name, value, samples in [('gain', gain[j], bg[:, j]),
                                ('deviation_a', deviation[j, 0], bd[:, j, 0]),
                                ('deviation_b', deviation[j, 1], bd[:, j, 1]),
                                ('mismatch', mismatch[j], bm[:, j])]:
                            low, high = np.nanquantile(samples, [.025, .975])
                            record[name], record[name+'_low'], record[name+'_high'] = value, low, high
                        rows.append(record)
                    for model_seed, group in pair.groupby('seed'):
                        seed_mean = group.groupby(['organoid_str', 'n'])[[a,b]].mean().groupby('n').mean().reindex(counts).to_numpy()
                        sg, sd, sm = scale_statistics(seed_mean, ref_index)
                        for j, n in enumerate(counts):
                            seed_rows.append(dict(route=route, hop=hop, metric=metric, marker_a=a, marker_b=b,
                                seed=model_seed, n=n, gain=sg[j], deviation_a=sd[j,0], deviation_b=sd[j,1], mismatch=sm[j]))
    return pd.DataFrame(rows), pd.DataFrame(seed_rows), pd.DataFrame(support)


def summarize_diagnostics(out, *, draws=1000, seed=42):
    out = Path(out)
    config = json.loads((out / 'settings.json').read_text())
    expected = [(fold, model_seed) for fold in range(5) for model_seed in config['seeds']]
    for fold, model_seed in expected:
        if not (out / f'fold_{fold}_seed_{model_seed}_done.json').exists():
            raise RuntimeError('Inference is incomplete; do not summarize partial checkpoints.')
    effects = pd.concat([pd.read_csv(out / f'fold_{f}_seed_{s}_effects.csv.gz') for f,s in expected], ignore_index=True)
    interactions = pd.concat([pd.read_csv(out / f'fold_{f}_seed_{s}_interactions.csv.gz') for f,s in expected], ignore_index=True)
    # Layer 1 cannot receive signals from hop 2. Suppress direction/ratio statistics
    # for structurally absent effects (and cancellation noise in factorial contrasts).
    effects.loc[effects.hop == 2, ['h1_ratio', 'h1_cosine']] = np.nan
    interactions.loc[interactions.hop == 2, 'h1_interaction_ratio'] = np.nan
    effect_keys=['route', 'source_marker_name', 'hop', 'n']
    inter_keys=['kind', 'marker_a', 'marker_b', 'hop', 'n']
    print('Summarize route and hidden-state effects', flush=True)
    tables={
        'effect_summary':cluster_summary(effects,effect_keys,EFFECT_METRICS,draws=draws,seed=seed),
        'interaction_summary':cluster_summary(interactions,inter_keys,INTERACTION_METRICS,draws=draws,seed=seed),
    }
    # Retain model-seed variation separately from conditional organoid bootstraps.
    tables['effect_seed_summary']=effects.groupby(['seed',*effect_keys,'organoid_str'])[EFFECT_METRICS].mean().groupby(['seed',*effect_keys]).mean().reset_index()
    tables['interaction_seed_summary']=interactions.groupby(['seed',*inter_keys,'organoid_str'])[INTERACTION_METRICS].mean().groupby(['seed',*inter_keys]).mean().reset_index()
    low, high=config['counts'][0],config['counts'][-1]
    for name,frame,keys,metrics in [('effect',effects,effect_keys,EFFECT_METRICS[:4]),
                                   ('interaction',interactions,inter_keys,INTERACTION_METRICS[:3])]:
        id_columns = [key for key in keys if key != 'n'] + ['organoid_str','fold','seed','orig_center']
        if name=='effect': id_columns+=['case_id']
        else: id_columns+=['source_a','source_b']
        paired=frame[frame.n==low].merge(frame[frame.n==high],on=id_columns,suffixes=('_low','_high'),validate='one_to_one')
        for metric in metrics: paired[metric]=paired[metric+'_high']-paired[metric+'_low']
        groupkeys=[key for key in keys if key!='n']
        tables[name+'_endpoint_summary']=cluster_summary(paired,groupkeys,metrics,draws=draws,seed=seed)
        tables[name+'_endpoint_seeds']=paired.groupby(['seed',*groupkeys,'organoid_str'])[metrics].mean().groupby(['seed',*groupkeys]).mean().reset_index()
    print('Fit matched-center common-scaling models', flush=True)
    scale,scale_seeds,scale_support=matched_scaling(effects,config['counts'],config['reference_n'],draws=draws,seed=seed)
    tables.update(common_scaling=scale, common_scaling_seeds=scale_seeds,common_scaling_support=scale_support)
    # Paired difference between full routing and the sum of isolated FiLM/head changes.
    # A nonzero result means route effects do not decompose additively.
    wide=effects.pivot(index=['fold','seed','organoid_str','case_id','source_marker_name','hop','n'],columns='route',values='delta_z').reset_index()
    ref=wide[wide.n==config['reference_n']][['fold','seed','case_id','full']].rename(columns={'full':'reference'})
    wide=wide.merge(ref,on=['fold','seed','case_id'],validate='many_to_one')
    wide['route_interaction_z']=wide.full-wide.film_only-wide.head_only+wide.reference
    tables['route_interaction_summary']=cluster_summary(wide,['source_marker_name','hop','n'],['route_interaction_z'],draws=draws,seed=seed)
    # LGR5-positive includes coexpressing centers; expose this context in support counts.
    tables['center_coexpression_support']=effects[effects.n==low].query("route == 'full'").drop_duplicates(['fold','organoid_str','orig_center']).groupby('center_markers').agg(n_centers=('orig_center','size'),n_organoids=('organoid_str','nunique')).reset_index()
    for name,table in tables.items():table.to_csv(out/f'{name}.csv',index=False)
    (out/'summary_settings.json').write_text(json.dumps(dict(bootstrap_samples=draws,seed=seed,minimum_profile_organoids=10),indent=2))
    return tables
