"""Observed exclusive-KI67 neighborhoods; no model predictions or interventions."""
from pathlib import Path
from dataclasses import dataclass, asdict
import json
import numpy as np
import pandas as pd
from scipy import sparse

MARKERS = ['Agr2', 'AldoB', 'Chroma', 'KI67', 'LGR5', 'Lysozyme', 'Serotonin', 'Unmarked']
BIN_LABELS = ['<150', '150–249', '250–349', '350–499', '500–799', '≥800']


@dataclass(frozen=True)
class ObservedConfig:
    size_edges: tuple = (0, 150, 250, 350, 500, 800, float('inf'))
    neck_low: float = .8
    neck_high: float = 1.2
    bootstrap_draws: int = 1000
    minimum_organoids: int = 10
    minimum_lgr5_per_group: int = 3
    seed: int = 1841


def adjacency_rings(edges, n):
    edges = np.asarray(edges, dtype=int).reshape(-1, 2)
    a = sparse.csr_matrix((np.ones(2*len(edges)),
        (np.r_[edges[:, 0], edges[:, 1]], np.r_[edges[:, 1], edges[:, 0]])), shape=(n, n))
    a.setdiag(0); a.eliminate_zeros(); a.data[:] = 1
    two = (a @ a).tocsr(); two.data[:] = 1
    two = two - two.multiply(a)
    two.setdiag(0); two.eliminate_zeros()
    return a, two


def ring_average(adjacency, values):
    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    result = np.asarray(adjacency @ values, dtype=float)
    denom = degree if result.ndim == 1 else degree[:, None]
    return np.divide(result, denom, out=np.full_like(result, np.nan), where=denom > 0)


def neck_regions(axis, low=.8, high=1.2):
    return np.select([~np.isfinite(axis), axis < low, axis <= high],
                     ['no_detected_crypt', 'crypt_body', 'neck_band'], default='beyond_neck')


def within_organoid_contrast(frame, minimum=3):
    """LGR5 with KI67 neighbor minus LGR5 without, requiring both groups."""
    rows = []
    measures = ['curvature', 'curvature_norm', 'curvature_rank', 'negative', 'axis',
                'ring1_fraction_AldoB', 'ring1_fraction_LGR5']
    for stratum, part in [('all', frame), *list(frame.groupby('region'))]:
        positive, negative = part[part.has_ki67_neighbor], part[~part.has_ki67_neighbor]
        if min(len(positive), len(negative)) < minimum:
            continue
        row = dict(stratum=stratum, with_count=len(positive), without_count=len(negative))
        for metric in measures:
            row[metric] = positive[metric].mean() - negative[metric].mean()
        rows.append(row)
    return rows


def bootstrap_summary(frame, groups, metrics, config):
    """One row per organoid per group before bootstrap; equal organoid weight."""
    rng = np.random.default_rng(config.seed)
    rows = []
    for key, part in frame.groupby(groups, observed=True, dropna=False):
        if part.organoid_str.duplicated().any():
            raise ValueError('Summarize organoids first')
        key = key if isinstance(key, tuple) else (key,)
        x = part[metrics].to_numpy(dtype=float)
        valid = np.isfinite(x); count = valid.sum(0)
        x0 = np.where(valid, x, 0)
        mean = np.divide(x0.sum(0), count, out=np.full(len(metrics), np.nan), where=count > 0)
        weights = rng.multinomial(len(part), np.full(len(part), 1/len(part)), size=config.bootstrap_draws)
        denominator = weights @ valid.astype(float)
        boot = np.divide(weights @ x0, denominator, out=np.full((len(weights), len(metrics)), np.nan), where=denominator > 0)
        for j, metric in enumerate(metrics):
            enough = count[j] >= config.minimum_organoids
            lo, hi = np.nanquantile(boot[:, j], [.025, .975]) if enough else (np.nan, np.nan)
            rows.append(dict(zip(groups, key)) | dict(metric=metric, mean=mean[j], low=lo, high=hi,
                n_organoids=int(count[j]), supported=bool(enough)))
    return pd.DataFrame(rows)


def run(root, config=ObservedConfig()):
    root = Path(root)
    run_dir = root / 'results_experiments/size_conditioned_exclusive/20260914_ordered_exclusive_film_depth2/exclusive'
    out = root / 'results_experiments/ki67_observed_neighborhoods/exclusive_v1'
    out.mkdir(parents=True, exist_ok=True)
    settings = json.loads((run_dir/'settings.json').read_text())
    data = root/'training_data'/settings['DATASET_NAME']
    original = root/'training_data'/'after_cleanup'
    metadata = json.loads(json.dumps(dict(config=asdict(config), dataset=str(data), run=str(run_dir))))
    if (out/'settings.json').exists() and json.loads((out/'settings.json').read_text()) != metadata:
        raise ValueError('Settings differ; use a new versioned output directory')
    (out/'settings.json').write_text(json.dumps(metadata, indent=2))
    if (out/'complete.json').exists():
        add_adjacent_lgr5_checks(out, config)
        return out
    membership = pd.read_csv(run_dir/'tables/split_membership.csv').query("role == 'val'")
    assert not membership.organoid_str.duplicated().any()
    all_nodes, organoids, contrasts = [], [], []
    for index, row in enumerate(membership.itertuples()):
        org = row.organoid_str
        meta = json.loads((data/f'{org}_aux.json').read_text())
        with np.load(data/f'{org}.npz', allow_pickle=False) as z:
            x, y, edges = z['x'], z['y'][:, 0], z['edges']
            d = z['d_crypts_graph']
        assert json.loads((data/f'{org}_markers.json').read_text()) == MARKERS[:-1]
        n = len(x)
        assert n == row.n_cells and np.all((x == 0) | (x == 1)) and np.all(x.sum(1) <= 1)
        assert d.ndim == 2 and d.shape[1] == n and np.isfinite(d).all()
        with np.load(original/f'{org}.npz', allow_pickle=False) as z:
            full_x = z['x']; np.testing.assert_array_equal(y, z['y'][:, 0])
            np.testing.assert_array_equal(edges, z['edges'])
        assert full_x.shape == x.shape
        area = float(meta['total_surface_area'])
        assert np.isfinite(area) and area > 0
        axis = d.min(0) if len(d) else np.full(n, np.nan)
        region = neck_regions(axis, config.neck_low, config.neck_high)
        fate = np.c_[x, x.sum(1) == 0].astype(float)
        names = np.asarray(MARKERS)[fate.argmax(1)]
        bin_id = int(np.searchsorted(config.size_edges, n, side='right')-1)
        # Keep collection label and biological day separate; day4p5-more is not later than day4p5.
        day = {'day3p5': 3.5, 'day4': 4., 'day4p5': 4.5, 'day4p5-more': 4.5}[meta['timepoint']]
        common = dict(organoid_str=org, n=n, size_bin=bin_id, timepoint=meta['timepoint'], day=day,
                      dataset=str(meta['dataset']), stratum_time=f"{meta['dataset']} / {meta['timepoint']}")
        frame = pd.DataFrame(dict(node=np.arange(n), marker=names, curvature=y,
            curvature_norm=y*area/(4*np.pi), curvature_centered=(y-y.mean())*area/(4*np.pi),
            curvature_rank=pd.Series(y).rank(method='average', pct=True), negative=(y<0).astype(float),
            axis=axis, region=region))
        frame = frame.assign(**common)
        for label in ['crypt_body', 'neck_band', 'beyond_neck']:
            frame[f'fraction_{label}'] = np.where(np.isfinite(axis), region == label, np.nan)
            frame[f'enrichment_{label}'] = frame[f'fraction_{label}'] - frame[f'fraction_{label}'].mean()
        for hop, adj in enumerate(adjacency_rings(edges, n), 1):
            frame[f'ring{hop}_degree'] = np.asarray(adj.sum(1)).ravel()
            fractions = ring_average(adj, fate)
            expected = (fate.sum(0)[None, :] - fate)/(n-1)
            for j, marker in enumerate(MARKERS):
                frame[f'ring{hop}_fraction_{marker}'] = fractions[:, j]
                frame[f'ring{hop}_enrichment_{marker}'] = fractions[:, j] - expected[:, j]
            frame[f'ring{hop}_curvature_norm'] = ring_average(adj, y*area/(4*np.pi))
            frame[f'ring{hop}_curvature_rank'] = ring_average(adj, frame.curvature_rank.to_numpy())
        frame['has_ki67_neighbor'] = frame.ring1_fraction_KI67 > 0
        frame['mixed_lgr5_aldob_neighbors'] = ((frame.ring1_fraction_LGR5 > 0) & (frame.ring1_fraction_AldoB > 0)).astype(float)
        for contrast in within_organoid_contrast(frame[frame.marker == 'LGR5'], config.minimum_lgr5_per_group):
            contrasts.append(common | contrast)
        full_ki = int(full_x[:, MARKERS.index('KI67')].sum())
        retained_ki = int(x[:, MARKERS.index('KI67')].sum())
        organoids.append(common | dict(n_crypts=len(d), n_ki67=retained_ki, n_lgr5=int((names=='LGR5').sum()),
            fraction_ki67=retained_ki/n, full_ki67_fraction=full_ki/n,
            ki67_retained=retained_ki/full_ki if full_ki else np.nan,
            detected_crypt=float(len(d)>0), surface_area=area))
        all_nodes.append(frame)
        if (index+1)%100 == 0:
            print(f'Observed census: {index+1}/{len(membership)} organoids', flush=True)
    nodes = pd.concat(all_nodes, ignore_index=True)
    organs = pd.DataFrame(organoids)
    paired = pd.DataFrame(contrasts)
    nodes.to_csv(out/'nodes.csv.gz', index=False)
    organs.to_csv(out/'organoids.csv', index=False)
    paired.to_csv(out/'lgr5_with_minus_without_ki67.csv', index=False)
    group_cols = ['organoid_str', 'n', 'size_bin', 'timepoint', 'day', 'dataset', 'stratum_time', 'marker']
    metrics = [c for c in nodes.select_dtypes(include=np.number).columns if c not in ['node','n','size_bin','day']]
    profiles = nodes.groupby(group_cols, observed=True)[metrics].mean().reset_index()
    profiles['n_centers'] = nodes.groupby(group_cols, observed=True).size().to_numpy()
    profiles.to_csv(out/'organoid_marker_profiles.csv', index=False)
    for name, groups in [('profiles_by_size',['marker','size_bin']), ('profiles_by_time',['marker','stratum_time']),
                         ('profiles_size_within_time',['marker','stratum_time','size_bin'])]:
        bootstrap_summary(profiles, groups, metrics, config).to_csv(out/f'{name}.csv',index=False)
    overview_metrics=['fraction_ki67','full_ki67_fraction','ki67_retained','detected_crypt']
    bootstrap_summary(organs,['size_bin'],overview_metrics,config).to_csv(out/'overview_by_size.csv',index=False)
    contrast_metrics=['curvature','curvature_norm','curvature_rank','negative','axis','ring1_fraction_AldoB','ring1_fraction_LGR5']
    for name, groups in [('lgr5_contrasts_by_size',['stratum','size_bin']), ('lgr5_contrasts_by_time',['stratum','stratum_time'])]:
        bootstrap_summary(paired,groups,contrast_metrics,config).to_csv(out/f'{name}.csv',index=False)
    # KI67 vs LGR5 in the SAME organoid controls for changing whole-organoid geometry/composition.
    pair_metrics=['curvature_norm','curvature_rank','axis','fraction_neck_band','enrichment_neck_band']
    ki = profiles[profiles.marker=='KI67'].set_index('organoid_str')
    lg = profiles[profiles.marker=='LGR5'].set_index('organoid_str')
    ids=ki.index.intersection(lg.index)
    diffs=ki.loc[ids,['n','size_bin','stratum_time']].copy()
    diffs[pair_metrics]=ki.loc[ids,pair_metrics]-lg.loc[ids,pair_metrics]
    diffs=diffs.reset_index()
    diffs.to_csv(out/'ki67_minus_lgr5_paired.csv',index=False)
    bootstrap_summary(diffs,['size_bin'],pair_metrics,config).to_csv(out/'ki67_minus_lgr5_by_size.csv',index=False)
    (out/'complete.json').write_text(json.dumps(dict(organoids=len(organs), cells=len(nodes),
        ki67_cells=int((nodes.marker=='KI67').sum()), ki67_organoids=int((organs.n_ki67>0).sum()),
        detected_crypt_organoids=int(organs.detected_crypt.sum()),
        lgr5_paired_organoids=int(paired[paired.stratum=='all'].organoid_str.nunique())), indent=2))
    print('Observed KI67 analysis complete', flush=True)
    add_adjacent_lgr5_checks(out, config)
    return out


def add_adjacent_lgr5_checks(out, config):
    """Relevance to the ablation cohort and sensitivity to the neck-band width."""
    out=Path(out)
    if (out/'context_checks_complete.json').exists():
        return
    nodes=pd.read_csv(out/'nodes.csv.gz')
    conditional=nodes[(nodes.marker=='KI67') & (nodes.ring1_fraction_LGR5>0)].copy()
    metrics=['curvature_norm','curvature_rank','axis','fraction_crypt_body','fraction_neck_band',
             'enrichment_neck_band','ring1_fraction_KI67','ring1_fraction_LGR5','ring1_fraction_AldoB']
    profiles=conditional.groupby(['organoid_str','size_bin','stratum_time'])[metrics].mean().reset_index()
    profiles.to_csv(out/'ki67_adjacent_lgr5_profiles.csv',index=False)
    bootstrap_summary(profiles,['size_bin'],metrics,config).to_csv(out/'ki67_adjacent_lgr5_by_size.csv',index=False)
    sensitivity=[]
    for half_width in [.1,.2,.3]:
        for org, group in nodes[np.isfinite(nodes.axis)].groupby('organoid_str'):
            for cohort, subset in [('all_KI67',group[group.marker=='KI67']),
                                  ('KI67_adjacent_LGR5',group[(group.marker=='KI67')&(group.ring1_fraction_LGR5>0)])]:
                if subset.empty:
                    continue
                fraction=subset.axis.between(1-half_width,1+half_width).mean()
                sensitivity.append(dict(organoid_str=org,size_bin=int(group.size_bin.iloc[0]),cohort=cohort,
                    half_width=half_width,fraction=fraction,
                    enrichment=fraction-group.axis.between(1-half_width,1+half_width).mean()))
    bootstrap_summary(pd.DataFrame(sensitivity),['cohort','half_width','size_bin'],['fraction','enrichment'],config).to_csv(out/'neck_band_sensitivity.csv',index=False)
    (out/'context_checks_complete.json').write_text(json.dumps(dict(ki67_adjacent_lgr5_cells=len(conditional),organoids=len(profiles))))
