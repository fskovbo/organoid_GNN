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
