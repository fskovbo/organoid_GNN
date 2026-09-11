"""Post-training area-based normalization of saved size-ablation cases.

No model or graph loading. Area references are fitted using training organoids
only. Normalization precedes aggregation and retains equal organoid weighting.
"""
import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd


def fit_area_references(cohort, membership):
    """Fit log(A) = alpha + beta log(N) independently in each training fold.

    Back-transformed predictions are geometric-mean references, without a
    lognormal smearing correction. Validation observations only assess the fit.
    """
    if cohort.organoid_str.duplicated().any():
        raise ValueError('Expected one metadata row per organoid.')
    if not np.isfinite(cohort[['n_cells', 'surface_area']]).all().all() or (cohort[['n_cells', 'surface_area']] <= 0).any().any():
        raise ValueError('Cell counts and areas must be positive and finite.')
    joined = membership[['fold', 'role', 'organoid_str']].merge(
        cohort[['organoid_str', 'n_cells', 'surface_area']], on='organoid_str', validate='many_to_one')
    if len(joined) != len(membership) or joined.duplicated(['fold', 'role', 'organoid_str']).any():
        raise ValueError('Missing or repeated split metadata.')
    fits, diagnostics = [], []
    for fold, frame in joined.groupby('fold', sort=True):
        train = frame[frame.role == 'train']
        val = frame[frame.role == 'val']
        if set(train.organoid_str) & set(val.organoid_str):
            raise ValueError('Training and validation organoids overlap.')
        x, y = np.log(train.n_cells.to_numpy()), np.log(train.surface_area.to_numpy())
        if len(x) < 3 or np.ptp(x) == 0:
            raise ValueError('Area reference requires at least three training organoids and varying N.')
        alpha, beta = np.linalg.lstsq(np.column_stack([np.ones(len(x)), x]), y, rcond=None)[0]
        record = dict(fold=int(fold), alpha=float(alpha), beta=float(beta), n_train=len(train),
                      n_val=len(val), train_min_n=train.n_cells.min(), train_max_n=train.n_cells.max())
        for role, group in [('train', train), ('val', val)]:
            actual = np.log(group.surface_area.to_numpy())
            predicted = alpha + beta * np.log(group.n_cells.to_numpy())
            residual = actual - predicted
            denom = np.sum((actual - actual.mean()) ** 2)
            record[f'{role}_r2_log_area'] = 1 - np.sum(residual ** 2) / denom if denom > 0 else np.nan
            record[f'{role}_rmse_log_area'] = float(np.sqrt(np.mean(residual ** 2)))
            diagnostics.append(group.assign(area_reference=np.exp(predicted), log_area_residual=residual))
        fits.append(record)
    return pd.DataFrame(fits), pd.concat(diagnostics, ignore_index=True)


def normalize_cases(frame, references):
    """Use evaluated N (observed or swept), with the case's training-fold fit."""
    result = frame.merge(references[['fold', 'alpha', 'beta']], on='fold', how='left', validate='many_to_one')
    if result[['alpha', 'beta', 'evaluated_n', 'delta_mu']].isna().any().any() or (result.evaluated_n <= 0).any():
        raise ValueError('Missing fold calibration or invalid ablation values.')
    result['area_reference'] = np.exp(result.alpha + result.beta * np.log(result.evaluated_n))
    result['curvature_reference'] = 4 * np.pi / result.area_reference
    result['delta_relative'] = result.delta_mu / result.curvature_reference
    return result


def load_normalized_effects(run_dir, data_dir, models, *, bootstrap_samples=1000, seed=42, progress=None):
    """Read saved metadata/cases and return fits, diagnostics, organoid means, summaries.

    Files are reduced one at a time to bound memory. The reduction preserves
    case counts and sums; all cases/seeds within an organoid are averaged before
    organoids receive equal weight. Confidence intervals condition on the fits.
    """
    from src.analysis.pseudotime import summarize_by_organoid

    run_dir, data_dir = Path(run_dir), Path(data_dir)
    settings = json.loads((run_dir / 'settings.json').read_text())
    tables = run_dir / 'tables'
    cohort = pd.read_csv(tables / 'cohort.csv')
    areas = []
    for row in cohort.itertuples():
        metadata = json.loads((data_dir / f'{row.organoid_str}_aux.json').read_text())
        if float(metadata['num_nodes']) != float(row.n_cells):
            raise ValueError(f'Metadata count changed since the run: {row.organoid_str}')
        area = metadata['total_surface_area']
        areas.append(float(area[0] if isinstance(area, list) else area))
    cohort['surface_area'] = areas
    membership = pd.read_csv(tables / 'split_membership.csv')
    references, diagnostics = fit_area_references(cohort, membership)
    edges = np.unique(np.quantile(cohort.n_cells, np.linspace(0, 1, settings['N_SIZE_BINS'] + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    bins = pd.cut(cohort.n_cells, edges, labels=False, include_lowest=True)
    bin_lookup = dict(zip(cohort.organoid_str, bins))
    bin_centers = cohort.assign(size_bin=bins).groupby('size_bin').n_cells.median()
    counts = json.loads((run_dir / 'sweep_grid.json').read_text())
    paths = []
    for fold in references.fold:
        for model in models:
            if model not in settings['MODEL_NAMES']:
                raise ValueError(f'Model absent from saved run: {model}')
            for model_seed in settings['MODEL_SEEDS']:
                prefix = tables / f'fold_{fold}_seed_{model_seed}_{model}'
                paths.append((Path(str(prefix) + '_observed.csv'), fold, model, model_seed, None))
                paths.extend((Path(str(prefix) + f'_N{count}_sweep.csv'), fold, model, model_seed, count) for count in counts)
    missing = [str(path) for path, *_ in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f'Incomplete saved ablation run: {missing[:3]}')
    keys = ['model', 'fold', 'organoid_str', 'center_marker', 'source_marker_name', 'hop', 'analysis', 'coordinate']
    pieces = []
    required = ['organoid_str', 'center_marker_names', 'source_marker_name', 'hop', 'analysis',
                'evaluated_n', 'observed_n', 'delta_mu', 'fold', 'seed', 'model', 'case_id']
    for index, (path, fold, model, model_seed, count) in enumerate(paths):
        frame = pd.read_csv(path, usecols=required)
        if not ((frame.fold == fold) & (frame.model == model) & (frame.seed == model_seed)).all():
            raise ValueError(f'Unexpected file identity: {path}')
        allowed = membership[(membership.fold == fold) & (membership.role == 'val')].organoid_str
        if not frame.organoid_str.isin(allowed).all() or frame.case_id.duplicated().any():
            raise ValueError(f'Invalid validation case membership: {path}')
        mode = 'observed' if count is None else 'sweep'
        expected_n = frame.observed_n if count is None else count
        if not (frame.analysis == mode).all() or not np.allclose(frame.evaluated_n, expected_n):
            raise ValueError(f'Incorrect size coordinates: {path}')
        frame = normalize_cases(frame, references)
        frame['coordinate'] = frame.organoid_str.map(bin_lookup) if count is None else frame.evaluated_n
        frame['center_marker'] = frame.center_marker_names.map(lambda value: ast.literal_eval(value) or ['unmarked'])
        frame = frame.explode('center_marker')
        pieces.append(frame.groupby(keys, observed=True).agg(
            raw_sum=('delta_mu', 'sum'), relative_sum=('delta_relative', 'sum'), n_cases=('delta_mu', 'size')).reset_index())
        if progress and (index + 1) % 50 == 0:
            progress(f'Read {index + 1}/{len(paths)} saved case tables')
    organoids = pd.concat(pieces, ignore_index=True).groupby(keys, observed=True).agg(
        raw_sum=('raw_sum', 'sum'), relative_sum=('relative_sum', 'sum'), n_cases=('n_cases', 'sum')).reset_index()
    organoids['delta_mu'] = organoids.raw_sum / organoids.n_cases
    organoids['delta_relative'] = organoids.relative_sum / organoids.n_cases
    groups = ['model', 'center_marker', 'source_marker_name', 'hop', 'coordinate']
    summaries = []
    for mode in ['observed', 'sweep']:
        subset = organoids[organoids.analysis == mode]
        support = subset.groupby(groups).n_cases.sum().rename('n_rows').reset_index()
        for metric in ['delta_mu', 'delta_relative']:
            table = summarize_by_organoid(subset, groups, metric, bootstrap_samples=bootstrap_samples, seed=seed)
            table = table.drop(columns='n_rows').merge(support, on=groups, validate='one_to_one')
            table['analysis'], table['metric'] = mode, metric
            table['n'] = table.coordinate.map(bin_centers) if mode == 'observed' else table.coordinate
            summaries.append(table)
    return dict(cohort=cohort, references=references, diagnostics=diagnostics,
                organoids=organoids, summary=pd.concat(summaries, ignore_index=True))
