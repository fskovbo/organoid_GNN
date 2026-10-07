"""Per-node prediction tables from restored runs, independent of architecture."""
import numpy as np
import pandas as pd
from src.inference.predict import predict_targets


def organoid_mse_summary(scores, by=(), *, value='mse'):
    """Equal-organoid mean and SEM, averaging repeated predictions per organoid.

    Seeds/repeated splits are not independent organoids. The SEM is descriptive
    held-out organoid variation conditional on the fitted models, not retraining
    uncertainty. With fewer than two organoids, SEM remains undefined (NaN).
    """
    by = [by] if isinstance(by, str) else list(by)
    units = scores.groupby(by + ['organoid_str'], observed=True, dropna=False)[value].mean()
    if by:
        return units.groupby(level=by, observed=True, dropna=False).agg(
            ['mean', 'std', 'sem', 'count']).reset_index()
    return pd.DataFrame([dict(mean=units.mean(), std=units.std(),
                              sem=units.sem(), count=units.count())])


def compare_baseline_mse(scores, baseline_scores):
    """Pair physical-curvature MSEs by fold/organoid, preserving model weighting."""
    baseline = pd.DataFrame(baseline_scores)[['fold', 'organoid_str', 'mse']].rename(
        columns={'mse': 'baseline_mse'})
    # Resumed runs may reload the previously paired table. Recompute derived
    # columns instead of creating baseline_mse_x / baseline_mse_y on a merge.
    model_scores = pd.DataFrame(scores).drop(
        columns=['baseline_mse', 'mse_minus_baseline'], errors='ignore')
    result = model_scores.merge(baseline, on=['fold', 'organoid_str'],
                                       how='left', validate='many_to_one')
    if result['baseline_mse'].isna().any():
        raise ValueError('Missing baseline MSE for one or more validation organoids')
    result['mse_minus_baseline'] = result['mse'] - result['baseline_mse']
    return result


def prediction_table(selection, *, role='val', device='cpu', batch_size=64):
    """Physical curvature means/errors plus variance in fitted target units.

    Physical uncertainty after a nonlinear inverse is not Gaussian; retain
    transformed variance for exact model-scale calibration instead of labelling
    a delta-method approximation as physical predictive variance.
    """
    group = selection['groups'][role]
    z, mu, lv, _ = predict_targets(group, selection['model'], device=device,
                                   batch_size=batch_size, return_log_var=True)
    y, pred, _ = selection['transform'].inverse_distribution(z, mu, lv, graphs=group)
    rows, start = [], 0
    for g in group:
        stop = start + len(g.x)
        offset = np.asarray(selection['baseline_offsets'][g.organoid_str]).reshape(-1)
        sl = slice(start, stop)
        truth = np.asarray(y)[sl].reshape(-1) + offset
        estimate = np.asarray(pred)[sl].reshape(-1) + offset
        frame = pd.DataFrame(dict(organoid_str=str(g.organoid_str), node=np.arange(len(g.x)),
            n_cells=len(g.x), y_true=truth, y_pred=estimate, error=estimate-truth,
            squared_error=(estimate-truth)**2, y_transformed=np.asarray(z)[sl].reshape(-1),
            pred_transformed=np.asarray(mu)[sl].reshape(-1), variance_transformed=np.exp(np.asarray(lv)[sl].reshape(-1))))
        baseline_predictions = selection.get('baseline_predictions')
        if baseline_predictions is not None:
            frame['baseline_prediction'] = np.asarray(baseline_predictions[g.organoid_str]).reshape(-1)
            frame['baseline_squared_error'] = (frame.y_true-frame.baseline_prediction)**2
            frame['baseline_absolute_error'] = np.abs(frame.y_true-frame.baseline_prediction)
        frame['absolute_error'] = np.abs(frame.error)
        rows.append(frame)
        start = stop
    return pd.concat(rows, ignore_index=True)


def regional_mse(predictions, annotations):
    """Physical MSE per organoid/region, preserving node identity and baseline.

    Expects predictions from one checkpoint. No variance or curvature matching
    enters the region assignment. Organoids without cells in a region contribute
    no row, rather than a zero error.
    """
    keys = ['organoid_str', 'node']
    if predictions.duplicated(keys).any() or annotations.duplicated(keys).any():
        raise ValueError('Regional evaluation requires unique organoid/node keys.')
    cells = predictions.merge(annotations[keys + ['region']], on=keys, how='left', validate='one_to_one')
    if cells.region.isna().any():
        raise ValueError('Missing region annotations for predicted cells.')
    aggregations = dict(n_cells=('node', 'size'), mse=('squared_error', 'mean'))
    if 'baseline_squared_error' in cells:
        aggregations['baseline_mse'] = ('baseline_squared_error', 'mean')
    return cells.groupby(['organoid_str', 'region'], observed=True).agg(**aggregations).reset_index()
