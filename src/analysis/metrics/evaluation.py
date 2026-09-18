"""Per-node prediction tables from restored runs, independent of architecture."""
import numpy as np
import pandas as pd
from src.inference.predict import predict_targets


def compare_baseline_mse(scores, baseline_scores):
    """Pair physical-curvature MSEs by fold/organoid, preserving model weighting."""
    baseline = pd.DataFrame(baseline_scores)[['fold', 'organoid_str', 'mse']].rename(
        columns={'mse': 'baseline_mse'})
    result = pd.DataFrame(scores).merge(baseline, on=['fold', 'organoid_str'],
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
        rows.append(frame)
        start = stop
    return pd.concat(rows, ignore_index=True)
