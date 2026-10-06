"""Rare-marker metrics with organoid-level support and uncertainty."""
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


def classification_metrics(y, p, threshold):
    y = np.asarray(y, dtype=bool)
    called = np.asarray(p) >= threshold
    tp, fp = int((called & y).sum()), int((called & ~y).sum())
    fn, tn = int((~called & y).sum()), int((~called & ~y).sum())
    return dict(n_cells=len(y), positives=int(y.sum()), calls=int(called.sum()), tp=tp, fp=fp, fn=fn, tn=tn,
        precision=tp/(tp+fp) if tp+fp else np.nan, recall=tp/(tp+fn) if tp+fn else np.nan,
        fp_per_1000=1000*fp/len(y), false_positive_rate=fp/(fp+tn) if fp+tn else np.nan,
        average_precision=average_precision_score(y, p) if y.any() else np.nan,
        roc_auc=roc_auc_score(y, p) if len(np.unique(y)) == 2 else np.nan,
        brier=brier_score_loss(y, p))


def organoid_calls(y, p, graph_ids, threshold):
    frame = pd.DataFrame(dict(graph_id=graph_ids, positive=np.asarray(y, int), probability=p,
        call=(np.asarray(p) >= threshold).astype(int)))
    frame['tp'] = frame.positive * frame.call
    frame['fp'] = (1-frame.positive) * frame.call
    return frame.groupby('graph_id').agg(n_cells=('positive', 'size'), positives=('positive', 'sum'),
        expected_positives=('probability', 'sum'), calls=('call', 'sum'), tp=('tp', 'sum'), fp=('fp', 'sum')).reset_index()


def bootstrap_precision_recall(calls, n_bootstrap=500, seed=42):
    """Resample whole test organoids; intervals condition on fitted classifiers."""
    rng = np.random.default_rng(seed)
    values = calls[['tp', 'fp', 'positives', 'n_cells']].to_numpy(float)
    samples = np.asarray([values[rng.integers(len(values), size=len(values))].sum(axis=0) for _ in range(n_bootstrap)])
    tp, fp, positive, n = samples.T
    prec = np.divide(tp, tp+fp, out=np.full_like(tp, np.nan), where=tp+fp>0)
    recall = np.divide(tp, positive, out=np.full_like(tp, np.nan), where=positive>0)
    result = {}
    for name, vector in [('precision', prec), ('recall', recall), ('fp_per_1000', 1000*fp/n)]:
        valid = vector[np.isfinite(vector)]
        result[name+'_low'], result[name+'_high'] = np.quantile(valid, [.025, .975]) if len(valid) else (np.nan, np.nan)
    return result
