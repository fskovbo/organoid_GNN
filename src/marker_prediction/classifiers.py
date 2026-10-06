"""Organoid-disjoint classification, calibration and high-precision decisions."""
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import precision_recall_curve


def organoid_splits(cohort, target_names, n_folds=5, calibration_fraction=.25, seed=42):
    """Outer held-out organoids; an inner held-out calibration group per fold.

    Stratify on joint positive-marker presence when supported. Otherwise use the
    rarest marker's presence, rather than allowing a cell-level fallback split.
    The caller supplies only organoids with all requested stains measured.
    """
    if not np.all(cohort[[f'measured_{m}' for m in target_names]].values):
        raise ValueError('Unmeasured organoids cannot be split as negative examples.')
    positive = cohort[[f'positive_{m}' for m in target_names]].to_numpy() > 0
    strata = positive.astype(int) @ (2 ** np.arange(len(target_names)))
    _, counts = np.unique(strata, return_counts=True)
    if counts.min() < max(n_folds, int(np.ceil(1/calibration_fraction))):
        strata = positive[:, np.argmin(positive.sum(axis=0))].astype(int)
    for fold, (development, test) in enumerate(StratifiedKFold(n_folds, shuffle=True, random_state=seed).split(cohort, strata)):
        train, calibration = train_test_split(development, test_size=calibration_fraction,
            random_state=seed+fold, stratify=strata[development])
        split = {role: cohort.iloc[indices].graph_id.astype(int).tolist()
                 for role, indices in [('train', train), ('calibration', calibration), ('test', test)]}
        for role, ids in split.items():
            rows = cohort[cohort.graph_id.isin(ids)]
            for marker in target_names:
                if rows[f'positive_{marker}'].sum() == 0:
                    raise ValueError(f'No {marker} positives in fold {fold} {role}.')
        yield dict(fold=fold, **split)


def fit_classifier(x, y, x_calibration, y_calibration, *, seed, max_iter=120,
                   learning_rate=.08, max_leaf_nodes=15, min_samples_leaf=40,
                   l2_regularization=10., balanced=True):
    """Fit boosting on training cells and sigmoid calibration on different organoids.

    Disable sklearn's automatic cell-wise early-stopping split. Calibration keeps
    the natural prevalence even when the training loss is class-balanced.
    """
    if len(np.unique(y)) != 2 or len(np.unique(y_calibration)) != 2:
        raise ValueError('Training and calibration must each contain both classes.')
    estimator = HistGradientBoostingClassifier(max_iter=max_iter, learning_rate=learning_rate,
        max_leaf_nodes=max_leaf_nodes, min_samples_leaf=min_samples_leaf,
        l2_regularization=l2_regularization, class_weight='balanced' if balanced else None,
        early_stopping=False, random_state=seed)
    estimator.fit(x, y)
    calibration_scores = estimator.decision_function(x_calibration).reshape(-1, 1)
    calibrator = LogisticRegression(C=1000., max_iter=1000, random_state=seed)
    calibrator.fit(calibration_scores, y_calibration)
    return dict(estimator=estimator, calibrator=calibrator)


def probabilities(model, x):
    return model['calibrator'].predict_proba(model['estimator'].decision_function(x).reshape(-1, 1))[:, 1]


def precision_threshold(y, probability, graph_ids, target=.95, min_calls=20, min_organoids=5):
    """Maximize calibration recall subject to empirical precision and support.

    This is a threshold-selection rule, not a precision guarantee. The outer test
    determines achieved precision. Return infinity when no supported rule exists.
    """
    precision, recall, thresholds = precision_recall_curve(y, probability)
    eligible = np.flatnonzero(precision[:-1] >= target)
    if not len(eligible):
        return np.inf
    order = eligible[np.argsort(-recall[eligible], kind='stable')]
    for i in order:
        called = probability >= thresholds[i]
        if called.sum() >= min_calls and len(np.unique(graph_ids[called])) >= min_organoids:
            return float(thresholds[i])
    return np.inf
