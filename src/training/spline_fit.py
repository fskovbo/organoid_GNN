"""Penalized spline fitting and organoid-level hyperparameter selection.

The outer validation fold is never used for reference fitting, knot placement,
penalty selection or residual-variance estimation. Graph features can be reused
across radii without storing a large cell-by-spline design matrix.
"""
import numpy as np
import pandas as pd
import torch
from scipy.linalg import solve
from threadpoolctl import threadpool_limits
from src.data.neighborhood_counts import exact_hop_counts, fate_identities
from src.models.fate_spline import FateSplineCurvature


def graph_samples(graphs, radius):
    """Compute exact shells once, keeping physical residual targets and IDs."""
    samples = []
    for g in graphs:
        y = g.y.detach().cpu().numpy().reshape(-1).astype(float)
        if len(y) != len(g.x) or not np.isfinite(y).all():
            raise ValueError('Finite scalar physical targets required for every cell.')
        samples.append(dict(organoid_str=str(g.organoid_str), N=len(g.x),
                            identity=fate_identities(g.x),
                            counts=exact_hop_counts(g.x, g.edge_index, radius), y=y))
    ids = [s['organoid_str'] for s in samples]
    if len(set(ids)) != len(ids) or not samples:
        raise ValueError('Expected nonempty, unique organoid samples.')
    return samples


def normal_equations(model, samples):
    """Sufficient statistics for mean organoid MSE; no cell-sized spline matrix."""
    p = model.weights.numel()
    gram, rhs = np.zeros((p, p)), np.zeros(p)
    for sample in samples:
        z = model.local_design(sample['identity'], sample['counts'])
        basis = model.basis([sample['N']])[0]
        gram += np.kron(z.T @ z / len(z), np.outer(basis, basis)) / len(samples)
        rhs += np.kron(z.T @ sample['y'] / len(z), basis) / len(samples)
    return gram, rhs


def solve_spline(model, equations, *, smoothness, ridge, pair_ridge):
    """Solve one constrained-in-parameterization, strictly penalized quadratic.

    Objective: mean organoid MSE + smoothness * sum(second coefficient
    differences squared) + ridge * non-pair coefficient norm squared
    + pair_ridge * pair coefficient norm squared. A 1e-10 numerical ridge
    acts on every coefficient, including the shared intercept smooth.
    """
    if not all(np.isfinite(v) and v >= 0 for v in (smoothness, ridge, pair_ridge)):
        raise ValueError('Penalties must be finite and nonnegative.')
    gram, rhs = equations
    k, d, r = model.n_splines, model.n_markers, model.radius
    second = np.diff(np.eye(k), n=2, axis=0)
    penalty = np.kron(np.eye(model.hidden_dim), smoothness * (second.T @ second))
    diagonal = np.full(model.hidden_dim, ridge)
    diagonal[0] = 0.
    if model.variant == 'pairwise':
        diagonal[1+d+r*d:] = pair_ridge
    penalty.flat[::len(penalty)+1] += np.repeat(diagonal, k) + 1e-10
    solution = solve(gram + penalty, rhs, assume_a='pos')
    model.weights.copy_(torch.as_tensor(solution.reshape(model.weights.shape), dtype=torch.float64))
    model.is_fitted.fill_(True)
    return model


def sample_mse(model, samples):
    return np.asarray([np.mean((model.predict_sample(s) - s['y'])**2) for s in samples])


def fit_spline(samples, *, n_markers, radius, variant, n_splines=6,
               smoothness_grid=(.01, .1, 1.), ridge=.001,
               pair_ridge_grid=(.001, .01), inner_fraction=.2, seed=42,
               blas_threads=2, callback=None):
    """Tune penalties on an inner organoid holdout, then refit all outer training.

    Shared/center models test each smoothness only once. The same seeded inner
    membership is used for every radius/variant; full training membership and
    penalty trials are returned for the saved checkpoint.
    """
    if len(samples) < 4 or not 0 < inner_fraction < 1:
        raise ValueError('Need >=4 training organoids and 0 < inner_fraction < 1.')
    if not smoothness_grid or not pair_ridge_grid:
        raise ValueError('Penalty grids must be nonempty.')
    if blas_threads < 1:
        raise ValueError('blas_threads must be positive.')
    order = np.random.default_rng(seed).permutation(len(samples))
    n_val = max(1, min(len(samples)-2, round(len(samples)*inner_fraction)))
    inner_val = [samples[i] for i in order[:n_val]]
    inner_train = [samples[i] for i in order[n_val:]]
    kwargs = dict(n_markers=n_markers, radius=radius, variant=variant, n_splines=n_splines)
    model = FateSplineCurvature(**kwargs).configure(inner_train)
    trials = []
    # Limit BLAS oversubscription for the many moderate-size matrix operations.
    with threadpool_limits(limits=blas_threads):
        if callback:
            callback('Building inner-training sufficient statistics')
        equations = normal_equations(model, inner_train)
        for smoothness in smoothness_grid:
            for pair_ridge in (pair_ridge_grid if variant == 'pairwise' else [ridge]):
                if callback:
                    callback(f'Inner holdout: smoothness={smoothness:g}, pair ridge={pair_ridge:g}')
                solve_spline(model, equations, smoothness=smoothness, ridge=ridge, pair_ridge=pair_ridge)
                trials.append(dict(smoothness=float(smoothness), ridge=float(ridge),
                                   pair_ridge=float(pair_ridge), inner_mse=float(sample_mse(model, inner_val).mean())))
        best = min(trials, key=lambda row: row['inner_mse'])
        if callback:
            callback('Refitting selected penalties on all outer-training organoids')
        model = FateSplineCurvature(**kwargs).configure(samples)
        equations = normal_equations(model, samples)
        solve_spline(model, equations, **{key:best[key] for key in ('smoothness', 'ridge', 'pair_ridge')})
        train_mse = float(sample_mse(model, samples).mean())
        model.log_variance.fill_(np.log(max(train_mse, 1e-30)))
    metadata = dict(best=best, train_mse=train_mse,
                    inner_train=[s['organoid_str'] for s in inner_train],
                    inner_validation=[s['organoid_str'] for s in inner_val],
                    n_coefficients=model.weights.numel(), loss='equal-organoid physical residual MSE',
                    extrapolation='clamp to training log-N range',
                    variance='constant training residual MSE; not calibrated uncertainty')
    return model, metadata, pd.DataFrame(trials)


def fit_fraction_spline(samples, *, n_markers, radius=2, n_splines=5,
                        fraction_degree=1, center_nonlinear=True, n_interactions=0,
                        smoothness_grid=(1e-5, 1e-3, .01),
                        ridge_grid=(1e-6, 1e-5, 1e-4, 1e-3),
                        extension_multipliers=(1., 10.), inner_fraction=.2,
                        seed=42, blas_threads=2, callback=None):
    """Fit explicit nonlinear ring-fraction responses with an untouched outer fold.

    For degree 1 without interactions, the multiplier controls center-specific
    linear terms; otherwise it controls nonlinear abundance/product terms.
    Smoothness and ridge apply in the saved projected feature basis. Inner
    validation sufficient statistics accelerate repeated quadratic fits.
    Pair screening, projections, reference compositions and knot placement are
    learned separately on inner training, then refitted on full outer training.
    """
    from scipy.linalg import cho_factor, cho_solve
    from src.models.fate_fraction import FateFractionSpline
    if len(samples) < 4 or not 0 < inner_fraction < 1 or blas_threads < 1:
        raise ValueError('Need >=4 organoids, 0<inner_fraction<1 and positive BLAS threads.')
    if any(not len(grid) or any(not np.isfinite(v) or v < 0 for v in grid)
           for grid in (smoothness_grid,ridge_grid,extension_multipliers)):
        raise ValueError('Penalty grids must contain finite nonnegative values.')
    order = np.random.default_rng(seed).permutation(len(samples))
    n_val = max(1,min(len(samples)-2,round(len(samples)*inner_fraction)))
    inner_val = [samples[i] for i in order[:n_val]]
    inner_train = [samples[i] for i in order[n_val:]]
    kwargs = dict(n_markers=n_markers,radius=radius,n_splines=n_splines,
                  fraction_degree=fraction_degree,center_nonlinear=center_nonlinear,n_interactions=n_interactions)
    def message(s):
        if callback:
            callback(s)
    def fit_weights(model,equations,smoothness,ridge,extension_multiplier):
        gram,rhs = equations
        k = model.n_splines
        second = np.diff(np.eye(k),n=2,axis=0)
        block = second.T@second
        diagonal = np.full(model.hidden_dim,ridge,dtype=float)
        diagonal[0] = 0.
        boundary = model.linear_dim if model.abundance_dim or model.interaction_dim else 1+n_markers+radius*n_markers
        diagonal[boundary:] *= extension_multiplier
        penalty = smoothness*np.kron(np.eye(model.hidden_dim),block)
        penalty.flat[::len(penalty)+1] += np.repeat(diagonal,k)+1e-10
        matrix = gram+penalty
        solution = cho_solve(cho_factor(matrix,lower=True,check_finite=False),rhs,check_finite=False)
        model.weights.copy_(torch.tensor(solution.reshape(model.weights.shape)))
        model.is_fitted.fill_(True)
        return solution
    with threadpool_limits(limits=blas_threads):
        message('Configure inner-training fraction basis and selected interactions')
        model = FateFractionSpline(**kwargs).configure(inner_train)
        selected_inner = model.interaction_pairs.tolist()
        message(f'Accumulate inner-training statistics ({model.weights.numel()} coefficients)')
        equations = normal_equations(model,inner_train)
        validation = normal_equations(model,inner_val)
        yy = np.mean([np.mean(s['y']**2) for s in inner_val])
        trials = []
        for smoothness in smoothness_grid:
            for ridge in ridge_grid:
                for multiplier in extension_multipliers:
                    solution = fit_weights(model,equations,smoothness,ridge,multiplier)
                    mse = float(solution@validation[0]@solution-2*solution@validation[1]+yy)
                    trials.append(dict(smoothness=float(smoothness),ridge=float(ridge),
                        extension_multiplier=float(multiplier),inner_mse=mse))
            message(f'Inner tuning: smoothness={smoothness:g}, best MSE={min(t["inner_mse"] for t in trials):.7g}')
        best = min(trials,key=lambda t:t['inner_mse'])
        del equations,validation
        message('Refit selected penalties/basis on all outer-training organoids')
        model = FateFractionSpline(**kwargs).configure(samples)
        equations = normal_equations(model,samples)
        fit_weights(model,equations,**{k:best[k] for k in ('smoothness','ridge','extension_multiplier')})
        train_mse = float(sample_mse(model,samples).mean())
        model.log_variance.fill_(np.log(max(train_mse,1e-30)))
    metadata = dict(best=best,train_mse=train_mse,inner_train=[s['organoid_str'] for s in inner_train],
        inner_validation=[s['organoid_str'] for s in inner_val],n_coefficients=model.weights.numel(),
        selected_inner_pairs=selected_inner,selected_final_pairs=model.interaction_pairs.tolist(),
        loss='equal-organoid physical residual MSE',features='exclusive center fate, exact-ring fate fractions, log N',
        screening='training-only residual correlation; screening ridge 1e-5; fixed pair count',
        extrapolation='clamp to training log-N range',variance='constant training residual MSE; not calibrated')
    return model,metadata,pd.DataFrame(trials)
