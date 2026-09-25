"""Identifiable additive curvature regression on exclusive graph neighborhoods.

    K - baseline = b(N) + a[center](N)
                   + sum((fraction - reference) * (c(N) + P[center](N)))

All coefficients use a common cubic B-spline basis in log N. Weighted
sum-to-zero contrasts remove intercept/center/source/pair redundancies.
Parameters are physical curvature values, not nonlinear transformed targets.
"""
import numpy as np
import torch
from scipy.interpolate import BSpline
from scipy.linalg import null_space
from torch import nn
from src.data.neighborhood_counts import exact_hop_counts, fate_identities


class FateSplineCurvature(nn.Module):
    """CPU spline regression with the repository's graph-inference interface.

    ``center`` has radius zero; ``shared`` adds center-independent shell
    composition; ``pairwise`` also adds center-specific shell composition.
    No learned GNN embedding, masking input, or geometry head is present.
    Extrapolation clamps log N to the training range rather than inventing tails.
    """
    def __init__(self, n_markers, radius=2, variant='pairwise', n_splines=6):
        super().__init__()
        if not isinstance(n_markers, int) or n_markers < 1:
            raise ValueError('n_markers must be a positive integer.')
        if not isinstance(radius, int) or radius < 0 or variant not in ('center', 'shared', 'pairwise'):
            raise ValueError('Invalid radius or model variant.')
        if (variant == 'center') != (radius == 0):
            raise ValueError('Use center only at radius 0; shared/pairwise require radius >= 1.')
        if not isinstance(n_splines, int) or n_splines < 4:
            raise ValueError('Cubic splines require at least four basis functions.')
        self.n_markers, self.radius, self.variant, self.n_splines = n_markers, radius, variant, n_splines
        self.num_layers = radius
        t, d = n_markers + 1, n_markers
        self.hidden_dim = 1 + d + radius * d + (radius * d * d if variant == 'pairwise' else 0)
        self.register_buffer('weights', torch.zeros(self.hidden_dim, n_splines, dtype=torch.float64))
        self.register_buffer('knots', torch.zeros(n_splines + 4, dtype=torch.float64))
        self.register_buffer('center_reference', torch.full((t,), 1 / t, dtype=torch.float64))
        self.register_buffer('source_reference', torch.full((radius, t), 1 / t, dtype=torch.float64))
        self.register_buffer('center_contrast', torch.zeros(t, d, dtype=torch.float64))
        self.register_buffer('source_contrast', torch.zeros(radius, t, d, dtype=torch.float64))
        self.register_buffer('log_count_range', torch.zeros(2, dtype=torch.float64))
        self.register_buffer('log_variance', torch.tensor(0., dtype=torch.float64))
        self.register_buffer('is_fitted', torch.tensor(False))

    @staticmethod
    def _array(value):
        return value.detach().cpu().numpy()

    def configure(self, samples):
        """Fit reference compositions and log-N bounds from training organoids only.

        Organoids have equal weight. For source references, only nonempty
        shells contribute, averaging centers within each organoid first.
        Entirely absent identities receive a tiny reference weight, not data.
        """
        if not samples:
            raise ValueError('No training organoids.')
        t = self.n_markers + 1
        z = np.log([s['N'] for s in samples])
        lo, hi = float(z.min()), float(z.max())
        if hi <= lo:
            raise ValueError('Size-dependent fitting requires at least two distinct training sizes.')
        interior = np.linspace(lo, hi, self.n_splines - 2)[1:-1]
        knots = np.r_[np.repeat(lo, 4), interior, np.repeat(hi, 4)]
        center = np.mean([np.bincount(s['identity'], minlength=t) / len(s['identity']) for s in samples], axis=0)
        center = np.maximum(center, 1e-8); center /= center.sum()
        sources = []
        for r in range(self.radius):
            means = []
            for sample in samples:
                counts = sample['counts'][:, r]
                total = counts.sum(axis=1)
                if (total > 0).any():
                    means.append((counts[total > 0] / total[total > 0, None]).mean(axis=0))
            ref = np.mean(means, axis=0) if means else np.ones(t)
            ref = np.maximum(ref, 1e-8); ref /= ref.sum()
            sources.append(ref)
        source = np.asarray(sources).reshape(self.radius, t)
        values = dict(knots=knots, log_count_range=[lo, hi], center_reference=center,
                      source_reference=source, center_contrast=null_space(center[None, :]),
                      source_contrast=np.asarray([null_space(p[None, :]) for p in source]).reshape(self.radius, t, t-1))
        for name, value in values.items():
            getattr(self, name).copy_(torch.as_tensor(value, dtype=torch.float64, device=self.weights.device))
        self.is_fitted.fill_(False)
        return self

    def basis(self, counts):
        counts = np.asarray(counts, dtype=float).reshape(-1)
        if not np.isfinite(counts).all() or np.any(counts <= 0):
            raise ValueError('N must be finite and positive.')
        lo, hi = self._array(self.log_count_range)
        if hi <= lo:
            raise RuntimeError('Configure the training spline basis first.')
        return BSpline.design_matrix(np.clip(np.log(counts), lo, hi), self._array(self.knots), 3).toarray()

    def composition(self, counts):
        counts = np.asarray(counts)[:, :self.radius]
        if counts.ndim != 3 or counts.shape[1:] != (self.radius, self.n_markers + 1):
            raise ValueError('Counts do not cover the requested shells/identities.')
        if not np.isfinite(counts).all() or (counts < 0).any():
            raise ValueError('Invalid neighborhood counts.')
        total = counts.sum(axis=2, keepdims=True)
        fraction = np.divide(counts, total, out=np.zeros_like(counts, dtype=float), where=total > 0)
        # An empty shell contributes zero, rather than negative reference fates.
        return np.where(total > 0, fraction - self._array(self.source_reference), 0.)

    def local_design(self, identities, counts):
        center = self._array(self.center_contrast)[np.asarray(identities)]
        q = self.composition(counts)
        source = np.einsum('nrt,rtd->nrd', q, self._array(self.source_contrast))
        parts = [np.ones((len(center), 1)), center, source.reshape(len(center), -1)]
        if self.variant == 'pairwise':
            parts.append(np.einsum('na,nrb->nrab', center, source).reshape(len(center), -1))
        return np.concatenate(parts, axis=1)

    def predict_sample(self, sample, N=None):
        if not self.is_fitted.item():
            raise RuntimeError('Model has not been fitted.')
        coefficients = self._array(self.weights) @ self.basis([sample['N'] if N is None else N])[0]
        return self.local_design(sample['identity'], sample['counts']) @ coefficients

    def coefficients(self, N):
        """Read b, a, c, P in physical units with N as the leading dimension."""
        if not self.is_fitted.item():
            raise RuntimeError('Model has not been fitted.')
        w = self.basis(N) @ self._array(self.weights).T
        d, r, t = self.n_markers, self.radius, self.n_markers + 1
        a = w[:, 1:1+d] @ self._array(self.center_contrast).T
        stop = 1 + d + r*d
        c = np.einsum('nrd,rtd->nrt', w[:, 1+d:stop].reshape(-1, r, d), self._array(self.source_contrast)) if r else np.zeros((len(w), 0, t))
        p = np.zeros((len(w), r, t, t))
        if self.variant == 'pairwise':
            p = np.einsum('ad,nrde,rbe->nrab', self._array(self.center_contrast),
                          w[:, stop:].reshape(-1, r, d, d), self._array(self.source_contrast))
        return dict(b=w[:, 0], a=a, c=c, P=p)

    def contributions(self, sample, N=None):
        """Exact per-cell decomposition, excluding any saved organoid baseline."""
        coef = self.coefficients([sample['N'] if N is None else N])
        identity = sample['identity']
        q = self.composition(sample['counts'])
        shared = q * coef['c'][0]
        pair = q * coef['P'][0].transpose(1, 0, 2)[identity]
        center = coef['a'][0, identity]
        mean = coef['b'][0] + center + shared.sum(axis=(1, 2)) + pair.sum(axis=(1, 2))
        return dict(b=np.full(len(identity), coef['b'][0]), a=center, shared=shared,
                    pairwise=pair, prediction=mean)

    def forward(self, x, edge_index, data=None):
        """Recompute shell counts so edited fates never use stale cached features.

        The second return is the local design, not a learned embedding. A
        constant training-residual variance supports existing evaluation APIs;
        it is not a calibrated predictive uncertainty estimate.
        """
        if not self.is_fitted.item():
            raise RuntimeError('Model has not been fitted.')
        identities = fate_identities(x)
        counts = exact_hop_counts(x, edge_index, self.radius)
        local = self.local_design(identities, counts)
        batch = getattr(data, 'batch', None) if data is not None else None
        batch = self._array(batch).astype(int) if batch is not None else np.zeros(len(x), dtype=int)
        supplied = getattr(data, 'full_num_cells', None) if data is not None else None
        size = (self._array(supplied) if torch.is_tensor(supplied) else np.asarray(supplied)).reshape(-1) if supplied is not None else np.bincount(batch)
        if len(size) != int(batch.max()) + 1:
            raise ValueError('Provide one full_num_cells value per graph.')
        w = self.basis(size) @ self._array(self.weights).T
        mu = np.einsum('nf,nf->n', local, w[batch])
        mu = torch.as_tensor(mu, dtype=torch.float64, device=x.device)
        return (mu, self.log_variance.to(x.device).expand_as(mu)), torch.as_tensor(local, device=x.device)
