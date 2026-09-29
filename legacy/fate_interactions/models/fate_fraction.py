"""Smooth, explicit nonlinear functions of exclusive-fate ring fractions.

The predictor remains a linear regression in named polynomial/spline features.
Training-only projections separate new terms from the lower-order design;
these projections, scales and selected fraction pairs are saved in the model.
No shell size, graph motif, curvature annotation or geometry is a predictor.
"""
import numpy as np
import torch
from scipy.linalg import pinvh
from legacy.fate_interactions.models.fate_spline import FateSplineCurvature


class FateFractionSpline(FateSplineCurvature):
    """Add polynomial abundance responses and selected two-fraction products.

    ``fraction_degree`` is 1, 2 or 3. Nonlinear abundance terms can be shared
    or center dependent. Selected products are center dependent. All terms
    have smooth cubic-spline coefficients in log N. Fraction interactions are
    screened using only training targets, after removing the lower-order fit.
    Their selection must be repeated inside every hyperparameter holdout.
    """
    def __init__(self, n_markers, radius=2, n_splines=5, fraction_degree=2,
                 center_nonlinear=True, n_interactions=0):
        super().__init__(n_markers, radius, 'pairwise', n_splines)
        if fraction_degree not in (1, 2, 3) or n_interactions < 0:
            raise ValueError('Use degree 1–3 and a nonnegative interaction count.')
        self.fraction_degree = fraction_degree
        self.center_nonlinear = center_nonlinear
        self.n_interactions = n_interactions
        self.linear_dim = self.hidden_dim
        self.n_fraction_features = radius * (n_markers + 1)
        self.abundance_dim = self.n_fraction_features * (fraction_degree-1) * (n_markers+1 if center_nonlinear else 1)
        self.interaction_dim = n_interactions * (n_markers+1)
        candidates = self.n_fraction_features*(self.n_fraction_features-1)//2
        if n_interactions > candidates:
            raise ValueError('More requested interactions than distinct fraction pairs.')
        self.hidden_dim += self.abundance_dim + self.interaction_dim
        self.weights = torch.zeros(self.hidden_dim, n_splines, dtype=torch.float64)
        self.register_buffer('abundance_projection', torch.zeros(self.linear_dim, self.abundance_dim, dtype=torch.float64))
        self.register_buffer('abundance_scale', torch.ones(self.abundance_dim, dtype=torch.float64))
        self.register_buffer('interaction_pairs', torch.zeros(n_interactions, 2, dtype=torch.int64))
        self.register_buffer('interaction_projection', torch.zeros(self.linear_dim+self.abundance_dim, self.interaction_dim, dtype=torch.float64))
        self.register_buffer('interaction_scale', torch.ones(self.interaction_dim, dtype=torch.float64))

    def fractions(self, counts):
        counts = np.asarray(counts, dtype=float)[:, :self.radius]
        total = counts.sum(axis=-1, keepdims=True)
        return np.divide(counts, total, out=np.zeros_like(counts), where=total>0).reshape(len(counts), -1)

    def _condition(self, raw, identities, conditioned=True):
        if not conditioned:
            return raw
        center = np.c_[np.ones(len(identities)), self._array(self.center_contrast)[identities]]
        return np.einsum('na,nb->nab', center, raw).reshape(len(raw), -1)

    def _abundance(self, identities, counts):
        p = self.fractions(counts)
        parts = [p*(1-p)] if self.fraction_degree >= 2 else []
        if self.fraction_degree >= 3:
            parts.append(p*(1-p)*(2*p-1))
        raw = np.concatenate(parts, axis=1) if parts else np.empty((len(p), 0))
        return self._condition(raw, identities, self.center_nonlinear)

    def _interactions(self, identities, counts, pairs=None):
        pairs = self._array(self.interaction_pairs) if pairs is None else pairs
        p = self.fractions(counts)
        return self._condition(p[:, pairs[:,0]]*p[:, pairs[:,1]], identities)

    def _lower_design(self, identities, counts):
        base = super().local_design(identities, counts)
        raw = self._abundance(identities, counts)
        extra = (raw - base@self._array(self.abundance_projection))*self._array(self.abundance_scale)
        return np.c_[base, extra]

    def local_design(self, identities, counts):
        lower = self._lower_design(identities, counts)
        raw = self._interactions(identities, counts)
        extra = (raw-lower@self._array(self.interaction_projection))*self._array(self.interaction_scale)
        return np.c_[lower, extra]

    def _project(self, samples, design, raw, projection_name, scale_name):
        a = getattr(self, projection_name).shape[0]
        b = getattr(self, projection_name).shape[1]
        if not b:
            return
        xx, xy, yy = np.zeros((a,a)), np.zeros((a,b)), np.zeros(b)
        for s in samples:
            x, y = design(s['identity'], s['counts']), raw(s['identity'], s['counts'])
            w = 1/(len(samples)*len(x))
            xx += x.T@x*w
            xy += x.T@y*w
            yy += (y*y).sum(axis=0)*w
        projection = pinvh(xx, rtol=1e-10)@xy
        variance = np.maximum(yy-2*(projection*xy).sum(axis=0)+(projection*(xx@projection)).sum(axis=0), 0)
        # Equalize extension scales without amplifying numerically absent terms.
        scale = np.divide(.1, np.sqrt(variance), out=np.zeros_like(variance), where=variance>1e-10)
        getattr(self, projection_name).copy_(torch.tensor(projection))
        getattr(self, scale_name).copy_(torch.tensor(scale))

    def configure(self, samples):
        super().configure(samples)
        self._project(samples, super().local_design, self._abundance, 'abundance_projection', 'abundance_scale')
        if self.n_interactions:
            pairs = np.asarray([(a,b) for a in range(self.n_fraction_features) for b in range(a+1, self.n_fraction_features)])
            # Rank pair blocks by their residual association with the target,
            # conditional on all existing local terms AND their N dependence.
            # Screening is training-only, not evidence for biological coupling.
            from scipy.linalg import solve
            lower_dim = self.linear_dim+self.abundance_dim
            size = lower_dim*self.n_splines
            xx, xy = np.zeros((size,size)), np.zeros(size)
            for s in samples:
                x = self._lower_design(s['identity'],s['counts'])
                b = self.basis([s['N']])[0]
                w = 1/(len(samples)*len(x))
                xx += np.kron(x.T@x, np.outer(b,b))*w
                xy += np.kron(x.T@s['y'],b)*w
            xx.flat[::size+1] += 1e-5
            beta = solve(xx,xy,assume_a='pos').reshape(lower_dim,self.n_splines)
            numerator = np.zeros((self.n_markers+1,len(pairs),self.n_splines))
            denominator = np.zeros_like(numerator)
            for s in samples:
                b = self.basis([s['N']])[0]
                residual = s['y']-self._lower_design(s['identity'],s['counts'])@beta@b
                raw = self._interactions(s['identity'],s['counts'],pairs).reshape(len(residual),self.n_markers+1,len(pairs))
                w = 1/(len(samples)*len(raw))
                numerator += np.einsum('nap,n,b->apb',raw,residual,b)*w
                denominator += np.einsum('nap,b->apb',raw*raw,b*b)*w
            scores = (numerator*numerator/(denominator+1e-10)).sum(axis=(0,2))
            selected = np.argsort(-scores,kind='stable')[:self.n_interactions]
            self.interaction_pairs.copy_(torch.tensor(pairs[selected]))
            self._project(samples,self._lower_design,self._interactions,'interaction_projection','interaction_scale')
        return self

    def coefficients(self, N):
        """Coefficients on ORIGINAL explicit features, undoing projections.

        Columns are the original linear design, conditioned abundance features,
        then conditioned selected products. They reconstruct the prediction
        exactly. The lower-order zero-sum linear contrasts remain recorded.
        Individual polynomial coefficients are basis-dependent; inspect complete
        fraction-response functions, not an isolated polynomial coefficient.
        """
        w = self.basis(N)@self._array(self.weights).T
        stop = self.linear_dim+self.abundance_dim
        pair = w[:,stop:]*self._array(self.interaction_scale)
        lower = w[:,:stop]-pair@self._array(self.interaction_projection).T
        abundance = lower[:,self.linear_dim:]*self._array(self.abundance_scale)
        linear = lower[:,:self.linear_dim]-abundance@self._array(self.abundance_projection).T
        return dict(linear=linear, abundance=abundance, interactions=pair)

    def contributions(self, sample, N=None):
        co = self.coefficients([sample['N'] if N is None else N])
        ids, counts = sample['identity'],sample['counts']
        terms = dict(linear=super().local_design(ids,counts)*co['linear'][0],
                     abundance=self._abundance(ids,counts)*co['abundance'][0],
                     interactions=self._interactions(ids,counts)*co['interactions'][0])
        terms['prediction'] = sum(value.sum(axis=1) for value in terms.values())
        return terms
