"""Weighted representation projections and finite head-readout decompositions.

The former hardcoded KI67/N=300 experiment is retired. Its reusable numerical
operations and serialized projection/config classes remain available.
"""
from dataclasses import dataclass
import numpy as np
import pandas as pd
import torch

@dataclass(frozen=True)
class PCAStudyConfig:
    counts: tuple = (80, 220, 250, 275, 300, 325, 350, 400, 550, 700)
    display_counts: tuple = (80, 300, 700)
    reference_n: int = 300
    dimensions: tuple = (2, 8, 16, 32, 64, 128, 256)
    probe_checkpoint: tuple = (0, 42)
    bootstrap_draws: int = 1000
    batch_size: int = 128
    probe_seed: int = 7301

    def validate(self):
        if sorted(set(self.counts)) != list(self.counts) or min(self.counts) <= 0:
            raise ValueError("Supply increasing positive N values")
        if self.reference_n not in self.counts or not set(self.display_counts) <= set(self.counts):
            raise ValueError("Reference and displayed N values must be evaluated")
        if 2 not in self.dimensions or min(self.dimensions) < 1:
            raise ValueError("Include 2-D display and positive retained dimensions")


@dataclass
class WeightedPCA:
    mean: np.ndarray
    components: np.ndarray
    variance_ratio: np.ndarray

    def transform(self, h, dimensions=2):
        return (np.asarray(h) - self.mean) @ self.components[:dimensions].T

    def displacement(self, delta, dimensions=2):
        return np.asarray(delta) @ self.components[:dimensions].T

    def reconstruct_displacement(self, delta, dimensions):
        return self.displacement(delta, dimensions) @ self.components[:dimensions]


def fit_weighted_pca(states, organoids):
    """Equal organoid mass, equal case weight within organ, equal N/condition mass.

    Orthogonal components come from a weighted covariance of centered, unscaled
    features. No whitening or N-specific scaling is applied.
    """
    organs = pd.Series(organoids)
    weights = 1 / organs.map(organs.value_counts()).to_numpy(dtype=float)
    weights /= weights.sum()
    mean = sum(np.sum(np.asarray(h, dtype=float) * weights[:, None], axis=0) for h in states) / len(states)
    covariance = np.zeros((len(mean), len(mean)))
    for h in states:
        centered = np.asarray(h, dtype=float) - mean
        covariance += centered.T @ (centered * weights[:, None]) / len(states)
    values, vectors = np.linalg.eigh(covariance)
    order = np.argsort(values)[::-1]
    values, components = np.maximum(values[order], 0), vectors[:, order].T
    # Resolve arbitrary signs deterministically using the largest loading.
    signs = np.sign(components[np.arange(len(components)), np.abs(components).argmax(axis=1)])
    components *= np.where(signs == 0, 1, signs)[:, None]
    ratio = values / values.sum() if values.sum() else np.zeros_like(values)
    return WeightedPCA(mean, components, ratio)


def fit_response_svd(deltas, organoids):
    """Uncentered weighted SVD of displacements; zero remains no intervention.

    This is a response-oriented linear projection, not centered state PCA.
    Its axes maximize retained ablation energy, and are fitted without targets.
    """
    organs = pd.Series(organoids)
    weights = 1/organs.map(organs.value_counts()).to_numpy(dtype=float)
    weights /= weights.sum()
    moment = sum(np.asarray(d, dtype=float).T @ (np.asarray(d, dtype=float)*weights[:, None]) for d in deltas)/len(deltas)
    values, vectors = np.linalg.eigh(moment)
    order = np.argsort(values)[::-1]
    values, components = np.maximum(values[order], 0), vectors[:, order].T
    signs = np.sign(components[np.arange(len(components)), np.abs(components).argmax(axis=1)])
    components *= np.where(signs == 0, 1, signs)[:, None]
    return WeightedPCA(np.zeros(len(values)), components, values/values.sum() if values.sum() else np.zeros_like(values))


def exact_readout(model, base, delta, size):
    """Finite-ablation secant readout for Linear -> ReLU -> Linear in eval mode.

    Δz = g(h, Δh, N) dot Δh exactly (up to floating point). This g is an
    activation-averaged readout along the finite ablation segment, not the
    gradient at the intact point. It can change with both h and Δh.
    """
    net = model.head.net
    if not (isinstance(net[0], torch.nn.Linear) and isinstance(net[1], torch.nn.ReLU)
            and isinstance(net[2], torch.nn.Dropout) and isinstance(net[3], torch.nn.Linear)):
        raise ValueError("Exact readout requires the current single-ReLU head")
    w = net[0].weight.detach().cpu().numpy().astype(float)
    b = net[0].bias.detach().cpu().numpy().astype(float)
    output = net[3].weight[0].detach().cpu().numpy().astype(float)
    a = np.asarray(base, dtype=float) @ w[:, :-1].T + float(size) * w[:, -1] + b
    da = np.asarray(delta, dtype=float) @ w[:, :-1].T
    activation_delta = np.maximum(a + da, 0) - np.maximum(a, 0)
    gate = np.divide(activation_delta, da, out=(a > 0).astype(float), where=np.abs(da) > 1e-12)
    gate = np.clip(gate, 0, 1)
    g = (gate * output) @ w[:, :-1]
    contributions = activation_delta * output
    effect = np.sum(g * delta, axis=1)
    np.testing.assert_allclose(effect, contributions.sum(axis=1), atol=1e-9, rtol=1e-7)
    return dict(g=g, effect=effect, positive=np.maximum(contributions, 0).sum(axis=1),
                negative=np.minimum(contributions, 0).sum(axis=1))


def bilinear_change(g_a, d_a, g_b, d_b):
    g_a, d_a, g_b, d_b = (np.asarray(a, dtype=float) for a in (g_a, d_a, g_b, d_b))
    response = np.sum((g_a + g_b) / 2 * (d_b - d_a), axis=1)
    readout = np.sum((d_a + d_b) / 2 * (g_b - g_a), axis=1)
    np.testing.assert_allclose(response + readout, np.sum(g_b*d_b - g_a*d_a, axis=1), atol=1e-9, rtol=1e-7)
    return response, readout
