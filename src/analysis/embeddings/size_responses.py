"""Paired size sweeps of exclusive FiLM embeddings, without training.

Each checkpoint owns its scaler, PCA, GMM and visualization. Cell identities,
source edits and reference cluster membership remain fixed through a sweep.
"""
from contextlib import nullcontext
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Batch
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from src.analysis.conditioning.film import independent_conditioning


@dataclass(frozen=True)
class EmbeddingConfig:
    anchors: tuple = (80, 300, 550, 700)
    counts: tuple = (80, 120, 165, 220, 250, 275, 300, 325, 350, 361,
                     400, 450, 475, 500, 525, 550, 575, 600, 650, 700, 791)
    reference_n: int = 361
    folds: tuple = (0, 1, 2, 3, 4)
    seeds: tuple = (42, 43, 44)
    all_centers_per_organoid: int = 16
    lgr5_centers_per_organoid: int = 8
    markers: tuple = ("KI67", "LGR5")
    hops: tuple = (1, 2)
    sampling_seed: int = 4207
    max_organoids_per_fold: int | None = None
    n_clusters: int = 6
    sensitivity_k: tuple = (4, 6, 8)
    pca_dim: int = 32
    covariance_type: str = "full"
    cluster_seed: int = 69
    gmm_n_init: int = 3
    tsne_checkpoint: tuple = (0, 42)
    tsne_max_cells: int = 2500
    tsne_perplexity: float = 40.
    tsne_max_iter: int = 1000
    tsne_seed: int = 420
    bootstrap_draws: int = 1000
    observed_window_fold: float = 1.25
    local_size_fold: float = 2.
    min_organoids: int = 5
    batch_size: int = 128

    def validate(self):
        if sorted(set(self.counts)) != list(self.counts) or min(self.counts) <= 0:
            raise ValueError("counts must be positive, unique and increasing")
        if not set((*self.anchors, self.reference_n)) <= set(self.counts):
            raise ValueError("Include all anchors and reference_n in counts")
        if len(self.anchors) < 2 or len(set(self.anchors)) != len(self.anchors):
            raise ValueError("Use at least two distinct anchors")
        if not self.folds or not self.seeds or min(self.all_centers_per_organoid, self.lgr5_centers_per_organoid) < 1:
            raise ValueError("Nonempty folds, seeds and positive sample sizes are required")
        if not set(self.hops) <= {1, 2} or not self.hops:
            raise ValueError("This depth-2 analysis supports exact hops 1 and 2")
        if min(self.n_clusters, self.pca_dim, self.tsne_max_cells, self.batch_size, self.bootstrap_draws) < 1:
            raise ValueError("Clustering, sampling and bootstrap sizes must be positive")


def _json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, default=lambda x: x.item() if isinstance(x, np.generic) else str(x)))


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _rng(seed, organoid):
    digest = hashlib.sha256(f"{seed}:{organoid}".encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "little"))


@torch.no_grad()
def infer_states(model, subgraphs, requests, pre, *, count=None, route="full", reference_n=361, device="cpu", batch_size=128):
    """Final post-residual local embedding, excluding head globals. Never edit inputs."""
    if route not in ("full", "film_only", "head_only") or (count is None and route != "full"):
        raise ValueError("Observed-size inference supports the full route only")
    def standardize(n):
        return float((np.log(n) - pre["size_center"]) / pre["size_scale"])
    anchor = standardize(reference_n)
    model.to(device).eval()
    outputs, hidden = [], []
    for start in range(0, len(requests), batch_size):
        items = []
        for si, edits in requests[start:start + batch_size]:
            g = subgraphs[si].clone()
            for node, marker in edits:
                if node == int(g.center_idx):
                    raise ValueError("Center identity must be preserved")
                if g.x[node, marker] <= .5:
                    raise ValueError("Ablation source is not positive")
                g.x[node, marker] = 0
            size = standardize(g.full_num_cells if count is None else count)
            g.global_feat = g.x.new_tensor([[anchor if route == "film_only" else size]])
            items.append(g)
        batch = Batch.from_data_list(items).to(device)
        centers = batch.ptr[:-1] + batch.center_idx
        film = anchor if route == "head_only" else (standardize(count) if count is not None else None)
        context = independent_conditioning(model, [film] * model.num_layers) if route != "full" else nullcontext()
        with context:
            (mu, _), h = model(batch.x, batch.edge_index, batch)
        outputs.append(mu[centers].reshape(-1).cpu().numpy())
        hidden.append(h[centers, :model.hidden_dim].cpu().numpy())
    return dict(z=np.concatenate(outputs), h=np.concatenate(hidden))


def fit_shared_atlas(states, config):
    """One transform and clustering fit on intact states pooled across anchor N."""
    pooled = np.concatenate(states)
    if len(pooled) <= max(config.sensitivity_k + (config.n_clusters,)):
        raise ValueError("Too few sampled embeddings for the requested cluster counts")
    scaler = StandardScaler().fit(pooled)
    scaled = scaler.transform(pooled)
    pca = PCA(n_components=min(config.pca_dim, len(pooled) - 1, pooled.shape[1]),
              svd_solver="full", random_state=config.cluster_seed).fit(scaled)
    reduced = pca.transform(scaled)
    models, rows = {}, []
    for k in sorted(set(config.sensitivity_k + (config.n_clusters,))):
        gmm = GaussianMixture(n_components=k, covariance_type=config.covariance_type,
            reg_covar=1e-5, n_init=config.gmm_n_init, max_iter=300, random_state=config.cluster_seed).fit(reduced)
        models[k] = gmm
        rows.append(dict(k=k, bic=gmm.bic(reduced), aic=gmm.aic(reduced), converged=bool(gmm.converged_),
                         pca_variance=float(pca.explained_variance_ratio_.sum())))
    primary = models[config.n_clusters]
    labels = primary.predict(reduced)
    for row in rows:
        row["ari_to_primary"] = adjusted_rand_score(labels, models[row["k"]].predict(reduced))
    # Cluster numbers are arbitrary, frozen once; never reorder them at each N.
    return dict(scaler=scaler, pca=pca, gmm=primary), pd.DataFrame(rows)


def project_atlas(atlas, h):
    reduced = atlas["pca"].transform(atlas["scaler"].transform(h))
    probability = atlas["gmm"].predict_proba(reduced)
    return reduced, probability.argmax(axis=1), probability


def representation_change(h, reference):
    """Separate common translation/positive scale from relative cell arrangement.

    Uses original hidden coordinates within one model, independent of t-SNE.
    Zero-variance references have undefined scale/shape metrics.
    """
    centered, ref_centered = h - h.mean(0), reference - reference.mean(0)
    denom = float(np.sum(ref_centered ** 2))
    gain = max(0., float(np.sum(centered * ref_centered)) / denom) if denom > 1e-12 else np.nan
    scale = float(np.linalg.norm(centered))
    mismatch = float(np.linalg.norm(centered - gain * ref_centered) / scale) if scale > 1e-8 else np.nan
    return dict(translation_norm=float(np.linalg.norm(h.mean(0) - reference.mean(0))),
        common_positive_gain=gain, relative_shape_mismatch=mismatch,
        relative_displacement_rms=float(np.sqrt(np.mean(np.sum((centered - ref_centered) ** 2, axis=1)))))


def _transitions(assignments, nodes, config):
    rows = []
    for a, b in zip(config.anchors[:-1], config.anchors[1:]):
        left = assignments[assignments.state == f"N{a}"].set_index("node_id")
        right = assignments[assignments.state == f"N{b}"].set_index("node_id")
        if not left.index.equals(right.index):
            raise ValueError("Transitions require exactly paired cell identities")
        frame = pd.DataFrame(dict(node_id=left.index, source=left.cluster.to_numpy(), target=right.cluster.to_numpy()))
        frame = frame.merge(nodes[["node_id", "organoid_str"]], on="node_id", validate="one_to_one")
        # Equal organoid mass, then row-normalize. Export joint and conditional flows.
        frame["weight"] = 1 / frame.groupby("organoid_str").node_id.transform("size")
        mass = frame.groupby(["source", "target"]).weight.sum().unstack(fill_value=0).reindex(
            index=range(config.n_clusters), columns=range(config.n_clusters), fill_value=0)
        total = mass.to_numpy().sum()
        for i in range(config.n_clusters):
            for j in range(config.n_clusters):
                rows.append(dict(from_n=a, to_n=b, source=i, target=j,
                    joint_fraction=float(mass.loc[i, j] / total),
                    conditional_fraction=float(mass.loc[i, j] / mass.loc[i].sum()) if mass.loc[i].sum() else np.nan))
    return pd.DataFrame(rows)


def paired_bootstrap_curves(frame, group_cols, value_cols, *, draws=1000, seed=420, min_organoids=5):
    """Average seeds/cases within organs first; resample paired organ curves.

    Reports pointwise conditional intervals. Same bootstrap indices are used
    across N; cluster numbers must be grouped by checkpoint if included.
    """
    org = frame.groupby([*group_cols, "organoid_str", "n"], dropna=False)[value_cols].mean().reset_index()
    rng = np.random.default_rng(seed)
    rows = []
    for key, group in org.groupby(group_cols, dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        organs, ns = sorted(group.organoid_str.unique()), sorted(group.n.unique())
        # In observed-size windows support can differ; NaNs are retained explicitly.
        indices = rng.integers(0, len(organs), size=(draws, len(organs)))
        for metric in value_cols:
            matrix = group.pivot(index="organoid_str", columns="n", values=metric).reindex(index=organs, columns=ns).to_numpy()
            count = np.isfinite(matrix).sum(0)
            mean = np.divide(np.nansum(matrix, axis=0), count, out=np.full(len(ns), np.nan), where=count > 0)
            sampled = matrix[indices]
            denominator = np.isfinite(sampled).sum(1)
            boot = np.divide(np.nansum(sampled, axis=1), denominator,
                             out=np.full((draws, len(ns)), np.nan), where=denominator > 0)
            for j, n in enumerate(ns):
                valid = boot[:, j][np.isfinite(boot[:, j])]
                lo, hi = np.quantile(valid, [.025, .975]) if count[j] >= min_organoids and len(valid) else (np.nan, np.nan)
                rows.append(dict(zip(group_cols, key)) | dict(n=n, metric=metric, mean=mean[j], low=lo, high=hi,
                    n_organoids=int(count[j]), sufficient_support=bool(count[j] >= min_organoids)))
    return pd.DataFrame(rows, columns=[*group_cols, "n", "metric", "mean", "low", "high", "n_organoids", "sufficient_support"])


def assign_observed_windows(ns, anchors, width):
    """Non-overlapping nearest-anchor windows on log N; -1 means out of window."""
    distance = np.abs(np.log(np.asarray(ns)[:, None] / np.asarray(anchors)[None, :]))
    nearest = distance.argmin(axis=1)
    return np.where(distance[np.arange(len(nearest)), nearest] <= np.log(width), np.asarray(anchors)[nearest], -1)


