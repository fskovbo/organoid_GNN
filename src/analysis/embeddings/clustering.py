from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import torch
from torch_geometric.loader import DataLoader

from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


@dataclass
class EmbeddingExtractionResult:
    embeddings_full: np.ndarray
    embeddings_local: np.ndarray
    y_true: np.ndarray
    y_pred: np.ndarray
    log_var: np.ndarray | None
    x_markers: np.ndarray
    global_features: np.ndarray | None
    graph_index: np.ndarray
    local_node_index: np.ndarray
    center_only: bool
    organoid_ids: list[str | None]


@dataclass
class ClusteringResult:
    labels: np.ndarray
    probabilities: np.ndarray | None
    model: Any
    scaler: StandardScaler | None
    pca: PCA | None
    embeddings_used: np.ndarray
    cluster_centers: np.ndarray | None
    bic_by_k: dict[int, float] | None = None
    aic_by_k: dict[int, float] | None = None


@torch.no_grad()
def extract_node_embeddings(
    graphs: list,
    model: torch.nn.Module,
    *,
    device: str | None = None,
    batch_size: int = 32,
    num_workers: int = 0,
    pin_memory: bool = True,
    center_only: bool = False,
    strip_global_from_embedding: bool = True,
) -> EmbeddingExtractionResult:
    """
    Run a trained model on a list of PyG graphs/subgraphs and collect node embeddings.

    Important for your current GINCurvature layout:
    the model returns `h` AFTER optional concatenation with graph-level features.
    Therefore we return both:
      - embeddings_full  : exactly what the model returned
      - embeddings_local : with the last `global_dim` columns removed if requested

    This makes it easy to cluster on the current final embedding now, while keeping a
    clean path to later regress out or exclude graph-level features.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = model.to(device).eval()
    loader = DataLoader(
        graphs,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    Ys, Ymu, Ylv = [], [], []
    Xs = []
    Hfull, Hlocal = [], []
    Gfeat = []
    graph_index = []
    local_node_index = []
    organoid_ids: list[str | None] = []

    global_dim = int(getattr(model, "global_dim", 0) or 0)
    global_attr = getattr(model, "global_attr", "global_feat")

    graph_counter = 0

    for batch in loader:
        batch = batch.to(device, non_blocking=True)

        try:
            (mu, log_var), h = model(batch.x, batch.edge_index, data=batch)
        except TypeError:
            (mu, log_var), h = model(batch.x, batch.edge_index)

        if log_var is not None and getattr(log_var, "ndim", 0) == 3:
            cov = log_var @ log_var.transpose(-1, -2)
            log_var = torch.log(torch.diagonal(cov, dim1=-2, dim2=-1).clamp_min(1e-12))
        if mu.ndim == 2 and mu.shape[1] == 1:
            mu = mu.view(-1)
        if log_var is not None and log_var.ndim == 2 and log_var.shape[1] == 1:
            log_var = log_var.view(-1)

        if center_only:
            if not hasattr(batch, "center_idx"):
                raise ValueError("center_only=True requires graphs/subgraphs with a .center_idx attribute")
            sel = batch.ptr[:-1] + batch.center_idx.to(batch.ptr.device)
        else:
            sel = torch.arange(batch.num_nodes, device=batch.x.device)

        y_sel = batch.y[sel]
        mu_sel = mu[sel]
        x_sel = batch.x[sel]
        h_full_sel = h[sel]

        if strip_global_from_embedding and global_dim > 0:
            h_local_sel = h_full_sel[:, :-global_dim]
        else:
            h_local_sel = h_full_sel

        if global_dim > 0:
            if not hasattr(batch, global_attr):
                raise AttributeError(
                    f"Model expects graph-level features in batch.{global_attr!r}, but they are missing."
                )
            g = getattr(batch, global_attr)
            if g.ndim == 1:
                g = g.unsqueeze(-1)
            g_node = g[batch.batch]
            g_sel = g_node[sel]
        else:
            g_sel = None

        if center_only:
            n_graphs_batch = batch.num_graphs
            graph_ids_this = np.arange(graph_counter, graph_counter + n_graphs_batch, dtype=np.int64)
            graph_counter += n_graphs_batch
            node_ids_this = batch.center_idx.detach().cpu().numpy().astype(np.int64)

            if hasattr(batch, "organoid_str"):
                orgs = list(batch.organoid_str)
            else:
                orgs = [None] * n_graphs_batch
        else:
            ptr = batch.ptr.detach().cpu().numpy()
            n_graphs_batch = batch.num_graphs
            counts = np.diff(ptr)
            graph_ids_this = np.repeat(
                np.arange(graph_counter, graph_counter + n_graphs_batch, dtype=np.int64),
                counts,
            )
            graph_counter += n_graphs_batch
            node_ids_this = np.concatenate([np.arange(c, dtype=np.int64) for c in counts], axis=0)

            if hasattr(batch, "organoid_str"):
                orgs = []
                batch_orgs = list(batch.organoid_str)
                for og, c in zip(batch_orgs, counts):
                    orgs.extend([og] * int(c))
            else:
                orgs = [None] * int(sel.numel())

        Ys.append(y_sel.detach().cpu().numpy())
        Ymu.append(mu_sel.detach().cpu().numpy())
        Xs.append(x_sel.detach().cpu().numpy())
        Hfull.append(h_full_sel.detach().cpu().numpy())
        Hlocal.append(h_local_sel.detach().cpu().numpy())
        graph_index.append(graph_ids_this)
        local_node_index.append(node_ids_this)
        organoid_ids.extend(orgs)

        if log_var is not None:
            Ylv.append(log_var[sel].detach().cpu().numpy())
        if g_sel is not None:
            Gfeat.append(g_sel.detach().cpu().numpy())

    y_true = np.concatenate(Ys, axis=0).astype(np.float64)
    y_pred = np.concatenate(Ymu, axis=0).astype(np.float64)
    x_markers = np.concatenate(Xs, axis=0).astype(np.float32)
    embeddings_full = np.concatenate(Hfull, axis=0).astype(np.float32)
    embeddings_local = np.concatenate(Hlocal, axis=0).astype(np.float32)
    graph_index = np.concatenate(graph_index, axis=0).astype(np.int64)
    local_node_index = np.concatenate(local_node_index, axis=0).astype(np.int64)
    log_var = np.concatenate(Ylv, axis=0).astype(np.float64) if len(Ylv) > 0 else None
    global_features = np.concatenate(Gfeat, axis=0).astype(np.float32) if len(Gfeat) > 0 else None

    return EmbeddingExtractionResult(
        embeddings_full=embeddings_full,
        embeddings_local=embeddings_local,
        y_true=y_true,
        y_pred=y_pred,
        log_var=log_var,
        x_markers=x_markers,
        global_features=global_features,
        graph_index=graph_index,
        local_node_index=local_node_index,
        center_only=center_only,
        organoid_ids=organoid_ids,
    )


def regress_out_global_features(
    embeddings: np.ndarray,
    global_features: np.ndarray | None,
    *,
    add_intercept: bool = True,
    ridge: float = 1e-6,
    return_details: bool = False,
):
    """
    Residualize embeddings against graph-level features with a small ridge penalty.

    This is optional and not used by default, but lets you switch later from
    clustering the raw final embedding to clustering the globally-residualized embedding.
    """
    if global_features is None:
        if return_details:
            return embeddings.copy(), {"beta": None, "fitted": None}
        return embeddings.copy()

    G = np.asarray(global_features, dtype=np.float64)
    H = np.asarray(embeddings, dtype=np.float64)

    if G.ndim == 1:
        G = G[:, None]

    if add_intercept:
        G = np.concatenate([np.ones((G.shape[0], 1), dtype=G.dtype), G], axis=1)

    gtg = G.T @ G
    beta = np.linalg.solve(gtg + ridge * np.eye(gtg.shape[0], dtype=G.dtype), G.T @ H)
    fitted = G @ beta
    resid = H - fitted

    resid = resid.astype(np.float32)
    if return_details:
        return resid, {"beta": beta, "fitted": fitted}
    return resid


def _prepare_embeddings_for_clustering(
    embeddings: np.ndarray,
    *,
    standardize: bool = True,
    pca_dim: int | None = None,
):
    X = np.asarray(embeddings, dtype=np.float32)

    scaler = None
    if standardize:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

    pca = None
    if pca_dim is not None:
        if pca_dim <= 0:
            raise ValueError("pca_dim must be positive or None")
        pca = PCA(n_components=min(pca_dim, X.shape[1]), random_state=0)
        X = pca.fit_transform(X)

    return X, scaler, pca


def fit_gmm_clustering(
    embeddings: np.ndarray,
    *,
    n_clusters: int,
    covariance_type: str = "full",
    standardize: bool = True,
    pca_dim: int | None = None,
    seed: int = 0,
) -> ClusteringResult:
    X, scaler, pca = _prepare_embeddings_for_clustering(
        embeddings,
        standardize=standardize,
        pca_dim=pca_dim,
    )

    model = GaussianMixture(
        n_components=n_clusters,
        covariance_type=covariance_type,
        random_state=seed,
    )
    model.fit(X)
    labels = model.predict(X)
    probs = model.predict_proba(X)

    centers = None
    if hasattr(model, "means_"):
        centers = model.means_.copy()

    return ClusteringResult(
        labels=labels.astype(np.int64),
        probabilities=probs.astype(np.float32),
        model=model,
        scaler=scaler,
        pca=pca,
        embeddings_used=X.astype(np.float32),
        cluster_centers=centers.astype(np.float32) if centers is not None else None,
    )


def select_gmm_by_bic(
    embeddings: np.ndarray,
    *,
    k_values: Iterable[int],
    covariance_type: str = "full",
    standardize: bool = True,
    pca_dim: int | None = None,
    seed: int = 0,
) -> ClusteringResult:
    X, scaler, pca = _prepare_embeddings_for_clustering(
        embeddings,
        standardize=standardize,
        pca_dim=pca_dim,
    )

    best_model = None
    best_bic = None
    bic_by_k: dict[int, float] = {}
    aic_by_k: dict[int, float] = {}

    for k in k_values:
        model = GaussianMixture(
            n_components=int(k),
            covariance_type=covariance_type,
            random_state=seed,
        )
        model.fit(X)
        bic = float(model.bic(X))
        aic = float(model.aic(X))
        bic_by_k[int(k)] = bic
        aic_by_k[int(k)] = aic

        if best_bic is None or bic < best_bic:
            best_bic = bic
            best_model = model

    assert best_model is not None
    labels = best_model.predict(X)
    probs = best_model.predict_proba(X)
    centers = best_model.means_.copy() if hasattr(best_model, "means_") else None

    return ClusteringResult(
        labels=labels.astype(np.int64),
        probabilities=probs.astype(np.float32),
        model=best_model,
        scaler=scaler,
        pca=pca,
        embeddings_used=X.astype(np.float32),
        cluster_centers=centers.astype(np.float32) if centers is not None else None,
        bic_by_k=bic_by_k,
        aic_by_k=aic_by_k,
    )


def fit_kmeans_clustering(
    embeddings: np.ndarray,
    *,
    n_clusters: int,
    standardize: bool = True,
    pca_dim: int | None = None,
    seed: int = 0,
    n_init: int = 20,
) -> ClusteringResult:
    X, scaler, pca = _prepare_embeddings_for_clustering(
        embeddings,
        standardize=standardize,
        pca_dim=pca_dim,
    )

    model = KMeans(n_clusters=n_clusters, random_state=seed, n_init=n_init)
    labels = model.fit_predict(X)

    return ClusteringResult(
        labels=labels.astype(np.int64),
        probabilities=None,
        model=model,
        scaler=scaler,
        pca=pca,
        embeddings_used=X.astype(np.float32),
        cluster_centers=model.cluster_centers_.astype(np.float32),
    )


def summarize_clusters(
    extraction: EmbeddingExtractionResult,
    labels: np.ndarray,
    *,
    marker_names: list[str] | None = None,
) -> list[dict[str, Any]]:
    """
    Produce a compact per-cluster summary for interpretation.

    The summary is deliberately simple and dataframe-friendly. It includes:
      - size
      - mean/std of true and predicted curvature
      - mean marker positivity per marker
      - mean global features if present
    """
    labels = np.asarray(labels)
    K = int(labels.max()) + 1 if labels.size else 0
    out: list[dict[str, Any]] = []

    for k in range(K):
        mask = labels == k
        if not np.any(mask):
            continue

        row: dict[str, Any] = {
            "cluster": k,
            "n_nodes": int(mask.sum()),
            "y_true_mean": float(extraction.y_true[mask].mean()),
            "y_true_std": float(extraction.y_true[mask].std()),
            "y_pred_mean": float(extraction.y_pred[mask].mean()),
            "y_pred_std": float(extraction.y_pred[mask].std()),
        }

        if extraction.log_var is not None:
            row["log_var_mean"] = float(extraction.log_var[mask].mean())

        marker_means = extraction.x_markers[mask].mean(axis=0)
        for j, val in enumerate(marker_means):
            name = marker_names[j] if marker_names is not None else f"marker_{j}"
            row[f"frac_{name}"] = float(val)

        if extraction.global_features is not None:
            gmean = extraction.global_features[mask].mean(axis=0)
            for j, val in enumerate(gmean):
                row[f"global_{j}_mean"] = float(val)

        out.append(row)

    return out


def run_embedding_clustering(
    graphs: list,
    model: torch.nn.Module,
    *,
    device: str | None = None,
    batch_size: int = 32,
    center_only: bool = False,
    embedding_variant: str = "full",
    residualize_global: bool = False,
    clustering: str = "gmm",
    n_clusters: int | None = None,
    k_values: Iterable[int] | None = None,
    covariance_type: str = "full",
    standardize: bool = True,
    pca_dim: int | None = None,
    seed: int = 0,
    marker_names: list[str] | None = None,
):
    """
    End-to-end helper.

    embedding_variant:
      - "full"  : cluster exactly the returned final embedding h
      - "local" : strip appended global features from h before clustering

    residualize_global:
      - if True, regress graph-level features out of the chosen embedding before clustering
    """
    extraction = extract_node_embeddings(
        graphs,
        model,
        device=device,
        batch_size=batch_size,
        center_only=center_only,
        strip_global_from_embedding=True,
    )

    if embedding_variant == "full":
        emb = extraction.embeddings_full
    elif embedding_variant == "local":
        emb = extraction.embeddings_local
    else:
        raise ValueError("embedding_variant must be 'full' or 'local'")

    if residualize_global:
        emb = regress_out_global_features(emb, extraction.global_features)

    if clustering == "gmm":
        if n_clusters is not None:
            clustering_result = fit_gmm_clustering(
                emb,
                n_clusters=n_clusters,
                covariance_type=covariance_type,
                standardize=standardize,
                pca_dim=pca_dim,
                seed=seed,
            )
        else:
            if k_values is None:
                k_values = range(2, 11)
            clustering_result = select_gmm_by_bic(
                emb,
                k_values=k_values,
                covariance_type=covariance_type,
                standardize=standardize,
                pca_dim=pca_dim,
                seed=seed,
            )
    elif clustering == "kmeans":
        if n_clusters is None:
            raise ValueError("n_clusters must be specified for kmeans")
        clustering_result = fit_kmeans_clustering(
            emb,
            n_clusters=n_clusters,
            standardize=standardize,
            pca_dim=pca_dim,
            seed=seed,
        )
    else:
        raise ValueError("clustering must be 'gmm' or 'kmeans'")

    summary = summarize_clusters(extraction, clustering_result.labels, marker_names=marker_names)
    return extraction, clustering_result, summary
