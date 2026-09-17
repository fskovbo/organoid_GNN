"""Paired size sweeps of exclusive FiLM embeddings, without training.

Each checkpoint owns its scaler, PCA, GMM and visualization. Cell identities,
source edits and reference cluster membership remain fixed through a sweep.
"""
from contextlib import nullcontext
from dataclasses import asdict, dataclass
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Batch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from src.analysis.exclusive_size_ablation import make_model
from src.analysis.lgr5_film import hidden_comparison, independent_conditioning
from src.analysis.niche_hypotheses import crypt_membership, rings
from src.data.io import build_pyg_graph, load_organoid_npz
from src.data.metadata import load_marker_names_from_dir
from src.data.subgraphs import build_ego_subgraphs_for_graph


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


def experiment_paths(root, run_dir):
    root, run_dir = Path(root).resolve(), Path(run_dir).resolve()
    settings = json.loads((run_dir / "settings.json").read_text())
    if settings.get("FEATURE_ENCODING") != "exclusive_ordered":
        raise ValueError("An exclusive-marker run is required")
    if settings["NUM_LAYERS"] != 2 or settings["MODEL_GLOBAL_FEATURES"]["gin_film_size"] != ["log_num_cells"]:
        raise ValueError("Expected depth-2 FiLM with only log N in the head")
    if settings["NORM"] != "batch":
        raise ValueError("Ego inference is validated for frozen batch normalization only")
    data_dir = root / "training_data" / settings["DATASET_NAME"]
    markers = load_marker_names_from_dir(str(data_dir))
    return settings, data_dir, markers


def inspect_inputs(root, run_dir, config):
    """Read-only checkpoint availability and training/validation size support."""
    config.validate()
    settings, _, markers = experiment_paths(root, run_dir)
    if not set(config.markers) <= set(markers) or "LGR5" not in markers:
        raise ValueError("Requested markers are absent")
    run_dir = Path(run_dir)
    membership = pd.read_csv(run_dir / "tables/split_membership.csv")[["fold", "role", "organoid_str"]]
    cohort = pd.read_csv(run_dir / "tables/cohort.csv")[["organoid_str", "n_cells"]]
    joined = membership.merge(cohort, on="organoid_str", validate="many_to_one")
    rows = []
    for fold in config.folds:
        for seed in config.seeds:
            for file in [f"fold_{fold}_seed_{seed}_gin_film_size.pt", f"fold_{fold}_preprocessing.pkl"]:
                if not (run_dir / "checkpoints" / file).is_file():
                    raise FileNotFoundError(run_dir / "checkpoints" / file)
        for role in ("train", "val"):
            ns = joined.loc[(joined.fold == fold) & (joined.role == role), "n_cells"].to_numpy()
            if len(ns) == 0:
                raise ValueError(f"Empty {role} split for fold {fold}")
            for n in config.anchors:
                rows.append(dict(fold=fold, role=role, n=n, n_organoids=len(ns), min_n=int(ns.min()),
                    max_n=int(ns.max()), within_range=bool(ns.min() <= n <= ns.max()),
                    near_anchor=int((np.abs(np.log(ns / n)) <= np.log(config.observed_window_fold)).sum())))
    return pd.DataFrame(rows)


def _regions(data_dir, org, root):
    """Missing segmentation is explicit; no detection is not an early crypt label."""
    meta = json.loads((data_dir / f"{org}_aux.json").read_text())
    with np.load(data_dir / f"{org}.npz") as archive:
        n = len(archive["x"])
        projection = archive["proj_vertex_ids"] if "proj_vertex_ids" in archive else None
    path = Path(meta.get("segmentation_path", "__missing__"))
    if not path.is_absolute():
        path = Path(root) / path
    labels = np.full(n, -1, dtype=int)
    regions = np.full(n, "annotation_unavailable", dtype=object)
    if projection is not None and path.is_file():
        # Trusted local segmentation contains object arrays of vertex indices.
        with np.load(path, allow_pickle=True) as archive:
            if "crypts_ll" in archive:
                crypts = archive["crypts_ll"]
                labels = crypt_membership(projection, crypts)
                regions[:] = "outside_detected_crypt" if len(crypts) else "no_crypt_detected"
                regions[labels >= 0] = "detected_crypt"
    return meta, regions, labels


def prepare_fold(root, run_dir, fold, config):
    """Separate uniform all-cell sample and LGR5 sample, fixed across N/seeds."""
    settings, data_dir, markers = experiment_paths(root, run_dir)
    membership = pd.read_csv(Path(run_dir) / "tables/split_membership.csv")
    orgs = sorted(membership.loc[(membership.fold == fold) & (membership.role == "val"), "organoid_str"])
    orgs = orgs[:config.max_organoids_per_fold] if config.max_organoids_per_fold else orgs
    subgraphs, nodes, cases, exemplars, full_checks = [], [], [], {}, []
    li = markers.index("LGR5")
    for org in orgs:
        rng = _rng(config.sampling_seed, org)
        arr = load_organoid_npz(str(data_dir / f"{org}.npz"), strict=True, target_indices=[0])
        graph = build_pyg_graph(arr["x"], arr["edges"], arr["y"])
        graph.organoid_str = org
        x = graph.x.numpy()
        if not np.isin(x, [0, 1]).all() or (x.sum(axis=1) > 1).any():
            raise ValueError(f"Nonexclusive/nonbinary input in {org}")
        all_sample = set(map(int, rng.choice(len(x), min(config.all_centers_per_organoid, len(x)), replace=False)))
        eligible = np.flatnonzero(x[:, li] > .5)
        lgr_sample = set(map(int, rng.choice(eligible, min(config.lgr5_centers_per_organoid, len(eligible)), replace=False)))
        meta, regions, crypts = _regions(data_dir, org, root)
        adjacency = [set() for _ in x]
        for a, b in graph.edge_index.numpy().T:
            adjacency[a].add(int(b))
        subs = build_ego_subgraphs_for_graph(graph, num_hops=2, centers=sorted(all_sample | lgr_sample))
        for sub in subs:
            center, si = int(sub.orig_center), len(subgraphs)
            sub.full_num_cells = len(x)
            subgraphs.append(sub)
            near = rings(adjacency, center)
            row = dict(node_id=si, organoid_str=org, orig_center=center, observed_n=len(x),
                timepoint=str(meta.get("timepoint", "unknown")), region=regions[center], crypt_id=int(crypts[center]),
                measured_curvature=float(arr["y"][center]), sample_all=center in all_sample, sample_lgr5=center in lgr_sample,
                center_marker=markers[int(x[center].argmax())] if x[center].sum() else "Unmarked")
            for hop, ring in enumerate(near, 1):
                row[f"hop{hop}_size"] = len(ring)
                for mi, marker in enumerate([*markers, "Unmarked"]):
                    count = int(x[ring, mi].sum()) if mi < len(markers) else int((x[ring].sum(axis=1) == 0).sum())
                    row[f"hop{hop}_count_{marker}"] = count
                    row[f"hop{hop}_fraction_{marker}"] = count / len(ring) if ring else np.nan
            nodes.append(row)
            exemplars[str(si)] = dict(orig_nodes=sub.orig_nodes.tolist(), edges=sub.edge_index.tolist(),
                marker_index=np.where(sub.x.sum(1).numpy() > 0, sub.x.argmax(1).numpy(), len(markers)).tolist(),
                center_idx=int(sub.center_idx))
            if center in lgr_sample:
                for hop in config.hops:
                    for marker in config.markers:
                        mi = markers.index(marker)
                        positive = [j for j in near[hop - 1] if x[j, mi] > .5]
                        if not positive:
                            continue
                        source = int(rng.choice(positive))
                        local = int(torch.nonzero(sub.orig_nodes == source).item())
                        cases.append(dict(case_id=len(cases), node_id=si, organoid_str=org, orig_center=center,
                            orig_source=source, source_local=local, source_marker=mi, marker=marker, hop=hop,
                            observed_n=len(x), eligible_sources=len(positive)))
        if len(full_checks) < 3 and subs:
            full_checks.append((graph, len(subgraphs) - len(subs)))
    if not subgraphs or not cases:
        raise ValueError("No centers/ablation cases; increase the sample")
    nodes, cases = pd.DataFrame(nodes), pd.DataFrame(cases)
    paired = cases.groupby(["node_id", "hop"]).marker.nunique().eq(len(config.markers))
    cases["matched_markers"] = [bool(paired.loc[(r.node_id, r.hop)]) for r in cases.itertuples()]
    requests = [(i, ()) for i in range(len(subgraphs))]
    for case in cases.itertuples():
        requests.append((case.node_id, ((case.source_local, case.source_marker),)))
    cases["base_request"] = cases.node_id
    cases["edit_request"] = np.arange(len(subgraphs), len(requests))
    return subgraphs, nodes, cases, requests, exemplars, full_checks


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


def effect_table(pred, reference, cases, pre, geometric_reference, n, reference_n, route):
    base, edit = cases.base_request.to_numpy(), cases.edit_request.to_numpy()
    delta = pred["h"][edit] - pred["h"][base]
    delta_ref = reference["h"][edit] - reference["h"][base]
    norm, ratio, cosine = hidden_comparison(delta, delta_ref)
    raw = np.asarray(pre["residual_transform"].inverse(pred["z"].astype(float))).reshape(-1)
    ns = cases.observed_n.to_numpy() if n is None else np.full(len(cases), n)
    area_factor = np.exp(geometric_reference.alpha + geometric_reference.beta * np.log(ns)) / (4 * np.pi)
    fixed_factor = np.exp(geometric_reference.alpha + geometric_reference.beta * np.log(reference_n)) / (4 * np.pi)
    table = cases.copy().assign(n=ns, mode="observed" if n is None else "sweep", route=route,
        delta_z=pred["z"][edit] - pred["z"][base], delta_raw=raw[edit] - raw[base],
        hidden_norm=norm, hidden_ratio=ratio, hidden_cosine=cosine)
    table["delta_relative"] = table.delta_raw * area_factor
    table["delta_fixed_reference"] = table.delta_raw * fixed_factor
    return table


def _check_full_graph(model, checks, subs, pre, config, device):
    """Verify the extracted ego's center is identical to full-graph inference."""
    size = float((np.log(config.reference_n) - pre["size_center"]) / pre["size_scale"])
    errors = []
    for graph, si in checks:
        graph = graph.clone()
        graph.global_feat = graph.x.new_tensor([[size]])
        batch = Batch.from_data_list([graph]).to(device)
        with torch.no_grad():
            (mu, _), h = model(batch.x, batch.edge_index, batch)
        pred = infer_states(model, subs, [(si, ())], pre, count=config.reference_n, device=device)
        center = int(subs[si].orig_center)
        np.testing.assert_allclose(pred["z"][0], mu[center].item(), atol=3e-6, rtol=3e-5)
        np.testing.assert_allclose(pred["h"][0], h[center, :model.hidden_dim].detach().cpu(), atol=3e-5, rtol=3e-5)
        errors.append(float(abs(pred["z"][0] - mu[center].item())))
    return max(errors, default=0.)


def run_inference(root, run_dir, output_dir, config=EmbeddingConfig(), device="cpu"):
    """Resume by checkpoint; persist embeddings only at anchors, reference and observed N.

    Dense-grid scalar effects and route controls are retained at every N. Outputs
    have a settings/source fingerprint to prevent incompatible cache reuse.
    """
    support = inspect_inputs(root, run_dir, config)
    root, run_dir, out = Path(root), Path(run_dir), Path(output_dir)
    settings, data_dir, markers = experiment_paths(root, run_dir)
    out.mkdir(parents=True, exist_ok=True)
    payload = dict(config=asdict(config), run_dir=str(run_dir.resolve()), source_sha256=_sha(__file__),
        run_settings_sha256=_sha(run_dir / "settings.json"), markers=markers,
        split_sha256=_sha(run_dir / "tables/split_membership.csv"),
        geometry_sha256=_sha(run_dir / "geometric_normalization/references.csv"))
    encoded = json.loads(json.dumps(payload))
    if (out / "settings.json").exists() and json.loads((out / "settings.json").read_text()) != encoded:
        raise ValueError("Output settings/code changed. Choose a new output directory to preserve cached results.")
    _json(out / "settings.json", payload)
    support.to_csv(out / "size_support.csv", index=False)
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable in this kernel; select a CUDA-capable environment or CPU")
    refs = pd.read_csv(run_dir / "geometric_normalization/references.csv").set_index("fold")
    for fold in config.folds:
        folder = out / f"fold_{fold}"
        folder.mkdir(exist_ok=True)
        print(f"Prepare fixed held-out cells: fold {fold}", flush=True)
        subs, nodes, cases, requests, exemplars, checks = prepare_fold(root, run_dir, fold, config)
        nodes.to_csv(folder / "nodes.csv", index=False)
        cases.to_csv(folder / "cases.csv", index=False)
        _json(folder / "neighborhoods.json", exemplars)
        with (run_dir / f"checkpoints/fold_{fold}_preprocessing.pkl").open("rb") as handle:
            pre = pickle.load(handle)
        fingerprint = hashlib.sha256(pd.util.hash_pandas_object(nodes, index=True).values.tobytes() +
            pd.util.hash_pandas_object(cases, index=True).values.tobytes()).hexdigest()
        # Hash selected graph inputs and metadata, not just the selected node IDs.
        dataset_hash = hashlib.sha256()
        for org in sorted(nodes.organoid_str.unique()):
            for suffix in (".npz", "_aux.json"):
                dataset_hash.update((data_dir / f"{org}{suffix}").read_bytes())
        for seed in config.seeds:
            dest = folder / f"seed_{seed}"
            dest.mkdir(exist_ok=True)
            checkpoint = run_dir / f"checkpoints/fold_{fold}_seed_{seed}_gin_film_size.pt"
            signature = dict(checkpoint=_sha(checkpoint), preprocessing=_sha(run_dir / f"checkpoints/fold_{fold}_preprocessing.pkl"),
                             sample=fingerprint, dataset=dataset_hash.hexdigest())
            if (dest / "inference_done.json").exists():
                done = json.loads((dest / "inference_done.json").read_text())
                if done["signature"] != signature:
                    raise ValueError("Checkpoint or input data changed; choose a new output directory")
                print(f"Cached inference: fold {fold}, seed {seed}", flush=True)
                continue
            model = make_model(settings, markers)
            model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
            model.to(device).eval()
            kwargs = dict(model=model, subgraphs=subs, requests=requests, pre=pre, reference_n=config.reference_n,
                          device=device, batch_size=config.batch_size)
            ref = infer_states(**kwargs, count=config.reference_n)
            max_error = _check_full_graph(model, checks, subs, pre, config, device)
            tables = []
            for n in [*config.counts, None]:
                pred = ref if n == config.reference_n else infer_states(**kwargs, count=n)
                tag = "observed" if n is None else f"N{n}"
                if n is None or n in (*config.anchors, config.reference_n):
                    np.savez_compressed(dest / f"states_{tag}.npz", **pred)
                tables.append(effect_table(pred, ref, cases, pre, refs.loc[fold], n, config.reference_n, "full"))
                if n is not None:
                    for route in ("film_only", "head_only"):
                        routed = infer_states(**kwargs, count=n, route=route)
                        expected = ref["h"] if route == "head_only" else pred["h"]
                        np.testing.assert_allclose(routed["h"], expected, atol=2e-5, rtol=5e-5)
                        tables.append(effect_table(routed, ref, cases, pre, refs.loc[fold], n, config.reference_n, route))
                print(f"Embeddings/ablations: fold {fold}, seed {seed}, {tag}", flush=True)
            pd.concat(tables, ignore_index=True).to_csv(dest / "effects.csv.gz", index=False)
            _json(dest / "inference_done.json", dict(signature=signature, max_full_graph_z_error=max_error,
                nodes=len(nodes), cases=len(cases), head_only_hidden_invariant=True, film_only_hidden_matches_full=True))
    return out


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


def _profiles(assignments, nodes, markers):
    frame = assignments.merge(nodes, on="node_id", validate="many_to_one")
    for name in [*markers, "Unmarked"]:
        frame[f"center_fraction_{name}"] = (frame.center_marker == name).astype(float)
    for region in sorted(frame.region.unique()):
        frame[f"region_fraction_{region}"] = (frame.region == region).astype(float)
    cols = [c for c in frame if c.startswith(("hop1_", "hop2_", "center_fraction_", "region_fraction_"))]
    cols += ["observed_n", "measured_curvature", "prediction_z"]
    # Equal organoid weight within each cluster, rather than letting large organs dominate.
    org = frame.groupby(["state", "cluster", "organoid_str"])[cols].mean().reset_index()
    means = org.groupby(["state", "cluster"])[cols].mean().reset_index()
    support = frame.groupby(["state", "cluster"]).agg(n_cells=("node_id", "size"), n_organoids=("organoid_str", "nunique"))
    return means.merge(support.reset_index(), on=["state", "cluster"])


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


def _joint_tsne(dest, atlas, ids, cases, config, include_edits):
    """Sample identities once; all their anchor states share one t-SNE fit."""
    rng = np.random.default_rng(config.tsne_seed)
    ids = np.sort(rng.choice(ids, min(len(ids), config.tsne_max_cells), replace=False))
    records, embeddings = [], []
    for n in config.anchors:
        with np.load(dest / f"states_N{n}.npz") as archive:
            h = archive["h"]
        for i in ids:
            records.append(dict(node_id=i, n=n, condition="intact", hop=0))
            embeddings.append(h[i])
        if include_edits:
            # Hop 1 is the primary t-SNE overlay; hop 2 retained in numerical tables.
            selected = cases[cases.node_id.isin(ids) & (cases.hop == 1)]
            for c in selected.itertuples():
                records.append(dict(node_id=c.node_id, n=n, condition=c.marker, hop=c.hop))
                embeddings.append(h[c.edit_request])
    h = np.stack(embeddings)
    reduced, labels, _ = project_atlas(atlas, h)
    tsne = TSNE(n_components=2, perplexity=min(config.tsne_perplexity, max(1., (len(h) - 1) / 3)),
        learning_rate="auto", max_iter=config.tsne_max_iter, init="pca", random_state=config.tsne_seed)
    coordinates = tsne.fit_transform(reduced)
    return pd.DataFrame(records).assign(cluster=labels, tsne_1=coordinates[:, 0], tsne_2=coordinates[:, 1])


def run_clustering(output_dir, config=EmbeddingConfig()):
    """Fit two atlases per checkpoint; no cross-checkpoint hidden-vector pooling."""
    out = Path(output_dir)
    saved = json.loads((out / "settings.json").read_text())
    if saved["config"] != json.loads(json.dumps(asdict(config))):
        raise ValueError("Clustering settings must match the inference configuration")
    markers = saved["markers"]
    for fold in config.folds:
        folder = out / f"fold_{fold}"
        nodes, cases = pd.read_csv(folder / "nodes.csv"), pd.read_csv(folder / "cases.csv")
        for seed in config.seeds:
            dest = folder / f"seed_{seed}"
            if not (dest / "inference_done.json").exists():
                raise FileNotFoundError(f"Run inference first: {dest}")
            if (dest / "clustering_done.json").exists():
                print(f"Cached clustering: fold {fold}, seed {seed}", flush=True)
                continue
            states = {}
            for tag in [*[f"N{n}" for n in sorted(set((*config.anchors, config.reference_n)))], "observed"]:
                with np.load(dest / f"states_{tag}.npz") as archive:
                    states[tag] = {key: archive[key] for key in archive.files}
            for population, col in [("all", "sample_all"), ("lgr5", "sample_lgr5")]:
                print(f"Shared GMM: fold {fold}, seed {seed}, {population}", flush=True)
                ids = nodes.loc[nodes[col], "node_id"].to_numpy()
                atlas, sensitivity = fit_shared_atlas([states[f"N{n}"]["h"][ids] for n in config.anchors], config)
                atlas_dir = dest / population
                atlas_dir.mkdir(exist_ok=True)
                with (atlas_dir / "atlas.pkl").open("wb") as handle:
                    pickle.dump(atlas, handle)
                sensitivity.to_csv(atlas_dir / "cluster_sensitivity.csv", index=False)
                reference_h = states[f"N{config.reference_n}"]["h"][ids]
                _, reference_labels, _ = project_atlas(atlas, reference_h)
                assignments, geometry, edited = [], [], []
                for tag, pred in states.items():
                    _, labels, probability = project_atlas(atlas, pred["h"][ids])
                    frame = pd.DataFrame(dict(node_id=ids, state=tag, cluster=labels,
                        reference_cluster=reference_labels, prediction_z=pred["z"][ids], confidence=probability.max(1)))
                    for k in range(config.n_clusters):
                        frame[f"prob_C{k}"] = probability[:, k]
                    assignments.append(frame)
                    if tag != "observed":
                        geometry.append(dict(n=int(tag[1:]), **representation_change(pred["h"][ids], reference_h)))
                    if population == "lgr5":
                        _, edit_labels, edit_prob = project_atlas(atlas, pred["h"][cases.edit_request])
                        _, base_labels, base_prob = project_atlas(atlas, pred["h"][cases.base_request])
                        edited.append(cases[["case_id", "node_id", "marker", "hop", "matched_markers"]].assign(state=tag,
                            base_cluster=base_labels, ablated_cluster=edit_labels,
                            changed_cluster=edit_labels != base_labels,
                            membership_total_variation=.5 * np.abs(edit_prob - base_prob).sum(1)))
                assigned = pd.concat(assignments, ignore_index=True)
                assigned.to_csv(atlas_dir / "assignments.csv.gz", index=False)
                _profiles(assigned, nodes, markers).to_csv(atlas_dir / "profiles.csv", index=False)
                _transitions(assigned, nodes, config).to_csv(atlas_dir / "transitions.csv", index=False)
                pd.DataFrame(geometry).to_csv(atlas_dir / "representation_change.csv", index=False)
                if edited:
                    pd.concat(edited).to_csv(atlas_dir / "ablation_membership.csv.gz", index=False)
                if (fold, seed) == config.tsne_checkpoint:
                    _joint_tsne(dest, atlas, ids, cases, config, population == "lgr5").to_csv(atlas_dir / "tsne.csv.gz", index=False)
            _json(dest / "clustering_done.json", dict(fold=fold, seed=seed, separate_checkpoint_atlases=True))
    return out


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


def summarize(output_dir, config=EmbeddingConfig()):
    """Export fixed-cluster curves, matched sensitivities, seed checks and extrema."""
    out = Path(output_dir)
    tables = out / "tables"
    tables.mkdir(exist_ok=True)
    frames = []
    for fold in config.folds:
        nodes = pd.read_csv(out / f"fold_{fold}/nodes.csv")
        for seed in config.seeds:
            dest = out / f"fold_{fold}/seed_{seed}"
            assignments = pd.read_csv(dest / "lgr5/assignments.csv.gz")
            fixed = assignments[assignments.state == f"N{config.reference_n}"][["node_id", "reference_cluster"]]
            effects = pd.read_csv(dest / "effects.csv.gz").merge(fixed, on="node_id", validate="many_to_one")
            effects = effects.merge(nodes[["node_id", "region", "timepoint"]], on="node_id", validate="many_to_one")
            frames.append(effects.assign(fold=fold, seed=seed))
    effects = pd.concat(frames, ignore_index=True)
    effects.to_csv(tables / "effects_with_fixed_clusters.csv.gz", index=False)
    metrics = ["delta_relative", "delta_z", "delta_raw", "delta_fixed_reference", "hidden_norm", "hidden_ratio", "hidden_cosine"]
    bootstrap = dict(draws=config.bootstrap_draws, seed=config.sampling_seed, min_organoids=config.min_organoids)
    sweep = effects[effects["mode"] == "sweep"].copy()
    # Individual support and the strict common KI67/LGR5 cohort are exported separately.
    cohorts = pd.concat([sweep.assign(cohort="eligible"), sweep[sweep.matched_markers].assign(cohort="matched")])
    group = ["cohort", "route", "marker", "hop"]
    local_support = cohorts[cohorts.route == "full"].copy()
    local_support["within_local_size"] = np.abs(np.log(local_support.n / local_support.observed_n)) <= np.log(config.local_size_fold)
    local_support["log_size_ratio"] = np.log(local_support.n / local_support.observed_n)
    support_rows = []
    for key, part in local_support.groupby(["cohort", "marker", "hop", "n"]):
        distinct = part.drop_duplicates(["organoid_str", "case_id"])
        near = distinct[distinct.within_local_size]
        support_rows.append(dict(zip(["cohort", "marker", "hop", "n"], key)) | dict(
            n_organoids=distinct.organoid_str.nunique(), n_local_organoids=near.organoid_str.nunique(),
            n_cases=len(distinct), n_local_cases=len(near), median_log_size_ratio=distinct.log_size_ratio.median()))
    pd.DataFrame(support_rows).to_csv(tables / "sweep_distance_support.csv", index=False)
    paired_bootstrap_curves(cohorts, group, metrics, **bootstrap).to_csv(tables / "overall_curves.csv", index=False)
    # Cluster IDs have meaning within one fitted checkpoint only.
    fixed = cohorts[cohorts.route == "full"].copy()
    paired_bootstrap_curves(fixed, ["fold", "seed", "cohort", "reference_cluster", "marker", "hop"], metrics,
                            **bootstrap).to_csv(tables / "fixed_cluster_curves.csv", index=False)
    seed_org = cohorts.groupby(["seed", *group, "organoid_str", "n"])[metrics].mean().reset_index()
    seed_curves = seed_org.groupby(["seed", *group, "n"])[metrics].mean().reset_index()
    seed_curves.to_csv(tables / "seed_curves.csv", index=False)
    observed = effects[(effects["mode"] == "observed") & (effects.route == "full")].copy()
    observed["n"] = assign_observed_windows(observed.observed_n.to_numpy(), config.anchors, config.observed_window_fold)
    observed = observed[observed.n > 0]
    paired_bootstrap_curves(observed, ["marker", "hop"], metrics, **bootstrap).to_csv(tables / "observed_curves.csv", index=False)
    # Fixed local-support subset per anchor for an honest observed/sweep comparison.
    local_parts = []
    for anchor in config.anchors:
        ids = observed.loc[observed.n == anchor, ["fold", "seed", "case_id"]].drop_duplicates()
        part = sweep[(sweep.n == anchor) & (sweep.route == "full")].merge(ids, on=["fold", "seed", "case_id"], validate="one_to_one")
        local_parts.append(part)
    local = pd.concat(local_parts, ignore_index=True)
    paired_bootstrap_curves(local, ["marker", "hop"], metrics, **bootstrap).to_csv(tables / "local_support_sweep_curves.csv", index=False)
    # A per-center share is about perturbation sensitivity, not additive prediction attribution.
    matched = sweep[sweep.matched_markers & (sweep.route == "full")]
    index = ["fold", "seed", "organoid_str", "node_id", "hop", "n", "reference_cluster"]
    wide = matched.pivot(index=index, columns="marker", values="delta_relative")
    if set(config.markers) <= set(wide.columns):
        denominator = wide[list(config.markers)].abs().sum(axis=1)
        for marker in config.markers:
            share = wide.reset_index()[index].copy()
            share["share"] = np.divide(wide[marker].abs().to_numpy(), denominator.to_numpy(),
                out=np.full(len(wide), np.nan), where=denominator.to_numpy() > 1e-10)
            share["marker"] = marker
            paired_bootstrap_curves(share, ["marker", "hop"], ["share"], **bootstrap).to_csv(tables / f"sensitivity_share_{marker}.csv", index=False)
    # Locate extrema on the sampled grid, independently for each seed and output scale.
    extrema = []
    for key, group_frame in seed_curves[(seed_curves.route == "full")].groupby(["seed", "cohort", "marker", "hop"]):
        group_frame = group_frame.sort_values("n")
        for metric in ["delta_relative", "delta_z", "delta_raw"]:
            y = group_frame[metric].to_numpy()
            for kind, index_ in [("minimum", int(np.argmin(y))), ("maximum", int(np.argmax(y)))]:
                extrema.append(dict(zip(["seed", "cohort", "marker", "hop"], key)) | dict(metric=metric, kind=kind,
                    n=int(group_frame.iloc[index_].n), effect=float(y[index_]), interior=0 < index_ < len(y) - 1))
    pd.DataFrame(extrema).to_csv(tables / "grid_extrema_by_seed.csv", index=False)
    local_extrema = []
    for key, group_frame in seed_curves[seed_curves.route == "full"].groupby(["seed", "cohort", "marker", "hop"]):
        group_frame = group_frame.sort_values("n")
        for metric in ["delta_relative", "delta_z", "delta_raw"]:
            y, ns = group_frame[metric].to_numpy(), group_frame.n.to_numpy()
            slope = np.diff(y) / np.diff(np.log(ns))
            for j in range(1, len(ns) - 1):
                kind = "minimum" if slope[j-1] < 0 < slope[j] else "maximum" if slope[j-1] > 0 > slope[j] else None
                if kind:
                    local_extrema.append(dict(zip(["seed", "cohort", "marker", "hop"], key)) |
                        dict(metric=metric, kind=kind, n=int(ns[j]), effect=float(y[j]),
                             left_n=int(ns[j-1]), right_n=int(ns[j+1]), left_slope=float(slope[j-1]), right_slope=float(slope[j])))
    pd.DataFrame(local_extrema, columns=["seed", "cohort", "marker", "hop", "metric", "kind", "n", "effect", "left_n", "right_n", "left_slope", "right_slope"]).to_csv(
        tables / "local_turning_points_by_seed.csv", index=False)
    # Paired organoid bootstraps estimate the location of the pooled curve's extremum.
    extrema_boot = []
    rng = np.random.default_rng(config.sampling_seed)
    full = cohorts[cohorts.route == "full"]
    for key, group_frame in full.groupby(["cohort", "marker", "hop"]):
        organ = group_frame.groupby(["organoid_str", "n"])[["delta_relative", "delta_z", "delta_raw"]].mean().reset_index()
        for metric in ["delta_relative", "delta_z", "delta_raw"]:
            matrix = organ.pivot(index="organoid_str", columns="n", values=metric).sort_index(axis=1)
            if matrix.isna().any().any():
                raise ValueError("Extremum bootstraps require fixed support across N")
            counts = matrix.columns.to_numpy()
            indices = rng.integers(0, len(matrix), (config.bootstrap_draws, len(matrix)))
            curves = matrix.to_numpy()[indices].mean(axis=1)
            for kind, arg in [("minimum", curves.argmin(1)), ("maximum", curves.argmax(1))]:
                for n in counts:
                    extrema_boot.append(dict(zip(["cohort", "marker", "hop"], key)) | dict(metric=metric, kind=kind,
                        n=int(n), probability=float((counts[arg] == n).mean()), n_organoids=len(matrix)))
    pd.DataFrame(extrema_boot, columns=["cohort", "marker", "hop", "metric", "kind", "n", "probability", "n_organoids"]).to_csv(
        tables / "grid_extrema_bootstrap.csv", index=False)
    _json(tables / "summary_done.json", dict(checkpoints=len(config.folds) * len(config.seeds),
        uncertainty="Paired organoid bootstrap, conditional on fitted models, atlas and geometric calibration; seeds averaged within organoid.",
        cluster_ids="Never pooled across checkpoints", observed_windows="Nearest anchor within multiplicative window"))
    return tables
