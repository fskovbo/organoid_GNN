"""One-hop KI67 response geometry, with a fixed N=300 reference.

Uses unchanged held-out cases and checkpoints. PCA is an orthogonal projection
of unscaled local hidden features, with equal organoid weight. Existing GMM
clusters remain annotations; no clustering is fitted to the 2-D display.
"""
from dataclasses import dataclass, asdict
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.analysis.exclusive_size_ablation import make_model
from src.analysis.lgr5_film import hidden_comparison
from src.analysis.size_embedding import infer_states, paired_bootstrap_curves, project_atlas
from src.analysis.size_embedding_focus import load_config, curvature_order, head_prediction
from src.data.io import load_organoid_npz, build_pyg_graph
from src.data.subgraphs import build_ego_subgraphs_for_graph


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


def _write(path, value):
    Path(path).write_text(json.dumps(value, indent=2))


def _read_npz(path):
    with np.load(path) as archive:
        return {key: archive[key] for key in archive.files}


def _subgraphs(data_dir, cases):
    """Reconstruct just the unchanged saved KI67 cases; center and source verified."""
    result = {}
    for org, group in cases.groupby("organoid_str", sort=True):
        arr = load_organoid_npz(str(data_dir / f"{org}.npz"), strict=True, target_indices=[0])
        graph = build_pyg_graph(arr["x"], arr["edges"], arr["y"])
        graph.organoid_str = org
        lookup = {int(c.orig_center): c for c in group.itertuples()}
        for sub in build_ego_subgraphs_for_graph(graph, num_hops=2, centers=sorted(lookup)):
            row = lookup[int(sub.orig_center)]
            sub.full_num_cells = len(graph.x)
            local = int(torch.nonzero(sub.orig_nodes == row.orig_source).item())
            if local != row.source_local or sub.x[local, row.source_marker] != 1:
                raise ValueError("Saved source no longer matches the exclusive graph")
            if int(sub.center_idx) == local or (sub.x.sum(1) > 1).any():
                raise ValueError("Expected exclusive input and a distinct neighboring source")
            result[row.case_id] = sub
    return [result[c] for c in cases.case_id]


def _slice_states(saved, cases):
    return dict(h_base=saved["h"][cases.base_request], h_abl=saved["h"][cases.edit_request],
                z_base=saved["z"][cases.base_request], z_abl=saved["z"][cases.edit_request])


def _inferred_states(model, subs, cases, pre, n, config, device):
    requests = [(i, ()) for i in range(len(cases))]
    requests += [(i, ((int(c.source_local), int(c.source_marker)),)) for i, c in enumerate(cases.itertuples())]
    pred = infer_states(model, subs, requests, pre, count=n, reference_n=config.reference_n,
                        device=device, batch_size=config.batch_size)
    return dict(h_base=pred["h"][:len(cases)], h_abl=pred["h"][len(cases):],
                z_base=pred["z"][:len(cases)], z_abl=pred["z"][len(cases):])


def _profile_table(nodes, fixed_clusters, order):
    frame = nodes.copy().assign(cluster=fixed_clusters)
    cols = [c for c in frame if c.startswith("hop1_")]
    organ = frame.groupby(["cluster", "organoid_str"])[cols].mean().reset_index()
    result = organ.groupby("cluster")[cols].mean()
    result["n_organoids"] = organ.groupby("cluster").organoid_str.nunique()
    result["n_centers"] = frame.groupby("cluster").size()
    return result.reset_index().merge(order, on="cluster", validate="one_to_one")


def analyze_checkpoint(dest, original, cases, nodes, model, pre, geometry, config):
    states = {n: _read_npz(dest / f"N{n}.npz") for n in config.counts}
    pca = fit_weighted_pca([states[n][key] for n in config.counts for key in ("h_base", "h_abl")], cases.organoid_str)
    with (dest / "pca.pkl").open("wb") as handle:
        pickle.dump(pca, handle)
    with (original / "lgr5/atlas.pkl").open("rb") as handle:
        atlas = pickle.load(handle)
    reference = states[config.reference_n]
    _, raw_fixed, _ = project_atlas(atlas, reference["h_base"])
    physical_reference = np.asarray(pre["residual_transform"].inverse(reference["z_base"].astype(float))).reshape(-1)
    order = curvature_order(raw_fixed, physical_reference, atlas["gmm"].n_components)
    order.to_csv(dest / "curvature_order.csv", index=False)
    mapping = order.set_index("raw_cluster").cluster.to_dict()
    fixed = np.array([mapping[c] for c in raw_fixed])
    _profile_table(nodes, fixed, order).to_csv(dest / "neighborhood_profiles.csv", index=False)
    points, effects, fidelity, readouts = [], [], [], {}
    ref_delta = reference["h_abl"] - reference["h_base"]
    ref_size = (np.log(config.reference_n) - pre["size_center"]) / pre["size_scale"]
    ref_g = exact_readout(model, reference["h_base"], ref_delta, ref_size)["g"]
    for n, state in states.items():
        base, delta = state["h_base"], state["h_abl"] - state["h_base"]
        size = (np.log(n) - pre["size_center"]) / pre["size_scale"]
        factor = np.exp(geometry.alpha + geometry.beta * np.log(n)) / (4 * np.pi)
        raw_base = np.asarray(pre["residual_transform"].inverse(state["z_base"].astype(float))).reshape(-1)
        raw_abl = np.asarray(pre["residual_transform"].inverse(state["z_abl"].astype(float))).reshape(-1)
        norm, ratio, cosine = hidden_comparison(delta, ref_delta)
        ro = exact_readout(model, base, delta, size)
        readouts[n] = ro
        dz = state["z_abl"].astype(float) - state["z_base"].astype(float)
        np.testing.assert_allclose(ro["effect"], dz, atol=3e-6, rtol=5e-5)
        _, _, g_cosine = hidden_comparison(ro["g"], ref_g)
        data = cases[["case_id", "node_id", "organoid_str", "orig_center", "observed_n"]].copy()
        data = data.assign(n=n, cluster=fixed, delta_z=dz, delta_raw=raw_abl-raw_base,
            delta_relative=(raw_abl-raw_base)*factor, hidden_norm=norm, hidden_ratio=ratio,
            hidden_cosine=cosine, readout_cosine=g_cosine,
            frozen_reference_readout=np.sum(ref_g*delta, axis=1), positive_units=ro["positive"], negative_units=ro["negative"])
        effects.append(data)
        _, raw_clusters, _ = project_atlas(atlas, base)
        a, b = pca.transform(base), pca.transform(state["h_abl"])
        points.append(data[["case_id", "node_id", "organoid_str", "n", "delta_relative"]].assign(
            pc1=a[:, 0], pc2=a[:, 1], ablated_pc1=b[:, 0], ablated_pc2=b[:, 1],
            prediction_residual=raw_base, cluster=[mapping[c] for c in raw_clusters], reference_cluster=fixed))
        energy = np.sum(delta.astype(float)**2, axis=1)
        for dim in sorted(set(min(d, len(pca.components)) for d in config.dimensions)):
            projected = pca.reconstruct_displacement(delta, dim)
            pred_z = head_prediction(model, base + projected, size)
            projected_raw = np.asarray(pre["residual_transform"].inverse(pred_z)).reshape(-1)
            projected_effect = (projected_raw - raw_base) * factor
            true_effect = (raw_abl - raw_base) * factor
            part = cases[["organoid_str"]].copy()
            part["retained_energy"] = np.divide(np.sum(projected**2, axis=1), energy,
                out=np.full(len(cases), np.nan), where=energy > 1e-16)
            part["squared_error"] = (projected_effect - true_effect)**2
            part["squared_effect"] = true_effect**2
            part["sign_agreement"] = np.where(np.abs(true_effect) > 1e-8, np.sign(projected_effect) == np.sign(true_effect), np.nan)
            org = part.groupby("organoid_str").mean().reset_index()
            fidelity.append(org.assign(n=n, dimensions=dim, pooled_variance_retained=float(pca.variance_ratio[:dim].sum())))
    frame = pd.concat(effects, ignore_index=True)
    # Exact two-factor decomposition of neighboring steps, cumulatively anchored at N=300.
    response_steps, readout_steps = [np.zeros(len(cases))], [np.zeros(len(cases))]
    for a, b in zip(config.counts[:-1], config.counts[1:]):
        d_a, d_b = states[a]["h_abl"]-states[a]["h_base"], states[b]["h_abl"]-states[b]["h_base"]
        dr, dg = bilinear_change(readouts[a]["g"], d_a, readouts[b]["g"], d_b)
        response_steps.append(response_steps[-1]+dr)
        readout_steps.append(readout_steps[-1]+dg)
    ref_idx = config.counts.index(config.reference_n)
    for i, n in enumerate(config.counts):
        frame.loc[frame.n == n, "response_change_from_reference"] = response_steps[i] - response_steps[ref_idx]
        frame.loc[frame.n == n, "readout_change_from_reference"] = readout_steps[i] - readout_steps[ref_idx]
    frame.to_csv(dest / "effects.csv.gz", index=False)
    pd.concat(points).to_csv(dest / "pca_points.csv.gz", index=False)
    pd.concat(fidelity).to_csv(dest / "projection_fidelity_organoids.csv.gz", index=False)
    pd.DataFrame(dict(component=np.arange(1, len(pca.components)+1), variance_ratio=pca.variance_ratio,
        cumulative_variance=np.cumsum(pca.variance_ratio))).to_csv(dest / "pca_variance.csv", index=False)


def marker_probes(dest, subs, cases, model, pre, geometry, markers, config, device):
    """Other one-hop marker removals at the same KI67-eligible centers (one checkpoint).

    PCA axes are unchanged. Each marker has its own fixed support across N; its
    direction is compared to KI67 at the same center, rather than to an unmatched
    population mean. The source selection is deterministic and recorded.
    """
    with (dest / "pca.pkl").open("rb") as handle:
        pca = pickle.load(handle)
    requests, records = [], []
    for i, (sub, case) in enumerate(zip(subs, cases.itertuples())):
        center = int(sub.center_idx)
        ring = sub.edge_index[0, sub.edge_index[1] == center].unique().numpy()
        for mi, marker in enumerate(markers):
            if marker == "KI67":
                continue
            eligible = ring[sub.x[ring, mi].numpy() > .5]
            eligible = eligible[eligible != center]
            if not len(eligible):
                continue
            seed = int.from_bytes(hashlib.sha256(f"{config.probe_seed}:{case.organoid_str}:{case.orig_center}:{marker}".encode()).digest()[:8], "little")
            source = int(np.random.default_rng(seed).choice(sorted(eligible)))
            requests.append((i, ((source, mi),)))
            records.append(dict(case_id=case.case_id, row=i, node_id=case.node_id, organoid_str=case.organoid_str,
                marker=marker, source=int(sub.orig_nodes[source])))
    manifest = pd.DataFrame(records)
    manifest.to_csv(dest / "probe_manifest.csv", index=False)
    if not len(manifest):
        return
    rows = []
    for n in config.display_counts:
        state = _read_npz(dest / f"N{n}.npz")
        pred = infer_states(model, subs, requests, pre, count=n, reference_n=config.reference_n,
                            device=device, batch_size=config.batch_size)
        base = state["h_base"][manifest.row]
        delta = pred["h"] - base
        ki_delta = (state["h_abl"]-state["h_base"])[manifest.row]
        _, _, cosine = hidden_comparison(delta, ki_delta)
        pc = pca.displacement(delta)
        ki_pc = pca.displacement(ki_delta)
        raw = np.asarray(pre["residual_transform"].inverse(pred["z"].astype(float))).reshape(-1)
        raw_base = np.asarray(pre["residual_transform"].inverse(state["z_base"].astype(float))).reshape(-1)[manifest.row]
        factor = np.exp(geometry.alpha + geometry.beta*np.log(n))/(4*np.pi)
        energy = np.sum(delta.astype(float)**2, axis=1)
        keep = np.divide(np.sum(pc**2, axis=1), energy, out=np.full(len(delta),np.nan), where=energy>1e-16)
        rows.append(manifest.assign(n=n, delta_pc1=pc[:,0], delta_pc2=pc[:,1], ki67_pc1=ki_pc[:,0], ki67_pc2=ki_pc[:,1],
            full_space_cosine_to_KI67=cosine, retained_energy_2pc=keep, delta_relative=(raw-raw_base)*factor))
    pd.concat(rows).to_csv(dest / "marker_probes.csv.gz", index=False)


def run(source_dir, root, config=PCAStudyConfig(), *, device="cpu"):
    config.validate()
    source, root = Path(source_dir).resolve(), Path(root).resolve()
    original_config = load_config(source)
    old_settings = json.loads((source / "settings.json").read_text())
    run_dir = Path(old_settings["run_dir"])
    settings = json.loads((run_dir / "settings.json").read_text())
    data_dir = root / "training_data" / settings["DATASET_NAME"]
    out = source / "ki67_hop1_pca_v1"
    out.mkdir(exist_ok=True)
    metadata = dict(version=1, config=asdict(config), source=str(source), markers=old_settings["markers"],
        source_settings_sha256=hashlib.sha256((source / "settings.json").read_bytes()).hexdigest())
    metadata = json.loads(json.dumps(metadata))
    if (out / "settings.json").exists() and json.loads((out / "settings.json").read_text()) != metadata:
        raise ValueError("Incompatible PCA configuration; use a different versioned output directory")
    _write(out / "settings.json", metadata)
    if (out / "complete.json").exists():
        add_response_projection(out, config)
        print("Using completed KI67 PCA analysis", flush=True)
        return out
    geometry = pd.read_csv(run_dir / "geometric_normalization/references.csv").set_index("fold")
    for fold in original_config.folds:
        folder = source / f"fold_{fold}"
        all_cases = pd.read_csv(folder / "cases.csv")
        cases = all_cases[(all_cases.marker == "KI67") & (all_cases.hop == 1)].reset_index(drop=True)
        if cases.node_id.duplicated().any() or cases.empty:
            raise ValueError("Expected one saved hop-1 KI67 source per eligible center")
        all_nodes = pd.read_csv(folder / "nodes.csv").set_index("node_id")
        nodes = all_nodes.loc[cases.node_id].reset_index()
        if not nodes.center_marker.eq("LGR5").all():
            raise ValueError("Only unchanged exclusive LGR5 centers are eligible")
        folded = out / f"fold_{fold}"
        folded.mkdir(exist_ok=True)
        cases.to_csv(folded / "cases.csv", index=False)
        nodes.to_csv(folded / "nodes.csv", index=False)
        with (run_dir / f"checkpoints/fold_{fold}_preprocessing.pkl").open("rb") as handle:
            pre = pickle.load(handle)
        subs = None
        for seed in original_config.seeds:
            dest, original = folded / f"seed_{seed}", folder / f"seed_{seed}"
            dest.mkdir(exist_ok=True)
            if (dest / "done.json").exists():
                print(f"Cached PCA: fold {fold}, seed {seed}", flush=True)
                continue
            checkpoint = run_dir / f"checkpoints/fold_{fold}_seed_{seed}_gin_film_size.pt"
            digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            if digest != json.loads((original / "inference_done.json").read_text())["signature"]["checkpoint"]:
                raise ValueError("Checkpoint changed since the saved embedding experiment")
            model = make_model(settings, old_settings["markers"])
            model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
            existing = pd.read_csv(original / "effects.csv.gz")
            existing = existing[(existing["mode"] == "sweep") & (existing.route == "full") & (existing.marker == "KI67") & (existing.hop == 1)]
            max_error = 0.
            for n in config.counts:
                path = dest / f"N{n}.npz"
                if path.exists():
                    state = _read_npz(path)
                elif (original / f"states_N{n}.npz").exists():
                    state = _slice_states(_read_npz(original / f"states_N{n}.npz"), cases)
                    np.savez_compressed(path, **state)
                else:
                    if subs is None:
                        subs = _subgraphs(data_dir, cases)
                    state = _inferred_states(model, subs, cases, pre, n, config, device)
                    np.savez_compressed(path, **state)
                old = existing[existing.n == n].set_index("case_id").reindex(cases.case_id)
                if old.delta_z.isna().any():
                    raise ValueError("N grid must be present in the saved scalar sweep for verification")
                difference = state["z_abl"]-state["z_base"]
                np.testing.assert_allclose(difference, old.delta_z, atol=3e-6, rtol=5e-5)
                max_error = max(max_error, float(np.max(np.abs(difference-old.delta_z.to_numpy()))))
                print(f"KI67 hidden states: fold {fold}, seed {seed}, N={n}", flush=True)
            model.cpu().eval()
            analyze_checkpoint(dest, original, cases, nodes, model, pre, geometry.loc[fold], config)
            if (fold, seed) == config.probe_checkpoint:
                if subs is None:
                    subs = _subgraphs(data_dir, cases)
                marker_probes(dest, subs, cases, model, pre, geometry.loc[fold], old_settings["markers"], config, device)
            _write(dest / "done.json", dict(checkpoint_sha256=digest, cases=len(cases), max_saved_effect_error=max_error))
            print(f"PCA COMPLETE: fold {fold}, seed {seed}", flush=True)
    summarize(out, config)
    _write(out / "complete.json", dict(checkpoints=len(original_config.folds)*len(original_config.seeds), reference_n=config.reference_n,
        core_marker="KI67", hop=1, probe_checkpoint=list(config.probe_checkpoint)))
    add_response_projection(out, config)
    return out


def summarize(out, config):
    out = Path(out)
    tables = out / "tables"
    tables.mkdir(exist_ok=True)
    effects, fidelity = [], []
    for dest in sorted(out.glob("fold_*/seed_*")):
        fold, seed = int(dest.parent.name.split("_")[1]), int(dest.name.split("_")[1])
        effects.append(pd.read_csv(dest / "effects.csv.gz").assign(fold=fold, seed=seed))
        fidelity.append(pd.read_csv(dest / "projection_fidelity_organoids.csv.gz").assign(fold=fold, seed=seed))
    frame = pd.concat(effects, ignore_index=True)
    frame.to_csv(tables / "effects.csv.gz", index=False)
    metrics = ["delta_relative", "delta_z", "delta_raw", "hidden_norm", "hidden_ratio", "hidden_cosine", "readout_cosine",
               "frozen_reference_readout", "positive_units", "negative_units", "response_change_from_reference", "readout_change_from_reference"]
    frame["cohort"] = "KI67_hop1"
    kwargs = dict(draws=config.bootstrap_draws, seed=config.probe_seed, min_organoids=5)
    paired_bootstrap_curves(frame, ["cohort"], metrics, **kwargs).to_csv(tables / "curves.csv", index=False)
    paired_bootstrap_curves(frame, ["fold", "seed", "cluster"], ["delta_relative"], **kwargs).to_csv(tables / "cluster_curves.csv", index=False)
    org = frame.groupby(["seed", "organoid_str", "n"])[metrics].mean().reset_index()
    org.groupby(["seed", "n"])[metrics].mean().reset_index().to_csv(tables / "seed_curves.csv", index=False)
    # Describe each unchanged pair separately; seed averages are not independent cells.
    individual = frame.groupby(["fold", "case_id", "organoid_str", "n"])[["delta_relative", "delta_z"]].mean().reset_index()
    individual.to_csv(tables / "individual_curves.csv.gz", index=False)
    patterns = []
    for key, part in individual.groupby(["fold", "case_id", "organoid_str"]):
        part = part.sort_values("n")
        y, ns = part.delta_relative.to_numpy(), part.n.to_numpy()
        z = part.delta_z.to_numpy()
        slopes = np.diff(y)
        local = ns[1:-1][(slopes[:-1] < 0) & (slopes[1:] > 0)]
        # A small transformed-effect dead band excludes floating-point sign flips.
        signs = np.sign(z[np.abs(z) > 1e-6])
        patterns.append(dict(fold=key[0], case_id=key[1], organoid_str=key[2],
            minimum_n=int(ns[np.argmin(y)]), interior_minimum=bool(0 < np.argmin(y) < len(y)-1),
            local_minimum_near_300=bool(((local >= 250) & (local <= 350)).any()),
            sign_changes=int(np.sum(signs[1:] != signs[:-1])) if len(signs)>1 else 0))
    pd.DataFrame(patterns).to_csv(tables / "individual_patterns.csv", index=False)
    fd = pd.concat(fidelity, ignore_index=True)
    fd.to_csv(tables / "projection_fidelity_organoids.csv.gz", index=False)
    org = fd.groupby(["organoid_str", "n", "dimensions"])[["retained_energy", "squared_error", "squared_effect", "sign_agreement", "pooled_variance_retained"]].mean().reset_index()
    final = org.groupby(["n", "dimensions"]).mean(numeric_only=True).reset_index()
    final["relative_rmse"] = np.sqrt(final.squared_error / final.squared_effect)
    final.to_csv(tables / "projection_fidelity.csv", index=False)


def add_response_projection(out, config):
    """Add a projection of perturbations when state PCs hide most of their energy.

    Reuses all hidden arrays; no GNN evaluations or extra marker edits required.
    """
    out = Path(out)
    if (out / "response_projection_complete.json").exists():
        return
    source = Path(json.loads((out / "settings.json").read_text())["source"])
    source_settings = json.loads((source / "settings.json").read_text())
    run = Path(source_settings["run_dir"])
    settings = json.loads((run / "settings.json").read_text())
    geometry = pd.read_csv(run / "geometric_normalization/references.csv").set_index("fold")
    all_fidelity = []
    for dest in sorted(out.glob("fold_*/seed_*")):
        fold, seed = int(dest.parent.name.split("_")[1]), int(dest.name.split("_")[1])
        cases = pd.read_csv(dest.parent / "cases.csv")
        states = {n:_read_npz(dest / f"N{n}.npz") for n in config.counts}
        deltas = [states[n]["h_abl"]-states[n]["h_base"] for n in config.counts]
        projection = fit_response_svd(deltas, cases.organoid_str)
        with (dest / "response_svd.pkl").open("wb") as handle:
            pickle.dump(projection, handle)
        with (run / f"checkpoints/fold_{fold}_preprocessing.pkl").open("rb") as handle:
            pre = pickle.load(handle)
        model = make_model(settings, source_settings["markers"])
        model.load_state_dict(torch.load(run / f"checkpoints/fold_{fold}_seed_{seed}_gin_film_size.pt", map_location="cpu", weights_only=True))
        points, fidelity = [], []
        for n, delta in zip(config.counts, deltas):
            state = states[n]
            size = (np.log(n)-pre["size_center"])/pre["size_scale"]
            area = np.exp(geometry.loc[fold].alpha+geometry.loc[fold].beta*np.log(n))/(4*np.pi)
            raw_base = np.asarray(pre["residual_transform"].inverse(state["z_base"].astype(float))).reshape(-1)
            raw_abl = np.asarray(pre["residual_transform"].inverse(state["z_abl"].astype(float))).reshape(-1)
            actual = (raw_abl-raw_base)*area
            uv = projection.displacement(delta)
            points.append(cases[["case_id", "node_id", "organoid_str"]].assign(n=n, response1=uv[:,0], response2=uv[:,1],delta_relative=actual))
            energy = np.sum(delta.astype(float)**2,axis=1)
            for dim in sorted(set(min(d,len(projection.components)) for d in config.dimensions)):
                projected = projection.reconstruct_displacement(delta,dim)
                z = head_prediction(model,state["h_base"]+projected,size)
                raw = np.asarray(pre["residual_transform"].inverse(z)).reshape(-1)
                part=cases[["organoid_str"]].copy()
                part["retained_energy"]=np.divide(np.sum(projected**2,axis=1),energy,out=np.full(len(cases),np.nan),where=energy>1e-16)
                part["squared_error"]=((raw-raw_base)*area-actual)**2
                part["squared_effect"]=actual**2
                part["sign_agreement"]=np.where(np.abs(actual)>1e-8,np.sign(raw-raw_base)==np.sign(actual),np.nan)
                fidelity.append(part.groupby("organoid_str").mean().reset_index().assign(n=n,dimensions=dim))
        fd=pd.concat(fidelity).assign(fold=fold,seed=seed)
        fd.to_csv(dest / "response_projection_fidelity_organoids.csv.gz",index=False)
        pd.concat(points).to_csv(dest / "response_points.csv.gz",index=False)
        all_fidelity.append(fd)
        print(f"Response projection: fold {fold}, seed {seed}",flush=True)
    fd=pd.concat(all_fidelity)
    org=fd.groupby(["organoid_str","n","dimensions"])[["retained_energy","squared_error","squared_effect","sign_agreement"]].mean().reset_index()
    final=org.groupby(["n","dimensions"]).mean(numeric_only=True).reset_index()
    final["relative_rmse"]=np.sqrt(final.squared_error/final.squared_effect)
    final.to_csv(out / "tables/response_projection_fidelity.csv",index=False)
    _write(out / "response_projection_complete.json",dict(checkpoints=len(all_fidelity),centered=False,uses_curvature_targets=False))
