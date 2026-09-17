"""A compact, curvature-ordered interpretation of saved exclusive FiLM sweeps.

No GNN inference, training, GMM fitting or t-SNE fitting is repeated. The only
new model evaluations pass saved hidden states through the trained output head.
"""
import hashlib
import itertools
import json
import math
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.analysis.exclusive_size_ablation import make_model
from src.analysis.size_embedding import EmbeddingConfig, paired_bootstrap_curves

FACTORS = ("ablation_response", "intact_embedding", "head_size", "normalization")


def load_config(source):
    values = json.loads((Path(source) / "settings.json").read_text())["config"]
    return EmbeddingConfig(**{k: tuple(v) if isinstance(v, list) else v for k, v in values.items()})


def curvature_order(labels, predictions, n_clusters):
    """Ascending median predicted residual curvature; empty reference groups last.

    IDs break ties deterministically. Empty groups have no reference curvature;
    their placement at the end is bookkeeping, not a high-curvature claim.
    """
    labels, predictions = np.asarray(labels), np.asarray(predictions)
    rows = []
    for old in range(n_clusters):
        values = predictions[labels == old]
        if not np.isfinite(values).all():
            raise ValueError("Cluster-order predictions must be finite")
        rows.append(dict(raw_cluster=old, median_predicted_residual=float(np.median(values)) if len(values) else np.nan,
                         reference_cells=len(values), empty_at_reference=not len(values)))
    table = pd.DataFrame(rows).sort_values(["empty_at_reference", "median_predicted_residual", "raw_cluster"],
                                          na_position="last").reset_index(drop=True)
    table["cluster"] = np.arange(n_clusters)
    return table


def remap_table(frame, order):
    """Remap every label and responsibility column without changing membership."""
    frame = frame.copy()
    mapping = order.set_index("raw_cluster").cluster.to_dict()
    for column in ("cluster", "reference_cluster", "base_cluster", "ablated_cluster", "source", "target"):
        if column in frame:
            mapped = frame[column].map(mapping)
            if mapped.isna().any():
                raise ValueError(f"Unknown cluster in {column}")
            frame[column] = mapped.astype(int)
    probability_columns = [f"prob_C{k}" for k in range(len(order))]
    if set(probability_columns) <= set(frame):
        original = frame[probability_columns].copy()
        for row in order.itertuples():
            frame[f"prob_C{row.cluster}"] = original[f"prob_C{row.raw_cluster}"]
    return frame


def shapley_changes(values):
    """Exact symmetric endpoint decomposition for a complete binary factor cube.

    Axis 0 encodes masks of factors changed from endpoint A to endpoint B.
    The returned rows sum to values[all changed] - values[none changed].
    """
    values = np.asarray(values, dtype=float)
    n = int(round(np.log2(len(values))))
    if 2 ** n != len(values) or n == 0:
        raise ValueError("Provide all 2**n states for at least one factor")
    contributions = np.zeros((n, *values.shape[1:]), dtype=float)
    for i in range(n):
        for mask in range(2 ** n):
            if mask & (1 << i):
                continue
            k = mask.bit_count()
            weight = math.factorial(k) * math.factorial(n - k - 1) / math.factorial(n)
            contributions[i] += weight * (values[mask | (1 << i)] - values[mask])
    np.testing.assert_allclose(contributions.sum(axis=0), values[-1] - values[0], atol=1e-10, rtol=1e-9)
    return contributions


@torch.no_grad()
def head_prediction(model, hidden, standardized_size, batch_size=4096):
    model.cpu().eval()
    values = []
    for start in range(0, len(hidden), batch_size):
        h = torch.as_tensor(hidden[start:start + batch_size], dtype=torch.float32)
        inputs = torch.cat([h, h.new_full((len(h), 1), float(standardized_size))], dim=1)
        values.append(model.head(inputs)[:, 0].numpy())
    return np.concatenate(values).astype(float)


def decompose_interval(model, states_a, states_b, cases, pre, geometric_reference, n_a, n_b):
    """Separate changes of Δh, intact h, explicit head N, and area normalization.

    E(h,d,t,a) = a * [T^-1(head(h+d,t)) - T^-1(head(h,t))].
    Hybrid states are internal diagnostics, not plausible biological tissues.
    Averaging all factor orders distributes their nonlinear interactions
    symmetrically. Components are signed contributions, not causal percentages.
    """
    base, edit = cases.base_request.to_numpy(), cases.edit_request.to_numpy()
    intact = [s["h"][base] for s in (states_a, states_b)]
    delta = [s["h"][edit] - s["h"][base] for s in (states_a, states_b)]
    size = [(np.log(n) - pre["size_center"]) / pre["size_scale"] for n in (n_a, n_b)]
    area = [np.exp(geometric_reference.alpha + geometric_reference.beta * np.log(n)) / (4 * np.pi) for n in (n_a, n_b)]
    cubes = {metric: np.empty((16, len(cases))) for metric in ("delta_z", "delta_raw", "delta_relative")}
    for response_bit, intact_bit, size_bit in itertools.product((0, 1), repeat=3):
        h = intact[intact_bit]
        z = head_prediction(model, np.concatenate([h, h + delta[response_bit]]), size[size_bit])
        raw = np.asarray(pre["residual_transform"].inverse(z)).reshape(-1)
        dz, dk = z[len(cases):] - z[:len(cases)], raw[len(cases):] - raw[:len(cases)]
        for norm_bit in (0, 1):
            mask = response_bit + 2 * intact_bit + 4 * size_bit + 8 * norm_bit
            cubes["delta_z"][mask] = dz
            cubes["delta_raw"][mask] = dk
            cubes["delta_relative"][mask] = dk * area[norm_bit]
    # Check reconstructed endpoints against the saved GNN's actual predictions.
    max_error = 0.
    for mask, states in [(0, states_a), (15, states_b)]:
        saved = states["z"][edit] - states["z"][base]
        np.testing.assert_allclose(cubes["delta_z"][mask], saved, atol=3e-6, rtol=5e-5)
        max_error = max(max_error, float(np.max(np.abs(cubes["delta_z"][mask] - saved))))
    return {metric: shapley_changes(cube) for metric, cube in cubes.items()}, max_error


def prepare_focus(source_dir, *, force=False):
    """Generate a separate consistent set of ordered tables; preserve old artifacts."""
    source = Path(source_dir).resolve()
    config = load_config(source)
    settings = json.loads((source / "settings.json").read_text())
    run = Path(settings["run_dir"])
    out = source / "focused_analysis"
    out.mkdir(exist_ok=True)
    signature = dict(version=1, source_settings=hashlib.sha256((source / "settings.json").read_bytes()).hexdigest(),
        code=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), reference_n=config.reference_n,
        ordering="Median predicted baseline-subtracted physical curvature at reference N; empty clusters last")
    if (out / "complete.json").exists() and not force:
        if json.loads((out / "complete.json").read_text())["signature"] != signature:
            raise ValueError("Focused analysis changed; call with force=True to rebuild only these derived results")
        print("Using cached focused analysis", flush=True)
        return out
    model_settings = json.loads((run / "settings.json").read_text())
    references = pd.read_csv(run / "geometric_normalization/references.csv").set_index("fold")
    intervals = sorted(set((*config.anchors, config.reference_n)))
    orders, component_tables, validation = [], [], []
    for fold in config.folds:
        folder = source / f"fold_{fold}"
        cases = pd.read_csv(folder / "cases.csv")
        nodes = pd.read_csv(folder / "nodes.csv")
        with (run / f"checkpoints/fold_{fold}_preprocessing.pkl").open("rb") as handle:
            pre = pickle.load(handle)
        for seed in config.seeds:
            print(f"Curvature ordering and cached-head diagnostics: fold {fold}, seed {seed}", flush=True)
            original, dest = folder / f"seed_{seed}", out / f"fold_{fold}/seed_{seed}"
            if not (original / "clustering_done.json").exists():
                raise FileNotFoundError(f"Complete the embedding analysis first: {original}")
            dest.mkdir(parents=True, exist_ok=True)
            states = {}
            for n in intervals:
                with np.load(original / f"states_N{n}.npz") as archive:
                    states[n] = {key: archive[key] for key in archive.files}
            predictions = np.asarray(pre["residual_transform"].inverse(states[config.reference_n]["z"].astype(float))).reshape(-1)
            for population in ("all", "lgr5"):
                old, new = original / population, dest / population
                new.mkdir(exist_ok=True)
                assigned = pd.read_csv(old / "assignments.csv.gz")
                ref = assigned[assigned.state == f"N{config.reference_n}"]
                order = curvature_order(ref.cluster.to_numpy(), predictions[ref.node_id], config.n_clusters)
                order.to_csv(new / "curvature_order.csv", index=False)
                orders.append(order.assign(fold=fold, seed=seed, population=population))
                for name in ("assignments.csv.gz", "profiles.csv", "transitions.csv", "ablation_membership.csv.gz", "tsne.csv.gz"):
                    if (old / name).exists():
                        remap_table(pd.read_csv(old / name), order).to_csv(new / name, index=False)
            model = make_model(model_settings, settings["markers"])
            model.load_state_dict(torch.load(run / f"checkpoints/fold_{fold}_seed_{seed}_gin_film_size.pt", map_location="cpu", weights_only=True))
            for a, b in zip(intervals[:-1], intervals[1:]):
                parts, error = decompose_interval(model, states[a], states[b], cases, pre, references.loc[fold], a, b)
                validation.append(dict(fold=fold, seed=seed, from_n=a, to_n=b, max_saved_effect_error=error))
                for metric, contributions in parts.items():
                    for i, factor in enumerate(FACTORS):
                        frame = cases[["organoid_str", "marker", "hop"]].copy()
                        frame["contribution"] = contributions[i]
                        organ = frame.groupby(["organoid_str", "marker", "hop"]).contribution.mean().reset_index()
                        component_tables.append(organ.assign(fold=fold, seed=seed, from_n=a, to_n=b, metric=metric, factor=factor))
    tables = out / "tables"
    tables.mkdir(exist_ok=True)
    ordering = pd.concat(orders, ignore_index=True)
    ordering.to_csv(tables / "curvature_order.csv", index=False)
    # Keep the old raw IDs in the mapping, not mixed into relabelled plot tables.
    fixed = pd.read_csv(source / "tables/fixed_cluster_curves.csv")
    mapped = fixed.merge(ordering[ordering.population == "lgr5"][["fold", "seed", "raw_cluster", "cluster"]],
        left_on=["fold", "seed", "reference_cluster"], right_on=["fold", "seed", "raw_cluster"], validate="many_to_one")
    mapped["reference_cluster"] = mapped.cluster
    mapped.drop(columns=["raw_cluster", "cluster"]).to_csv(tables / "fixed_cluster_curves.csv", index=False)
    components = pd.concat(component_tables, ignore_index=True)
    components.to_csv(tables / "mechanism_organoids.csv.gz", index=False)
    # These are endpoint changes. For the bootstrap helper, n denotes the interval end;
    # from_n remains a group key. Seeds are averaged within organoid before resampling.
    summary = paired_bootstrap_curves(components.rename(columns={"to_n": "n", "metric": "scale"}),
        ["from_n", "marker", "hop", "scale", "factor"], ["contribution"],
        draws=config.bootstrap_draws, seed=config.sampling_seed, min_organoids=config.min_organoids)
    summary.to_csv(tables / "mechanism_summary.csv", index=False)
    pd.DataFrame(validation).to_csv(tables / "validation.csv", index=False)
    (out / "complete.json").write_text(json.dumps(dict(signature=signature, checkpoints=len(config.folds) * len(config.seeds),
        intervals=intervals, max_saved_effect_error=max(v["max_saved_effect_error"] for v in validation)), indent=2))
    return out
