"""A compact, curvature-ordered interpretation of saved exclusive FiLM sweeps.

No GNN inference, training, GMM fitting or t-SNE fitting is repeated. The only
new model evaluations pass saved hidden states through the trained output head.
"""
import itertools
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.analysis.embeddings.size_responses import EmbeddingConfig

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


