#!/usr/bin/env python3
"""Export marker-composition patch tables from organoid graph datasets.

The script writes two CSV files:

1. A raw patch-composition table for all non-excluded graphs.
2. A validation-set patch-composition table with predictions from depth-2 and
   depth-4 GIN models plus cluster labels from the depth-2 final node embedding.

Patch centers are sampled greedily so selected centers are at least
``min_center_distance`` graph hops apart. For each center, marker composition and
curvature summaries are computed over the ``patch_radius``-hop neighborhood.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import sys
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.motif_clustering import run_embedding_clustering
from src.data.filters import (
    filter_graphs_by_blacklist,
    filter_graphs_by_marker_diversity,
    filter_graphs_by_metadata,
    filter_graphs_by_numeric_metadata,
    filter_graphs_by_sphericity,
    load_graph_blacklist_from_dir,
)
from src.data.io import load_graph_dataset_from_dir
from src.data.metadata import (
    add_log_metadata_features,
    attach_metadata_to_graphs,
    fill_missing_metadata_for_group,
    get_graph_metadata,
    get_graph_metadata_value,
    infer_global_dim,
    load_aux_metadata_for_dir,
    load_marker_names_from_dir,
    promote_metadata_to_graph_tensors,
    strip_graph_metadata,
)
from src.data.preprocessing import interpolate_target_outliers_from_neighbors
from src.data.splits import graph_metadata_key, train_val_split_graphs
from src.data.target_transforms import (
    AsinhStandardizeTransform,
    ChainedTargetTransform,
    GlobalBaselineResidualTransform,
    standardize_graph_global_features,
)
from src.inference.predict import predict_targets
from src.models.gnn import GINCurvature
from src.training.loop import TrainConfig, train
from src.training.losses import WeightedLossTerm, edge_loss_term


DEFAULT_GLOBAL_FIELD_SPECS = [
    {
        "meta_keys": [
            "log_surface_area",
            "log_volume",
            "log_volume_over_area",
            "log_num_cells",
        ],
        "attr_name": "global_feat",
        "kind": "graph_vector",
        "dtype": torch.float32,
    },
]


def parse_int_list(text: str) -> list[int]:
    if text is None or str(text).strip() == "":
        return []
    return [int(x.strip()) for x in str(text).split(",") if x.strip()]


def slugify(value: str) -> str:
    slug = re.sub(r"[^0-9A-Za-z]+", "_", str(value)).strip("_")
    return slug or "marker"


def as_numpy(x: Any) -> np.ndarray:
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def target_vector(graph) -> np.ndarray:
    y = as_numpy(graph.y)
    if y.ndim == 2:
        if y.shape[1] != 1:
            raise ValueError(
                "This export script expects a single target. "
                f"Got graph target shape {y.shape}."
            )
        y = y[:, 0]
    return np.asarray(y, dtype=np.float64).reshape(-1)


def build_adjacency(graph) -> list[set[int]]:
    n_nodes = int(graph.x.shape[0])
    adjacency = [set() for _ in range(n_nodes)]
    edge_index = as_numpy(graph.edge_index)
    if edge_index.size == 0:
        return adjacency

    src = edge_index[0].astype(np.int64, copy=False)
    dst = edge_index[1].astype(np.int64, copy=False)
    for u, v in zip(src, dst):
        if 0 <= u < n_nodes and 0 <= v < n_nodes and u != v:
            adjacency[int(u)].add(int(v))
            adjacency[int(v)].add(int(u))
    return adjacency


def nodes_within_hops(adjacency: list[set[int]], center: int, radius: int) -> list[int]:
    seen = {int(center)}
    frontier = {int(center)}
    for _ in range(int(radius)):
        next_frontier: set[int] = set()
        for u in frontier:
            next_frontier.update(adjacency[u])
        next_frontier.difference_update(seen)
        if not next_frontier:
            break
        seen.update(next_frontier)
        frontier = next_frontier
    return sorted(seen)


def sample_centers_dense(
    graph,
    *,
    min_center_distance: int = 3,
) -> list[int]:
    """Greedily sample many centers with pairwise distance >= min_center_distance."""
    n_nodes = int(graph.x.shape[0])
    if n_nodes == 0:
        return []

    adjacency = build_adjacency(graph)
    exclusion_radius = max(0, int(min_center_distance) - 1)
    exclusion_balls = [
        nodes_within_hops(adjacency, i, exclusion_radius)
        for i in range(n_nodes)
    ]

    available = np.ones(n_nodes, dtype=bool)
    centers: list[int] = []

    # Favor nodes with smaller exclusion balls. This deterministic greedy rule
    # produces a maximal dense packing without solving an expensive MIS problem.
    order = sorted(range(n_nodes), key=lambda i: (len(exclusion_balls[i]), i))
    for i in order:
        if not available[i]:
            continue
        centers.append(i)
        available[np.asarray(exclusion_balls[i], dtype=np.int64)] = False
    return centers


def metadata_value(graph, keys: Iterable[str], default=np.nan):
    for key in keys:
        value = get_graph_metadata_value(graph, key, default=None, strict=False)
        if value is not None:
            return value
    return default


def graph_identity(graph) -> dict[str, Any]:
    md = get_graph_metadata(graph, strict=False, default={})
    return {
        "organoid_str": getattr(graph, "organoid_str", None),
        "organoid_id": md.get("organoid_id", getattr(graph, "organoid_str", None)),
        "label_uid": md.get("label_uid", md.get("organoid_id", getattr(graph, "organoid_str", None))),
        "dataset": md.get("dataset", None),
        "timepoint": md.get("timepoint", None),
    }


def original_node_id(graph, node_id: int):
    md = get_graph_metadata(graph, strict=False, default={})
    mapping = md.get("new_to_old_index", None)
    if mapping is None:
        kept_ids = md.get("kept_node_ids", None)
        mapping = kept_ids
    if mapping is None:
        return int(node_id)
    try:
        return int(mapping[int(node_id)])
    except Exception:
        return int(node_id)


def rows_for_graph(
    graph,
    *,
    graph_index: int,
    marker_names: list[str],
    patch_radius: int,
    min_center_distance: int,
    predictions_by_name: dict[str, np.ndarray] | None = None,
    cluster_labels: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    x = np.asarray(as_numpy(graph.x), dtype=np.float64)
    y = target_vector(graph)
    n_nodes = int(x.shape[0])
    if x.shape[1] != len(marker_names):
        raise ValueError(
            f"Graph has {x.shape[1]} marker columns, but marker_names has "
            f"{len(marker_names)} entries."
        )

    adjacency = build_adjacency(graph)
    centers = sample_centers_dense(graph, min_center_distance=min_center_distance)
    identity = graph_identity(graph)
    surface_area = metadata_value(graph, ["total_surface_area", "surface_area"])
    volume = metadata_value(graph, ["total_volume", "volume"])

    pred_map = predictions_by_name or {}
    rows: list[dict[str, Any]] = []
    marker_slugs = [slugify(name) for name in marker_names]

    for center in centers:
        pool = nodes_within_hops(adjacency, center, int(patch_radius))
        pool_arr = np.asarray(pool, dtype=np.int64)
        y_pool = y[pool_arr]
        x_pool = x[pool_arr]

        row = {
            "graph_index": int(graph_index),
            **identity,
            "center_node_id": int(center),
            "center_node_original_id": original_node_id(graph, center),
            "n_cells_pool": int(len(pool)),
            "n_cells_organoid": int(n_nodes),
            "total_surface_area": surface_area,
            "total_volume": volume,
            "center_curvature": float(y[center]),
            "pool_mean_curvature": float(np.mean(y_pool)),
            "pool_std_curvature": float(np.std(y_pool, ddof=1)) if len(y_pool) > 1 else 0.0,
        }

        for j, (marker, slug) in enumerate(zip(marker_names, marker_slugs)):
            positives = x_pool[:, j] > 0.5
            row[f"n_{slug}_positive"] = int(np.sum(positives))
            row[f"frac_{slug}_positive"] = float(np.mean(positives)) if len(positives) else np.nan
            row[f"center_{slug}_positive"] = int(x[center, j] > 0.5)

        for name, pred in pred_map.items():
            row[f"prediction_{name}"] = float(np.asarray(pred).reshape(-1)[center])

        if cluster_labels is not None:
            row["cluster_depth2"] = int(np.asarray(cluster_labels).reshape(-1)[center])

        rows.append(row)

    return rows


def export_patch_table(
    graphs: list,
    marker_names: list[str],
    *,
    patch_radius: int,
    min_center_distance: int,
    predictions_by_graph: dict[str, list[np.ndarray]] | None = None,
    clusters_by_graph: list[np.ndarray] | None = None,
) -> pd.DataFrame:
    all_rows: list[dict[str, Any]] = []
    for gi, graph in enumerate(graphs):
        pred_for_graph = None
        if predictions_by_graph is not None:
            pred_for_graph = {
                name: values[gi]
                for name, values in predictions_by_graph.items()
            }
        cluster_for_graph = None
        if clusters_by_graph is not None:
            cluster_for_graph = clusters_by_graph[gi]

        all_rows.extend(
            rows_for_graph(
                graph,
                graph_index=gi,
                marker_names=marker_names,
                patch_radius=patch_radius,
                min_center_distance=min_center_distance,
                predictions_by_name=pred_for_graph,
                cluster_labels=cluster_for_graph,
            )
        )
    return pd.DataFrame(all_rows)


def split_flat_by_graph(values: np.ndarray, graphs: list) -> list[np.ndarray]:
    values = np.asarray(values)
    out = []
    offset = 0
    for graph in graphs:
        n = int(graph.x.shape[0])
        out.append(values[offset:offset + n].copy())
        offset += n
    if offset != len(values):
        raise ValueError(f"Flat array length mismatch: consumed {offset}, got {len(values)}.")
    return out


def load_graphs_with_metadata(data_dir: Path, target_indices: list[int]) -> tuple[list, list[str]]:
    marker_names = load_marker_names_from_dir(str(data_dir))
    if marker_names is None:
        raise FileNotFoundError(f"No marker-name JSON sidecar found in {data_dir}.")

    graphs = load_graph_dataset_from_dir(
        str(data_dir),
        strict=False,
        target_indices=target_indices or None,
    )
    meta_by_stem = load_aux_metadata_for_dir(str(data_dir))
    n_attached = attach_metadata_to_graphs(graphs, meta_by_stem)
    print(f"Loaded {len(graphs)} graphs; attached metadata to {n_attached}.")
    return graphs, list(marker_names)


def apply_model_training_filters(graphs: list, data_dir: Path, args) -> list:
    out = [copy.copy(g) for g in graphs]

    if args.filter_blacklist:
        blacklist = load_graph_blacklist_from_dir(data_dir)
        out = filter_graphs_by_blacklist(out, blacklist, print_summary=True)

    if args.apply_quality_filters:
        out = fill_missing_metadata_for_group(
            out,
            field="complexity",
            fill_value=args.missing_complexity_fill_value,
            dataset=args.missing_complexity_dataset,
            timepoint=args.missing_complexity_timepoint,
        )
        out, g_spherical = filter_graphs_by_sphericity(
            out,
            max_sphericity=args.sphericity_max,
            print_summary=True,
            return_rejected=True,
        )
        _ = filter_graphs_by_marker_diversity(
            g_spherical,
            min_score=args.spherical_marker_diversity_min,
            print_summary=True,
        )
        out = filter_graphs_by_numeric_metadata(
            out,
            key="complexity",
            min_value=args.complexity_min,
            allow_missing=False,
            print_summary=True,
        )

    if args.interpolate_target_outliers:
        out, info = interpolate_target_outliers_from_neighbors(
            out,
            target_indices=None,
            clip_quantiles=(args.outlier_clip_low, args.outlier_clip_high),
        )
        print(f"Interpolated target outliers in {len(info)} graph entries.")

    return out


def attach_global_features(graphs: list) -> list:
    out = add_log_metadata_features(graphs, inplace=False)
    out = promote_metadata_to_graph_tensors(out, DEFAULT_GLOBAL_FIELD_SPECS, inplace=False)
    return out


def prepare_training_data(graphs: list, args):
    if args.use_global_features:
        graphs = attach_global_features(graphs)

    g_train_raw, g_val_raw, split_info = train_val_split_graphs(
        graphs,
        val_frac=args.val_frac,
        seed=args.split_seed,
        force_val_keys=None,
        key_fn=graph_metadata_key,
    )
    print(f"Split -> train: {len(g_train_raw)} | val: {len(g_val_raw)}")

    g_train = strip_graph_metadata([copy.deepcopy(g) for g in g_train_raw], inplace=False)
    g_val = strip_graph_metadata([copy.deepcopy(g) for g in g_val_raw], inplace=False)

    if args.use_global_features:
        standardize_graph_global_features(
            g_train,
            g_val,
            attr_name="global_feat",
            robust=False,
        )

    if args.subtract_global_baseline:
        target_transform = ChainedTargetTransform([
            GlobalBaselineResidualTransform(
                hidden_dim=args.baseline_hidden_dim,
                dropout=args.dropout,
                lr=args.lr,
                batch_size=args.batch_size,
                max_epochs=args.max_epochs,
                patience=args.patience,
                num_workers=args.num_workers,
            ),
            AsinhStandardizeTransform(robust=True),
        ])
        print("Training target: residual after global-feature baseline, then asinh-standardized.")
    else:
        target_transform = AsinhStandardizeTransform(robust=True)
        print("Training target: asinh-standardized raw target.")

    target_transform.fit(g_train)
    target_transform.transform_graphs(g_train)
    target_transform.transform_graphs(g_val)
    return g_train, g_val, g_val_raw, split_info, target_transform


def make_gin_model(graphs: list, depth: int, args) -> GINCurvature:
    return GINCurvature(
        n_markers=int(graphs[0].x.size(1)),
        global_dim=infer_global_dim(graphs),
        hidden_dim=args.hidden_dim,
        num_layers=int(depth),
        dropout=args.dropout,
        residual=args.residual,
        norm=args.norm,
    )


def train_models(g_train: list, g_val: list, args) -> dict[int, GINCurvature]:
    aux_losses = []
    if args.edge_loss_weight > 0:
        aux_losses.append(
            WeightedLossTerm(
                name="edge",
                fn=edge_loss_term,
                weight=args.edge_loss_weight,
                params={
                    "weighted": args.edge_loss_weighted,
                    "alpha": args.edge_loss_alpha,
                    "normalize_by": args.edge_loss_normalize_by,
                    "clip_weight": args.edge_loss_clip_weight,
                },
            )
        )

    cfg = TrainConfig(
        lr=args.lr,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        patience=args.patience,
        num_workers=args.num_workers,
        aux_losses=aux_losses,
    )

    trained: dict[int, GINCurvature] = {}
    for depth in args.depths:
        print(f"\n=== Training GIN depth={depth} ===")
        model = make_gin_model(g_train, depth, args)
        model, metrics, _history = train(model, g_train, g_val, cfg)
        print(f"Depth {depth} best val MAE: {metrics['val_mae']:.6g}")
        trained[int(depth)] = model
    return trained


def predict_models_by_graph(
    models: dict[int, GINCurvature],
    g_val: list,
    g_val_raw: list,
    target_transform,
    args,
) -> dict[str, list[np.ndarray]]:
    predictions: dict[str, list[np.ndarray]] = {}
    for depth, model in sorted(models.items()):
        y_true, y_pred, _log_var, _x = predict_targets(
            g_val,
            model,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            return_log_var=True,
            target_transform=target_transform,
        )
        y_pred = np.asarray(y_pred)
        if y_pred.ndim == 2 and y_pred.shape[1] == 1:
            y_pred = y_pred[:, 0]
        predictions[f"depth_{depth}"] = split_flat_by_graph(y_pred, g_val_raw)
        if depth == args.depths[0]:
            y_true = np.asarray(y_true)
            if y_true.ndim == 2 and y_true.shape[1] == 1:
                y_true = y_true[:, 0]
            raw_y = np.concatenate([target_vector(g) for g in g_val_raw])
            max_abs = float(np.max(np.abs(raw_y - y_true))) if len(raw_y) else 0.0
            print(f"Validation target inverse-transform check max abs diff: {max_abs:.3e}")
    return predictions


def cluster_depth2_embeddings(
    model: GINCurvature,
    g_val: list,
    g_val_raw: list,
    marker_names: list[str],
    args,
) -> list[np.ndarray]:
    extraction, clustering_result, _summary = run_embedding_clustering(
        graphs=g_val,
        model=model,
        batch_size=args.batch_size,
        center_only=False,
        embedding_variant=args.embedding_variant,
        residualize_global=args.residualize_global,
        clustering=args.clustering,
        n_clusters=args.n_clusters,
        covariance_type=args.covariance_type,
        standardize=args.cluster_standardize,
        pca_dim=args.pca_dim,
        seed=args.cluster_seed,
        marker_names=marker_names,
    )
    if len(clustering_result.labels) != len(extraction.local_node_index):
        raise RuntimeError(
            "Clustering labels do not match extracted node embeddings: "
            f"{len(clustering_result.labels)} labels for {len(extraction.local_node_index)} nodes."
        )
    return split_flat_by_graph(clustering_result.labels.astype(np.int64), g_val_raw)


def json_safe(value: Any):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, set):
        return sorted(json_safe(v) for v in value)
    if isinstance(value, tuple):
        return [json_safe(v) for v in value]
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    return value


def write_export_readme(
    output_dir: Path,
    *,
    raw_csv: Path,
    validation_csv: Path | None,
    marker_names: list[str],
    args,
) -> Path:
    marker_cols = ", ".join(marker_names)
    excluded = args.exclude_timepoint if args.exclude_timepoint else "none"
    model_table_text = (
        f"- `{validation_csv.name}`: validation-set patch table with model predictions "
        "and depth-2 embedding cluster labels.\n"
        if validation_csv is not None
        else "- Model/prediction table was skipped for this run.\n"
    )

    readme = f"""# Patch Composition Table Export

This folder contains CSV tables exported by `scripts/export_patch_composition_tables.py`.

## Files

- `{raw_csv.name}`: raw patch-composition table for all graphs after the timepoint exclusion.
{model_table_text}- `export_config.json`: command settings, marker names, and split information for this run.

## Sampling

Each row corresponds to one sampled center node in one organoid graph.

- Excluded timepoint: `{excluded}`
- Center sampling: greedy dense sampling with centers at least `{args.min_center_distance}` graph hops apart.
- Patch/pool radius: all cells within `{args.patch_radius}` graph hops of the center node, including the center.
- The greedy sampling creates a maximal dense packing of centers. It is deterministic, but not guaranteed to be the mathematically maximum possible packing.

## Marker Columns

Markers in this export:

`{marker_cols}`

For each marker `<marker>`, the tables include:

- `n_<marker>_positive`: number of marker-positive cells in the patch.
- `frac_<marker>_positive`: fraction of cells in the patch that are marker-positive.
- `center_<marker>_positive`: whether the center node is marker-positive.

Marker names are sanitized for column names, for example spaces and punctuation are replaced by underscores.

## Shared Columns

Both tables contain:

- `graph_index`: index of the graph within the graph list used for that table.
- `organoid_str`, `organoid_id`, `label_uid`, `dataset`, `timepoint`: organoid identifiers and metadata.
- `center_node_id`: node index in the loaded graph.
- `center_node_original_id`: original node index when the metadata contains `new_to_old_index` or `kept_node_ids`; otherwise equal to `center_node_id`.
- `n_cells_pool`: number of cells in the `{args.patch_radius}`-hop patch.
- `n_cells_organoid`: number of cells in the full organoid graph.
- `total_surface_area`, `total_volume`: organoid-level metadata.
- `center_curvature`: target curvature at the center node.
- `pool_mean_curvature`: mean target curvature over the patch.
- `pool_std_curvature`: sample standard deviation of target curvature over the patch.

## Validation Prediction Table

When the model table is generated, it additionally contains:

- `prediction_depth_2`: prediction at the center node from the trained depth-2 GIN.
- `prediction_depth_4`: prediction at the center node from the trained depth-4 GIN.
- `cluster_depth2`: cluster label assigned from `run_embedding_clustering()` on the final depth-2 node embedding.

Predictions are inverse-transformed back to original curvature units before export.

## Training and Clustering Defaults

The model table uses settings patterned after `experiments/scan_model_depth_motifs.ipynb`:

- GIN depths: `{args.depths}`
- Hidden dimension: `{args.hidden_dim}`
- Global features enabled: `{args.use_global_features}`
- Global baseline residual target transform: `{args.subtract_global_baseline}`
- Quality filters enabled: `{args.apply_quality_filters}`
- Blacklist filtering enabled: `{args.filter_blacklist}`
- Clustering method: `{args.clustering}`
- Number of clusters: `{args.n_clusters}`
- Embedding variant: `{args.embedding_variant}`
- PCA dimension: `{args.pca_dim}`
- Cluster standardization: `{args.cluster_standardize}`

Exact run settings are stored in `export_config.json`.
"""

    readme_path = output_dir / "README.md"
    readme_path.write_text(readme)
    return readme_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export organoid patch marker-composition CSV tables.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--dataset-name", default="mean_curvature_smooth")
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "training_data")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--target-indices", default="0", help="Comma-separated target columns.")
    parser.add_argument("--exclude-timepoint", default="day4", help="Exact timepoint to drop; set empty to keep all.")

    parser.add_argument("--patch-radius", type=int, default=2)
    parser.add_argument("--min-center-distance", type=int, default=3)

    parser.add_argument("--filter-blacklist", action="store_true")
    parser.add_argument("--apply-quality-filters", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--missing-complexity-dataset", default="20251201")
    parser.add_argument("--missing-complexity-timepoint", default="day4p5")
    parser.add_argument("--missing-complexity-fill-value", type=float, default=2.1)
    parser.add_argument("--sphericity-max", type=float, default=0.92)
    parser.add_argument("--spherical-marker-diversity-min", type=float, default=0.5)
    parser.add_argument("--complexity-min", type=float, default=2.0)
    parser.add_argument("--interpolate-target-outliers", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--outlier-clip-low", type=float, default=0.005)
    parser.add_argument("--outlier-clip-high", type=float, default=0.995)

    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--depths", type=parse_int_list, default=[2, 4])
    parser.add_argument("--hidden-dim", type=int, default=4 * 64)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--norm", default="batch")
    parser.add_argument("--residual", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-global-features", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--subtract-global-baseline", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--baseline-hidden-dim", type=int, default=64)

    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-epochs", type=int, default=2000)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--edge-loss-weight", type=float, default=0.20)
    parser.add_argument("--edge-loss-weighted", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--edge-loss-alpha", type=float, default=2.0)
    parser.add_argument("--edge-loss-normalize-by", default="graph_std")
    parser.add_argument("--edge-loss-clip-weight", type=float, default=4.0)

    parser.add_argument("--clustering", choices=["gmm", "kmeans"], default="gmm")
    parser.add_argument("--n-clusters", type=int, default=9)
    parser.add_argument("--embedding-variant", choices=["full", "local"], default="local")
    parser.add_argument("--residualize-global", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--covariance-type", default="full")
    parser.add_argument("--cluster-standardize", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pca-dim", type=int, default=32)
    parser.add_argument("--cluster-seed", type=int, default=0)
    parser.add_argument(
        "--skip-model-table",
        action="store_true",
        help="Only export the raw patch-composition table.",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    args.depths = list(args.depths)
    if 2 not in args.depths:
        raise ValueError("Depth 2 must be included because clusters are built from the depth-2 model.")
    if args.subtract_global_baseline and not args.use_global_features:
        raise ValueError("--subtract-global-baseline requires --use-global-features.")

    target_indices = parse_int_list(args.target_indices)
    data_dir = args.data_root / args.dataset_name
    if not data_dir.exists():
        raise FileNotFoundError(data_dir)

    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = (
            PROJECT_ROOT
            / "results_experiments"
            / "patch_composition_export"
            / f"run_{timestamp}"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    graphs, marker_names = load_graphs_with_metadata(data_dir, target_indices)

    if args.exclude_timepoint:
        graphs_for_export = filter_graphs_by_metadata(
            graphs,
            key="timepoint",
            drop_values={args.exclude_timepoint},
            missing="keep",
            print_summary=True,
        )
    else:
        graphs_for_export = [copy.copy(g) for g in graphs]

    print("\n=== Exporting raw patch-composition table ===")
    raw_table = export_patch_table(
        graphs_for_export,
        marker_names,
        patch_radius=args.patch_radius,
        min_center_distance=args.min_center_distance,
    )
    if args.exclude_timepoint:
        raw_name = f"patch_composition_all_graphs_except_{slugify(args.exclude_timepoint)}.csv"
    else:
        raw_name = "patch_composition_all_graphs.csv"
    raw_csv = args.output_dir / raw_name
    raw_table.to_csv(raw_csv, index=False)
    print(f"Wrote {len(raw_table):,} rows -> {raw_csv}")

    if args.skip_model_table:
        readme_path = write_export_readme(
            args.output_dir,
            raw_csv=raw_csv,
            validation_csv=None,
            marker_names=marker_names,
            args=args,
        )
        config = {
            "args": vars(args),
            "target_indices": target_indices,
            "marker_names": marker_names,
            "raw_csv": raw_csv,
            "validation_csv": None,
            "readme": readme_path,
        }
        config_path = args.output_dir / "export_config.json"
        config_path.write_text(json.dumps(json_safe(config), indent=2))
        print(f"Wrote README -> {readme_path}")
        print(f"Wrote config -> {config_path}")
        return 0

    print("\n=== Preparing model training data ===")
    graphs_model = apply_model_training_filters(graphs_for_export, data_dir, args)
    g_train, g_val, g_val_raw, split_info, target_transform = prepare_training_data(
        graphs_model,
        args,
    )

    print("\n=== Training models ===")
    trained_models = train_models(g_train, g_val, args)

    print("\n=== Predicting validation nodes ===")
    predictions_by_graph = predict_models_by_graph(
        trained_models,
        g_val,
        g_val_raw,
        target_transform,
        args,
    )

    print("\n=== Clustering depth-2 final node embeddings ===")
    clusters_by_graph = cluster_depth2_embeddings(
        trained_models[2],
        g_val,
        g_val_raw,
        marker_names,
        args,
    )

    print("\n=== Exporting validation prediction/cluster patch table ===")
    val_table = export_patch_table(
        g_val_raw,
        marker_names,
        patch_radius=args.patch_radius,
        min_center_distance=args.min_center_distance,
        predictions_by_graph=predictions_by_graph,
        clusters_by_graph=clusters_by_graph,
    )
    val_csv = args.output_dir / "patch_composition_validation_predictions_clusters.csv"
    val_table.to_csv(val_csv, index=False)
    print(f"Wrote {len(val_table):,} rows -> {val_csv}")

    readme_path = write_export_readme(
        args.output_dir,
        raw_csv=raw_csv,
        validation_csv=val_csv,
        marker_names=marker_names,
        args=args,
    )
    config = {
        "args": vars(args),
        "target_indices": target_indices,
        "marker_names": marker_names,
        "split_info": split_info,
        "raw_csv": raw_csv,
        "validation_csv": val_csv,
        "readme": readme_path,
    }
    config_path = args.output_dir / "export_config.json"
    config_path.write_text(json.dumps(json_safe(config), indent=2))
    print(f"Wrote README -> {readme_path}")
    print(f"Wrote config -> {config_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
