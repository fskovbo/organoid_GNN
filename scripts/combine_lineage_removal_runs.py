#!/usr/bin/env python3
"""Combine result tables from multiple ``lineage_removal.ipynb`` runs.

The output is a synthetic lineage-removal result directory that can be loaded by
``experiments/plot_experiment_results.ipynb`` via ``LINEAGE_RUN_DIR`` or by
auto-discovery if it is the newest complete run.
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd


REQUIRED_TABLES = {
    "mse": "mse_summary_across_organoids.csv",
    "baseline": "global_baseline_summary_across_organoids.csv",
    "organoid_mse": "organoid_level_mse.csv",
    "baseline_organoid_mse": "global_baseline_organoid_mse.csv",
    "zero_counts": "marker_zero_counts.csv",
    "training": "training_summary.csv",
    "crypt_shape": "crypt_neck_shape_assignments.csv",
}

OPTIONAL_TABLES = {
}

# ---------------------------------------------------------------------
# Editable default configuration
# ---------------------------------------------------------------------
#
# Fill these in to run the script without command-line arguments:
#
#   /home/fmoller/miniforge3/bin/conda run -n organoid-gnn \
#       python scripts/combine_lineage_removal_runs.py
#
# Command-line arguments still override these defaults.

RUN_DIRS = [
    Path("results_experiments/lineage_removal/marker_isolation_plus_targeted_removal_folded_depths-0-1-2-3-4_20260708_180507"),
    Path("results_experiments/lineage_removal/marker_panel_isolation_folded_depths-0-1-2-3-4_20260723_174336"),
]

# Leave as None to create a timestamped folder under
# results_experiments/lineage_removal/.
OUTPUT_DIR = None

ANALYSIS_VARIANT = "combined_marker_panel_isolation"
ALLOW_OVERLAP = False
OVERWRITE = False


def read_json(path: Path) -> dict:
    with path.open() as handle:
        return json.load(handle)


def settings_path(run_dir: Path) -> Path:
    for name in ("settings.json", "config.json"):
        path = run_dir / name
        if path.exists():
            return path
    raise FileNotFoundError(f"No settings.json or config.json in {run_dir}")


def read_table(run_dir: Path, filename: str) -> pd.DataFrame:
    path = run_dir / "tables" / filename
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def load_run(run_dir: Path) -> dict:
    run_dir = run_dir.expanduser().resolve()
    settings = read_json(settings_path(run_dir))
    tables = {
        key: read_table(run_dir, filename)
        for key, filename in REQUIRED_TABLES.items()
    }
    optional_tables = {
        key: read_table(run_dir, filename)
        for key, filename in OPTIONAL_TABLES.items()
        if (run_dir / "tables" / filename).exists()
    }
    return {
        "run_dir": run_dir,
        "settings": settings,
        "tables": tables,
        "optional_tables": optional_tables,
    }


def jsonable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [jsonable(v) for v in value]
    return value


def add_source_column(df: pd.DataFrame, run: dict) -> pd.DataFrame:
    out = df.copy()
    out.insert(0, "source_run_dir", str(run["run_dir"]))
    return out


def concat_with_source(runs: list[dict], table_key: str) -> pd.DataFrame:
    return pd.concat(
        [add_source_column(run["tables"][table_key], run) for run in runs],
        ignore_index=True,
        sort=False,
    )


def concat_optional_with_source(runs: list[dict], table_key: str) -> pd.DataFrame | None:
    frames = [
        add_source_column(run["optional_tables"][table_key], run)
        for run in runs
        if table_key in run["optional_tables"]
    ]
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True, sort=False)


def drop_source_duplicates(df: pd.DataFrame, subset: list[str] | None = None) -> pd.DataFrame:
    no_source = df.drop(columns=["source_run_dir"], errors="ignore")
    if subset is None:
        subset = list(no_source.columns)
    else:
        subset = [col for col in subset if col in no_source.columns]
    return no_source.drop_duplicates(subset=subset).reset_index(drop=True)


def merge_removal_sets(runs: list[dict]) -> list[dict]:
    merged = []
    seen = {}
    for run in runs:
        for spec in run["settings"].get("removal_sets", []):
            key = spec.get("key")
            if key is None:
                continue
            spec_norm = jsonable(spec)
            if key in seen:
                if seen[key] != spec_norm:
                    raise ValueError(
                        f"Conflicting removal_set definition for key {key!r}."
                    )
                continue
            seen[key] = spec_norm
            merged.append(spec_norm)
    return merged


def non_original_condition_keys(df: pd.DataFrame) -> set[str]:
    if not {"removal_key", "train_eval_key"}.issubset(df.columns):
        return set()
    mask = df["train_eval_key"].ne("orig_orig")
    return set(df.loc[mask, "removal_key"].dropna().astype(str))


def validate_no_overlapping_conditions(runs: list[dict], *, allow_overlap: bool) -> None:
    if allow_overlap:
        return
    seen = {}
    for run in runs:
        keys = non_original_condition_keys(run["tables"]["mse"])
        for key in keys:
            if key in seen:
                raise ValueError(
                    f"Condition/removal_key {key!r} appears in both "
                    f"{seen[key]} and {run['run_dir']}. Pass --allow-overlap "
                    "if this is intentional."
                )
            seen[key] = run["run_dir"]


def merge_settings(runs: list[dict], output_dir: Path, analysis_variant: str) -> dict:
    base = dict(runs[0]["settings"])
    base["created_at"] = datetime.now().isoformat()
    base["analysis_variant"] = analysis_variant
    base["save_dir"] = str(output_dir)
    base["combined_from_runs"] = [
        {
            "run_dir": str(run["run_dir"]),
            "analysis_variant": run["settings"].get("analysis_variant"),
            "created_at": run["settings"].get("created_at"),
            "n_removal_sets": len(run["settings"].get("removal_sets", [])),
        }
        for run in runs
    ]
    base["removal_sets"] = merge_removal_sets(runs)
    base["combination_note"] = (
        "Synthetic lineage_removal result created by "
        "scripts/combine_lineage_removal_runs.py. Marker-subset result rows "
        "are concatenated across input runs; baseline tables are taken from "
        "the first input run."
    )
    return jsonable(base)


def write_readme(output_dir: Path, runs: list[dict]) -> None:
    lines = [
        "# Combined Lineage Removal Run",
        "",
        "Created by `scripts/combine_lineage_removal_runs.py`.",
        "",
        "Input runs:",
    ]
    for run in runs:
        lines.append(f"- `{run['run_dir']}`")
    lines.extend(
        [
            "",
            "This directory preserves the table names expected by "
            "`experiments/plot_experiment_results.ipynb`.",
            "",
            "Global-baseline tables are taken from the first input run to "
            "avoid double-counting organoids in downstream SEM calculations.",
        ]
    )
    (output_dir / "README.md").write_text("\n".join(lines) + "\n")


def combine_runs(
    run_dirs: list[Path],
    output_dir: Path,
    *,
    overwrite: bool = False,
    allow_overlap: bool = False,
    analysis_variant: str = "combined_marker_panel_isolation",
) -> Path:
    runs = [load_run(path) for path in run_dirs]
    validate_no_overlapping_conditions(runs, allow_overlap=allow_overlap)

    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"{output_dir} already exists. Pass --overwrite to replace it."
            )
        shutil.rmtree(output_dir)
    tables_dir = output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    combined_mse = concat_with_source(runs, "mse")
    combined_organoid = concat_with_source(runs, "organoid_mse")
    combined_zero_counts = concat_with_source(runs, "zero_counts")
    combined_training = concat_with_source(runs, "training")

    # Use one baseline source only. The baseline is independent of marker
    # subset, and concatenating separately trained baselines would double-count
    # organoids in downstream SEM calculations.
    combined_baseline = runs[0]["tables"]["baseline"].copy()
    combined_baseline_organoid = runs[0]["tables"]["baseline_organoid_mse"].copy()

    combined_mse.to_csv(tables_dir / REQUIRED_TABLES["mse"], index=False)
    combined_baseline.to_csv(tables_dir / REQUIRED_TABLES["baseline"], index=False)
    combined_organoid.to_csv(tables_dir / REQUIRED_TABLES["organoid_mse"], index=False)
    combined_baseline_organoid.to_csv(
        tables_dir / REQUIRED_TABLES["baseline_organoid_mse"],
        index=False,
    )
    combined_zero_counts.to_csv(tables_dir / REQUIRED_TABLES["zero_counts"], index=False)
    combined_training.to_csv(tables_dir / REQUIRED_TABLES["training"], index=False)

    crypt_shape = concat_optional_with_source(runs, "crypt_shape")
    if crypt_shape is not None:
        crypt_shape = drop_source_duplicates(
            crypt_shape,
            subset=[
                "fold",
                "organoid_id",
                "node_id",
                "node_original_id",
                "region",
            ],
        )
        crypt_shape.to_csv(tables_dir / OPTIONAL_TABLES["crypt_shape"], index=False)

    settings = merge_settings(runs, output_dir, analysis_variant)
    for name in ("settings.json", "config.json"):
        with (output_dir / name).open("w") as handle:
            json.dump(settings, handle, indent=2)
    write_readme(output_dir, runs)
    return output_dir


def default_output_dir(project_root: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return (
        project_root
        / "results_experiments"
        / "lineage_removal"
        / f"combined_marker_panel_isolation_{timestamp}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Combine two or more lineage_removal.ipynb result directories into "
            "one plot_experiment_results-compatible run."
        )
    )
    parser.add_argument(
        "run_dirs",
        nargs="*",
        type=Path,
        help=(
            "Input lineage_removal result directories. If omitted, RUN_DIRS "
            "from the editable configuration block are used."
        ),
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Output directory. Defaults to a timestamped directory under "
            "results_experiments/lineage_removal/."
        ),
    )
    parser.add_argument(
        "--analysis-variant",
        default=None,
        help=(
            "analysis_variant value written to settings.json. Defaults to "
            "ANALYSIS_VARIANT from the editable configuration block."
        ),
    )
    parser.add_argument(
        "--allow-overlap",
        action="store_true",
        help="Allow the same non-original removal_key to appear in multiple runs.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace output directory if it already exists.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    run_dirs = args.run_dirs or RUN_DIRS
    if not run_dirs:
        raise SystemExit(
            "No input runs specified. Either edit RUN_DIRS at the top of this "
            "script or pass run directories on the command line."
        )
    output_dir = args.output_dir or OUTPUT_DIR or default_output_dir(project_root)
    output_dir = combine_runs(
        [Path(path) for path in run_dirs],
        output_dir,
        overwrite=bool(args.overwrite or OVERWRITE),
        allow_overlap=bool(args.allow_overlap or ALLOW_OVERLAP),
        analysis_variant=args.analysis_variant or ANALYSIS_VARIANT,
    )
    print(f"Wrote combined lineage-removal run to {output_dir}")
    print(
        "Set LINEAGE_RUN_DIR to this path in experiments/plot_experiment_results.ipynb "
        "or leave LINEAGE_RUN_DIR=None if this is the newest complete lineage run."
    )


if __name__ == "__main__":
    main()
