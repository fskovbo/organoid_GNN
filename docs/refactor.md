# Notebook and artifact organization

The active layout has 22 notebooks: four training workflows and eighteen analysis/inspection
notebooks. There is no `experiments/analysis/` nesting and no archive of duplicate notebooks.
The [notebook index](../experiments/README.md) is the entry point. Historical reports and
checkpoints remain in their original ignored results directories.

## Training contract

The GIN/FiLM, graph-control and lineage workflows expose settings and training loops
in the notebooks. Shared fold preparation lives in `src/training/preparation.py`:

1. Select one cohort with the shared `timepoints` setting (None means all), then split it into disjoint organoid partitions.
2. Apply ordered marker exclusivity when requested, through `src/data/marker_exclusivity.py`.
3. Fit global standardization and a global-only baseline on training organoids only, for every run.
4. Optionally subtract the baseline for residual targets. Keep baseline predictions separate from
   reconstruction offsets (which are zero without residualization).
5. Fit the target asinh transform on training cells and apply it to held-out graphs.
6. Train/evaluate the visible model grid and save each completed checkpoint immediately.

The baseline uses training-only stopping as in the established implementation; ordinary
models select epochs using the outer validation partition. These are validation results,
not a claim of nested preprocessing or untouched test performance. Optional target-outlier
interpolation retains the prior cohort-level behavior and is recorded in settings.

Every new training run lives in `training_results/<notebook_name>/<tag>_<timestamp>/`;
`SETTINGS['tag']` is optional. The main depth-scan notebook fits intact all-marker GIN/FiLM;
graph permutations and ring controls have their own notebook, as do marker combinations.
The separate FiLM training notebook is retired. A new standard run contains:

- `settings.json`, `splits.json`, `provenance.json`, `exclusivity_rules.json`, `cohort.csv`;
- `cohort/`: exact raw cohort/metadata and preprocessing audit;
- `inputs/`: prepared graphs, fitted transforms, global statistics and offsets per fold/input variant;
- `models/`: weights, constructor specifications, optimization history and model record;
- `models.json`, `validation_mse.csv`, and a final `complete.json`.

The baseline model is always included in each fold input bundle. `baseline_validation_mse.csv`
records its physical-curvature MSE per held-out organoid; `validation_mse.csv` includes the
paired `baseline_mse` and `mse_minus_baseline` columns for every trained model. Training
plots and model-comparison analyses display these baselines. Masking copies its reference
baseline (fitting one if an older reference lacks it), and saves the same MSE comparison.

Bundles use checksummed CPU tensor/state dictionaries and trusted-local pickles for PyG
objects and fitted transformations. Run directories are immutable for training purposes;
analyses write into their own subdirectories. Do not load untrusted pickle artifacts.

FiLM shares the main GIN bundle format and cohort snapshots. The size-reference adapter
exports a selected FiLM depth/width grid under its run's `analysis_inputs/` directory for
existing replacement/masking consumers. It preserves exact prepared graphs, fitted target
transforms, weights and baseline offsets; training-only area–N calibration is derived for
geometric normalization. Historical masking runs with copied references remain readable;
older runs without cohort snapshots still require their original dataset files.
Historical run folders are not moved.

Masking training is now standalone: its own settings define the dataset, shared timepoints,
filters, exclusive fate encoding, outer/inner splits, residualization and optimizer.
`model_types=['gin']`, `['film']`, or both select the architecture; `depths`, `hidden_dims`,
`seeds`, folds and masking rates define the grid. No settings or preprocessing are copied
from another training notebook or run. The default remains FiLM at depth 2, explicitly
configurable rather than hard-coded.

Each fold always fits/saves a global baseline, even without residualization. Baseline and
preprocessing fitting use outer-training organoids, including the inner early-stopping
group; this preserves the earlier masking protocol rather than claiming fully nested
preprocessing. New masking runs use the same `models.json`, `models/`, `inputs/`, `cohort/`
and `splits.json` bundle format as other training notebooks, plus `inner_splits.json`.
Saved analysis inputs carry the observed missingness flag. The model catalog records
family, depth, width, fold, seed and masking rate. Initialization preserves unmasked
predictions with zero flag weights; narrow-width residual projections are recorded in
checkpoint specifications so restoration preserves the actual trained architecture.

The masking quality benchmark explicitly selects one family/depth/width, comparing rates
and seeds within it. Its reports live under `analysis/masking_quality/<family>_d<depth>_h<width>`.
Modern masking models also load directly in the general ablation and embedding notebooks.
Depth 0 has no neighborhood interactions. `RUN_TRAINING` guards fitting, not independent
analysis notebooks; all new run settings should be reviewed before enabling it.

## Model-independent restoration

```python
from src.artifacts.runs import AnalysisRun
run = AnalysisRun('/absolute/path/to/run')
print(run.records)
selected = run.select(run.records.iloc[0].key, device='cpu')
model = selected['model']
validation = selected['groups']['val']
```

Notebook run selection uses `resolve_training_run`: an explicit name under the chosen
training workflow, or an explicit historical path. It never silently picks the newest of
several runs. The catalog selects one depth/fold/seed checkpoint; all-marker GIN/FiLM
training saves one shared input bundle per fold, independent of depth, width or architecture.

The same interface reads standard GIN/ring/subset bundles, historical size-conditioned GIN
and FiLM runs, and masked FiLM runs. It restores feature order, marker encoding, globals,
target transforms and split membership without fitting. `prediction_table` reconstructs
physical curvature using saved offsets and reports model uncertainty in transformed units.

Final local embeddings are extracted with `extract_node_embeddings`; appended global head
columns are removed. FiLM modulation remains in the representation. Different checkpoints
are not concatenated into one coordinate system without explicit alignment. The general
clustering notebook fits its atlas on training cells and orders labels by training predicted
curvature; the retained patch workflow keeps its original validation clustering and measured
curvature ordering. These are explicit, distinct atlas choices.

## Reusable source boundaries

`src/analysis/` contains substantive packages for interventions, embeddings, conditioning,
normalization, metrics, spatial operations and encoding comparisons. The 25 four-line
compatibility modules were deleted. Historical serialized class paths are translated only
inside `src/artifacts/pickle_compat.py`, rather than through filesystem import shims.
Ordinary imports use canonical modules. Fixed KI67/N=300 PCA orchestration was removed;
weighted PCA/SVD and exact finite-readout utilities remain reusable.

Study sequences for masking training/benchmarking, fate replacement, masking ablation,
exclusive comparison, observed-cell census, neck qualification and unassigned-cell analysis
are visible in their notebooks. Shared inference/calibration and summaries used by multiple
studies stay in source modules. The notebook-specific training handoff/global-namespace
injection was removed.

Notebook configuration uses one assignment or mapping field per line, with a short
inline explanation of its meaning or allowed options. Changing the presentation does
not change the configured values.

`src/plotting/` retains six reusable modules: cluster, motif, mesh, influence-map,
training-history and depth-scan plots. Fixed KI67, unassigned-cell, niche, masking,
replacement, encoding-comparison and size-conditioning figures are defined in visible
notebook sections. Unused figures from retired embedding studies were removed.

`spatial/niche_inference.py` evaluates held-out checkpoints and constructs case tables.
The former `niche_summary.py` selected a fixed collection of niche-study comparisons;
that procedure now lives in the niche and encoding-comparison notebooks. Its shared
bootstrap/statistical operations remain in `conditioning/response_statistics.py`.
The optional matched-encoding inference workflow receives the notebook's summarizer
explicitly as a callback.

Other clarified module names are `size_conditioning/cohort_inputs.py` (formerly
`data.py`), `size_conditioning/encoding_effects.py` (formerly `comparison.py`),
`embeddings/size_responses.py`, and `models/size_models.py`. Old serialized class
paths are translated by the artifact reader; new code imports the canonical names.

## Validation and migration

Tests cover notebook syntax/structure, training-analysis separation, saved-model/transform
round trips, historical checkpoint formats and pickle class paths, exclusivity, intervention
invariants, and small end-to-end workhorse training followed by fresh analysis kernels with
training disabled. Production scientific experiments are not rerun by this refactor.

`notebook_migration.json` maps former notebooks to their current workflows. An empty list
means the notebook was explicitly retired; saved results remain separately available.

Verified after consolidation: 64 tests pass. Fresh synthetic analysis runs include the main
benchmark, marker zeroing, sampling, PCA, clustering and unchanged patch sampling workflow.
The reader also reproduces original inputs, forward predictions and hidden representations
exactly for a checked validation organoid in saved head-only GIN, exclusive FiLM and masking
runs. No production training was executed. Historical inline outputs are preserved separately
in `results_experiments/refactor_20260918/previous_notebook_outputs.json.gz`.

After moving figure definitions, 48 cached-data figures were compared with their former
source-module implementations: plotted coordinates, confidence-band paths, images and
axis labels match exactly. Notebook settings were reformatted with AST-equivalence
checks, so the configuration values and evaluation order are unchanged. The 64-test
suite continues to pass, including pickle compatibility for the renamed embedding module.

The unified training layout passes 69 tests. Synthetic checks exercise both GIN and FiLM
at multiple depths, marker combinations with identical fold membership, tagged run names,
unambiguous analysis selection, and a new FiLM bundle exported into masking training,
masking benchmarks and replacement inference. Production training was not repeated.

## Ablation notebook separation and settings placement

The ablation folder now contains exactly `total_analysis`, `size_dependent_ablation`,
`ablation_comparison` and `sampling_diagnostics`. Both inference notebooks select any of
marker zeroing, trained masking or exclusive-fate replacement. They restore every saved
fold at a chosen depth and probe exact hops 1..depth, never the recipient center. Total
analysis pools observed-size effects; size analysis separates observed-size bins from
fixed-neighborhood N sweeps. Shared sampling and inference live in `interventions/fate_edits.py`;
restoration, donor fitting and checkpoint loops remain visible in notebooks.

Ordinary GIN and FiLM are supported. Replacement requires exclusive identities and uses
training-only matched donors; it no longer requires a log-N-only FiLM head. Sweeps require
`log_num_cells`. Masking requires a positive-rate masking-trained model, which can also be
used for zeroing/replacement. Comparison intersects supported physical source cases across
methods, preserves constant sweep cohorts, and flags different fitted models. Summaries
average cases/seeds within organoids before equal-organoid averaging and bootstrapping.
Settings are the first analysis code cell after bootstrap in every analysis notebook.

Validation: 75 tests pass, including all three methods executed through both analysis
notebooks and saved-output comparison on synthetic data. Higher-hop sampling/matching,
training-only donor selection, unchanged graphs/recipients and all-fold selection are
checked. No production analyses or training were rerun.

The historical niche/route report viewer lives in `visualizations/niche_reports.ipynb`
and no longer launches old model inference. Removed full/exclusive historical appendices,
old normalization panels and their notebook outputs are preserved in the gzipped notebook
archive under `results_experiments/refactor_20260918/`; scientific result directories and
checkpoints were not deleted or changed. Modern size-dependent inference uses the selected
training run, not any hard-coded historical checkpoint.

## Historical coverage sampling restored

Commit `9ae3657` is the reference for the ablation center/pair coverage sampler, its
2000/50/25 defaults, per-pair/per-hop case-level source non-overlap, and the 20-case
hatched heatmap threshold. The original reusable `sample_subgraphs_coverage` and
`flag_non_overlapping_source_cases` functions are called directly. The no-subsampling
branch now also returns full coverage diagnostics. Current ablation adapters additionally
include unassigned identities and support all selected model hops. Pair memberships are
filtered independently for coexpressing recipients without modifying their input features.
Output summaries distinguish unique physical cases from repeated seed rows. Coverage
settings are mirrored across inference and diagnostics, and checked when comparing runs.

Verification: 78 unittest checks pass, including all three intervention methods through
both analysis notebooks and the comparison notebook; coverage selection and non-overlap
are checked against the historical helpers. Plot checks cover below-threshold, exactly-at-
threshold and missing pairs, plus gaps in unsupported size curves. No production analysis
or model training was run.
