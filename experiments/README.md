# Notebook guide

Training notebooks create saved runs. Analysis notebooks select a run and checkpoint from
its catalog; they do not depend on a training kernel or refit target transformations.
Inspect the settings and execution switches before running. Executed experiment notebooks
may retain an enabled training switch and an explicit resume directory.

## Training

| Notebook | Purpose |
| --- | --- |
| [GIN/FiLM depth training](training/gin_depth_training.ipynb) | Full-panel GIN, FiLM or both: depths 0–4, folds, width/seed grids, shared timepoints, filters, optional residualization and exclusive markers. No subset, permutation or masking training. |
| [Accommodation scan training](training/energy_model_training.ipynb) | Explicit fresh-data settings; mean or mean+SD targets; configurable presence pairs and an accommodation grid on identical folds. |
| [Fixed-accommodation training](training/shape_conditioned_energy_training.ipynb) | Default γ=0.5; center-only, selectable interaction ranges/pairs and plain GIN; either or both target normalizations. |
| [Fate masking training](training/fate_masking_training.ipynb) | Missing-fate indicator training and matched zero-mask controls across rates/seeds. |
| [Graph controls training](training/graph_controls_training.ipynb) | Controls matched to a completed GIN/FiLM run: copy its cohort, exact folds, preprocessing, baseline and training settings; copy original reference checkpoints byte-for-byte and train ring or altered-signal controls. `max_folds` limits this to original folds without resplitting. |
| [Lineage removal training](training/lineage_removal_training.ipynb) | Any number of named marker combinations on shared folds and preprocessing, with immediate validation results. |

GIN, control and lineage training use the same artifact schema. Settings, split membership,
fitted transforms, baseline offsets, raw cohort, exact model inputs, constructor specifications,
weights and histories are recorded. The raw cohort and fold inputs are shared on disk rather
than duplicated for every depth. FiLM now uses this same format in the main notebook.
Masking retains its checkpoint/reference layout, which the same analysis reader understands.

Every run saves a fitted global baseline and compares held-out MSE against it, including
when `residualize=False`. That switch controls target subtraction only. The baseline scores
are saved in `baseline_validation_mse.csv`; `validation_mse.csv` also contains paired baseline
MSE and the model-minus-baseline difference. Negative differences indicate improvement.

All training notebooks save to `training_results/<notebook_name>/<tag>_<timestamp>/`.
Set `SETTINGS['tag']` to an optional purpose label (letters, digits, underscores or hyphens);
an empty tag omits the prefix. The chosen destination is `RUN_DIR` (`MASKING_RUN` for masking).
The separate size-conditioned FiLM training notebook has been merged into GIN depth training.

In general analysis notebooks, set `TRAINING_NOTEBOOK` and `TRAINING_RUN` to the workflow
and run-directory name. If `TRAINING_RUN=None`, loading succeeds only when exactly one run
is available; otherwise it lists the choices. An explicit historical path remains supported.
`AnalysisRun.records` lists all saved model keys, including their depth, fold and seed.

Masking training uses exclusive FiLM with only log cell count in the head. It exposes
depth (including 0), width, dropout, normalization, residual connections, optimizer/edge-loss
settings, folds, seeds and masking rates in `SETTINGS`. One architecture is trained per run;
cohort selection and preprocessing are inherited. Select a `reference_model_key` from the
main training run. A derived `analysis_inputs/film_d<depth>_h<width>/` package adapts its
saved weights and exact fold inputs without retraining or refitting target transforms.

`RUN_TRAINING=False` stops at the fitting cell after configuration/data inspection;
`True` permits fitting, validation summaries and artifact saving when that cell runs.
It does not launch separate analysis notebooks. The ablation notebooks probe every exact
hop from 1 through the selected model depth; the recipient center is never edited.

## Analysis

| Folder | Notebooks |
| --- | --- |
| Ablation | [Total effects](ablation/total_analysis.ipynb); [size-dependent effects](ablation/size_dependent_ablation.ipynb); [interactive size viewer](ablation/size_ablation_viewer.ipynb); [method comparison](ablation/ablation_comparison.ipynb); [sampling diagnostics](ablation/sampling_diagnostics.ipynb) |
| Benchmarks | [Model comparison](benchmarks/model_comparison.ipynb); [graph controls](benchmarks/graph_signal_controls.ipynb); [masking quality and robustness](benchmarks/masking_quality_and_robustness.ipynb) |
| Interpretable fate models | [Simplified model stability](../legacy/energy_models/notebooks/benchmarks/simple_fate_energy_evaluation.ipynb): supported replacement contrasts, fold stability, spatial prediction errors, accommodation compensation and collinearity; [Pooled interaction comparison](../legacy/energy_models/notebooks/benchmarks/pooled_interaction_evaluation.ipynb): historical count/fraction and centering tests; [Historical mean-curvature energy evaluation](../legacy/energy_models/pre_standardization_streamline/mean_curvature_energy_evaluation.ipynb): distance-first regions, matched FiLM MSE, fitted preferences, accommodation, activation and hop signs; [Coupled fate evaluation and coefficients](benchmarks/coupled_fate_evaluation.ipynb): all-region MSE, FiLM accuracy reference, coupling, center and ordered pair coefficients. |
| Marker subsets | [Marker informativeness](marker_subsets/marker_informativeness.ipynb) |
| Embeddings | [General clustering](embeddings/clustering.ipynb); [patch composition](embeddings/patch_composition.ipynb); [embedding responses and PCA](embeddings/embedding_responses.ipynb) |
| Neighborhoods | [observed KI67 neighborhoods](neighborhoods/ki67_observed_neighborhoods.ipynb); [unassigned cells](neighborhoods/unassigned_cells.ipynb) |
| Data quality | [Cohort review](data_quality/cohort_review.ipynb); [graph/mesh checks](data_quality/graph_and_mesh_checks.ipynb); [marker complexity](data_quality/marker_complexity.ipynb) |
| Visualizations | [Interactive energy-model organoid viewer](visualizations/energy_organoid_viewer.ipynb); [Organoid predictions and mesh export](visualizations/organoid_predictions.ipynb); [archived niche and FiLM-route reports](visualizations/niche_reports.ipynb) |

Generic analyses accept both GIN and FiLM through `AnalysisRun`. Masking/replacement donor
matching and independent FiLM-layer routes have specific mathematical/input requirements,
which their introductions state explicitly. Clustering extracts the final local embedding
before global-head concatenation and analyzes independent checkpoints separately. Patch
composition keeps its original non-overlapping patch sampling and accuracy-selection analysis.

Regional evaluation reads node-aligned crypt distances and circumference profiles from the
source data. The mean-curvature energy analysis uses the distance-first partition documented
in `docs/mean_curvature_energy.md`; earlier analyses retain their recorded definitions.

Each notebook opens with its purpose and describes the steps under section headings.
Configuration entries each have their own line and a brief inline explanation.
Related settings are grouped by purpose, with blank lines separating groups. Keep
consecutive imports together without intervening blank lines.
Study-specific figure definitions are visible in the notebook; shared plotting helpers
in `src/plotting/` cover reusable mesh, cluster, motif, influence and training plots.
Outputs from the old notebook arrangement were cleared to avoid presenting them as results
of revised code; the underlying scientific reports, figures and checkpoints remain in their
original result directories. Distinct historical cell outputs are also preserved in
`results_experiments/refactor_20260918/previous_notebook_outputs.json.gz` with their origin
notebook, cell position and section heading. Optional historical comparisons load explicit run directories.
The obsolete archive and all `legacy/` notebooks were removed.

Ablation model selection is explicit: set `TRAINING_SUBFOLDER`, `TRAINING_RUN`, `MODEL_DEPTH`, `MODEL_NAME`,
and (if ambiguous) `HIDDEN_DIM` in the first settings cell. Every saved fold at that depth
is required; `MODEL_SEEDS=None` includes every training seed. Missing checkpoints are an
error, not a reason to silently analyze fewer folds. Each checkpoint restores its own
validation membership and fitted transformations. Overall effects pool observed sizes;
the N-dependent notebook separates observed-size bins from fixed-neighborhood N sweeps.
All analysis notebooks expose their user settings immediately after the root bootstrap,
before plotting definitions and data/model loading. Historical report figures remain
readable; pre-restructure ablation notebook contents and outputs are preserved in
`results_experiments/refactor_20260918/ablation_notebooks_before_restructure.json.gz`.

The ablation folder has four computation/diagnostic notebooks and a saved-results viewer:

- `total_analysis`: choose zeroing, masking or replacement; plot observed-size effects pooled
  over N as recipient-by-source heatmaps at each hop 1..depth.
- `size_dependent_ablation`: choose the same interventions; plot observed-size bins and
  fixed-neighborhood N sweeps. Replacement weights can remain anchored at observed N or
  adapt to the supplied N. At observed size these choices are identical by construction.
- `size_ablation_viewer`: browse saved N-dependent summaries with center/perturbation marker,
  observed/sweep and normalization controls; overlay selected curves in one panel per hop.
  Uses saved means, bootstrap intervals and support thresholds without inference.
- `ablation_comparison`: load completed analysis directories and compare methods using
  common supported source-cell cases. Size-dependent outputs include observed cases, so
  they support both total and size comparisons without rerunning total inference.
- `sampling_diagnostics`: compare coverage-targeted, uniform and marker-stratified
  sampling, including post-filter shortfalls, without curvature inference.

Both inference notebooks accept ordinary GIN and FiLM. Replacement requires exclusive fates
and uses only training-fold donors, treating unassigned cells as an identity. Masking requires
a checkpoint trained with a positive masking rate. Zeroing and replacement can also use that
same checkpoint with all unedited fates observed, allowing a comparison of methods within one
model. Sweeps require a saved `log_num_cells` input. Other head globals, if present, stay fixed.
Ablation outputs and settings are saved beneath the selected training run's `analysis/` folder.
The comparison notebook reports when the selected analyses also differ in fitted models.
Previous ablation notebooks are preserved in
`results_experiments/refactor_20260918/ablation_before_four_roles.json.gz`.

The default ablation sampler is restored from `9ae3657` (`ablation_analysis.ipynb`):
`SAMPLING_SCHEME='coverage'`, `CENTER_SAMPLE_SIZE=2000` per fold,
`CENTER_MIN_MARKER_COUNT=50`, `CENTER_MIN_PAIR_COUNT=25` per marker pair and exact hop,
and `SAMPLING_SEED=0` (the former `SUBGRAPH_SEED`). Greedy coverage selection is followed
by weighted filling of the remaining budget. Unassigned cells are an additional sampling
identity. The first positive source in each ring is edited, as in the original single-source
analysis. `CENTER_APPLY_NON_OVERLAP=True` then filters source cells independently within
`CENTER_NON_OVERLAP_GROUP_COLUMNS=('hop','center_marker','source_marker')`.

These settings are mirrored in total, size-dependent and sampling-diagnostic notebooks;
comparison reads, displays and checks them from the saved results. Coverage targets apply
before non-overlap and donor support checks, so insufficient availability or budget can
leave shortfalls. Per-checkpoint `center_coverage`, `pair_coverage`, and `case_non_overlap`
audit tables are saved under `support/`. Selection does not use curvature or anatomy.

`HEATMAP_DISPLAY_MIN_CASES=20` restores white, diagonally hatched low-support heatmap
cells. It accepts a scalar or per-hop dictionary. The threshold uses retained unique
physical cases (`n_cases`), never seed-replicated inference rows; method comparisons count
only common supported cases. Size curves have gaps where the same threshold is not met.
An additional `MIN_PLOT_ORGANOIDS` threshold remains available (default 1).
Changed sampling requires new ablation inference, but no model retraining. If the chosen
`ANALYSIS_TAG` already contains different settings or code, the inference notebooks select
a separate folder with a deterministic configuration suffix. Matching configurations resume
their own results; existing output files are preserved. Copy the printed resolved path into
the comparison notebook. Total analysis has no N-sweep settings: it evaluates only observed
inputs and pools effects over N. Replacement donors are matched using observed properties;
`N_DEPENDENT_REPLACEMENT` belongs only to the size-dependent notebook.

Total, size-dependent and comparison notebooks include annotated sample-count heatmaps
for every hop. The blue log scale and in-cell integers show retained unique physical
cases, including low counts and zeros. Size analysis separates pooled observed bins
from the fixed all-N sweep cohort, without multiplying support by model seeds or swept
N values. Comparison uses common supported cases. Figures and `pair_sample_counts.csv`
are saved alongside effect plots. These plotting sections use loaded case tables only.

The clustering notebook includes separate sections for t-SNE, both marker-composition
normalizations (`P(marker | cluster)` and `P(cluster | marker)`), true/predicted physical
curvature minus the saved baseline, and nearest-crypt distance categories by cluster.
Full validation-population figures are always shown; optional historical accuracy-filtered
figures retain the lowest-error fraction within each fixed cluster. The atlas stays fitted
on training cells, with IDs ordered by training predicted curvature. Missing crypt distances
remain separate from villus, and the intermediate distance band is not a confirmed neck.

`embedding_responses` is a separate single-organoid N-sweep probe: fixed graph and fates,
shared reference PCA, trajectories for selected cells, predicted curvature and full embedding
displacement. Optional source-fate editing adds paired displacement and prediction differences.
Its optional historical LGR5/FiLM panels read saved reports; they do not control the main PCA.

Clustering selects one checkpoint using `MODEL_DEPTH`, `MODEL_FOLD`, `MODEL_NAME`,
`MODEL_SEED` and `HIDDEN_DIM`; `MODEL_KEY` is an optional exact override. It no longer
chooses the first catalog entry silently. The default is GIN depth 2, fold 0, with a
unique seed/width required. Training and validation population counts are printed before
extraction, and each scatter reports displayed cells and organoids. All validation cells
are encoded; `MAX_PLOT_CELLS` caps only scatter points (`None` includes them all).

Cohort review selects a saved model depth and one fold explicitly, prints the resolved
checkpoint, and ranks its validation organoids by physical-curvature MSE. Its interactive
mesh browser filters inclusive global MSE ranks (1 = worst) and sphericity Q = 36πV²/A³,
then selects an organoid to show ground truth and prediction side by side. Rank numbers
remain fixed after filtering; missing-Q inclusion is explicit. `TOP_N` limits the initial
table/export, not the browser. Browsing never reruns predictions or updates the blacklist.

Regional validation MSE versus depth is included in the GIN, masking, graph-control and lineage-removal training notebooks. The coupled-fate workflow reports regional MSE in its separate evaluation notebook. Regions are rebuilt from source crypt distances and circumference profiles, with evaluation artifacts saved under each run’s `regional_evaluation/` folder.

Training notebooks provide a persistent fitting-progress panel (`SHOW_PROGRESS=True`). For epoch-based training, `PRINT_EPOCHS=False` suppresses the epoch stream; coupled-fate fitting instead reports penalty trials and coupling-optimizer progress. Baseline fits and reused checkpoints are counted separately; completed, failed, or interrupted task records are saved in `training_progress.csv` under the run directory.

The active explicit fate model solves predicted curvatures jointly using normalized
neighbor coupling. The fitting notebook compares center-only and linear-pair variants,
with and without coupling; selected ordered pairs can later use presence or smooth
saturating responses. Coupling reaches the whole connected component, so FiLM depth 2
is an accuracy reference, not a matched-receptive-field control. Measured-neighbor variants are separate conditional reconstruction models; their
extra target information is explicit. Predicted-neighbor models retain a post-inference
measured-versus-predicted neighbor diagnostic. The evaluation includes every
anatomical category rather than showing only profile-qualified crypts and necks.

See [coupled-model conventions](../docs/coupled_fate.md). The earlier additive spline
and fraction workflows are [archived](../legacy/fate_interactions/README.md), including
their model code and notebooks. Their saved training results remain in place and
continue loading through the artifact compatibility mappings.

The [curvature-model decision record](../docs/curvature_model_decisions.md) records scientific priorities, previous findings, rejected interpretations and the current small joint model. Read it before extending these models.

## Curvature-energy workflow

The active workflow uses measured-organoid-mean subtraction, optionally followed by division by measured SD. Select `target_modes=['mean_only']`, `['standardized']`, or both in either training notebook. These are conditional reconstruction tasks, not fate-only shape prediction.

| Purpose | Training | Analysis |
| --- | --- | --- |
| Scan accommodation | [Accommodation scan](training/energy_model_training.ipynb) | [MSE and coefficient profiles](benchmarks/energy_model_evaluation.ipynb) |
| Fixed accommodation (default γ=0.5) | [Fixed-accommodation training](training/shape_conditioned_energy_training.ipynb) | [Model/normalization comparison](benchmarks/shape_conditioned_energy_evaluation.ipynb) |
| Inspect one fitted configuration | Load the fixed run | [Coefficients and artificial neighborhoods](benchmarks/energy_model_inspection.ipynb) |

Both training notebooks expose data, folds, cleanup and optimization settings. They save the raw cohort, physical targets, observed moments, models, predictions and memberships together. Fixed-accommodation configurations support radius 0 (no pairs), 1 or 2, source/recipient panels, explicit ordered pair lists, and GIN depth selection. Radius 0 retains accommodation. Presence within two hops counts an identity once across the complete neighborhood.

The artificial-neighborhood notebook inserts secretory identities into periodic six-neighbor KI67/LGR5/Unassigned tissue. It separates local preference, accommodation and uniform output centering, verifies the uniform-tissue control, and checks finite-size effects. Synthetic fields remain in fitted target units because no measured organoid SD is available.

Previous versions of these notebooks are in [the pre-standardization archive](../legacy/energy_models/pre_standardization_streamline/). Historical fitted models and reports remain available; active analyses no longer append those legacy comparisons.
