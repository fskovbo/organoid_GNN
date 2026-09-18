# Notebook guide

Training notebooks create saved runs. Analysis notebooks select a run and checkpoint from
its catalog; they do not depend on a training kernel or refit target transformations.
All training switches are initially off. Configure the settings, inspect the cohort/splits,
then enable training when ready.

## Training

| Notebook | Purpose |
| --- | --- |
| [GIN/FiLM depth training](training/gin_depth_training.ipynb) | Full-panel GIN, FiLM or both: depths 0–4, folds, width/seed grids, shared timepoints, filters, optional residualization and exclusive markers. No subset, permutation or masking training. |
| [Fate masking training](training/fate_masking_training.ipynb) | Missing-fate indicator training and matched zero-mask controls across rates/seeds. |
| [Graph controls training](training/graph_controls_training.ipynb) | GIN, ring and ring-size controls with intact, permuted or constant fate inputs. |
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

All four notebooks save to `training_results/<notebook_name>/<tag>_<timestamp>/`.
Set `SETTINGS['tag']` to an optional purpose label (letters, digits, underscores or hyphens);
an empty tag omits the prefix. The chosen destination is `RUN_DIR` (`MASKING_RUN` for masking).
The separate size-conditioned FiLM training notebook has been merged into GIN depth training.

In general analysis notebooks, set `TRAINING_NOTEBOOK` and `TRAINING_RUN` to the workflow
and run-directory name. If `TRAINING_RUN=None`, loading succeeds only when exactly one run
is available; otherwise it lists the choices. An explicit historical path remains supported.
`AnalysisRun.records` lists all saved model keys, including their depth, fold and seed.

Masking and replacement interventions use exclusive FiLM with only log cell count in the
head. Masking training exposes depth (including 0), width, dropout, normalization, residual
connections, optimizer/edge-loss settings, folds, seeds and masking rates in `SETTINGS`.
One architecture is trained per run; cohort selection and preprocessing are inherited.
Select a `reference_model_key` (masking) or `REFERENCE_MODEL_KEY` (ablation) from the main training run. A
derived `analysis_inputs/film_d<depth>_h<width>/` package adapts its saved weights and exact
fold inputs to these existing workflows without retraining or refitting target transforms.
Geometric normalization references are fitted from training-fold areas/counts during export.
Historical cached-study sections remain explicitly tied to their original result directories.

`RUN_TRAINING=False` stops at the fitting cell after configuration/data inspection;
`True` permits fitting, validation summaries and artifact saving when that cell runs.
It does not launch separate analysis notebooks. Masking benchmark/ablation graph contexts
cover the configured depth, while their single-neighbor probes remain at hops 1 and 2.

## Analysis

| Folder | Notebooks |
| --- | --- |
| Ablation | [Marker zeroing and saved normalization](ablation/marker_zeroing.ipynb); [replacement/masking and full/exclusive comparisons](ablation/fate_interventions.ipynb); [sampling diagnostics](ablation/sampling_diagnostics.ipynb); [niche and FiLM-route diagnostics](ablation/niche_hypotheses.ipynb) |
| Benchmarks | [Model comparison](benchmarks/model_comparison.ipynb); [graph controls](benchmarks/graph_signal_controls.ipynb); [masking quality and robustness](benchmarks/masking_quality_and_robustness.ipynb) |
| Marker subsets | [Marker informativeness](marker_subsets/marker_informativeness.ipynb) |
| Embeddings | [General clustering](embeddings/clustering.ipynb); [patch composition](embeddings/patch_composition.ipynb); [embedding responses and PCA](embeddings/embedding_responses.ipynb) |
| Neighborhoods | [Region predictability](neighborhoods/region_predictability.ipynb); [observed KI67 neighborhoods](neighborhoods/ki67_observed_neighborhoods.ipynb); [unassigned cells](neighborhoods/unassigned_cells.ipynb) |
| Data quality | [Cohort review](data_quality/cohort_review.ipynb); [graph/mesh checks](data_quality/graph_and_mesh_checks.ipynb); [marker complexity](data_quality/marker_complexity.ipynb) |
| Visualizations | [Organoid predictions and mesh export](visualizations/organoid_predictions.ipynb) |

Generic analyses accept both GIN and FiLM through `AnalysisRun`. Masking/replacement donor
matching and independent FiLM-layer routes have specific mathematical/input requirements,
which their introductions state explicitly. Clustering extracts the final local embedding
before global-head concatenation and analyzes independent checkpoints separately. Patch
composition keeps its original non-overlapping patch sampling and accuracy-selection analysis.

Region predictability expects all-marker models and node-aligned anatomical annotations.
The default is the saved circumference-qualified crypt/neck table. Missing/undetected crypts
are not relabeled as villus. There is no marker-coherence analysis in that notebook.

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
