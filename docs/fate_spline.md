# Exclusive-fate spline curvature model

`FateSplineCurvature` is a varying-coefficient additive regression, not a modified GIN.
Existing GIN, FiLM, masking and ring implementations are unchanged.

For identity A, exact shell r, source identity B, and shell fraction p:

```
prediction = saved baseline offset + b(N) + a_A(N)
             + sum_{r,B} (p_B^r - pi_B^r) [c_B^r(N) + P_AB^r(N)]
```

The `center` variant includes only b+a and has radius zero. `shared` adds c;
`pairwise` adds P. A radius includes all exact shells 1 through that radius.
Repeated paths, duplicate edges and self loops never increase counts. Edges are
treated as undirected. Cells without a positive marker have identity Unassigned.
An empty shell contributes zero, not a negative reference composition.

## Interpretation and fitting

Center weights are training identity proportions averaged equally over organoids.
Each shell's source weights are mean fractions over nonempty shells within each
organoid, then equally over organoids with that shell. Weights are bounded below
by 1e-8 before normalization so absent categories have a defined convention;
this does not supply evidence for them. Reference weights remain fixed over N.

Orthonormal null-space contrasts enforce weighted zero sums for a over centers,
c over sources, and P over both centers and sources. The convention resolves
structural main-effect redundancies, but not weak empirical support or correlated
neighborhoods. P is ordered: curvature is predicted at the recipient, so AB and BA
need not agree. c+P is the total sensitivity; P alone is a deviation from the
shared response. Multiplying by p-pi yields the actual contribution.

All coefficient curves use a clamped cubic B-spline basis with uniform interior
knots in training log N. Predictions outside the fitted N range use the nearest
boundary value. Reference weights and knot bounds are fitted independently on
inner-training data for tuning, then refitted on all outer-training inputs for
the final fit. Outer validation never selects penalties. Each organoid has equal
weight in the squared-error objective. Second differences penalize wiggles;
separate ridge penalties control non-pair and pair coefficients. A 1e-10 ridge
stabilizes every coefficient, including b. The notebook saves the actual penalty
grid, selected penalties, inner membership, and model buffers.

Fitting accumulates organoid-level sufficient statistics rather than a dense
cell-by-spline matrix. With T identities, R shells and L basis functions, a
pair model has L[T + R(T-1) + R(T-1)^2] coefficients. Solves run on CPU; a
configurable BLAS thread limit prevents oversubscription. No SGD epochs apply.
The existing progress panel reports folds, variants, radii and penalty trials.

The first implementation is linear in neighborhood fractions. Count-dependent
saturation and separate topology/ring-size terms are future model extensions,
not secretly imposed nonlinearities. Center effects are adjusted predictions at
the reference neighborhood, not intrinsic curvature. Pair coefficients are
conditional predictive associations, potentially including positional information.

## Targets and artifacts

Physical residuals use an identity target transform. Baseline offsets are added
once on prediction reconstruction. With residualization disabled, offsets are
zero but a fitted comparison baseline is still saved. Reference reuse restores
physical targets from the source transform and offset before applying the newly
selected residualization. It copies the baseline and saved predictions into the
new run; the source run is not modified and need not remain available for reload.

Runs use `training_results/fate_spline_training/<tag>_<timestamp>/`, the existing
immutable model/input bundles, `models.json`, exact `splits.json`, `settings.json`,
`exclusivity_rules.json`, and `provenance.json`. They also save validation and
baseline MSE, `penalty_trials.csv`, per-model `fit_settings.json`, progress records,
and optional source-derived regional evaluation. Knots, contrasts, reference
weights, coefficients and a fitted flag are all in the checkpoint state dict.

The inference interface returns physical residual means and a constant variance
estimated from training residual MSE for compatibility with prediction tables.
This variance is **not calibrated predictive uncertainty**, and it is not a
coefficient standard error. The second forward return is the local regression
design, **not a learned GNN embedding**. Generic FiLM-route/hidden-embedding
analyses and global-vector N interventions should not be applied to this model.
Use `coefficients(N)`, `predict_sample(sample, N)`, and `contributions(sample, N)`.
Graph forward inference recomputes counts to avoid stale features after fate edits.

## Notebooks

- `experiments/training/fate_spline_training.ipynb`: review settings and cohort;
  train radius zero once, then shared/pairwise radii 1–4; save all artifacts.
  Default data source is `fate_masking_training/Ndependent_20260922_115100`.
  Only `REFERENCE_RUN` is selected: training loads shared intact full-panel fold
  inputs via `AnalysisRun.fold_inputs`, not reference predictor checkpoints.
  Depth, width, model seed and masking rate are irrelevant to dataset reuse.
  Multiple different input bundles within a fold are rejected as ambiguous.
  The observed missingness flag is removed from spline inputs.
  Its exclusive-fate cohort, all saved timepoints, exact five folds and baselines
  are reused; the spline grid remains
  radii 1–4 plus center-only radius 0. `REFERENCE_RUN` is required. Dataset,
  filters, target selection, outer splitting and baseline training settings are
  inherited metadata, not editable settings in this notebook. No fresh cohort,
  refiltering, outer resplitting or baseline fitting path is present.
- `experiments/benchmarks/fate_spline_comparison.ipynb`: verify identical train/val
  memberships, node targets, topology, and exclusivity mapping; evaluate total and
  region MSE and paired MSE differences against selected GIN/FiLM depths.
  Reference model settings belong here; the default is the zero-masking FiLM
  control at depth 2, width 128 and seed 42, across every saved fold.
  Available reference depths are detected automatically; this default run has
  only depth 2. Spline curves still show all requested radii, with reference
  points and paired comparisons only where reference checkpoints exist.
- `experiments/neighborhoods/fate_spline_coefficients.ipynb`: one explicit fold,
  radius and seed; plot b/a/c/P, c+P, identity contrasts, training pair support,
  and an exact individual prediction decomposition. No confidence bands are
  claimed without refitting-based uncertainty analysis.

MSE shading is one SEM across organoids after averaging repeated seeds or
validation appearances within organoid. It is not uncertainty across retrained
models. Radius and GNN depth share a nominal receptive-field extent but do not
otherwise imply equivalent architectures, inputs, objectives or capacity.

## Nonlinear fraction benchmark

`experiments/training/fate_fraction_benchmark.ipynb` is a dedicated, executed
training-and-benchmark workflow for **radius/depth 2 only**. It restores the
zero-masking FiLM reference's exact cohort, folds, cleaned targets and fitted
baselines. It does not retrain the GNN or change graph-control implementations.
Outputs use `training_results/fate_fraction_benchmark/<tag>_<timestamp>/` and
retain the same portable model/input bundle interface.

`src/models/fate_fraction.py::FateFractionSpline` extends the linear model with
explicit polynomial abundance functions: `p(1-p)` and, optionally,
`p(1-p)(2p-1)`. These supplement the existing linear fraction terms. Abundance
functions can be shared or conditioned on center identity. A further option
adds a fixed number of distinct fraction products, within or across rings,
with center-dependent coefficients. Each coefficient uses the existing cubic
B-spline basis in log N. There is no MLP or nonlinear output transformation.

Only fractions, center identity and N enter the predictor. Multiplying every
count in a ring by the same positive factor leaves its prediction unchanged.
Unassigned is one of the identities. The graph is used to identify exact-hop
rings, not to provide ring size or graph motifs as additional predictors.

New features are residualized against the lower-order local design using
training-only, equal-organoid moments. Nondegenerate residual columns are
rescaled to RMS 0.1; numerically unsupported columns are zeroed. This prevents
new constant/linear dependencies from being mistaken for additional effects.
Because fractions are compositional, individual polynomial or product
coefficients remain basis-dependent. Interpret full response functions or
contrasts at supported compositions, not isolated raw coefficients.
`coefficients(N)` reverses the stored projections/scales and returns weights
on explicit original features; `contributions(sample, N)` reconstructs the
prediction exactly. These coefficient dictionaries differ from the original
linear model's `b/a/c/P` API; the old coefficient notebook is not implicitly
reused for this expanded family.

`fit_fraction_spline` tunes ridge, N-smoothness and extension shrinkage on an
inner organoid holdout. For the linear candidate the extension multiplier
shrinks center-dependent linear effects; for nonlinear candidates it shrinks
all additional terms. Product screening ranks training-only residual
associations after a lower-order size-dependent fit (screening ridge 1e-5).
It is repeated separately on inner training and on all outer training during
refitting. Outer validation never controls screening, normalization, knots or
penalty selection. An additional inner-selected benchmark chooses the
candidate family using the same inner holdout, without selecting a family on
outer validation MSE.

The experiment saves every tuning trial, inner membership, final screened
pairs, source artifact hashes, fold audits, per-organoid regional errors and
an independent reload check. Figures report physical-curvature MSE and paired
differences from FiLM, not biological interpretation. SEMs describe variation
across held-out organoids conditional on these fitted models; they do not
measure retraining uncertainty. Exact comparison requires using the same
radius, folds, target reconstruction and organoid weighting.

The dedicated notebook also includes a conservative cubic-model check using
exactly the FiLM optimizer's saved fitting subset (795 organoids per fold for
the current reference), excluding its early-stopping organoids. This check
tunes within that smaller set, whereas the main spline grid refits on all
outer-training organoids. Actual fitting IDs are recorded in
`matched_fit_membership.json` and the model's `fit_settings.json`; its shared
analysis input bundle still contains the complete outer fold. Different loss
functions and inner selection remain explicit protocol differences.
