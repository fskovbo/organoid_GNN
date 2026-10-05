# Mean-curvature preference and accommodation

For current scientific priorities and the simplified-model specification, read
[curvature_model_decisions.md](curvature_model_decisions.md). The formulations
below remain available for historical model restoration.

The current workflow uses measured mean subtraction or mean+SD standardization. Use `energy_model_training.ipynb` / `energy_model_evaluation.ipynb` for accommodation scans, `shape_conditioned_energy_training.ipynb` / `shape_conditioned_energy_evaluation.ipynb` for fixed-gamma comparisons, and `energy_model_inspection.ipynb` for single-model coefficients and synthetic tissues. Training notebooks are under `experiments/training`; analysis notebooks are under `experiments/benchmarks`.

Current predictions center the local preference before accommodation: `(I + gamma L) h = u - mean(u)`. Physical reconstruction adds the measured organoid mean and, for standardized models, multiplies by its measured SD. Earlier fitted-baseline workflows are archived under `legacy/energy_models/`; the equations below document historical model options as well as the underlying accommodation formulation.

## Model

`MeanCurvatureEnergy` predicts mean-curvature residuals by minimizing

```
E(h) = 0.5 sum_i (h_i - u_i)^2 + 0.5 lambda(N) sum_{unordered edges} (h_i-h_j)^2
(I + lambda L) h = u
```

Here L is the unnormalized Laplacian of the unique undirected adjacency, with
self-loops removed. Isolated cells retain their preferred curvature. The solution
preserves the unweighted organoid mean of u. It uses equal cell weights and no
patch-area measurements. This is a scalar-field surrogate energy, not a surface
Helfrich calculation. Lambda is a relative accommodation strength, not an
absolute material modulus; neither lumen pressure nor a geometry is inferred.
The saved spatially constant baseline is added after solving the residual field.
Global baseline features still include the previous organoid-level geometry.

Center preferences are constant in N. For each ordered center/source pair,

```
A(N) = A0 + A1 log(N/N_ref)
hop1 = A(N)
hop2 = sigma * 0.5 * A(N),  sigma in {-1,+1}
lambda(N) = softplus(l0 + l1 log(N/N_ref))
```

N_ref is the geometric mean of the fitting organoid sizes. There are no splines
and no N dependence in saturation or relative signs. Outside the training range,
the affine functions extrapolate; figures use the fitted size range. Both hop
coefficients cross zero together if A crosses zero. The half-amplitude constraint
is on coefficients, not on actual aggregate ring contributions.

The response uses `tanh(alpha*x)/tanh(alpha)` where x is the source fraction in
the entire two-hop neighborhood, multiplied by the fraction of those source cells
in each ring. Training-only reference means are conditional on center identity,
with equal organoid weighting. The hop-1 coefficient table has training-weighted
zero row/column means; hop 2 is derived and has no independently imposed contrast
constraints. This differs from independently constrained tables at each hop.
These are reference-dependent associations, not identifiable causal interactions.

## Fitting and artifacts

Continuous fitting profiles out linear coefficients by penalized least squares.
Exact sparse propagation and analytic envelope gradients optimize accommodation
and log saturation. A cached unconstrained propagated Gram matrix permits sign
flips without repeated graph solves. Coordinate sign search alternates with
continuous optimization until no single sign improves the profiled training
objective. Multiple starts explore distinct solutions but do not guarantee a
global discrete optimum. Saturation is bounded to [0.02,100]. Training-only inner
holdouts select pair regularization; the selected setting refits all outer-training
organoids. Convergence and sign-search histories are saved per model.

The fresh FiLM reference has depth 2, width 128 and zero masking. Its architecture,
optimizer, dropout, auxiliary edge loss and early-stopping protocol are copied
from the original zero-mask FiLM run. The new mean-curvature target transform is
fitted on the saved outer-training organoids. The physical targets and baselines
match the energy model; FiLM retains its internal asinh scaling. It fits its inner
training subset with early stopping, whereas the profiled energy model refits the
full outer-training set after penalty selection. Reported MSE is in physical units,
equally weighted by organoid. Energy fits support CPU direct sparse solvers or the CUDA backend below; FiLM uses CUDA when available.

Outputs are under `training_results/mean_curvature_energy_training/<tag>_<timestamp>/`.
Bundles contain models, baselines, preprocessing and exact memberships; a catalog
records every fold/model. Training verifies saved checkpoint contents on resume.
Analysis saves node predictions, region labels, organoid MSE, paired conditional
bootstrap comparisons, all coefficients, figures and `REPORT.md`.

## Distance-first regions

`distance_regions` is an additional reusable partition; `graph_regions` retains
its previous behavior. Every cell keeps its original nearest-crypt assignment.

- s<0.75: crypt interior, regardless of profile shape; split by whether any
  exclusive LGR5-positive cell is assigned to this crypt with s<0.75.
- 0.75<=s<1.1: neck if the circumference has a qualifying local minimum or flat
  section; otherwise `boundary_without_qualified_neck`. No additional minimum
  crypt-cell count is imposed by default.
- s>=1.1: villus, regardless of profile qualification.
- No detected crypts: `no_detected_crypt` for the whole organoid.
- Missing/invalid distance annotations remain explicit. Missing profiles do not
  remove valid interior or villus assignments, and cannot qualify a neck.

All cells contribute to total MSE. Regional errors weighted by the cell fractions
within an organoid must reconstruct its total error exactly. The boundaries are
operational anatomical proxies, not perfect segmentation, and small undetected
crypts remain a source of selection uncertainty.

## CUDA backend

The training notebook selects `ENERGY_DEVICE='cuda'` when CUDA is available,
with one fit at a time on the GPU. Set it to `'cpu'` to use the original direct
sparse solver. `CUDA_SOLVER_RTOL=1e-11` controls the GPU solve certificate.
Execution-only settings do not invalidate existing scientific checkpoints; a
completed run is resumed without refitting already saved models.

`fit_energy(..., device='cuda')` performs response construction, conditional
reference centering, sparse propagation, profiled linear fitting and analytic
envelope-gradient accumulation with float64 PyTorch tensors. The small bounded
SciPy nonlinear search and sequential sign choices remain on CPU. The GPU
computes the large unconstrained Gram statistics used by that same sign search.
Graph-shell counting/preparation remains a one-time CPU step.

Accommodation uses Jacobi-preconditioned conjugate gradients on the disjoint
union of the graph Laplacians. RHS chunking limits memory, and every solve checks
its true residual against the requested tolerance. Failure raises an error; it
never silently accepts an unconverged curvature field. There is no float32 or
mixed-precision change, minibatch approximation, changed penalty, or altered
hop-sign rule. Floating-point optimizer paths need not be bitwise identical.

Checkpoints keep the same constructor and state format. Existing models support
`model.predict_samples(samples, device='cuda')` and `model.to('cuda')` followed by
inference. For repeated inference on unchanged graph/fate features, construct
`EnergyBatch(samples, model.n_markers, device='cuda')` once and call its
`predict(model)` method to avoid repeated preparation and transfer. Rebuild that
batch after changing any graph or fate features. Targets are never needed in an
inference batch.

Run `scripts/benchmark_energy_cuda.py <saved-run>` for a separate performance
report with full-fold objective/gradient and prediction parity, timing, GPU memory,
and complete matched CPU/CUDA fitting on a size-spread training subset. It does
not modify saved scientific models or their evaluation. Small graphs and cold
preparation overhead can limit acceleration.

## Fixed pooled exposure and optional reference centering

`interaction='pooled'` adds a separate, backward-compatible formulation. Its
exposure is `(p_ring1 + 0.5*p_ring2)/1.5`, where each p is the source identity's
fraction within that ring. Empty rings have zero fractions. Both distance weights
are fixed: no per-pair or N-dependent range parameter is fitted. Activation is
applied after pooling. The local preference is

```
u_i = a_A + sum_B beta_AB(N) * (F_AB(x_iB) - mu_AB)
F_AB(x) = tanh(alpha_AB*x)/tanh(alpha_AB)
```

`center_response=False` sets mu and its optimization derivative to zero;
otherwise mu is recomputed from the fitting samples at each activation trial,
conditional on center identity with equal organoid weighting. Neither variant
uses validation data to compute references. Both keep the existing amplitude
contrast convention, regularization, baseline, and accommodation solve. The
pooled coefficient readout returns `amplitude`, not separate `hop1`/`hop2` tables.
The legacy two-slot response buffer is retained for checkpoint compatibility;
pooled activation occupies slot zero and slot one is identically zero. Discrete
sign search is skipped for pooled models. Old checkpoint specifications default
to `interaction='signed_hops', center_response=True` and reproduce their original
predictions.

Centering and not centering are not generally just coordinate changes when beta
is affine in log N and a is constant. Equivalence at fixed beta and alpha would
require `a_uncentered,A(N) = a_centered,A - sum_B beta_AB(N)*mu_AB`. Regularization
also changes under that shift. The dedicated experiment therefore compares two
fitting conventions/families, not an isolated biochemical mechanism.

The dedicated notebooks are `legacy/energy_models/notebooks/training/pooled_interaction_training.ipynb`
and `legacy/energy_models/notebooks/benchmarks/pooled_interaction_evaluation.ipynb`. They fit only the
two requested accommodated, learned-saturation variants on the saved five folds.
Previous signed-hop and FiLM models are reused, not retrained. Results are under
`training_results/pooled_interaction_training/<tag>_<timestamp>/`.

Fold diagnostics use the same fate-selected contexts for every fitted fold,
restricted to common marginal exposure/size support. They export the pair term
relative to zero source exposure, with reference offsets canceled, before
accommodation. This is not a fate-replacement ablation: other identity fractions
are held fixed and zero exposure can be outside support. Diagnostic reference
contexts include training cells for some folds; only the separate outer-fold MSE
is a held-out performance measure. Overlapping training folds make agreement
descriptive, not independent replication. Report absolute disagreement and
response magnitude alongside relative dispersion, and flag near-zero responses.

## Count-exposure reference experiment

For pooled interactions, `exposure_kind='counts'` replaces each ring fraction
with the number of cells of that identity in the ring. The exposure is
`(n_ring1 + 0.5*n_ring2)/1.5`, including Unassigned as an identity. It is not
divided by ring population or clipped. `exposure_kind='fractions'` remains
the default, preserving earlier checkpoint predictions. Counts are intentionally
restricted to the pooled interaction formulation.

Activation remains `tanh(alpha*x)/tanh(alpha)`, normalized at one exposure unit.
With counts, x may exceed one and the activation can exceed one before saturating
at `1/tanh(alpha)`. Thus amplitudes and saturation parameters have different
units from fraction-based models. The reference experiment retains the same
numerical bounds, priors, initialization strengths and penalty grid; these are
not scale-invariant regularization choices. Absolute identity counts can also
implicitly reveal neighborhood size.

The pooled training notebook exposes this choice beside the centering variants.
The evaluation notebook compares both count variants with both saved fraction
variants on identical outer/inner memberships and with the existing FiLM.
Fold-consistency diagnostics intersect count and fraction support on identical
reference neighborhoods. Consequently, their common context cohort differs
from the earlier fraction-only report. Earlier executed notebooks are preserved
inside that run's `source_snapshot/executed_notebooks` directory.

## Independent amplitudes for selected sources

`pair_constraints='direct', source_indices=[...]` gives each selected ordered
center/source pair its own coefficient, without weighted-zero row or column
constraints. All center identities remain represented. Excluded sources have
zero explicit pair amplitude, but still contribute to fraction denominators,
center preferences and mechanical accommodation. An exhaustive linear fraction
panel can be intercept-confounded. The current all-source notebook uses an
explicit source reference for linear activation and reports design rank.

With `interaction='pooled', center_response=False, size_dependent=False`, the
model has constant uncentered preferences, pair coefficients and accommodation.
The `linear` and `fixed` activations leave only one nonlinear fitted scalar.
Old defaults and checkpoint tensor shapes are unchanged. Source selection and
constraint mode are stored in constructor specifications for exact restoration.

## Literal presence, interaction radius and linear source reference

`interaction='pooled', interaction_radius=1` uses first-ring fractions;
`interaction_radius=2` retains `(p1 + 0.5*p2)/1.5`. Graph preprocessing keeps two
rings cached for all variants. Only the selected rings enter responses/support;
mechanical accommodation always uses the full graph Laplacian.

`activation='presence'` returns exactly `1[exposure > 0]`, including zero for an
empty neighborhood. No alpha or threshold is fitted. Presence is unchanged if a
source's positive count is multiplied, and it ignores positive distance weights.
`activation='mixed', pair_activations=choices` accepts a square list of lists,
indexed by center and source identity, whose entries are `'linear'` or `'presence'`.
There is one amplitude per pair, not two competing linear/presence coefficients.
The map is a fixed constructor setting saved in the checkpoint. Automatic
activation selection is not implemented in this experiment.

For direct all-source linear models, `reference_source=unassigned_index` removes
that source column while retaining all center coefficients. On populated rings,
`a_ref = a + beta_ref` and `beta_B_ref = beta_B - beta_ref` preserve predictions.
The zero reference coefficient does not mean zero biological contribution.
Empty rings do not obey the exhaustive-composition sum; their frequency must be
audited before interpreting this as a pure gauge choice. Presence models use
`reference_source=None` because nonlinear indicator sums are not generally one.

Existing constructor defaults and tensor layouts are unchanged for old models.
CPU and CUDA paths share these semantics, with checks for gradients, exact
responses, radius isolation, checkpoint restoration and legacy predictions.

## Restricted recipients and fixed accommodation

`recipient_indices=[...]` selects the rows of direct pair amplitudes that exist.
Excluded recipients have exactly zero incoming pair terms, not merely a penalty
or a support mask. Center preferences remain for all identities. Sources and
recipients need not be disjoint in the reusable model, but the current notebook
requires them to be disjoint and exhaustive. Historical defaults retain all rows.

`fixed_strength=gamma` specifies exact nonnegative accommodation, including zero,
with no logit approximation. It requires constant accommodation and bypasses all
nonlinear strength optimization. Both CPU and CUDA solvers obey it. The helper
`fit_energy_fixed` selects only coefficient shrinkage within the supplied training
partition and then refits that entire partition. Spherical validation is never
passed to this helper. Checkpoints save both options and remain loadable through
`AnalysisRun.select`/`fold_inputs`.

The fresh training notebook fits baseline/preprocessing on ordinary training only,
then transforms ordinary and spherical validation with those same fitted objects.
Curvature cleanup learns quantiles on training graphs only. Regional merging is
local to this new workflow; historical region definitions are not changed globally.
