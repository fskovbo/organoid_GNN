# Spatially coupled fate prediction

The active notebooks use **`CoupledFateResponse`**:

- `legacy/energy_models/notebooks/training/coupled_fate_training.ipynb` fits and saves the planned model grid.
- `experiments/benchmarks/coupled_fate_evaluation.ipynb` restores checkpoints, evaluates them and plots coefficients.

For each target, the model predicts physical residual curvature by solving
`(I - gamma(N) W) prediction = (1 - gamma(N)) local`, then restores the saved
organoid baseline. `W` averages adjacent cells without using their measured
curvature. Local preferences combine center identity and ordered center/source
responses within two hops. Propagation is shared across cell identities and can
spread these local contributions over the connected component. It smooths a
predicted scalar field; it is not a fitted Helfrich energy or a bending modulus.

## Active experiment
The active notebooks now use `CoupledFateResponse`. The preceding
`CoupledFateCurvature` class and its checkpoints remain readable unchanged.
No measured-neighbor variants are trained in the new experiment.

For center identity A, source identity B and local radius R, define x as the
fraction of B cells among **all** cells within hops 1..R, and p_r as the fraction
of those B cells lying in exact ring r. The uncentered pair contribution is
`f_alpha(x) * sum_r beta_AB^r(N) p_r`, with zero contribution if B is absent.
`f_alpha(x)=tanh(alpha*x)/tanh(alpha)` for positive alpha, and f_0(x)=x.
It is bounded on x in [0,1]. The linear limit weights each ring's source count
by the total population across the neighborhood, unlike the earlier per-ring
fraction model. This change applies to **all** new activation controls.

Activation choices are linear, fixed alpha=3, and learned pair-specific alpha.
Alpha stays constant across N. Learned alpha is bounded to [0.02,100]; its log
has weak shrinkage toward log(3). Ordered pairs occurring in fewer than 20
training organoids retain fixed alpha and are flagged. Fixed saturation was
specified before outer evaluation. Alpha controls shape, not a count threshold.
Reference response means are recomputed on fitting data for each candidate
alpha, and their derivatives are included in the profiled objective. Coefficient
row/column constraints retain the original training-weighted convention.

Gaussian-only runs compare center, linear, fixed, learned families, with/without
propagation and N dependence: 16 configurations per fold. Mean-only and joint
runs compare the learned family under the same four propagation/size settings.
Joint models share only alpha. Each target has separate center preferences,
pair amplitudes, propagation coefficients and saved baseline. Target losses are
scaled by outer/inner-fitting residual RMS as appropriate and averaged equally;
all reported MSEs use physical target units. Constant local models retain an
N-dependent global baseline. Multiple optimizer starts are selected by penalized
training objective, and shrinkage by an inner organoid holdout. Fits require a
small projected gradient. Preconditioned full BFGS handles the different scales
of coupling-spline and saturation parameters, with smooth bounds on saturation.
A bounded SLSQP fallback handles nonconvergence. Independent initialization
paths continue through the penalty grid, and selected inner solutions warm-start
outer refits; no validation target initializes or selects a path.

Gaussian targets and baselines are restored exactly from the zero-mask FiLM
reference. New mean targets use the same source nodes and adjacency. Outlier
thresholds/medians and the new baseline are fitted on the outer-training fold,
then frozen across its configurations. Inner model selection conditions on this
outer-training preprocessing; it is not a fully nested refit of the baseline.
Joint inputs use the exact same two scalar baselines as individual-target fits.
Target 0 is Gaussian K; target 1 is signed H=(k1+k2)/2. The source exporter averages
both over cell patches. H_patch² >= K_patch is therefore not a mandatory
geometric constraint, even before outlier cleanup; evaluation compares its
observed/predicted diagnostic without imposing it.

New bundles save `baselines_by_target`, `preprocessing_by_target`, physical
residual graphs and per-target offsets. `predict_sample` always returns a
nodes-by-targets array and does not read target values. `contributions` provides
an exact physical-unit decomposition before and after propagation. The notebook
uses explicit target-aware evaluation because the existing generic scalar
`prediction_table` is appropriate only for the Gaussian FiLM reference.

## Historical implementation and checkpoint compatibility

The following documents the earlier `CoupledFateCurvature` experiment. Its
measured-neighbor variants, per-ring response families and optional response
screen are retained for restoring old checkpoints; they are **not** run by the
current notebooks. The additive spline models are archived under
`legacy/fate_interactions/`.

### Prediction equation and inputs

For one organoid, let `local` be the vector of local fate-based curvature
preferences, `W` the one-hop neighbor-averaging matrix, and `gamma(N)` a scalar
coupling strength shared by its cells. Predictions solve

```
(I - gamma W) prediction = (1 - gamma) local
```

`W` uses unique undirected edges, discards self-loops/duplicates, and has rows
summing to one. An isolated cell has `W_ii=1`, so it retains its local preference.
A sparse direct solve gives the equilibrium. Measured target curvature is never
an input to the predicted-neighbor solve. An explicitly separate
`neighbor_curvature="measured"` mode instead evaluates
`prediction = (1-gamma) local + gamma W observed`. This is conditional
reconstruction using neighboring target values, not a fate-only prediction.
Self-loops are excluded from measured input; isolated cells use `local`. Fitted coupling is bounded by `0 < gamma < gamma_max < 1`;
uncoupled controls set it exactly to zero. A uniform local preference is preserved.
The model smooths a scalar curvature field, not the surface geometry itself; it
is not Helfrich mechanics and gamma is not a bending modulus. Nonnegative local
preferences cannot generate negative predictions by positive averaging alone.

Local preferences comprise an identity-specific coefficient `a` and optional
ordered center/source terms in exact shortest-hop rings. Inputs are exclusive
fates (including Unassigned), source fractions in rings, adjacency and organoid N.
There is no shared source `c` term, MLP, learned hidden embedding, absolute ring
count predictor, or global geometry head. Topology does affect propagation.
Even radius-zero local preferences have global component reach when coupled.

All local coefficients use cubic B-splines in log N. Coupling uses the same
basis followed by `gamma_max * sigmoid(logit_spline)`. Both clamp N outside the
training range. The reported `depth` catalog field is the *local pair radius*,
not the effective propagation distance; `effective_reach` is saved separately.

### Pair responses and interpretation

Default responses are linear fractions. `pair_responses` stores JSON-compatible
ordered center/source/hop overrides, using names in the notebook and integer
indices in the model constructor. Implemented families are:

- `linear`: fraction p.
- `presence`: 1 if p>0, otherwise 0.
- `hill`: normalized p/(p+half), zero at p=0 and one at p=1.
- `threshold`: a normalized logistic transition with specified threshold/width.

Adding response families is confined to the validated `pair_response` function;
model fitting, serialization and response readout reuse the same interface.
Response shape parameters are specified, not silently learned. Marker edits
recompute fractions and graph predictions; there is no target-dependent cache.

Center and source contrast bases constrain each pair coefficient table to have
training-weighted zero row and column means, distinguishing it from unconstrained
center/shared linear effects. Reference weights give organoids equal weight.
Each response is additionally centered by its average among that center identity's
nonempty training rings, averaging within each organoid before across organoids.
These reference responses are fixed across N and refitted only on fitting data.
Empty rings contribute zero. With different response families across pairs,
zero-sum constraints concern coefficients, not a zero average of different
response functions. Treat these as constrained response coefficients, not causal
biological interaction energies.

`a` is a local preference under this reference convention, not mean observed
curvature or intrinsic curvature of an isolated fate. `P` changes local preference;
its eventual effect spreads through the graph. `coefficients(N)` returns a, P and
gamma; `contributions(sample)` returns both local and propagated decompositions,
which sum exactly to the final prediction. Different reference conventions across
folds make selected-fold coefficient inspection preferable to blindly averaging.
A one-fraction plot is a component readout; actual fate replacements change
multiple fractions and must respect their sum-to-one constraint.

### Fitting and saved data

`physical_fate_fold` restores the saved exclusive observed input bundle, inverses
its target transform and preserves its baseline residualization. The copied
baseline prediction must be constant within an organoid. Historical offsets
computed by float32 subtraction can contain rounding noise; the loader uses the
saved constant prediction and adjusts residual targets to preserve exactly the
restored physical targets. Disagreements beyond rounding tolerance are rejected.
Such offsets commute with
normalized propagation, so residual coupling plus that baseline is equivalent
to physical-curvature coupling with the baseline included in local preference.
No filtering, outlier cleanup, outer resplitting or baseline fitting is repeated.

For fixed gamma, the fate coefficients are fitted exactly by penalized,
equal-organoid least squares using the propagated design. Preconditioned BFGS optimizes
the coupling logits with an envelope gradient through the sparse solve (or
through the direct observed-neighbor predictor). The complete objective, including
penalties, is divided by mean squared training targets during optimization; this
leaves its minimizer unchanged. The initial inverse Hessian accounts for spline
smoothness stiffness. Convergence uses the scaled gradient, not a small function
change that can prematurely freeze an N-dependent curve. The sigmoid keeps
coupling bounded; arbitrary logit bounds are not needed. Multiple
initial coupling values are selected by training objective within each penalty
trial. Inner organoid holdouts select penalties; the selected model refits all
outer-training organoids. Failed inner trials are recorded and excluded by
default. A nonconverged final optimizer does not produce a completed saved model.

The fitting procedure can be more expensive than the archived spline regression:
each coupling optimizer evaluation solves graph systems for the fitting organoids.
Start with the center stages when checking settings/performance. Progress records
identify the fold, model and coupling-optimizer stage; epochs do not apply.

Outputs use `training_results/coupled_fate_training/<tag>_<timestamp>/`:
`settings.json`, `splits.json`, `cohort/`, `inputs/`, `models/`, `models.json`,
`penalty_trials.csv`, `training_progress.csv`, `validation_mse.csv`, and
`complete.json`. Input bundles include the copied baseline and identity target
transform. Model bundles include optimizer status, selected penalties, inner
membership and fitting IDs. `AnalysisRun.select` restores them independently.
A constant residual variance exists solely for the common inference interface;
it is not a calibrated uncertainty estimate.

### Evaluation

FiLM is an accuracy reference with different reach and its original training
objective/early-stopping protocol. The evaluation verifies outer membership,
validation fates/edges and physical targets. All regional categories are exported,
including unqualified crypt territories and no detected crypt. Regional errors,
weighted by cell counts within each organoid, must reconstruct total MSE exactly.
Figures show organoid SEM, conditional on fitted models, not retraining uncertainty.

For predicted-neighbor models, measured versus predicted one-hop curvature means
are compared only after held-out predictions are complete. The new measured-neighbor
models intentionally use neighboring validation curvature, so their lower error
does not demonstrate better fate-only generalization. Their scores are labeled
separately in plots. All such targets retain the reference run’s target cleanup.

### Optional biological response screen

The training notebook can trigger a bounded exploratory follow-up when the linear
predicted-neighbor pair model remains more than 2% above the saved depth-2 FiLM
MSE. For LGR5 centers, the Lysozyme and Serotonin responses in both exact rings
are changed to either presence or Hill saturation (scale fraction 0.1); all other
pairs remain linear. Linear is also a candidate. Shapes are screened using the
same inner organoid holdouts in uncoupled models, then the selected shape is
refitted with coupling. This saves expensive coupled optimization during shape
screening; it does not exhaustively optimize shapes jointly with coupling.

Both uncoupled candidates and the selected coupled model are saved. When linear
wins, the existing linear coupled checkpoint is reused. `response_search/`
contains settings, inner scores, choices, penalty trials and completion status.
The outer results only trigger the exploratory extension, never choose shapes
or penalties; another cohort is needed for confirmatory claims about its gains.
Presence/saturation reflect hypotheses about niche support, not established
curvature laws. Paneth support of LGR5 stem cells is supported by
[Sato et al. (2011)](https://www.nature.com/articles/nature09637); the parallel
Serotonin hypothesis comes from the user's observations.

