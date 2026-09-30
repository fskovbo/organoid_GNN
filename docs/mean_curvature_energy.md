# Mean-curvature preference and accommodation

The new notebooks are `experiments/training/mean_curvature_energy_training.ipynb`
and `experiments/benchmarks/mean_curvature_energy_evaluation.ipynb`. They reuse
saved mean-target preprocessing, global baselines and outer folds from the
previous coupled-fate experiment. Older models and notebooks are unchanged.

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
