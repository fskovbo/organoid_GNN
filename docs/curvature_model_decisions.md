# Curvature model: scientific decisions, evidence and working specification

Last updated: 2026-10-02. Read this document before modifying the interpretable
curvature model. This is a scientific decision record, not a claim that the
parameters have already been identified as physical material constants.

## Priority order and authorization

The user's central objective is to distinguish approximate contributions from
cell-identity preferences, local fate-dependent signalling, shared mechanical
accommodation, and global organoid conditions. Physically/biologically meaningful
proxies, stability across folds, and resistance to parameter compensation take
priority over small improvements in prediction MSE. Prefer a simpler model with
some accuracy loss to an expressive model with ambiguous parameters.

## Current workflow: conditional mean-curvature allocation (2026-10-02)

The active model predicts mean-subtracted curvature, with optional division by each organoid's measured population SD. Mean+SD is the preferred analysis; mean-only remains an explicit matched control. This intentionally conditions validation on measured global moments, without feeding local targets or moments into the predictor.

`h = (I + gamma L)^(-1) [u - mean(u)]`, with `u = a_center + sum_AB beta_AB presence(B within the configured radius)`.

- Constant coefficients and fixed gamma; no learned activation, N-dependence or training-reference interaction centering.
- Default fixed gamma is 0.5. Scan models use a separately exposed gamma grid. Pair shrinkage is selected on a training-only inner holdout; all models then refit the outer training set.
- All-to-all presence is the main current variant. Also expose center-only (radius 0, accommodation retained), radius 1/2, restricted source/recipient panels, arbitrary ordered pairs, and plain GIN. Radius 2 means presence anywhere in the two-hop neighborhood, counted once.
- Fresh-data settings are fully exposed: exclusive fates, day4/day4p5/day4p5-more, q<0.93 ordinary cohort, five matched folds. Spherical-like organoids are external validation only. Target cleanup is training-fitted; the SD floor uses only the inner-training subset.
- The measured organoid mean is the reference for current MSE; older fitted global baselines are not part of this conditional target. Both normalization variants reconstruct physical predictions for fair error comparisons.
- Overall MSE covers ordinary validation; spherical-like and ordinary no-detected-crypt share a regional panel. No separate spherical plot or unqualified-boundary panel. Fold traces/points and mean/SEM are shown.
- Center coefficients have an arbitrary common offset after output projection; pair amplitudes are conditional local preferences, not identified causal interactions. Measured SD rescales their physical effect between organoids.

Active notebooks: `energy_model_training` / `energy_model_evaluation` for accommodation scans; `shape_conditioned_energy_training` / `shape_conditioned_energy_evaluation` for fixed-gamma model/normalization comparisons; `energy_model_inspection` for one configuration's coefficients and artificial tissue.

Artificial tissue is a periodic triangular lattice (six neighbors per node), with one Agr2/Serotonin/Lysozyme/Chroma in KI67/LGR5/Unassigned. Keep adjacency fixed between insertion and uniform control. Display actual mean-centered predictions, unprojected local and accommodated fields versus the inserted type's coefficient, and an exact decomposition into center preference, changed pair contributions and uniform projection offset. A larger lattice verifies local finite-size convergence. Do not call the uniform centering offset mechanical propagation. There is no synthetic measured SD or physical shape ground truth.

Older notebook versions were archived under `legacy/energy_models/pre_standardization_streamline/`. Saved historical runs are preserved. The historical development and evidence below explain why these choices were made; they do not override this current contract.

## Historical specification before observed-moment standardization

The primary simplified specification is a fresh-cohort, uncentered joint model with
constant center preferences and pair amplitudes, **one-hop literal presence**,
and accommodation fixed at each point of an explicit grid. Sources are Agr2,
Chroma, Lysozyme and Serotonin. Recipients are their complement: AldoB, KI67,
LGR5 and Unassigned in the current dataset. Source cells have no incoming direct
pair terms, but every cell has a preferred curvature and participates mechanically.
No weighted-zero constraints, learned activation thresholds, N-dependent local
parameters or measured neighboring targets are authorized in this experiment.

All settings are exposed in `experiments/training/energy_model_training.ipynb`;
no reference run supplies data settings or baseline models. Defaults use mean H,
exclusive fates, day4/day4p5/day4p5-more, sphericity q<0.93, and n_folds=5. Four
older training notebooks were archived under `legacy/energy_models/`. Existing
models, result folders and historical evaluation notebooks are preserved.

Otherwise eligible organoids with q>=0.93 form **separate validation only**.
They never fit cleanup thresholds, scaling, the baseline or local coefficients,
and never select gamma or regularization. Earlier suggestions of using spherical
organoids as a calibration set are superseded by the user's latest instruction.
Exclude earlier timepoints, blacklist/quality failures and invalid sphericity
metadata from both cohorts. The ordinary and spherical sets share one timepoint
setting; only the sphericity decision separates them.

Reporting merges spherical validation with ordinary no_detected_crypt into
`spherical_or_no_detected_crypt`. Unqualified boundary cells are not a named
regional panel; they remain in total MSE and annotations as `excluded_boundary`.
Ordinary total MSE excludes the spherical cohort; an extra spherical-only MSE
panel/table remains available. Source annotation labels are also saved. Parameter
curves plot each fold thin and the fold mean thick. "Native cell curvature" is
labeled **preferred residual curvature** because the global baseline is subtracted.

## Experimental context and constraints

Graphs encode approximate cell adjacency. Binary marker vectors can coexpress;
current interpretable models use the established sequential exclusive-fate rules.
Unassigned is an observed identity, not a missing-information token. Zeroing a
marker changes cell identity and must not be described as removing all biological
contribution. Earlier replacement and trained-mask experiments were introduced
for this reason. Masking, replacement and conversion to Unassigned estimate
different model contrasts; none is automatically a causal biological ablation.

Organoids are destructive snapshots. N is cell count and an imperfect pseudotime,
not longitudinal follow-up. The user's hypotheses include Paneth/Lysozyme and
Serotonin niche support, possible redundancy, Agr2/Chroma progenitor-to-successor
changes, and KI67 transit-amplifying cells near mature necks. Serotonin constriction
is observed qualitatively but not measured sufficiently to calibrate mechanics.
These observations motivate hypotheses; Serotonin is not thereby established as
a Wnt-secreting cell in this dataset.

Crypt detection misses immature/small crypts. Current regions use s<0.75 for
crypt interiors, split by LGR5 presence in the assigned interior; 0.75<=s<1.1 is
neck only with a qualifying flat/minimum circumference profile; s>=1.1 is villus.
Unqualified boundaries, missing annotations and no-detected-crypt stay explicit.
Region summaries have different organoid memberships. Regional cell-weighted
errors reconstruct each organoid's total, but region-level organoid means cannot
be averaged directly to obtain overall MSE.

No cell-patch area weighting, lumen volume, measured neighboring curvature,
Gauss-Bonnet constraint, or exact surface Helfrich energy is assumed. Targets
are patch-averaged curvature: target 0 is Gaussian K and target 1 mean H.
Patch averages do not require H_patch^2>=K_patch. The main current target is H.

The saved global baseline uses organoid-level geometry and size. New fate models
have no geometry head and use fractions rather than ring counts, but baseline
geometry and graph connectivity still enter the complete prediction. Do not call
this literally topology-independent or geometry-free prediction.

## Model evolution and observations

Historical statements below summarize saved reports; they are exploratory and
condition on the fitted datasets. Do not compare numerical MSE across K and H.

1. **Additive/spline fate models and fraction nonlinearities.** The user found
   ring-composition MLPs close to depth-2 GNN performance, motivating models based
   on fate fractions rather than ring populations. Quadratic/cubic additions
   improved prediction but complicated interpretation. Those models/notebooks
   were archived under `legacy/fate_interactions/` when predicted curvature
   coupling became the main approach. Do not restore morphology shortcuts merely
   to improve MSE.
2. **Predicted versus measured neighboring curvature.** The [20260925 run](../training_results/coupled_fate_training/predicted_measured_bfgs_20260925_184706/REPORT.md)
   found Gaussian MSE 0.0021763 for linear predicted coupling versus 0.0020679
   FiLM. Measured-neighbor pair MSE was 0.0001861, using additional target
   information; an unfitted measured-neighbor mean did still better. This is
   reconstruction, not a superior fate-only model. The user explicitly deferred
   measured-neighbor models. LGR5 niche saturation improved a local subgroup more
   than global MSE, motivating activation tests. Premature optimizer stopping was
   corrected using scaling/preconditioning and gradient checks.
3. **Activation, N and curvature targets.** The [20260928 run](../training_results/coupled_fate_training/activation_curvatures_20260928_170831/REPORT.md)
   fit 120 models on five saved folds. Learned propagated Gaussian MSE 0.0020779
   approached FiLM 0.0020679. Joint K/H fitting shared activation parameters only,
   not a geometric energy; it gave little accuracy gain. For one main model,
   46/320 saturation estimates approached a bound and 11/64 pair estimates varied
   >10-fold across folds. Accurate predictions did not establish stable thresholds.
4. **Equal-cell mean-curvature energy.** The [20260929 run](../training_results/mean_curvature_energy_training/logN_signed_half_hop_20260929_145842/REPORT.md)
   replaced degree-weighted averaging by an unnormalized-Laplacian preference
   energy. Mean MSE was 0.0037916 versus matching mean-FiLM 0.0037764. Accommodation
   improved MSE about 3.1% versus the learned local model. Hop 2 had exactly half
   hop-1 absolute amplitude, initially allowing opposite signs. Signs disagreed
   across folds for 21/64 pairs; 26/64 saturation ranges exceeded tenfold. Large
   or opposing hop effects can encode anatomical position rather than signalling.
5. **Same-sign, pooled distance exposure.** The [fraction run](../training_results/pooled_interaction_training/centered_vs_uncentered_20260930_112935/REPORT.md)
   combined ring fractions before activation with fixed weights 1 and 0.5.
   No learned range or independent hop sign remained. Centered/uncentered mean
   MSEs were 0.0038461/0.0038171. Median relative fold dispersion was 0.238/0.148,
   versus 0.308 for the earlier signed-hop model on that diagnostic cohort.
   Uncentered saturation still hit the upper bound in 77/320 estimates. An imposed
   same-sign distance rule is not empirical evidence of stable biological signs.
6. **Counts reference.** The [count run](../training_results/pooled_interaction_training/counts_centered_vs_uncentered_20260930_121245/REPORT.md)
   used (n_ring1+0.5*n_ring2)/1.5 without population normalization. Centered and
   uncentered MSEs were 0.0039064/0.0038870. On common count/fraction contexts,
   median relative dispersions were 0.256/0.158, compared with 0.238/0.148 for
   fractions. Uncentered count/fraction saturation estimates varied >10-fold for
   19/64 and 10/64 pairs. Counts expose neighborhood size implicitly. Keeping
   the same numerical activation priors and penalties is not scale-invariant.
   Worse count MSE is not proof that biology senses fractions instead of counts.

All recent comparisons use 1,170 held-out organoids across five saved folds,
rooted in `fate_masking_training/Ndependent_20260922_115100` zero-mask data. The
matching H baseline/preprocessing and H FiLM are saved in the 20260929 mean-energy
run. Reuse those exact inputs and splits; do not refit baselines for this experiment.
FiLM is an accuracy reference: its training loss and effective receptive field
are different from an equilibrium model with component-wide accommodation.

## What the recent stability comparisons do and do not establish

The matched diagnostic responses are pair contributions before accommodation,
relative to zero source exposure, evaluated on common fate-only contexts. They
are not actual fate replacements. Marginal common support does not establish
joint support for every intervention. Folds overlap in training membership, so
agreement is descriptive, not independent replication. A low relative dispersion
must be read with response magnitude and absolute disagreement.

LGR5-centered Lysozyme and Serotonin contributions increased with N in all four
count/fraction, centered/uncentered formulations and all five folds. Neighbor
LGR5 also increased, but its absolute sign/magnitude changed more with centering.
KI67 decreased consistently in count models but lacked a stable N trend in the
uncentered fraction model. These are model associations, not evidence that Wnt
secretion or physical attraction changes with N. Log-linear shapes were imposed.

A particularly important contradiction: accommodation decreased with N in every
centered fit but increased in every uncentered fit, for both input representations.
For fractions, fold-average lambda was approximately 0.39 to 0.29 (centered)
versus 0.20 to 0.40 (uncentered) from N~112 to N~947. This shows formulation-level
compensation despite within-formulation fold agreement. Do not interpret those
slopes as developmental tissue stiffening/softening.

## Centering, N dependence and coefficient constraints

For center A and source B, let F_AB(x) be activation and mu_AB its training-only
conditional mean (equal organoid weighting). Centered local preference is

    u_i = a_A + sum_B beta_AB(N) [F_AB(x_iB) - mu_AB].

Uncentered preference is the same equation with mu=0. With constant beta and
fixed activation the exact re-expression is

    a_uncentered,A = a_centered,A - sum_B beta_AB mu_AB.

This preserves u, accommodated predictions, beta and lambda. Do not train both
as separate biological hypotheses in the simplified experiment. If coefficients
are penalized, transform the penalty too before claiming equivalent fits.
Centering may be used internally as conditioning, but must not alter the intended
objective. The new experiment fits the uncentered objective explicitly.

With beta=b+d*log(N/N_ref) and a constant, that shift requires an N-dependent
center term. The previous centered/uncentered fits were different restricted
families, not mere relabelings. Mu itself was not a free coefficient, but changed
with learned activation. It averaged across N, not separately at every N.

The historical pair table used weighted-zero rows and columns. These are not
physical conservation laws. For exhaustive *linear fractions*, sum_B x_B=1
(for nonempty rings), so a row offset in beta can be absorbed into a. A row
constraint can then select a reference. A column constraint on the entire table
excludes a shared source effect unless it is modeled elsewhere. Pair-specific
nonlinear activations, counts, and restricted N terms need not retain the simple
reference equivalence. No rule requires positive responses of some receivers
to be balanced by negative responses of other receivers.

The simplified model uses independent amplitudes for a limited source panel,
with all other explicit source effects zero. It has no weighted-zero row/column
constraints. This is an explicit restricted-interaction hypothesis. Avoid the
full linear-fraction intercept ambiguity by not modeling all exhaustive source
identities simultaneously; assess actual design rank, support and collinearity.
Do not pretend ridge regularization or dropping a column identifies an absolute
physical interaction when the data only identify contrasts.

The center coefficient is a residual preference. Since the complete prediction
restores baseline b_g, a constant a_A corresponds to b_g+a_A in absolute preferred
curvature. It is not a directly measured isolated-cell curvature. Zero modeled
sources is a meaningful reference only where sufficiently represented in the
data; even there, omitted biological processes still exist. Report source-free
support per center. Fitting a center-only model first and freezing its a absorbs
average interaction effects into a and imposes, rather than discovers, a split.

## Earlier small joint model (four sources, all recipients)

For each source B, x_iB=(p_iB,1+0.5*p_iB,2)/1.5, where p is its fraction in an
exact hop ring; an empty ring contributes zero. Exclude the center from both rings.
All identities contribute to denominators, including excluded interaction sources.

    u_i = a_identity(i) + sum_{B in selected sources} beta_identity(i),B F(x_iB)
    E(h) = 0.5 sum_i (h_i-u_i)^2 + 0.5 lambda sum_{unordered edges} (h_i-h_j)^2
    (I+lambda L) h = u
    H_prediction,i = saved_baseline_g + h_i

Fit two prespecified activation choices: F(x)=x and F(x)=tanh(3x)/tanh(3).
The value 3 is a fixed model assumption, not a learned biochemical threshold.
All a, beta and lambda are independent of N. Lambda is nonnegative and shared by
all cells/identities. Neither a nor beta is sign constrained. Distance weights
are fixed and positive: both rings have the same marginal effect sign per pair.
There is no shared-source coefficient in addition to beta; beta may itself be
positive for every receiver. No learned hidden representation is used.

The only nonlinear fitted parameter is scalar accommodation. At each trial
lambda, jointly solve penalized equal-organoid least squares for a and beta.
Use training-only inner membership for penalty selection, then refit all outer
training data. Keep outer validation untouched for selection. Saved preprocessing
is outer-training-fitted, so inner selection is conditional on it, not fully nested.
Use multiple scalar initialization strengths and inspect the profile objective.

The baseline may remain N-dependent. Neighborhood compositions and graph
connectivity also vary with N, so constant local laws do not imply constant
observed curvature. Conversely, sweeping supplied N on a fixed prepared graph
must leave the residual model prediction unchanged. Baseline-restoration changes
are a separate operation and must not be described as interaction sensitivity.

## Required evaluation and safeguards

Focus plots and interpretation on the simplified model, with previous models
only as compact accuracy references. Evaluate both fixed activations without
selecting a winner using outer validation. Include:

- Exact saved memberships/preprocessing, complete model/baseline bundles, settings,
  source snapshots, organoid-weighted MSE/SEM, and region/size summaries.
- All-fold center preferences, amplitudes, accommodation and response curves on
  common training-support exposure intervals. Fixed-alpha agreement is imposed,
  not learned threshold stability. Flag rare pairs and nearly zero responses.
- Source-free center support and activation/exposure diversity. Suppressed sources
  must not silently change the fractions used by included sources.
- Accommodation profiles: fix lambda over a declared range, refit all linear
  coefficients on outer training data at the saved penalty, and quantify the
  loss increase and compensating coefficient changes. Validation profiles are
  diagnostics only and must not select a new fit. A declared loss-tolerance
  interval is a sensitivity measure, not a confidence interval.
- Conditional linear-design rank/collinearity before ridge, plus penalty sensitivity.
  An inverse-Gram correlation is a design diagnostic, not an empirical biological
  parameter correlation or a calibrated covariance estimate.
- Numerical CPU/CUDA/gradient agreement, checkpoint round trips, constant-N
  invariance, structural zeros, no forced sign compensation, and exact reference
  re-expression. Preserve old checkpoint predictions and user edits.

Do not claim that convergence, narrow bootstrap MSE intervals, strong prediction,
regularization, or five overlapping folds establish mechanistic identifiability.
If compensation remains, report it and simplify or calibrate externally; do not
silently add flexibility to improve MSE.

## Biological grounding from the discussion

These papers motivate caution, not fixed quantitative coefficients:

- [Farin et al., 2016](https://pubmed.ncbi.nlm.nih.gov/26863187/): Wnt3 transfer,
  membrane binding, receptor regulation and division-dependent dilution complicate
  a simple secretion-plus-diffusion interpretation.
- [Matsu-Ura et al., 2016](https://doi.org/10.1016/j.molcel.2016.10.015): temporal
  Paneth-derived Wnt signalling does not establish a monotonic secretion-versus-N law.
- [Yang et al., 2021](https://www.nature.com/articles/s41556-021-00700-2): apical/basal
  tension and lumen changes coordinate morphogenesis; none directly calibrates
  the scalar graph accommodation ratio.
- [Pentinmikko et al., 2022](https://pmc.ncbi.nlm.nih.gov/articles/PMC9565803/): cell
  shape can influence niche-signal reception, complicating a strict separation
  between intrinsic shape and signalling.

Future additions should identify which unresolved observation requires them,
which parameter could compensate, and how that ambiguity will be tested.

## Completed simplified experiment (2026-09-30)

The [constant-source run](../training_results/simple_fate_energy_training/constant_secretory_sources_20260930_144417/REPORT.md)
implemented the proposed four-source panel (Agr2, Chroma, Lysozyme, Serotonin)
with all eight receiver identities. Ten models completed: linear and fixed
saturation on the same five saved folds. Each has 41 fitted parameters: eight
center preferences, 32 independent pair amplitudes and one accommodation scalar.
All 60 initialization/penalty/refit attempts converged. No existing scientific
model was overwritten and the historical contrast-basis defaults remain readable.

All 32 pairs have supported positive-exposure response curves. Their source-free
reference is assessed separately on all nodes: even the least represented
center type has source-free centers in at least 82 training organoids per fold.
The exposure-curve quantiles condition on source presence; using unconditional
quantiles initially hid rare interactions because their upper quantile was zero.
The corrected definition and a minimum sampled-positive-organoid requirement
are explicit in the executed evaluation notebook and tables.

Median relative fold disagreement of supported response increments is 0.110
(linear) and 0.122 (fixed saturation); 27/32 and 28/32 signs agree across all
folds. Accommodation is about 0.136 and 0.137, with fold SD about 0.003. These
metrics should not be compared directly with old full-panel N-sweep summaries:
the new source panel, support reference and response contrast are different.

All conditional propagated designs have full rank 40/40. Largest absolute
inverse-Gram correlations are about 0.69–0.70 (linear) and 0.79–0.80 (saturated).
Nevertheless, at least a twofold accommodation change can be compensated within
a declared 1% penalized-objective tolerance in all five folds of both models.
For fourfold accommodation, mean center-coefficient RMS shifts are about 0.045
and 0.031, with objective increases of about 0.67% and 0.79%, respectively.
The tested profile range is finite and the tolerance is not a confidence interval.
This is practical compensation, not proof of an exact rank deficiency or of a
statistically unidentifiable parameter. Do not mistake narrow fold dispersion
for a uniquely determined mechanical decomposition.

Mean-curvature MSE is 0.0040872 (linear) and 0.0040509 (saturated), versus the
matching FiLM's 0.0037764. Do not add complexity merely to close this gap. A
noiseless planted-model test recovers known preferences, amplitudes and lambda,
which supports numerical correctness under the assumed model, not biological
identifiability in real data. The current source panel and fixed response shapes
remain hypotheses, and the meaning of a stays baseline-relative.

Active notebooks for this experiment:
- `legacy/energy_models/notebooks/training/simple_fate_energy_training.ipynb`
- `legacy/energy_models/notebooks/benchmarks/simple_fate_energy_evaluation.ipynb`

Next work should address compensation or external calibration before claiming
that the fitted accommodation and local preferences independently measure
mechanics and intrinsic cell shape. No further complexity is authorized by this
record alone.

## All-source, one-/two-hop presence experiment (2026-09-30)

The new run is `training_results/simple_fate_energy_training/all_sources_linear_presence_20260930_153914`.
It preserves all earlier run folders, reuses the exact five H folds/preprocessing
and baselines, and fits only models retaining both interactions and accommodation.
The grid is linear/presence × radius 1/2; global families are prespecified, not
selected using outer validation. There are 65 fitted parameters for linear with
reference coding and 73 for presence (including one accommodation parameter).

Evaluation reports actual one-cell B→Unassigned replacements within an observed
ring, preserving center identity and ring populations. The sign convention is
intact minus replaced **local preferred residual H**, before accommodation.
Removing a B cell need not remove B presence; adding Unassigned can itself change
an indicator. These are conditional model associations, not causal ablations.
Primary contrasts use identical fate-only reference contexts for every fold,
with equal organoid weights. Absence of a zero-source observation is not repaired
by this replacement contrast; center preferences retain their reference caveat.

Accommodation profiles refit all local coefficients at fixed lambda multipliers;
penalty sensitivity refits them at fixed fitted lambda. Neither is an uncertainty
interval or a model-selection step. Novelty-stratified validation measures error
versus fate-composition distance to outer-training organoids; it is **not** a
composition-blocked training experiment. Edge-difference errors use measured
curvature only for evaluation, never as a prediction feature.

See the new run's REPORT.md and executed simple_fate_energy notebooks for numerical
findings. Maintain the priority: robust response contrasts and limited compensation
before extra flexibility. Do not present tighter fold agreement alone as physical
identification of accommodation or of source amplitudes.

### Results of the all-source presence comparison

All 20 models converged (120 successful optimization attempts). At radius 2,
presence MSE is 0.003869 versus 0.003980 for linear; the saved FiLM reference is
0.003776. One-hop values are 0.003953 (presence) and 0.004030 (linear).
Accuracy is still secondary. Fold-only median relative disagreement of supported,
non-negligible hop-1 replacement means is 15%/12% for linear radius 1/2 and
19%/38% for presence radius 1/2. Presence nevertheless has more robust nonzero
replacement signs across the specific tested penalty/accommodation settings:
35/55 and 37/55 versus 27/55 and 24/55. Do not conflate these diagnostics or
claim one response family is universally more identifiable.

All conditional designs have full rank after the linear reference convention,
but maximum inverse-Gram correlations reach ~0.996 for linear and ~0.9997 for
two-hop presence. AldoB preference versus neighboring AldoB presence is a leading
confounding pair. Every fold/variant allows a twofold accommodation change within
the descriptive 1% penalized-objective band after refitting local coefficients.
Source expansion has not solved physical identification; fitted lambda also
changes from ~0.14 in the restricted model to ~0.34–0.50 in the new specifications.

For LGR5 centers, positive Chroma and Serotonin replacement contrasts survive all
four specifications and the tested sensitivity settings. Lysozyme is positive
across primary folds for linear and one-hop presence, but two-hop presence crosses
zero across folds; both presence variants lose sign robustness in sensitivity
fits. A one-cell replacement often leaves another Lysozyme in the two-hop region,
so loss of a cell is not loss of presence. These remain conditional model
associations, not established molecular signaling mechanisms.

The implementation accepts fixed per-pair choices without learning thresholds;
automatic pair selection was not run. Check rank and presence/absence support
before interpreting any custom mixture, especially rows with only linear terms.
The original checkpoints reproduce predictions to <6e-13 in the saved regression
audit. Thirty-two affected-model tests passed. Earlier run folders are preserved.

## Fresh fixed-accommodation workflow (2026-10-02; completed)

The model is u_i=a_A+1[A not in S] sum_{B in S} beta_AB 1[B at hop 1],
followed by (I+gamma L)h=u and restoration of the fitted organoid baseline.
With the current eight identities it has 8 center preferences and 16 amplitudes.
Gamma is saved as an exact nonnegative constructor setting, including zero;
there is no nonlinear accommodation search. Local coefficients are profiled by
penalized least squares. An inner ordinary-training holdout selects pair shrinkage
at each grid point; every gamma checkpoint is saved, with no automatic selection
using either outer validation set. Inner selection is conditional on preprocessing
and baseline fitted on all outer-training organoids.

Initial single-holdout grid: gamma=[0,.01,.02,.05,.1,.2,.4,.8,1.6,3.2], center ridge=1e-4,
pair ridge candidates=[1e-4,1e-3]. All baseline, target cleanup and dataset options
are visible in the notebook. Training fits a new baseline per ordinary fold and
saves the raw cohort, exact splits, cleaned/prepared graphs, preprocessing,
baseline, coefficients, regional annotations and validation scores. Spherical
memberships are identical across folds and are not independent replicated data.

The user authorized training and a matched plain depth-1 GIN comparison. Completed
run: `training_results/energy_model_training/sources_to_other_presence_hop1_20261002_113114`.
There are 695 ordinary training, 174 ordinary validation and 324 separate spherical
validation organoids; one holdout, not a fold-stability experiment. The GIN has
no FiLM or global inputs and predicts the identical physical residuals using the
same saved baseline, preprocessing and memberships. Its epoch count is selected
on the same inner training holdout, followed by reinitialization and full-training
refit for 298 epochs. It uses equal-organoid MSE, no auxiliary edge loss, and its
variance output is unused. Energy accommodation extends beyond one hop, unlike
the GIN comparator's fate aggregation.

The ordinary-validation energy minimum is gamma=0.4 (MSE 0.004623), versus
GIN 0.004595 and baseline 0.005315. The minimum is broad: gamma=0.2 through 0.8
is within about 0.6% of its MSE. This does not independently identify a physical
accommodation strength. Energy is better in LGR5-containing crypts (0.008915 vs
GIN 0.009649), while GIN is better in necks and villus. In separate spherical
validation, baseline wins (0.001044 vs energy 0.001497 and GIN 0.001340).
Every tested energy gamma performs worse than the baseline on spheres. Keep this
as evidence of limited transfer of learned fate corrections, not proof that
spheres lack signaling interactions, and do not tune gamma on these organoids.

Artifacts include all 10 energy checkpoints, the GIN, baseline, exact inputs,
splits, predictions, regional scores, and coefficient profiles. Restored
predictions reproduce training scores. Single/folded synthetic end-to-end checks
also cover the GIN and excluded/spherical regional handling. Two older pooled
and simple-energy evaluation notebooks are archived in `legacy/energy_models/`.
No historical result folders were removed.


## Extended accommodation sweep across five folds (2026-10-02)

Authorized follow-up: energy only; no GIN retraining. The fresh training workflow
now defaults to five ordinary folds and gamma=[0,.1,.2,.4,.8,1.6,3.2,6.4,12.8,25.6].
All other scientific model settings and cohort filters are preserved. Run:
`training_results/energy_model_training/folded_presence_hop1_extended_gamma_20261002_120514`.
All 50 fixed-gamma energy checkpoints completed. Every one of the 869 ordinary
organoids is held out exactly once; all 324 spherical organoids remain separate
and are evaluated under each fold's model. The shared first fold reproduces the
prior run's scores exactly at common gamma values. All inner fits select pair
ridge=1e-4 from the unchanged candidate grid.

Evaluation adds whole-organoid identity/lineage composition and one-hop neighbor
composition around each source identity. Each original organoid contributes once,
not once per fold. Composition uses equal organoid weights. For source-centered
neighborhoods, fractions first average over non-isolated source centers within an
organoid, then over organoids; those without that source are excluded rather than
assigned zero. EXO=Agr2+Lysozyme and ENDO=Chroma+Serotonin are descriptive exclusive
marker groups. N/timepoint distributions are not matched, so cohort composition
contrasts must not be interpreted as controlled biological effects.

Coefficient plots show each fold thin/translucent and the mean thick. New tables
and heatmaps show sample fold SD, SD/max(RMS,1e-3), positive/negative/near-zero fold
counts and signed majority agreement at every gamma. The 1e-3 near-zero threshold
is a transparent descriptive convention in residual H units, not a statistical
significance test. Presence and absence each require at least 20 training
organoids in every fold for the supported-sign summary. Fold training overlaps;
spherical evaluations reuse exactly the same organoids. Fold agreement is not an
independent replication or a proof of mechanistic identification. Each fold fits
its own baseline, so preference dispersion includes baseline uncertainty.

### Findings of the five-fold extended sweep

Every fold has its lowest ordinary-validation MSE at gamma=0.4. The mean is
0.004471 versus baseline 0.005202. The minimum remains broad and does not precisely
identify a physical gamma. Spherical MSE at that gamma is 0.001433 versus baseline
0.001026; at gamma=25.6 it drops to 0.001178 but remains worse than baseline, while
ordinary MSE increases to 0.005069. Spheres do not select gamma.

At gamma=0.4 all 16 supported interactions have the same non-negligible sign across
five folds (tolerance 1e-3). Median relative fold SD/RMS is 0.0847 for interactions
and 0.0961 for center preferences. Seven of eight center preferences have unanimous
non-negligible signs; Agr2 is weak/near zero. Fixed-gamma stability does not remove
coefficient compensation across gamma. At gamma=25.6, AldoB receiving Chroma or
Serotonin has four positive and one negative amplitude across folds. All source
presence/absence pairs pass the declared organoid-support threshold.

Whole-organoid AldoB fractions average 0.7210 in spherical-like versus 0.3989 in
ordinary organoids. EXO fractions are 0.0345 vs 0.0568, ENDO 0.0034 vs 0.0076, and
LGR5 0.0675 vs 0.1320. Source-conditioned neighborhoods remain LGR5-rich: around
Lysozyme, 0.4325 vs 0.4519; around Serotonin, 0.5174 vs 0.5147 (spherical first).
These conditional figures use only 112/62 spherical organoids containing usable
Lysozyme/Serotonin centers, versus 746/567 ordinary organoids. Smaller source
prevalence therefore coexists with retained local LGR5 association; the data do
not justify equating spherical morphology with absence of these sources. N and
timepoint distributions remain potential confounders of the descriptive contrast.

The evaluation notebook saves per-organoid compositions, source support,
per-fold predictions, coefficient/sign tables, and a report. Three targeted
synthetic tests pass, including folded energy-only execution and sign/weighting
checks. All 50 saved checkpoints restore; shared fold-0 validation MSEs reproduce
the earlier single-split run exactly at common gamma values. No previous model
or result folder was replaced.


## All-to-all one-hop presence reference (2026-10-02)

At the user's request, test all eight exclusive identities (including Unassigned)
as both sources and recipients. This is a reference experiment, not evidence that
the restricted biological hypothesis should be replaced. There are 8 center
preferences and 64 ordered amplitudes, with no symmetry or weighted-zero rules.
Self-identity interactions refer to other adjacent cells, not the center itself.
Activation remains literal one-hop presence, uncentered, with constant local
parameters and the same fixed ten-point gamma grid. No model class implementation
was changed for this experiment; the existing explicit panel options were used.

Run: `training_results/energy_model_training/all_to_all_presence_hop1_matched_20261002_130854`.
All 50 models completed. The five fold input/baseline bundles are byte-for-byte
copies of `folded_presence_hop1_extended_gamma_20261002_120514`, validated against
the exposed data settings, raw graphs and exact splits before copying. Inner
membership also matches. All fits select pair ridge=1e-4 from the same candidate
grid. Spherical organoids remain strictly external; no GIN was trained.

Every fold's ordinary-validation minimum is now gamma=0.8, with fold-mean MSE
0.004313. At this SAME gamma, restricted MSE is 0.004495. The restricted model's
own minimum at gamma=0.4 is 0.004471. The shared baseline MSE is 0.005202.
Spherical MSE at gamma=0.8 is 0.001300 versus restricted 0.001423 and baseline
0.001026. At the largest gamma=25.6 it is 0.001108, still worse than baseline.
These are descriptive validation profiles, not separate tests of selected optima.

At gamma=0.8, 61/64 interaction amplitudes have unanimous non-negligible signs
across folds, using the same 1e-3 descriptive tolerance. All pairs pass the
20-organoid presence/absence support rule, but that does not establish joint
context support or independent parameter identification. Median relative fold
SD/RMS for all 64 pairs is 0.200. The shared 16 pairs have 16/16 unanimous signs
and median relative dispersion 0.10195, versus 0.10246 for the restricted model
at the same gamma. Thus predictive accuracy improves, while shared-pair stability
is almost unchanged, not demonstrably improved. At gamma=0.4 the expanded model
has only 14/16 unanimous shared signs versus restricted 16/16. These tests do not
justify treating the older all-source presence model's different replacement-
response dispersion as a numerically matched stability benchmark.

The three non-unanimous amplitude cases at gamma=0.8 are Lysozyme→Agr2,
AldoB→Chroma, and AldoB→Serotonin; the last has four positive and one near-zero
fold, not an actual negative estimate. Center signs agree in all eight identities
at this gamma, with median relative fold dispersion 0.09065. However, preferred
center coefficients now refer to no source identity being present. There are
ZERO source-free training organoids for every recipient in EVERY fold. These are
therefore extrapolated zero-neighbor intercepts, not directly supported estimates
of isolated-cell preferences. Their mean levels change greatly when adding the
other source identities (e.g. AldoB and LGR5), and apparent fold stability does
not repair this interpretational problem or remove gamma/preference compensation.

The main evaluation notebook shows matched MSE and stability comparisons, with
shared-pair denominators and the same gamma. Per-recipient amplitude panels group
source curves to keep 64 interactions readable. All checkpoints, copied inputs,
CSV comparisons and executed notebook snapshots are preserved. Workflow tests
cover matched all-to-all input copying, 72 coefficients, reference comparison and
synthetic fold/covariate support checks. Earlier checkpoints/results are untouched.

## 2026-10-02: matched all-to-all linear-fraction reference

User requested replacing one-hop presence with linear neighborhood fractions and
quantifying whether fold variance increases. Run:
`training_results/energy_model_training/all_to_all_linear_fraction_hop1_matched_20261002_133445`.
Reference: `all_to_all_presence_hop1_matched_20261002_130854`.
All 50 fits (five folds × ten fixed gamma values) completed successfully, with
finite coefficients and selected pair ridge=1e-4. Exact outer/inner memberships,
fold preprocessing and baseline bundles were reused; input bundles are byte-for-byte
identical. No model implementation changed: the existing linear activation option
was selected. All eight identities, including Unassigned, are sources and recipients;
72 coefficients, constant parameters, no response centering or weighted-zero constraints.

**A structural limitation matters more than the variance comparison.** For each
recipient A, local preference is `a_A + sum_B beta_AB*x_B`. On non-isolated cells
`sum_B x_B=1`, so subtracting any delta_A from a_A and adding it to every beta_AB
leaves predictions unchanged. All five unpenalized linear designs have rank 64/72,
compared with 72/72 for presence; there are no isolated training cells. Numerical
row-shift invariance was verified to 1.39e-17. Invertible mechanical accommodation
cannot remove the eight redundant directions. Ridge chooses a numerical convention,
not separately identified native curvatures and signaling amplitudes. Do not silently
add zero-row constraints and present the resulting coefficients as biological absolutes.

There is **no general variance increase**. Comparing sample variance across folds,
averaged over matched coefficients, at SAME gamma:

| Quantity | gamma=0.4: linear vs presence | gamma=0.8: linear vs presence |
| --- | ---: | ---: |
| Raw center coefficient variance | -42.1% | -37.4% |
| Raw pair coefficient variance | +15.0% | -8.9% |
| Reference-relative pair contrast variance | +9.6% | -12.1% |
| Final prediction variance on identical external spherical cells | -4.5% | -5.0% |

Reference-relative quantities use Unassigned: `a_A+beta_AU` and `beta_AB-beta_AU`.
These are invariant to the linear row ambiguity and describe pure-neighborhood
endpoints for either activation family. Pure-neighborhood support may be limited;
these are not causal effects or generic presence marginal effects in mixed neighborhoods.
At gamma=0.4, 49/56 linear contrasts versus 45/56 presence contrasts have unanimous
non-negligible signs (1e-3 tolerance). Raw linear amplitude signs themselves are
reference dependent. Changing from presence to fractions also changes exposure scale
and the effective regularization despite identical numerical penalties. Only five
overlapping training folds are available: these are descriptive variance changes,
not independent confidence statements.

Every linear fold has its ordinary-validation minimum at gamma=0.4 (mean MSE
0.004409), versus presence 0.004320 at the SAME gamma and 0.004313 at its own
minimum gamma=0.8. Baseline is 0.005202. At gamma=0.4, linear spherical MSE is
0.001326, versus presence 0.001323 and baseline 0.001026. Shared-sphere prediction
variance gives a useful common-context stability check, but the spherical cohort's
lower variance does not fix its worse MSE or validate extrapolation to all contexts.

Conclusion: linear fractions retain useful predictive associations and fairly stable
reference-relative contrasts. They do not improve identification of separate native
preferences and absolute pair amplitudes; in fact, all-to-all fractions make that
split exactly redundant. Given our priority of physical interpretability over small
MSE gains, this test gives no reason to prefer this parameterization over presence.
Full-rank presence is also not proof of mechanistic identification. Keep both runs,
reference-relative diagnostics, rank checks, and common-context prediction variance.
The executed training/evaluation notebooks, CSV tables, figures, audit and report
are saved with the linear run; existing results remain untouched.

## 2026-10-02: conditional allocation given measured mean and spread

User approved a new adjacent task: predict the spatial distribution of mean
curvature from exclusive fates while supplying the measured organoid mean and
variance. Implemented in `shape_conditioned_energy_training.ipynb` and
`shape_conditioned_energy_evaluation.ipynb`. This is explicitly conditional
reconstruction, not independent fate-only morphology prediction or causal
removal of developmental history. Curvature summaries can themselves contain
consequences of fate-dependent processes.

The target is z=(H-m_g)/max(s_g,s_floor), where s_g is the square root of the
population variance (ddof=0) across cells. The floor is 0.1 times the median SD
of the recorded INNER training organoids, bounded below by 1e-6. Means/SDs are
computed on the physical targets after the original training-fitted cleanup.
All cells are equally weighted; approximate cell areas are not used. Held-out
organoids intentionally supply their own mean and SD. Predicted variance is
not forced to match measured variance.

Use the five exact folds/inner splits and preprocessing of the all-to-all
presence reference. Every identity (including Unassigned) is a source and a
recipient, one-hop presence, constant coefficients, same accommodation grid.
Compare standardized center-only, standardized presence+accommodation, a
matched mean-subtraction-only presence model, and a standardized plain depth-1
GIN with no FiLM/global features. Keep previous global baselines as references,
not as the offsets defining the new targets. Select shrinkage, accommodation and
GIN epochs on inner training holdouts; never on outer/spherical validation.

Output projection is part of the fit: zhat=(I+gamma L)^(-1)*(u-mean(u)). GIN
outputs are centered per graph inside the differentiable loss and at inference.
This projection uses predicted values only, not measured target values. Mean
removal makes a common center-coefficient shift arbitrary; fitted coefficients
are relative allocation parameters, not isolated-cell native curvatures. The
new optional energy-model zero_mean_output flag defaults to False to preserve
historical checkpoints. Projection sums for float32 GIN output accumulate in
float64 to avoid mean drift. A numerical-debug GIN attempt is archived with the
new run; the corrected GIN fits were repeated, while float64 energy fits were
unchanged.

### User's next requested diagnostic — record now, execute later

Construct artificial graphs/lineage neighborhoods to isolate the response to
secretory cells in LGR5-rich and KI67-rich tissue. Plot field changes versus
hop distance, varying source identity, count, spacing and background composition.
Keep graph topology and externally specified mean/SD fixed between paired fate
interventions. Artificial graphs have no measured curvature summaries, so begin
with dimensionless predicted fields or explicitly chosen real-organoid scales.

Crucially, distinguish unprojected local-preference changes, their accommodated
field, and the final zero-mean projection. A local change causes a uniform
compensating offset under mean projection, even at gamma=0. This must not be
mistaken for long-range mechanical propagation. Test graph-size and boundary
sensitivity and compare folds at the SAME gamma. These are model-computation
diagnostics, not causal verification of secretory signaling.

### Conditional-allocation results

Completed run:
`training_results/shape_conditioned_energy_training/observed_mean_std_presence_20261002_144511`.
Five exact folds, 869 ordinary organoids and 324 external spherical organoids;
105 explicit-model fits and five GINs. All 110 checkpoints reproduce archived
predictions exactly. Data restoration, outer/inner membership, baseline weights
and training-only SD-floor computation were audited. CPU/CUDA, serialization,
normalization, differentiable projection and complete synthetic notebook workflow
checks pass alongside historical energy regression tests (19 tests total).

Using inner-selected gamma per fold, ordinary physical mean-curvature MSE:

| Model | MSE | Mean within-organoid R² |
| --- | ---: | ---: |
| Measured organoid mean only | 0.005131 | 0 |
| Standardized center identity only | 0.004425 | 0.114 |
| Presence energy, measured mean subtraction only | 0.004110 | 0.145 |
| Presence energy, measured mean and SD | 0.003943 | 0.192 |
| Plain depth-1 GIN, measured mean and SD | 0.004027 | 0.180 |

The normalized energy model improves physical MSE by 4.1% over the matched
mean-only control and 2.1% over this GIN, in every fold. Its improvement relative
to predicting the measured mean is 23.1% as a ratio of fold-averaged physical
MSEs, whereas averaging per-organoid explained fractions gives 19.2%; do not
conflate these statistics. Selected gamma is 0.8 for fold 0 and 0.4 for folds
1–4, identically for both target definitions. Its standardized coefficients are
scaled by the supplied SD at reconstruction, not absolute mechanical parameters.

In >=75% same-identity neighborhoods, normalization improves MSE over mean-only
fitting by 10.2% for LGR5 and 1.2% for KI67. Mean signed errors remain negative
(-0.02511 and -0.01617 physical units, respectively); lower MSE has not removed
systematic homogeneous-neighborhood underprediction. Figure 4 displays these
results, with organoid-weighted bins and support counts exported separately.

On the unseen spherical cohort, normalized energy MSE is 0.000967 versus measured
mean baseline 0.000947: 2.1% worse. It avoids much of the mean-only model's
extra variation (MSE 0.001174) but still does not reliably locate the small local
curvature variation there. This is a useful limitation of transfer across shape
states despite access to mean/SD. At common gamma=0.8, 56/64 pair amplitudes
have unanimous signs above a 0.01-standardized-unit tolerance; this is descriptive
stability, not mechanistic identification. The planned artificial-neighborhood
tests remain a next step, not something already performed in this run.

## Streamlined conditional workflow: completed execution

Accommodation scan: `training_results/energy_model_training/conditional_scan_20261002_162217` (110 fits). Fixed accommodation: `training_results/shape_conditioned_energy_training/fixed_gamma05_20261002_162517` (40 energy fits and 10 GIN fits). All five active training/analysis notebooks were executed; 21 focused tests passed. Exact physical targets, graph inputs, inner/outer memberships and normalization floors agree between the scan and fixed runs; matching gamma=0.5 one-hop coefficients also agree numerically.

Across the tested grid, both normalization modes attain their lowest mean ordinary-validation physical MSE at gamma=0.5. At fixed gamma, mean-only one-hop presence MSE is 0.004108; mean+SD is 0.003941; the matched depth-1 mean+SD GIN is 0.004027. Two-hop mean+SD presence reaches 0.003819, with the different direct interaction range explicitly recorded. These are descriptive outer-validation comparisons, not new inner-selection claims.

Single-model diagnostics default to one-hop mean+SD presence at gamma=0.5. Periodic 41×41 and 61×61 triangular lattices verify exactly six neighbors and convergence of the local unprojected response. Uniform tissue projects to zero. The insertion curves are exported alongside their exact center/pair/projection decomposition, so the small distant output-centering offset is distinguishable from accommodation. Coefficients retain the identifiability cautions above.
