# Model audit against `9ae3657`

Audit date: 2026-09-22. Reference: `9ae3657` (“Prepare for refactor”). This audit covers existing model implementations, relocated model/masking helpers, the shared training loop, and the training notebook's control-class selection. It does not claim that newly configured training runs reproduce historical runs with different datasets, filters or hyperparameters.

## What went wrong

I introduced an incorrect class mapping when consolidating control training. In `experiments/compare_models.ipynb` and `experiments/permutations_tests.ipynb` at the reference commit, `ring-size` selected `RingSizeMLP`. The consolidated notebook instead selected `RingFractionSizeMLP`. That mapping is already present in refactor commit `05f8f6b`; it predates the subsequent caching and progress changes.

The two classes existed before the refactor and have different input contracts. I failed to preserve that distinction. Existing workflow checks verified training, saving, loading and matching splits, but did not assert the scientific meaning of the `ring-size` label. This was an unauthorized change to the experiment, even though neither class's forward calculation was edited.

| Label | Required local input | Incorrectly selected input |
| --- | --- | --- |
| `ring` | Center markers and marker fractions in each exact-hop ring | Unchanged |
| `ring-size` | Center markers and node counts in each exact-hop ring | Center markers, ring counts **and neighboring marker fractions** |

The notebook now selects the original `RingSizeMLP` with `use_center_markers=True`. It receives no neighborhood fate composition. As before, the global vector is inherited from the reference run and passed to the head. Cached marker fractions can still be present in the shared data for the separate `ring` model; `RingSizeMLP` does not read them.

## Other differences found

| Component | Comparison with the reference |
| --- | --- |
| `src/models/gnn.py` | Byte-identical. Includes GIN, size-conditioned FiLM, SAGE, GAT, jumping-knowledge variants and their prediction heads. |
| `src/models/baseline.py`, `distribution.py`, `io.py` | Byte-identical. |
| `src/models/ring_mlp.py` | Three existing classes now record `hidden_dim`, `dropout` and `norm` as ordinary attributes for checkpoint reconstruction. No parameter, forward or aggregation calculation changed. `RingSizeMLP` itself is unchanged. |
| `src/training/losses.py`, `src/data/target_transforms.py`, `src/data/ring_features.py` | Byte-identical. The control notebook now precomputes ring features after signal edits, rather than recomputing them during each forward pass. |
| `src/models/size_models.py` | `make_model` and `seed_all` moved from `src/analysis/exclusive_size_ablation.py`; their ASTs are identical. |
| `src/data/fate_masking.py` | Encoding, mask sampling and adapters moved from `src/analysis/fate_masking.py`; their ASTs are identical. |
| `src/models/fate_masking.py` | Initialization was extended for configurable depth and width: depth 0 inserts the missingness column before global inputs; residual projections handle widths equal to the original or augmented input dimension. The augmented model explicitly gets a projection when its constructor would otherwise use an identity. These are actual initialization/architecture edge-case changes, not merely file moves. The usual depth-2, width-128 case is unaffected. |
| `src/training/masking.py` | Gradient clipping became configurable, retaining the original value 2.0 by default. Progress callbacks/log controls were added. Mask encoding, two-pass loss, optimizer, inner split and stopping criterion remain the same with default settings. |
| `src/training/loop.py` | Progress callbacks and optional epoch printing were added. The audit caught a change for `patience=0`: the loop could stop on an improving epoch. This is now restored to the original behavior of stopping on the first non-improving epoch. Positive-patience behavior is unchanged. |

The new artifact loaders and notebook orchestration are additional code. Training options, data selection, residualization, fold preparation and baseline fitting were reorganized and made configurable in response to the earlier requests. They should not be conflated with identical numerical model implementations; each run's saved settings and inputs determine that experiment.

## Removal of invalid results

In `training_results/graph_controls_training/controls_20260921_165106`, deleted all 15 incorrectly labelled `ring-size` model bundles: five depths × three signals, fold 0. Removed their entries from `models.json`, validation MSE, regional MSE, regional summaries and the training progress table. Replaced the regional figure using only retained controls and cleared the training notebook's saved outputs. No incorrect model was archived or relabelled as another control.

The remaining 30 checkpoints and all shared input/cohort bundles were verified byte-identical after cleanup (135 files checked). Baseline results and original settings remain. `ring_size_correction.json` records the deletion, without retaining the removed models' scores. The corrected controls need a new training run; no production models were retrained during this correction.

## Plot changes and validation

Overall MSE now has one panel per input signal. Regional MSE has one row per signal and one column per region. Model colors are consistent across panels, and model/baseline mean ± SEM bands are retained. Optimization histories also have separate signal panels.

Verification includes:

- Direct comparison with the reference ring classes across 24 configurations (four classes, three depths, two normalization types): exactly equal initialization and training/evaluation forward outputs with matched random seeds.
- Direct original/current training comparisons on a small CPU fixture: exactly equal weights, metrics and histories for both `patience=0` and `patience=2`.
- A regression test that changes all neighboring fate vectors and fills cached marker fractions with NaNs: the `ring-size` center prediction and embedding remain exactly unchanged. It checks depths 0, 1, 2 and 4, cached versus uncached inputs, and the actual encoder input.
- The real notebook workflow exercised with small synthetic datasets, including training controls, copying reference checkpoints, exact saved splits/baselines, and restored model class/input dimension checks.
- Signal-panel tests verify separate scores, consistent colors and SEM bands for regional plots.

The complete test suite passed: **101 tests**. No production training was performed.
