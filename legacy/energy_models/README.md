# Archived curvature-energy training workflows

Archived on 2026-10-02 at the user’s request. The four training notebooks in
`notebooks/training/` retain their settings and outputs; only their own notebook
provenance paths were repaired after moving. Their trained result folders remain
unchanged, and compatible model code stays in `src/models/` for checkpoint loading.

These workflows inherit settings from earlier runs and are historical references,
not the entry point for new experiments. They may execute training if run: inspect
historical execution switches before using them.

The active standalone workflow is
[energy_model_training](../../experiments/training/energy_model_training.ipynb),
with [energy_model_evaluation](../../experiments/benchmarks/energy_model_evaluation.ipynb).
It loads a fresh cohort, exposes all training settings, and sweeps fixed accommodation.
The pooled-interaction and simple-fate evaluation notebooks are now archived in
`notebooks/benchmarks/`, preserving their displayed outputs. Their own provenance
paths were repaired. Coupled-fate and mean-curvature evaluation notebooks remain
under `experiments/benchmarks/` for older results.
