# Archived additive fate-interaction models

The linear spline and nonlinear fraction models and their four notebooks were
archived on 2026-09-25 at the user's request. Their implementations live under
`models/` and `training/`; notebooks retain their displayed outputs and reside
under `notebooks/`. Existing `training_results/fate_spline_training/` and
`training_results/fate_fraction_benchmark/` runs are preserved in place.

Historical checkpoint and pickle class paths are remapped by `src/artifacts`;
there are no import-only shim files in the active model directory. Archived
notebooks use the archived implementation explicitly. They are historical
workflows, not the active entry point for new interaction studies.

Use `experiments/training/coupled_fate_training.ipynb` for the replacement and
`experiments/benchmarks/coupled_fate_evaluation.ipynb` for comparison/readout.
