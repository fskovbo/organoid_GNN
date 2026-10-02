# Learning organoid morphology from cell-fate graphs

Training and analysis are separate workflows. Start with the [notebook guide](experiments/README.md).

- `experiments/training/`: GIN/FiLM depths, graph controls, marker combinations, masking and interpretable fate models.
- `experiments/ablation/`: zeroing, replacement, trained masking, sampling and niche diagnostics.
- `experiments/benchmarks/`: saved-model comparisons and masking benchmarks.
- `experiments/marker_subsets/`: marker-panel informativeness.
- `experiments/embeddings/`: clustering, patch composition and representation responses.
- `experiments/neighborhoods/`: measured cell neighborhoods and regional predictability.
- `experiments/data_quality/`: graph, mesh and cohort inspection.
- `experiments/visualizations/`: curvature/marker mesh rendering and export.

`src/` contains reusable data, model, training, inference, analysis and plotting operations.
Notebook-specific choices and experiment sequences belong in notebooks. New trained
models go under `training_results/<notebook_name>/<tag>_<timestamp>/`.
An empty tag produces a timestamp-only run name. Historical models and scientific reports
remain under `results_experiments/`; datasets remain in `training_data/`.

Use the existing `organoid-gnn` environment (including the local `organograph` dependency).
Notebook bootstrap cells locate the repository from any notebook subdirectory. An editable
install is optional: `python -m pip install -e '.[notebooks]'`.

Run checks with `python -m unittest discover -s tests -p 'test_*.py'`.
The [artifact/refactor guide](docs/refactor.md) explains restoration and historical compatibility.

Before changing the interpretable curvature models, read the [scientific decision record](docs/curvature_model_decisions.md): priorities, past experiments, reference conventions and the current simplified model.
