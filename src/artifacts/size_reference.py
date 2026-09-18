"""Adapt one saved FiLM configuration for existing size-intervention workflows.

The modern run remains authoritative. This derived package uses saved graphs,
transforms and weights; no preprocessing is fitted and no model is trained.
"""
import json
import pickle
import shutil
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from .bundle import load_bundle, save_bundle
from .runs import AnalysisRun


def size_reference(directory, model_key=None, *, root=None):
    """Return a legacy run unchanged, or export one explicit modern FiLM grid.

    A grid fixes architecture/depth/width and includes its saved folds and seeds.
    Ambiguous grids require a model key; unsupported head inputs fail explicitly.
    """
    directory = Path(directory).resolve()
    if not (directory / 'models.json').exists():
        return directory
    run = AnalysisRun(directory, root=root)
    eligible = run.records.query("name == 'film' and subset == 'all' and signal == 'intact'")
    if model_key is None:
        if len(eligible[['depth', 'hidden_dim']].drop_duplicates()) != 1:
            raise ValueError('Select REFERENCE_MODEL_KEY from the saved FiLM model catalog; '
                             'one depth/width configuration is required.')
        model_key = eligible.iloc[0].key
    chosen = eligible[eligible.key == model_key]
    if len(chosen) != 1:
        raise ValueError('Select an intact, all-marker FiLM checkpoint for size interventions.')
    row = chosen.iloc[0]
    records = eligible[(eligible.depth == row.depth) & (eligible.hidden_dim == row.hidden_dim)]
    folds = sorted(int(x) for x in records.fold.unique())
    seeds = sorted(int(x) for x in records.seed.unique())
    if len(records) != len(folds) * len(seeds):
        raise ValueError('The selected FiLM grid has unfinished folds/seeds.')
    s = run.settings
    if s['global_features'] != ['log_num_cells']:
        raise ValueError('These size interventions require log_num_cells only in the FiLM head.')
    output = directory / 'analysis_inputs' / f'film_d{int(row.depth)}_h{int(row.hidden_dim)}'
    signature = dict(source_run=str(directory), keys=sorted(records.key), settings=s)
    # Verify bundles before accepting an existing derived reference.
    selected = {f: run.select(records[records.fold == f].iloc[0].key) for f in folds}
    signature['model_manifests'] = {r.key: json.loads((directory/r.bundle/'manifest.json').read_text())
                                    for r in records.itertuples()}
    signature['input_manifests'] = {r.input_bundle: json.loads((directory/r.input_bundle/'manifest.json').read_text())
                                    for r in records.itertuples()}
    signature['cohort_manifest'] = json.loads((directory/'cohort/manifest.json').read_text())
    if (output/'source.json').exists():
        if json.loads((output/'source.json').read_text()) != signature:
            raise ValueError('Saved training artifacts changed after size-reference export.')
        return output
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='.size-reference-', dir=output.parent))
    try:
        (staging/'checkpoints').mkdir(); (staging/'tables').mkdir()
        first = selected[folds[0]]
        raw = list(first['raw_graphs'].values())
        markers = first['marker_names']
        ids = {g.organoid_str: i for i, g in enumerate(raw)}
        cohort = pd.DataFrame([dict(organoid_str=g.organoid_str, n_cells=len(g.x),
            timepoint=str(g.meta.get('timepoint', 'unknown')),
            surface_area=float(g.meta['total_surface_area'])) for g in raw])
        splits, membership = [], []
        for fold, pack in selected.items():
            split = dict(fold=fold)
            for role, graphs in pack['groups'].items():
                split[role+'_indices'] = [ids[g.organoid_str] for g in graphs]
                membership.extend(dict(fold=fold, role=role, **cohort.iloc[ids[g.organoid_str]].to_dict()) for g in graphs)
            splits.append(split)
            names = pack['all_global_features']; col = names.index('log_num_cells')
            pre = dict(residual_transform=pack['transform'], baseline=pack['baseline'],
                global_center=pack['global_center'], global_scale=pack['global_scale'],
                global_feature_names=names, size_center=float(pack['global_center'][col]),
                size_scale=float(pack['global_scale'][col]), prepared_groups=pack['groups'],
                baseline_offsets=pack['baseline_offsets'])
            for name in ('baseline_predictions', 'baseline_validation_mse', 'baseline_features', 'residualized'):
                if name in pack:
                    pre[name] = pack[name]
            with (staging/f'checkpoints/fold_{fold}_preprocessing.pkl').open('wb') as handle:
                pickle.dump(pre, handle)
        settings = dict(DATASET_NAME=s['dataset'], TARGET_INDICES=s['target_indices'], TIMEPOINTS=s['timepoints'],
            MARKER_NAMES=markers, MODEL_NAMES=['gin_film_size'], MODEL_GLOBAL_FEATURES={'gin_film_size':['log_num_cells']},
            MODEL_SEEDS=seeds, NUM_LAYERS=int(row.depth), HIDDEN_DIM=int(row.hidden_dim),
            FILM_HIDDEN_DIM=s['film_hidden_dim'], DROPOUT=s['dropout'], RESIDUAL=s['residual'], NORM=s['norm'],
            LR=s['lr'], WEIGHT_DECAY=s['weight_decay'], EDGE_LOSS_WEIGHT=s['edge_loss_weight'],
            EDGE_LOSS_PARAMS=s['edge_loss_params'], SUBTRACT_CONSTANT_GLOBAL_BASELINE=s['residualize'],
            BASELINE_FEATURES=s['baseline_features'], EXCLUSIVE_MARKERS=s['exclusive_markers'],
            FEATURE_ENCODING='exclusive_ordered' if s['exclusive_markers'] else 'full_binary',
            INTERPOLATE_TARGET_OUTLIERS=False, OUTLIER_CLIP_QUANTILES=s['outlier_quantiles'],
            SOURCE_SETTINGS=s, SOURCE_RUN=str(directory))
        (staging/'settings.json').write_text(json.dumps(settings, indent=2))
        (staging/'splits.json').write_text(json.dumps(splits, indent=2))
        cohort.to_csv(staging/'tables/cohort.csv', index=False)
        membership = pd.DataFrame(membership)
        membership.to_csv(staging/'tables/split_membership.csv', index=False)
        if all('baseline_validation_mse' in pack for pack in selected.values()):
            pd.DataFrame([dict(fold=fold, **row) for fold, pack in selected.items()
                          for row in pack['baseline_validation_mse']]).to_csv(
                              staging/'tables/baseline_validation_mse.csv', index=False)
        save_bundle(staging/'cohort', dict(raw_graphs=raw, marker_names=markers), splits=splits)
        from src.analysis.normalization.geometry import fit_area_references
        refs, diagnostics = fit_area_references(cohort, membership)
        (staging/'geometric_normalization').mkdir()
        refs.to_csv(staging/'geometric_normalization/references.csv', index=False)
        diagnostics.to_csv(staging/'geometric_normalization/diagnostics.csv', index=False)
        # A suggested marginal grid only; analyses may provide their own counts.
        bounds = [np.quantile([len(g.x) for g in p['groups']['train']], [.1, .9]) for p in selected.values()]
        lo, hi = max(b[0] for b in bounds), min(b[1] for b in bounds)
        counts = np.unique(np.rint(np.geomspace(lo, hi, 9)).astype(int)) if hi >= lo else np.array([], int)
        (staging/'sweep_grid.json').write_text(json.dumps(counts.tolist()))
        for r in records.itertuples():
            payload = load_bundle(directory/r.bundle)
            torch.save(payload['model'].state_dict(), staging/f'checkpoints/fold_{r.fold}_seed_{r.seed}_gin_film_size.pt')
        (staging/'source.json').write_text(json.dumps(signature, indent=2))
        staging.rename(output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return output
