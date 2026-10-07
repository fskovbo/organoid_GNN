"""Read the existing size-conditioning and masking run formats without training.

The run directory is explicit. No latest-run discovery, preprocessing fitting,
or mutation of saved settings occurs in this reader.
"""
from src.artifacts import pickle_compat as artifact_pickle
import json
from pathlib import Path

import pandas as pd

from src.models.gnn import GINCurvature, SizeFiLMGINCurvature
from .checkpoints import load_weights


def select_depth_records(run, depth, *, model_name=None, hidden_dim=None, seeds=None, rate=None):
    """Select one intact full-panel architecture at a depth across every saved fold.

    Reject ambiguous widths/families and incomplete fold/seed grids. Validation
    inputs are subsequently restored with ``run.select`` for each returned key.
    """
    if not isinstance(depth, int) or isinstance(depth, bool) or depth < 0:
        raise ValueError('Model depth must be a nonnegative integer.')
    records = run.records
    rows = records[(records.depth == depth) & (records.subset == 'all') & (records.signal == 'intact')].copy()
    if model_name is not None:
        rows = rows[rows.name == model_name]
    if hidden_dim is not None:
        if 'hidden_dim' in rows:
            rows = rows[rows.hidden_dim == hidden_dim]
        elif run.legacy.settings['HIDDEN_DIM'] != hidden_dim:
            rows = rows.iloc[:0]
    if rate is not None:
        if 'rate' not in rows:
            raise ValueError('This run has no masking rates.')
        rows = rows[rows.rate == rate]
    elif 'rate' in rows and rows.rate.notna().any():
        raise ValueError('Select a saved masking rate explicitly.')
    if rows.empty:
        raise ValueError(f'No matching all-marker intact models at depth {depth}; inspect the run catalog.')
    family_columns = ['name'] + (['hidden_dim'] if 'hidden_dim' in rows else [])
    if len(rows[family_columns].drop_duplicates()) != 1:
        raise ValueError('Select MODEL_NAME and HIDDEN_DIM to identify one model architecture.')
    if run.modern:
        splits = json.loads((run.directory / 'splits.json').read_text())
        expected_folds = {s['fold'] for s in splits}
        expected_seeds = set(run.settings['seeds'])
    else:
        expected_folds = set(run.legacy.metadata.get('folds', [s['fold'] for s in run.legacy.splits]))
        expected_seeds = set(run.legacy.metadata.get('seeds', run.legacy.settings['MODEL_SEEDS']))
    if seeds is not None:
        if not seeds or len(set(seeds)) != len(seeds) or not set(seeds) <= expected_seeds:
            raise ValueError('Select distinct seeds present in the training settings.')
        expected_seeds = set(seeds)
        rows = rows[rows.seed.isin(seeds)]
    expected = {(fold, seed) for fold in expected_folds for seed in expected_seeds}
    actual = set(zip(rows.fold, rows.seed))
    if actual != expected or rows.duplicated(['fold', 'seed']).any():
        raise ValueError(f'Incomplete or duplicate model grid at depth {depth}; missing: {sorted(expected - actual)}')
    return rows.sort_values(['fold', 'seed']).reset_index(drop=True)


class SavedRun:
    def __init__(self, directory):
        self.directory = Path(directory).resolve()
        self.metadata = json.loads((self.directory / "settings.json").read_text())
        self.masking = "model_settings" in self.metadata
        self.settings = self.metadata["model_settings"] if self.masking else self.metadata
        self.reference = self.directory / "reference" if self.masking else self.directory
        self.splits = json.loads((self.reference / "splits.json").read_text())
        self.cohort = pd.read_csv(self.reference / "tables/cohort.csv")
        if self.cohort.organoid_str.duplicated().any():
            raise ValueError("Saved cohort contains duplicate organoid IDs")
        for split in self.splits:
            train, val = split["train_indices"], split["val_indices"]
            if len(set(train)) != len(train) or len(set(val)) != len(val) or set(train) & set(val):
                raise ValueError("Invalid saved split membership")
            if any(i < 0 or i >= len(self.cohort) for i in train + val):
                raise ValueError("Saved split index outside the cohort")

    def membership(self, fold):
        split = next(s for s in self.splits if s["fold"] == fold)
        membership = {role: self.cohort.iloc[split[role + "_indices"]].organoid_str.tolist()
                      for role in ("train", "val")}
        if self.masking:
            path = self.directory / f"fold_{fold}_membership.json"
            membership["inner"] = json.loads(path.read_text())
        return membership

    def preprocessing(self, fold):
        """Load fitted preprocessing from a trusted, local completed run."""
        with (self.reference / f"checkpoints/fold_{fold}_preprocessing.pkl").open("rb") as handle:
            return artifact_pickle.load(handle)

    def model(self, fold, seed, marker_names, *, name="gin_film_size", rate=None, device="cpu"):
        settings = self.settings
        features = settings["MODEL_GLOBAL_FEATURES"][name]
        kw = dict(n_markers=len(marker_names) + int(self.masking), hidden_dim=settings["HIDDEN_DIM"],
                  num_layers=settings["NUM_LAYERS"], dropout=settings["DROPOUT"],
                  residual=settings["RESIDUAL"], norm=settings["NORM"], global_dim=len(features))
        if name.startswith("gin_film_"):
            model = SizeFiLMGINCurvature(**kw, film_hidden_dim=settings["FILM_HIDDEN_DIM"],
                                        size_feature_index=features.index("log_num_cells"))
        elif name.startswith("gin_head_"):
            model = GINCurvature(**kw)
        else:
            raise ValueError(f"Unsupported model family {name}")
        if self.masking:
            from src.models.fate_masking import make_mask_model
            model = make_mask_model(settings, marker_names)
            if rate not in self.metadata["training"]["rates"]:
                raise ValueError("Select a masking rate present in the saved run")
            filename = f"fold_{fold}_seed_{seed}_p{float(rate):g}.pt"
        else:
            if rate is not None:
                raise ValueError("A masking rate was supplied for an ordinary model")
            filename = f"fold_{fold}_seed_{seed}_{name}.pt"
        return load_weights(model, self.directory / "checkpoints" / filename, device=device)


class AnalysisRun:
    """One analysis interface for new training runs and historical FiLM/masking runs.

    ``select`` returns explicitly named inputs, never a restored notebook namespace.
    Embeddings from independent models must be analyzed separately: their axes
    are not aligned merely because they share a depth or architecture.
    """
    def __init__(self, directory, *, root=None):
        from .paths import project_root
        self.directory = Path(directory).resolve()
        self.root = Path(root) if root else project_root()
        self.settings = json.loads((self.directory / 'settings.json').read_text())
        self._data_cache = {}
        self.modern = (self.directory / 'models.json').exists()
        if self.modern:
            self.records = pd.read_json(self.directory / 'models.json')
            self.legacy = None
        else:
            self.legacy = SavedRun(self.directory)
            rows = []
            r = self.legacy
            folds = r.metadata.get('folds', [s['fold'] for s in r.splits])
            seeds = r.metadata.get('seeds', r.settings['MODEL_SEEDS'])
            for fold in folds:
                for seed in seeds:
                    for name in (['gin_film_size'] if r.masking else r.settings['MODEL_NAMES']):
                        for rate in (r.metadata['training']['rates'] if r.masking else [None]):
                            rows.append(dict(key=f'{name}_f{fold}_s{seed}' + (f'_p{rate:g}' if rate is not None else ''),
                                             fold=fold, seed=seed, name=name, rate=rate,
                                             depth=r.settings['NUM_LAYERS'], subset='all', signal='intact'))
            self.records = pd.DataFrame(rows)

    def _cached_input(self, name):
        """Keep the cohort and at most one prepared fold/panel in memory.

        Returned selections own their deep copies; evicting an old input does
        not change a previously returned model or its data.
        """
        from .bundle import load_bundle
        if name != 'cohort':
            for old in list(self._data_cache):
                if old not in ('cohort', name):
                    del self._data_cache[old]
        if name not in self._data_cache:
            self._data_cache[name] = load_bundle(self.directory / name, device='cpu')
        return self._data_cache[name]

    def fold_inputs(self, fold):
        """Restore a fold's intact full-panel data without loading a predictor.

        Depths, widths, seeds and masking rates normally share an input bundle.
        Deduplicate those catalog references, but reject genuinely different
        bundles rather than arbitrarily choosing preprocessing. Saved baseline
        models are part of preprocessing and are restored with the fold data.
        """
        import copy
        from .bundle import load_bundle, graph_membership
        if not self.modern or 'input_bundle' not in self.records:
            raise ValueError('Fold-only loading requires a modern run with shared input bundles.')
        membership = json.loads((self.directory / 'splits.json').read_text())
        splits = [s for s in membership if s['fold'] == fold]
        if len(splits) != 1:
            raise ValueError(f'Expected one saved split for fold {fold}.')
        rows = self.records[(self.records.fold == fold) & (self.records.subset == 'all')
                            & (self.records.signal == 'intact')]
        bundles = rows.input_bundle.dropna().unique().tolist()
        if len(bundles) != 1:
            raise ValueError(f'Fold {fold} needs one unambiguous intact full-panel input bundle; found {bundles}.')
        for name in ('cohort', bundles[0]):
            self._cached_input(name)
        result = {**copy.deepcopy(self._data_cache['cohort']),
                  **copy.deepcopy(self._data_cache[bundles[0]])}
        actual = graph_membership(train=result['groups']['train'], validation=result['groups']['val'])
        if actual != {role:splits[0][role] for role in ('train', 'validation')}:
            raise ValueError(f'Fold {fold} inputs disagree with the saved split membership.')
        result['source_input_bundle'] = bundles[0]
        return result

    def select(self, key, *, device='cpu'):
        import copy
        import numpy as np
        import torch
        from .bundle import load_bundle
        rows = self.records[self.records.key == key]
        if len(rows) != 1:
            raise KeyError(f'Select one saved model key; available: {self.records.key.tolist()}')
        row = rows.iloc[0]
        if self.modern:
            selected = load_bundle(self.directory / row.bundle, device=device)
            if 'input_bundle' in row and pd.notna(row.input_bundle):
                for name in ('cohort', row.input_bundle):
                    self._cached_input(name)
                selected = {**copy.deepcopy(self._data_cache['cohort']),
                            **copy.deepcopy(self._data_cache[row.input_bundle]), **selected}
            return selected
        from src.analysis.size_conditioning.cohort_inputs import load_cohort
        from src.data.metadata import load_marker_names_from_dir
        from src.data.target_transforms import standardize_graph_global_features
        from src.training.preparation import global_values
        from torch_geometric.data import Data
        r = self.legacy
        dataset = self.root / 'training_data' / r.settings['DATASET_NAME']
        cohort_bundle = r.reference / 'cohort'
        if (cohort_bundle / 'manifest.json').exists():
            cohort_snapshot = load_bundle(cohort_bundle)
            raw = cohort_snapshot['raw_graphs']
            if isinstance(raw, dict):
                raw = list(raw.values())
            markers = cohort_snapshot['marker_names']
        else:
            raw = load_cohort(dataset, r.settings, r.reference)
            markers = load_marker_names_from_dir(str(dataset))
        pre = r.preprocessing(int(row.fold))
        if 'prepared_groups' in pre:
            groups = copy.deepcopy(pre['prepared_groups'])
            if r.masking:
                from src.data.fate_masking import encode_fates
                for group in groups.values():
                    for graph in group:
                        graph.x = encode_fates(graph.x)
            features = r.settings['MODEL_GLOBAL_FEATURES'][row['name']]
            model = r.model(int(row.fold), int(row.seed), markers, name=row['name'],
                            rate=float(row.rate) if r.masking else None, device=device)
            return dict(model=model, groups=groups, raw_graphs={g.organoid_str:g for g in raw},
                        marker_names=markers, transform=pre['residual_transform'],
                        baseline=pre.get('comparison_baseline') or pre.get('baseline'),
                        baseline_predictions=pre.get('baseline_predictions'),
                        baseline_validation_mse=pre.get('baseline_validation_mse'),
                        baseline_features=pre.get('comparison_baseline_features', pre.get('baseline_features')),
                        baseline_offsets=pre['baseline_offsets'], global_features=features,
                        all_global_features=pre['global_feature_names'], global_center=pre['global_center'],
                        global_scale=pre['global_scale'], record=row.to_dict(), settings=r.settings)
        split = next(s for s in r.splits if s['fold'] == row.fold)
        names = pre.get('global_feature_names', ['log_num_cells', 'log_surface_area', 'log_volume', 'log_volume_over_area'])
        features = r.settings['MODEL_GLOBAL_FEATURES'][row['name']]
        groups, offsets = {}, {}
        for role in ('train', 'val'):
            group = []
            for i in split[role + '_indices']:
                g = raw[i]
                glob = (torch.tensor(global_values(g, names), dtype=torch.float32)
                        - torch.as_tensor(pre['global_center'])) / torch.as_tensor(pre['global_scale'])
                group.append(Data(x=g.x.clone(), y=g.y.clone(), edge_index=g.edge_index.clone(),
                                  organoid_str=g.organoid_str, full_num_cells=float(len(g.x)),
                                  global_feat=torch.as_tensor(glob, dtype=torch.float32).reshape(1, -1)))
            original = [g.y.clone() for g in group]
            baseline = pre.get('baseline')
            if baseline is not None and pre.get('residualized', r.settings.get('SUBTRACT_CONSTANT_GLOBAL_BASELINE', True)):
                baseline.prediction_cache.clear()
                baseline.transform_graphs(group, in_place=True)
            for g, y in zip(group, original):
                offsets[g.organoid_str] = (y - g.y).numpy()
            pre['residual_transform'].transform_graphs(group, in_place=True)
            for g in group:
                g.global_feat = g.global_feat[:, [names.index(n) for n in features]].clone()
                if r.masking:
                    from src.data.fate_masking import encode_fates
                    g.x = encode_fates(g.x)
            groups[role] = group
        model = r.model(int(row.fold), int(row.seed), markers, name=row['name'],
                        rate=float(row.rate) if r.masking else None, device=device)
        return dict(model=model, groups=groups, raw_graphs={g.organoid_str: g for g in raw},
                    marker_names=markers, transform=pre['residual_transform'], baseline_offsets=offsets,
                    baseline=pre.get('comparison_baseline') or pre.get('baseline'),
                    baseline_predictions=pre.get('baseline_predictions'),
                    baseline_validation_mse=pre.get('baseline_validation_mse'),
                    baseline_features=pre.get('comparison_baseline_features', pre.get('baseline_features')),
                    global_features=features, all_global_features=names,
                    global_center=np.asarray(pre['global_center']), global_scale=np.asarray(pre['global_scale']),
                    record=row.to_dict(), settings=r.settings)
