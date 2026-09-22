"""Build historical masking artifacts for backward-compatibility tests only.

New experiments use the standalone notebook. This fixture keeps the old
reference-based file layout available to test its existing analysis readers.
"""
import json
import pickle
import shutil
from dataclasses import asdict
from pathlib import Path
import torch
from src.analysis.size_conditioning.cohort_inputs import load_cohort, fold_graphs
from src.analysis.interventions.masking import make_mask_model, train_mask_model, checkpoint_path, inner_split
from src.data.metadata import load_marker_names_from_dir


def train_masking_experiment(root, reference_run, output_dir, *, config, folds=None, seeds=None,
                             device='cpu', run_tag='', model_overrides=None):
    root,reference,out=map(Path,(root,reference_run,output_dir))
    settings=json.loads((reference/'settings.json').read_text())
    mapping=dict(depth='NUM_LAYERS',hidden_dim='HIDDEN_DIM',film_hidden_dim='FILM_HIDDEN_DIM',
        dropout='DROPOUT',norm='NORM',residual='RESIDUAL',lr='LR',weight_decay='WEIGHT_DECAY',
        edge_loss_weight='EDGE_LOSS_WEIGHT',edge_loss_params='EDGE_LOSS_PARAMS')
    settings.update({mapping[k]:v for k,v in (model_overrides or {}).items()})
    splits=json.loads((reference/'splits.json').read_text())
    folds=folds if folds is not None else [s['fold'] for s in splits]
    seeds=seeds if seeds is not None else settings['MODEL_SEEDS']
    markers=settings.get('MARKER_NAMES') or load_marker_names_from_dir(str(root/'training_data'/settings['DATASET_NAME']))
    if not out.exists():
        out.mkdir()
        shutil.copytree(reference,out/'reference')
        (out/'checkpoints').mkdir();(out/'history').mkdir();(out/'benchmarks').mkdir();(out/'figures').mkdir()
        (out/'settings.json').write_text(json.dumps(dict(tag=run_tag,model_settings=settings,markers=markers,
            training=asdict(config),folds=folds,seeds=seeds,model_overrides=model_overrides or {})))
    graphs=load_cohort(root/'training_data'/settings['DATASET_NAME'],settings,reference)
    for split in splits:
        fold=split['fold']
        if fold not in folds:continue
        with (reference/f'checkpoints/fold_{fold}_preprocessing.pkl').open('rb') as handle:pre=pickle.load(handle)
        groups=fold_graphs(graphs,split,pre,settings)
        fit_ids,early_ids=inner_split(len(groups['train']),config.inner_val_fraction,config.inner_split_seed+fold)
        fit=[groups['train'][i] for i in fit_ids];early=[groups['train'][i] for i in early_ids]
        (out/f'fold_{fold}_membership.json').write_text(json.dumps(dict(fit=[g.organoid_str for g in fit],
            early_stopping=[g.organoid_str for g in early],benchmark=[g.organoid_str for g in groups['val']])))
        for rate in config.rates:
            for seed in seeds:
                path=checkpoint_path(out,fold,seed,rate)
                if path.exists():continue
                model,history=train_mask_model(make_mask_model(settings,markers,seed=seed),fit,early,
                    settings,config,rate=rate,seed=seed,device=device,verbose=False)
                torch.save(dict(model_state=model.state_dict(),fold=fold,seed=seed,rate=rate,
                    history=history.to_dict('records')),path)
                history.to_csv(out/'history'/f'{path.stem}.csv',index=False)
    (out/'training_complete.json').write_text('{}')
    return out
