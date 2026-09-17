"""Reproduce the size-conditioned FiLM experiment with exclusive fate vectors.

Reuse immutable full-marker checkpoints and train-only target preprocessing;
train fifteen new, equivalently initialized FiLM models on exclusive inputs.
"""
import copy,json,pickle,random,shutil
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from src.data.marker_exclusivity import exclusive_markers, EXCLUSIVITY_RULES
from src.data.io import load_graph_dataset_from_dir, select_graph_targets
from src.data.metadata import attach_metadata_to_graphs, load_aux_metadata_for_dir, load_marker_names_from_dir
from src.data.preprocessing import interpolate_target_outliers_from_neighbors
from src.models.gnn import GINCurvature, SizeFiLMGINCurvature, count_parameters
from src.training.loop import TrainConfig, train
from src.training.losses import WeightedLossTerm, edge_loss_term
from src.inference.predict import predict_targets
from src.data.subgraphs import build_ego_subgraphs_for_dataset
from src.analysis.pseudotime import make_size_ablation_cases, evaluate_size_ablation

MODEL='gin_film_size'


def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)


def make_model(settings,markers,seed=None):
    kw=dict(n_markers=len(markers),hidden_dim=settings['HIDDEN_DIM'],num_layers=settings['NUM_LAYERS'],
        dropout=settings['DROPOUT'],residual=settings['RESIDUAL'],norm=settings['NORM'],global_dim=1)
    if seed is not None:seed_all(seed)
    model=SizeFiLMGINCurvature(**kw,film_hidden_dim=settings['FILM_HIDDEN_DIM'],size_feature_index=0)
    if seed is not None:
        seed_all(seed);normal=GINCurvature(**kw)
        # Exact common-weight initialization used by the original paired experiment.
        model.load_state_dict(normal.state_dict(),strict=False)
    return model


def prepare_dataset(root,reference,out):
    original_settings=json.loads((reference/'settings.json').read_text())
    source=root/'training_data'/original_settings['DATASET_NAME']
    dataset=root/'training_data'/('after_cleanup_exclusive_'+out.parent.name)
    dataset.mkdir(exist_ok=True)
    cohort=pd.read_csv(reference/'tables/cohort.csv')
    markers=load_marker_names_from_dir(str(source))
    audit=[]
    for org in cohort.organoid_str:
        src=source/f'{org}.npz';dest=dataset/src.name
        with np.load(src) as archive: arrays={k:archive[k] for k in archive.files}
        original=arrays['x'];exclusive=exclusive_markers(original,markers)
        arrays['x']=exclusive
        if not dest.exists():np.savez_compressed(dest,**arrays)
        else:
            with np.load(dest) as saved:
                assert set(saved.files)==set(arrays)
                for key in arrays:np.testing.assert_array_equal(saved[key],arrays[key])
        for suffix in ['_aux.json','_markers.json']:shutil.copy2(source/f'{org}{suffix}',dataset/f'{org}{suffix}')
        for i,marker in enumerate(markers):
            audit.append(dict(organoid_str=org,marker=marker,full_positive=int(original[:,i].sum()),
                exclusive_positive=int(exclusive[:,i].sum()),n_cells=len(original),
                original_multi=int((original.sum(1)>1).sum()),exclusive_multi=int((exclusive.sum(1)>1).sum()),
                original_unmarked=int((original.sum(1)==0).sum()),exclusive_unmarked=int((exclusive.sum(1)==0).sum())))
    pd.DataFrame(audit).to_csv(out/'tables/exclusivity_audit.csv',index=False)
    (out/'exclusivity_rules.json').write_text(json.dumps(EXCLUSIVITY_RULES,indent=2))
    return dataset,markers


def load_cohort(dataset,settings,reference):
    graphs=load_graph_dataset_from_dir(str(dataset))
    attach_metadata_to_graphs(graphs,load_aux_metadata_for_dir(str(dataset)),exclude_keys=None)
    graphs=select_graph_targets(graphs,target_indices=[0],inplace=False)
    # Preserve the precise original cohort order for target preprocessing and training shuffling.
    order=pd.read_csv(reference/'tables/cohort.csv').organoid_str.tolist()
    lookup={g.organoid_str:g for g in graphs};assert set(lookup)==set(order)
    graphs=[lookup[o] for o in order]
    if settings['INTERPOLATE_TARGET_OUTLIERS']:
        graphs,_=interpolate_target_outliers_from_neighbors(graphs,clip_quantiles=settings['OUTLIER_CLIP_QUANTILES'])
    return graphs


def fold_graphs(graphs,split,pre,settings):
    groups={}
    for role in ['train','val']:
        group=[]
        for idx in split[role+'_indices']:
            g=graphs[idx];a=float(g.meta['total_surface_area']);v=float(g.meta['total_volume']);n=len(g.x)
            raw=torch.tensor([[np.log(n),np.log(a),np.log(v),np.log(v/a)]],dtype=torch.float32)
            glob=(raw-torch.as_tensor(pre['global_center']))/torch.as_tensor(pre['global_scale'])
            group.append(Data(x=g.x.clone(),y=g.y.clone(),edge_index=g.edge_index.clone(),organoid_str=g.organoid_str,
                full_num_cells=float(n),timepoint_label=str(g.meta.get('timepoint','unknown')),global_feat=glob.float()))
        pre['baseline'].prediction_cache.clear()
        pre['baseline'].transform_graphs(group,in_place=True)
        pre['residual_transform'].transform_graphs(group,in_place=True)
        for g in group:g.global_feat=g.global_feat[:,:1].clone()
        groups[role]=group
    return groups


def mse_rows(model,group,transform,fold,seed,device):
    y,mu,_=predict_targets(group,model,batch_size=128,device=device,target_transform=transform)
    y,mu=np.asarray(y).reshape(-1),np.asarray(mu).reshape(-1)
    rows=[];start=0
    for g in group:
        end=start+len(g.x)
        rows.append(dict(fold=fold,seed=seed,model=MODEL,organoid_str=g.organoid_str,n_cells=int(g.full_num_cells),
            timepoint=g.timepoint_label,mse=float(np.mean((y[start:end]-mu[start:end])**2))))
        start=end
    return rows


def run(root,reference,comparison,device='cuda'):
    root,reference,comparison=Path(root).resolve(),Path(reference).resolve(),Path(comparison).resolve()
    out=comparison/'exclusive';out.mkdir(parents=True,exist_ok=True)
    for folder in ['tables','checkpoints','figures','geometric_normalization']: (out/folder).mkdir(exist_ok=True)
    (comparison/'matched_full').mkdir(exist_ok=True)
    settings=json.loads((reference/'settings.json').read_text())
    dataset,markers=prepare_dataset(root,reference,out)
    settings.update(DATASET_NAME=dataset.name,MODEL_NAMES=[MODEL],MODEL_GLOBAL_FEATURES={MODEL:['log_num_cells']},
        FEATURE_ENCODING='exclusive_ordered',REFERENCE_RUN=str(reference),SAVE_DIR=str(out),DEVICE=device,
        EXCLUSIVITY_RULES=EXCLUSIVITY_RULES,BASELINE_REUSED=True)
    (out/'settings.json').write_text(json.dumps(settings,indent=2))
    (comparison/'comparison_settings.json').write_text(json.dumps(dict(reference_run=str(reference),exclusive_run=str(out),
        representation='unchanged seven binary columns; at most one positive; zeros preserved',
        matched_ablation='same exclusive-eligible centers and source cells, evaluated in both encodings'),indent=2))
    for path in ['splits.json','sweep_grid.json','tables/cohort.csv','tables/split_membership.csv','tables/global_baseline.csv','geometric_normalization/references.csv']:
        shutil.copy2(reference/path,out/path)
    splits=json.loads((out/'splits.json').read_text());counts=json.loads((out/'sweep_grid.json').read_text())
    torch.set_num_threads(4)
    if device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    graphs=load_cohort(dataset,settings,reference)
    oldsettings=json.loads((reference/'settings.json').read_text())
    # Full x only is needed for matched inference; targets and edges are unchanged.
    full_x={}
    for g in graphs:
        with np.load(root/'training_data'/oldsettings['DATASET_NAME']/f'{g.organoid_str}.npz') as f:full_x[g.organoid_str]=torch.tensor(f['x'],dtype=torch.float32)
    for split in splits:
        fold=split['fold'];print(f'PREPARE fold={fold}',flush=True)
        shutil.copy2(reference/f'checkpoints/fold_{fold}_preprocessing.pkl',out/f'checkpoints/fold_{fold}_preprocessing.pkl')
        with (out/f'checkpoints/fold_{fold}_preprocessing.pkl').open('rb') as f:pre=pickle.load(f)
        groups=fold_graphs(graphs,split,pre,settings)
        cfg=TrainConfig(lr=settings['LR'],weight_decay=settings['WEIGHT_DECAY'],batch_size=settings['BATCH_SIZE'],
            max_epochs=settings['MAX_EPOCHS'],patience=settings['PATIENCE'],num_workers=settings['NUM_WORKERS'],device=device,
            loss_name=settings['LOSS_NAME'],aux_losses=[WeightedLossTerm(name='edge',fn=edge_loss_term,
            weight=settings['EDGE_LOSS_WEIGHT'],params=settings['EDGE_LOSS_PARAMS'])])
        for seed in settings['MODEL_SEEDS']:
            checkpoint=out/f'checkpoints/fold_{fold}_seed_{seed}_{MODEL}.pt'
            score=out/f'tables/fold_{fold}_seed_{seed}_validation.csv'
            if checkpoint.exists() and score.exists():print(f'SKIP trained {fold}/{seed}',flush=True);continue
            model=make_model(settings,markers,seed=seed);seed_all(seed)
            print(f'TRAIN exclusive fold={fold} seed={seed}',flush=True)
            model,metrics,history=train(model,groups['train'],groups['val'],cfg)
            rows=mse_rows(model,groups['val'],pre['residual_transform'],fold,seed,device)
            pd.DataFrame(rows).to_csv(score,index=False)
            torch.save(model.cpu().state_dict(),checkpoint)
            (checkpoint.with_suffix('.history.json')).write_text(json.dumps(history))
            print(f'TRAIN COMPLETE {fold}/{seed}: epochs={len(history["train_loss"])}',flush=True)
        # Fixed random center sample, independently of fate. Exclusive-eligible
        # source choices are reused verbatim for the full model comparison.
        subgraphs=build_ego_subgraphs_for_dataset(groups['val'],num_hops=2,
            max_centers_per_graph=settings['ABLATION_CENTERS_PER_ORGANOID'],seed=settings['ABLATION_SEED']+fold)
        cases=make_size_ablation_cases(subgraphs,markers,hops=settings['ABLATION_HOPS'],seed=settings['ABLATION_SEED']+fold)
        indices=[];used={}
        for i,g in enumerate(subgraphs):
            if used.get(g.organoid_str,0)<settings['SWEEP_CENTERS_PER_ORGANOID']:indices.append(i)
            used[g.organoid_str]=used.get(g.organoid_str,0)+1
        remap={i:j for j,i in enumerate(indices)}
        sweeps=[dict(c,subgraph_index=remap[c['subgraph_index']]) for c in cases if c['subgraph_index'] in remap]
        for name,manifest in [('ablation',cases),('sweep',sweeps)]:
            (out/f'tables/fold_{fold}_{name}_manifest.json').write_text(json.dumps(manifest,indent=2))
        fullsubs=[]
        for sub in subgraphs:
            g=copy.copy(sub);g.x=full_x[g.organoid_str][g.orig_nodes].clone();fullsubs.append(g)
        for seed in settings['MODEL_SEEDS']:
            for encoding,folder,subs,modelrun in [('exclusive',out/'tables',subgraphs,out),('full_matched',comparison/'matched_full',fullsubs,reference)]:
                model=make_model(settings,markers)
                model.load_state_dict(torch.load(modelrun/f'checkpoints/fold_{fold}_seed_{seed}_{MODEL}.pt',map_location='cpu',weights_only=True))
                kwargs=dict(model=model,target_transform=pre['residual_transform'],size_center=pre['size_center'],size_scale=pre['size_scale'],
                    batch_size=settings['PERTURB_BATCH_SIZE'],device=device,max_fold_change=settings['LOCAL_SIZE_FOLD_CHANGE'])
                for n in ['observed',*counts]:
                    filename=f'fold_{fold}_seed_{seed}_{MODEL}_'+('observed' if n=='observed' else f'N{n}_sweep')+'.csv'
                    path=folder/filename
                    if path.exists():continue
                    result=evaluate_size_ablation(subs if n=='observed' else [subs[i] for i in indices],
                        cases if n=='observed' else sweeps,count=None if n=='observed' else n,**kwargs)
                    result.assign(fold=fold,seed=seed,model=MODEL,encoding=encoding).to_csv(path,index=False)
                    print(f'ABLATION {encoding} fold={fold} seed={seed} N={n}',flush=True)
        # Reproduce the original full-model validation with reused preprocessing.
        fullval=[]
        for g in groups['val']:
            view=copy.copy(g);view.x=full_x[g.organoid_str];fullval.append(view)
        oldmse=pd.read_csv(reference/'tables/validation_organoid_mse.csv')
        for seed in settings['MODEL_SEEDS']:
            model=make_model(settings,markers)
            model.load_state_dict(torch.load(reference/f'checkpoints/fold_{fold}_seed_{seed}_{MODEL}.pt',map_location='cpu',weights_only=True))
            check=pd.DataFrame(mse_rows(model,fullval,pre['residual_transform'],fold,seed,device))
            match=check.merge(oldmse[(oldmse.fold==fold)&(oldmse.seed==seed)&(oldmse.model==MODEL)],on='organoid_str',suffixes=('_new','_old'))
            np.testing.assert_allclose(match.mse_new,match.mse_old,atol=2e-8,rtol=2e-4)
            check.to_csv(comparison/'matched_full'/f'fold_{fold}_seed_{seed}_validation.csv',index=False)
        print(f'FOLD COMPLETE {fold}',flush=True)
    pd.concat([pd.read_csv(p) for p in sorted((out/'tables').glob('*_validation.csv'))]).to_csv(out/'tables/validation_organoid_mse.csv',index=False)
    (out/'complete.json').write_text(json.dumps(dict(checkpoints=15,full_validation_reproduced=True)))
    return out


def full_fate_control(root,reference,comparison,device='cuda'):
    """Match the amount of source fate removed: zero the entire full-marker vector.

    Exclusive single-marker removal already yields an all-zero source vector.
    This supplemental intervention separates that fact from ordinary one-bit
    full-marker removal, without fitting another model.
    """
    from src.data.subgraphs import build_ego_subgraphs_for_graph
    from src.data.io import load_organoid_npz,build_pyg_graph
    root,reference,comparison=Path(root),Path(reference),Path(comparison)
    settings=json.loads((reference/'settings.json').read_text());markers=load_marker_names_from_dir(str(root/'training_data'/settings['DATASET_NAME']))
    out=comparison/'full_fate_control';out.mkdir(exist_ok=True)
    counts=json.loads((reference/'sweep_grid.json').read_text())
    torch.set_num_threads(4)
    for fold in range(5):
        cases=json.loads((comparison/f'exclusive/tables/fold_{fold}_sweep_manifest.json').read_text())
        # Reconstruct exact recipient egos and remap saved local indices explicitly.
        subgraphs=[];lookup={}
        for org in sorted({c['organoid_str'] for c in cases}):
            arr=load_organoid_npz(str(root/'training_data'/settings['DATASET_NAME']/f'{org}.npz'),strict=True,target_indices=[0])
            graph=build_pyg_graph(arr['x'],arr['edges'],arr['y']);graph.organoid_str=org
            for sub in build_ego_subgraphs_for_graph(graph,num_hops=2,centers=sorted({c['orig_center'] for c in cases if c['organoid_str']==org})):
                lookup[(org,int(sub.orig_center))]=len(subgraphs);subgraphs.append(sub)
        with (reference/f'checkpoints/fold_{fold}_preprocessing.pkl').open('rb') as f:pre=pickle.load(f)
        for c in cases:
            c['subgraph_index']=lookup[(c['organoid_str'],c['orig_center'])]
            sub=subgraphs[c['subgraph_index']]
            c['source_node']=int(torch.nonzero(sub.orig_nodes==c['orig_source_node']).item())
            c['source_markers_to_zero']=list(range(len(markers)))
            sub.global_feat=torch.tensor([[(np.log(c['observed_n'])-pre['size_center'])/pre['size_scale']]],dtype=torch.float32)
        for seed in settings['MODEL_SEEDS']:
            model=make_model(settings,markers)
            model.load_state_dict(torch.load(reference/f'checkpoints/fold_{fold}_seed_{seed}_gin_film_size.pt',map_location='cpu',weights_only=True))
            for n in counts:
                path=out/f'fold_{fold}_seed_{seed}_gin_film_size_N{n}_sweep.csv'
                if path.exists():continue
                frame=evaluate_size_ablation(subgraphs,cases,model,pre['residual_transform'],size_center=pre['size_center'],size_scale=pre['size_scale'],
                    count=n,batch_size=settings['PERTURB_BATCH_SIZE'],device=device)
                frame.assign(fold=fold,seed=seed,encoding='full_fate_control').to_csv(path,index=False)
        print(f'Full-fate control fold={fold} complete',flush=True)
    return out
