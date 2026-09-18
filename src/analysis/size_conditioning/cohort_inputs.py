"""Saved-cohort preparation and feature-encoding comparison primitives.

Model fitting and study-level execution live in training/analysis notebooks.
"""
from src.artifacts import pickle_compat as artifact_pickle
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
from src.inference.predict import predict_targets
from src.analysis.interventions.size_sweeps import evaluate_size_ablation

MODEL='gin_film_size'






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
    if (Path(reference) / 'cohort/manifest.json').exists():
        from src.artifacts.bundle import load_bundle
        snapshot = load_bundle(Path(reference) / 'cohort')
        raw = snapshot['raw_graphs']
        return list(raw.values()) if isinstance(raw, dict) else raw
    graphs=load_graph_dataset_from_dir(str(dataset))
    attach_metadata_to_graphs(graphs,load_aux_metadata_for_dir(str(dataset)),exclude_keys=None)
    graphs=select_graph_targets(graphs,target_indices=[0],inplace=False)
    # Preserve the precise original cohort order for target preprocessing and training shuffling.
    order=pd.read_csv(reference/'tables/cohort.csv').organoid_str.tolist()
    lookup={g.organoid_str:g for g in graphs}
    if len(lookup) != len(graphs) or len(set(order)) != len(order):
        raise ValueError('Duplicate organoid IDs in data or saved cohort')
    missing = set(order) - set(lookup)
    if missing:
        raise ValueError(f'Saved cohort organoids are missing from the dataset: {sorted(missing)}')
    # Full-marker datasets may contain organoids excluded by the saved filters.
    # Select the recorded cohort rather than rerunning filtering or requiring
    # the source directory itself to contain only the selected organoids.
    graphs=[lookup[o] for o in order]
    if settings['INTERPOLATE_TARGET_OUTLIERS']:
        graphs,_=interpolate_target_outliers_from_neighbors(graphs,clip_quantiles=settings['OUTLIER_CLIP_QUANTILES'])
    if settings.get('EXCLUSIVE_MARKERS', False):
        markers = load_marker_names_from_dir(str(dataset))
        for g in graphs:
            g.x = torch.as_tensor(exclusive_markers(g.x.numpy(), markers), dtype=g.x.dtype)
    return graphs


def fold_graphs(graphs,split,pre,settings):
    if 'prepared_groups' in pre:
        return copy.deepcopy(pre['prepared_groups'])
    groups={}
    for role in ['train','val']:
        group=[]
        for idx in split[role+'_indices']:
            g=graphs[idx];a=float(g.meta['total_surface_area']);v=float(g.meta['total_volume']);n=len(g.x)
            raw=torch.tensor([[np.log(n),np.log(a),np.log(v),np.log(v/a)]],dtype=torch.float32)
            glob=(raw-torch.as_tensor(pre['global_center']))/torch.as_tensor(pre['global_scale'])
            group.append(Data(x=g.x.clone(),y=g.y.clone(),edge_index=g.edge_index.clone(),organoid_str=g.organoid_str,
                full_num_cells=float(n),timepoint_label=str(g.meta.get('timepoint','unknown')),global_feat=glob.float()))
        if pre.get('residualized', settings.get('SUBTRACT_CONSTANT_GLOBAL_BASELINE', pre.get('baseline') is not None)):
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
        with (reference/f'checkpoints/fold_{fold}_preprocessing.pkl').open('rb') as f:pre=artifact_pickle.load(f)
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

from src.models.size_models import seed_all, make_model
