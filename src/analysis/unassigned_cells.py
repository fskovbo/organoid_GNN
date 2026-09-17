"""Observed all-zero fate cells, held-out predictions, and marker-to-zero edits."""
from dataclasses import dataclass, asdict
from pathlib import Path
import hashlib
import json
import pickle
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Batch
from src.data.io import build_pyg_graph
from src.analysis.exclusive_size_ablation import make_model
from src.analysis.ki67_observed import MARKERS, ObservedConfig, bootstrap_summary


@dataclass(frozen=True)
class UnassignedConfig:
    seeds: tuple = (42, 43, 44)
    folds: tuple = (0, 1, 2, 3, 4)
    minimum_cells_per_group: int = 3
    minimum_organoids: int = 10
    bootstrap_draws: int = 1000
    graph_batch_size: int = 8
    seed: int = 9041


def majority_context(frame):
    """A strict majority of the exact first ring; ties are mixed."""
    f=frame[[f'ring1_fraction_{m}' for m in MARKERS]].to_numpy(float)
    valid=np.isfinite(f).all(1)
    winner=np.nan_to_num(f,nan=-1).argmax(1)
    strength=np.nan_to_num(f,nan=-1).max(1)
    return np.where(~valid,'Isolated',np.where(strength>.5,np.asarray(MARKERS)[winner],'Mixed'))


def neighbor_contrasts(frame, metrics, config, degree_matched=False):
    """With vs without an unassigned neighbor, within organoid and center fate.

    In the degree-matched variant, contrasts are first calculated within exact
    ring degree, then weighted by min(n_with,n_without) within each organoid.
    This is a descriptive standardization, not an intervention estimate.
    """
    rows=[]
    base=['organoid_str','marker','size_bin','stratum_time']
    for hop in [1,2]:
        part=frame.copy()
        part['exposed']=part[f'ring{hop}_fraction_Unmarked']>0
        part=part[part[f'ring{hop}_degree']>0]
        part['degree']=part[f'ring{hop}_degree'].astype(int)
        groups=base+(['degree'] if degree_matched else [])
        agg=part.groupby(groups+['exposed'],observed=True)[metrics].mean()
        counts=part.groupby(groups+['exposed'],observed=True).size()
        if True not in agg.index.get_level_values('exposed') or False not in agg.index.get_level_values('exposed'):continue
        yes,no=agg.xs(True,level='exposed'),agg.xs(False,level='exposed')
        cy,cn=counts.xs(True,level='exposed'),counts.xs(False,level='exposed')
        ids=yes.index.intersection(no.index)
        ids=ids[(cy.reindex(ids)>=config.minimum_cells_per_group)&(cn.reindex(ids)>=config.minimum_cells_per_group)]
        diff=(yes.loc[ids]-no.loc[ids]).reset_index()
        diff['with_cells']=cy.loc[ids].to_numpy();diff['without_cells']=cn.loc[ids].to_numpy()
        if degree_matched:
            diff['weight']=np.minimum(diff.with_cells,diff.without_cells)
            output=[]
            for key,group in diff.groupby(base,observed=True):
                row=dict(zip(base,key))
                for metric in metrics:
                    valid=group[metric].notna()
                    row[metric]=np.average(group.loc[valid,metric],weights=group.loc[valid,'weight']) if valid.any() else np.nan
                output.append(row|dict(with_cells=int(group.with_cells.sum()),without_cells=int(group.without_cells.sum())))
            diff=pd.DataFrame(output,columns=base+metrics+['with_cells','without_cells'])
        rows.append(diff.assign(hop=hop,degree_matched=degree_matched))
    return pd.concat(rows,ignore_index=True)


def _stats(config):
    return ObservedConfig(bootstrap_draws=config.bootstrap_draws,minimum_organoids=config.minimum_organoids,seed=config.seed)


def observed_tables(nodes,organs,out,config):
    stats=_stats(config)
    metrics=['curvature_norm','curvature_centered','curvature_rank','negative',
             'ring1_curvature_rank','ring2_curvature_rank','fraction_neck_band','enrichment_neck_band']
    metrics += [f'ring{hop}_{kind}_{m}' for hop in [1,2] for kind in ['fraction','enrichment'] for m in MARKERS]
    keys=['organoid_str','marker','size_bin','stratum_time']
    profiles=nodes.groupby(keys)[metrics].mean().reset_index()
    profiles.to_csv(out/'organoid_profiles.csv',index=False)
    bootstrap_summary(profiles,['marker','size_bin'],metrics,stats).to_csv(out/'profiles_by_size.csv',index=False)
    bootstrap_summary(profiles,['marker','stratum_time'],metrics,stats).to_csv(out/'profiles_by_collection.csv',index=False)
    presence=nodes.assign(unassigned=nodes.marker.eq('Unmarked').astype(float)).groupby('organoid_str').unassigned.mean()
    organs=organs.copy();organs['fraction_unassigned']=organs.organoid_str.map(presence)
    organs.to_csv(out/'organoids.csv',index=False)
    bootstrap_summary(organs,['size_bin'],['fraction_unassigned'],stats).to_csv(out/'prevalence_by_size.csv',index=False)
    unassigned=nodes[nodes.marker=='Unmarked']
    contexts=unassigned.groupby(['organoid_str','size_bin','context'])[['curvature_norm','curvature_rank','curvature_centered']].mean().reset_index()
    contexts.to_csv(out/'unassigned_context_organoids.csv',index=False)
    bootstrap_summary(contexts,['context','size_bin'],['curvature_norm','curvature_rank','curvature_centered'],stats).to_csv(out/'unassigned_context_curves.csv',index=False)
    support=unassigned.groupby('context').agg(cells=('node','size'),organoids=('organoid_str','nunique')).reset_index()
    support.to_csv(out/'context_support.csv',index=False)
    for matched in [False,True]:
        c=neighbor_contrasts(nodes,['curvature_norm','curvature_rank'],config,matched)
        tag='degree_matched' if matched else 'unmatched'
        c.to_csv(out/f'neighbor_contrasts_{tag}.csv',index=False)
        bootstrap_summary(c,['marker','hop','size_bin'],['curvature_norm','curvature_rank'],stats).to_csv(out/f'neighbor_summary_{tag}.csv',index=False)
        bootstrap_summary(c,['marker','hop','stratum_time'],['curvature_norm','curvature_rank'],stats).to_csv(out/f'neighbor_collection_{tag}.csv',index=False)


@torch.no_grad()
def infer_fold(root,run_dir,data,settings,fold,nodes,config,out,device):
    membership=pd.read_csv(run_dir/'tables/split_membership.csv')
    orgs=sorted(membership[(membership.fold==fold)&(membership.role=='val')].organoid_str)
    with (run_dir/f'checkpoints/fold_{fold}_preprocessing.pkl').open('rb') as handle:pre=pickle.load(handle)
    graphs=[];org_metadata=[]
    for org in orgs:
        with np.load(data/f'{org}.npz') as z:
            graph=build_pyg_graph(z['x'],z['edges'],z['y'][:,0])
        graph.organoid_str=org
        meta=json.loads((data/f'{org}_aux.json').read_text())
        graph.global_feat=torch.tensor([[(np.log(len(graph.x))-pre['size_center'])/pre['size_scale']]],dtype=torch.float32)
        graphs.append(graph);org_metadata.append(meta)
    for seed in config.seeds:
        dest=out/f'fold_{fold}_seed_{seed}';dest.mkdir(exist_ok=True)
        checkpoint=run_dir/f'checkpoints/fold_{fold}_seed_{seed}_gin_film_size.pt'
        digest=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        if (dest/'complete.json').exists():
            if json.loads((dest/'complete.json').read_text())['checkpoint_sha256']!=digest:raise ValueError('Checkpoint changed')
            continue
        model=make_model(settings,MARKERS[:-1]).to(device).eval()
        model.load_state_dict(torch.load(checkpoint,map_location=device,weights_only=True))
        frames=[]
        for start in range(0,len(graphs),config.graph_batch_size):
            group=graphs[start:start+config.graph_batch_size]
            batch=Batch.from_data_list(group).to(device)
            (mu,_),hidden=model(batch.x,batch.edge_index,batch)
            z=mu.reshape(-1).cpu().numpy()
            residual=np.asarray(pre['residual_transform'].inverse(z.astype(float))).reshape(-1)
            hn=torch.linalg.vector_norm(hidden[:,:model.hidden_dim],dim=1).cpu().numpy()
            offset=0
            for j,g in enumerate(group):
                end=offset+len(g.x);raw=residual[offset:end]
                area=float(org_metadata[start+j]['total_surface_area'])/(4*np.pi)
                frames.append(pd.DataFrame(dict(organoid_str=g.organoid_str,node=np.arange(len(g.x)),
                    prediction_z=z[offset:end],prediction_centered=(raw-raw.mean())*area,
                    hidden_norm=hn[offset:end],fold=fold,seed=seed)))
                offset=end
        predictions=pd.concat(frames,ignore_index=True)
        # Intact full-graph inference must reproduce the archived ablation baselines.
        saved=pd.read_csv(run_dir/f'tables/fold_{fold}_seed_{seed}_gin_film_size_observed.csv')
        merged=saved.merge(predictions[['organoid_str','node','prediction_z']],left_on=['organoid_str','orig_center'],
            right_on=['organoid_str','node'],validate='many_to_one')
        assert len(merged)==len(saved)
        error=float(np.max(np.abs(merged.prediction_z-merged.base_mu_transformed)))
        np.testing.assert_allclose(merged.prediction_z,merged.base_mu_transformed,atol=4e-6,rtol=5e-5)
        predictions.to_csv(dest/'predictions.csv.gz',index=False)
        (dest/'complete.json').write_text(json.dumps(dict(checkpoint_sha256=digest,max_saved_baseline_error=error,nodes=len(predictions))))
        print(f'Unassigned study: completed fold {fold}, seed {seed}',flush=True)


def prediction_tables(nodes,out,config):
    predictions=pd.concat([pd.read_csv(path) for path in sorted(out.glob('fold_*/predictions.csv.gz'))],ignore_index=True)
    # Average seeds per cell BEFORE organoid summaries. Export seed diagnostics separately.
    joined=predictions.merge(nodes[['organoid_str','node','marker','size_bin','curvature_centered']],
        on=['organoid_str','node'],validate='many_to_one')
    joined['squared_error']=(joined.prediction_centered-joined.curvature_centered)**2
    seed_org=joined.groupby(['seed','organoid_str','marker','size_bin'])[['prediction_centered','curvature_centered','squared_error','hidden_norm']].mean().reset_index()
    seed_org.to_csv(out/'seed_organoid_predictions.csv',index=False)
    means=predictions.groupby(['organoid_str','node'])[['prediction_centered','prediction_z','hidden_norm']].mean().reset_index()
    merged=nodes.merge(means,on=['organoid_str','node'],validate='one_to_one')
    assert len(merged)==len(nodes)
    merged['squared_error']=(merged.prediction_centered-merged.curvature_centered)**2
    merged[['organoid_str','node','prediction_centered','prediction_z','hidden_norm','squared_error']].to_csv(out/'ensemble_predictions.csv.gz',index=False)
    metrics=['prediction_centered','curvature_centered','squared_error','hidden_norm']
    profiles=merged.groupby(['organoid_str','marker','size_bin'])[metrics].mean().reset_index()
    bootstrap_summary(profiles,['marker','size_bin'],metrics,_stats(config)).to_csv(out/'prediction_curves.csv',index=False)
    u=merged[merged.marker=='Unmarked'].groupby(['organoid_str','context','size_bin'])[metrics].mean().reset_index()
    bootstrap_summary(u,['context','size_bin'],metrics,_stats(config)).to_csv(out/'prediction_context_curves.csv',index=False)
    c=neighbor_contrasts(merged,['curvature_norm','prediction_centered','curvature_rank'],config)
    bootstrap_summary(c,['marker','hop','size_bin'],['curvature_norm','prediction_centered','curvature_rank'],_stats(config)).to_csv(out/'predicted_neighbor_contrasts.csv',index=False)
    return merged


def ablation_tables(nodes,organs,run_dir,out,config):
    """Reinterpret existing exclusive single-marker removals as marker -> zero."""
    area=organs.set_index('organoid_str').surface_area
    source_lookup=nodes[['organoid_str','node','marker','ring1_fraction_Unmarked']].rename(
        columns={'node':'orig_source_node','marker':'source_actual_fate','ring1_fraction_Unmarked':'source_unassigned_fraction'})
    recipient_lookup=nodes[['organoid_str','node','marker','size_bin']].rename(columns={'node':'orig_center','marker':'recipient'})
    frames=[]
    for fold in config.folds:
        for seed in config.seeds:
            f=pd.read_csv(run_dir/f'tables/fold_{fold}_seed_{seed}_gin_film_size_observed.csv')
            f=f.merge(source_lookup,on=['organoid_str','orig_source_node'],validate='many_to_one')
            f=f.merge(recipient_lookup,on=['organoid_str','orig_center'],validate='many_to_one')
            assert (f.source_actual_fate==f.source_marker_name).all() and (f.n_perturbed==1).all()
            f['delta_normalized']=f.delta_mu*f.organoid_str.map(area)/(4*np.pi)
            frames.append(f[['fold','seed','case_id','organoid_str','orig_center','orig_source_node','recipient',
                            'source_marker_name','size_bin','hop','delta_normalized','source_unassigned_fraction']])
    effects=pd.concat(frames,ignore_index=True)
    effects.to_csv(out/'marker_to_unassigned_cases.csv.gz',index=False)
    org=effects.groupby(['organoid_str','recipient','source_marker_name','hop','size_bin']).delta_normalized.mean().reset_index()
    bootstrap_summary(org,['recipient','source_marker_name','hop','size_bin'],['delta_normalized'],_stats(config)).to_csv(out/'ablation_by_size.csv',index=False)
    bootstrap_summary(org,['recipient','source_marker_name','hop'],['delta_normalized'],_stats(config)).to_csv(out/'ablation_matrix.csv',index=False)


def run(root,config=UnassignedConfig(),device='cpu'):
    root=Path(root)
    run_dir=root/'results_experiments/size_conditioned_exclusive/20260914_ordered_exclusive_film_depth2/exclusive'
    observed=root/'results_experiments/ki67_observed_neighborhoods/exclusive_v2_profile_necks'
    out=root/'results_experiments/unassigned_cells/exclusive_v1';out.mkdir(parents=True,exist_ok=True)
    settings=json.loads((run_dir/'settings.json').read_text());data=root/'training_data'/settings['DATASET_NAME']
    signature=json.loads(json.dumps(dict(config=asdict(config),model_run=str(run_dir),observed_source=str(observed))))
    if (out/'settings.json').exists() and json.loads((out/'settings.json').read_text())!=signature:raise ValueError('Incompatible saved configuration')
    (out/'settings.json').write_text(json.dumps(signature,indent=2))
    if (out/'complete.json').exists():
        add_curvature_qc(root,out,config)
        source_context_comparison(out,config)
        return out
    nodes=pd.read_csv(observed/'nodes.csv.gz');organs=pd.read_csv(observed/'organoids.csv')
    nodes['context']=majority_context(nodes)
    audit=pd.read_csv(run_dir/'tables/encoding_transition_counts.csv')
    zeros=audit[audit.exclusive_marker=='unmarked']
    assert zeros.n_cells.sum()==nodes.marker.eq('Unmarked').sum()
    provenance=zeros.groupby('original_markers').n_cells.sum().reset_index()
    provenance.to_csv(out/'unassigned_provenance.csv',index=False)
    if not (out/'observed_complete.json').exists():
        observed_tables(nodes,organs,out,config)
        (out/'observed_complete.json').write_text('{}')
    for fold in config.folds:infer_fold(root,run_dir,data,settings,fold,nodes,config,out,device)
    prediction_tables(nodes,out,config)
    ablation_tables(nodes,organs,run_dir,out,config)
    checks=[json.loads(p.read_text()) for p in out.glob('fold_*/complete.json')]
    (out/'complete.json').write_text(json.dumps(dict(cells=len(nodes),unassigned_cells=int(nodes.marker.eq('Unmarked').sum()),
        organoids=len(organs),unassigned_organoids=int(nodes[nodes.marker=='Unmarked'].organoid_str.nunique()),
        checkpoints=len(checks),maximum_saved_baseline_error=max(c['max_saved_baseline_error'] for c in checks)),indent=2))
    add_curvature_qc(root,out,config)
    source_context_comparison(out,config)
    return out


def source_context_comparison(out,config):
    if (out/'ablation_source_context.csv').exists():return
    effects=pd.read_csv(out/'marker_to_unassigned_cases.csv.gz')
    sources=effects.drop_duplicates(['organoid_str','orig_source_node','source_marker_name'])
    source_means=sources.groupby(['organoid_str','source_marker_name']).source_unassigned_fraction.mean().reset_index()
    native=pd.read_csv(out/'organoid_profiles.csv').query("marker == 'Unmarked'").set_index('organoid_str').ring1_fraction_Unmarked
    source_means['natural_unassigned_fraction']=source_means.organoid_str.map(native)
    source_means=source_means.dropna(subset=['natural_unassigned_fraction'])
    source_means['difference']=source_means.source_unassigned_fraction-source_means.natural_unassigned_fraction
    source_means.to_csv(out/'ablation_source_context_organoids.csv',index=False)
    bootstrap_summary(source_means,['source_marker_name'],['source_unassigned_fraction','natural_unassigned_fraction','difference'],_stats(config)).to_csv(out/'ablation_source_context.csv',index=False)


def add_curvature_qc(root,out,config):
    """Keep raw targets, and separately reproduce the configured interpolation.

    This is a sensitivity check using the existing preprocessing convention,
    not a correction inferred from unassigned-cell identity or a new model fit.
    """
    if (out/'curvature_qc_complete.json').exists():return
    from src.data.preprocessing import interpolate_target_outliers_from_neighbors
    meta=json.loads((out/'settings.json').read_text())
    model_run=Path(meta['model_run']);observed=Path(meta['observed_source'])
    settings=json.loads((model_run/'settings.json').read_text());data=root/'training_data'/settings['DATASET_NAME']
    nodes=pd.read_csv(observed/'nodes.csv.gz');nodes['context']=majority_context(nodes)
    graphs=[]
    for org in pd.read_csv(model_run/'tables/cohort.csv').organoid_str:
        with np.load(data/f'{org}.npz') as z:g=build_pyg_graph(z['x'],z['edges'],z['y'][:,0])
        g.organoid_str=org;graphs.append(g)
    cleaned,info=interpolate_target_outliers_from_neighbors(graphs,clip_quantiles=settings['OUTLIER_CLIP_QUANTILES'],report=False)
    area=pd.read_csv(out/'organoids.csv').set_index('organoid_str').surface_area
    frames=[]
    for original,g in zip(graphs,cleaned):
        y=g.y.numpy().astype(float);raw=original.y.numpy().astype(float);factor=area.loc[g.organoid_str]/(4*np.pi)
        frames.append(pd.DataFrame(dict(organoid_str=g.organoid_str,node=np.arange(len(y)),
            clean_curvature_norm=y*factor,clean_curvature_centered=(y-y.mean())*factor,
            clean_rank=pd.Series(y).rank(pct=True),target_changed=(np.abs(y-raw)>1e-8).astype(float))))
    clean=pd.concat(frames,ignore_index=True)
    clean.to_csv(out/'interpolated_measurements.csv.gz',index=False)
    merged=nodes.merge(clean,on=['organoid_str','node'],validate='one_to_one').merge(
        pd.read_csv(out/'ensemble_predictions.csv.gz'),on=['organoid_str','node'],validate='one_to_one')
    merged['clean_squared_error']=(merged.prediction_centered-merged.clean_curvature_centered)**2
    metrics=['curvature_norm','curvature_centered','curvature_rank','clean_curvature_norm','clean_curvature_centered',
        'clean_rank','target_changed','prediction_centered','squared_error','clean_squared_error']
    perorg=merged.groupby(['organoid_str','marker','size_bin'])[metrics].mean().reset_index()
    perorg.to_csv(out/'curvature_qc_organoids.csv',index=False)
    bootstrap_summary(perorg,['marker','size_bin'],metrics,_stats(config)).to_csv(out/'curvature_qc_summary.csv',index=False)
    contrasts=neighbor_contrasts(merged,['curvature_norm','clean_curvature_norm','curvature_rank','clean_rank','prediction_centered'],config)
    bootstrap_summary(contrasts,['marker','hop','size_bin'],['curvature_norm','clean_curvature_norm','curvature_rank','clean_rank','prediction_centered'],_stats(config)).to_csv(out/'neighbor_qc_summary.csv',index=False)
    (out/'curvature_qc_complete.json').write_text(json.dumps(dict(targets_changed=int(merged.target_changed.sum()),
        thresholds={str(k):[float(v) for v in bounds] for k,bounds in info['thresholds'].items()},
        quantiles=settings['OUTLIER_CLIP_QUANTILES']),indent=2))
