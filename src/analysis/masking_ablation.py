"""Add learned missing-fate ablations to a completed replacement experiment.

Keep old inference files immutable. Evaluate zero, both replacement mixtures,
and masking within the SAME new checkpoint, on the exact saved source cases.
The original checkpoint curves remain a separately labelled historical reference.
"""
import json
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import torch

from src.analysis.fate_masking import (
    ObservedFateAdapter, evaluate_single_mask, load_mask_checkpoint,
    checkpoint_path, checked_settings, _sha,
)
from src.analysis.exclusive_size_ablation import load_cohort, fold_graphs
from src.analysis.replacement_ablation import evaluate_replacements, contrast_table, _write_csv, load_comparison
from src.analysis.pseudotime import summarize_by_organoid
from src.data.subgraphs import build_ego_subgraphs_for_graph

MASK_METHODS = ('zero', 'replacement_fixed', 'replacement_size_dependent', 'masking')


def reconstruct_cases(graphs, manifest):
    """Reconstruct egos from saved original indices, never resample sources."""
    lookup={g.organoid_str:g for g in graphs}; subs=[]; mapping={}
    for org, frame in manifest.groupby('organoid_str',sort=True):
        if org not in lookup: raise ValueError(f'Case organoid is not in validation fold: {org}')
        centers=sorted(frame.orig_center.unique().astype(int).tolist())
        for sub in build_ego_subgraphs_for_graph(lookup[org],num_hops=2,centers=centers):
            mapping[(org,int(sub.orig_center))]=len(subs); subs.append(sub)
    cases=manifest.copy()
    for i,c in cases.iterrows():
        si=mapping[(c.organoid_str,int(c.orig_center))]; sub=subs[si]
        source=torch.nonzero(sub.orig_nodes==int(c.orig_source_node)).reshape(-1)
        if len(source)!=1: raise ValueError('Saved source is absent from reconstructed ego.')
        if len(lookup[c.organoid_str].x)!=int(c.observed_n): raise ValueError('Observed graph size changed.')
        cases.loc[i,'subgraph_index']=si; cases.loc[i,'source_node']=int(source[0])
    cases['subgraph_index']=cases.subgraph_index.astype(int);cases['source_node']=cases.source_node.astype(int)
    return subs,cases.reset_index(drop=True)


def run_masking_ablation(root, replacement_dir, masking_run, output_dir, *, rate=.02,
                          batch_size=128, device=None):
    root,old,training,out=map(lambda p:Path(p).resolve(),(root,replacement_dir,masking_run,output_dir))
    if not (old/'complete.json').exists() or not (training/'training_complete.json').exists():
        raise RuntimeError('Complete replacement inference and masking training first.')
    oldcfg=json.loads((old/'settings.json').read_text()); cfg=json.loads((training/'settings.json').read_text())
    if rate<=0 or rate not in cfg['training']['rates']: raise ValueError('Select a trained positive masking rate.')
    if not set(oldcfg['folds'])<=set(cfg['folds']) or not set(oldcfg['seeds'])<=set(cfg['seeds']):
        raise ValueError('Masking models must cover every replacement fold and seed.')
    if Path(oldcfg['run_dir']).resolve()!=Path(cfg['reference_run']).resolve():
        raise ValueError('Replacement and masking runs must use the same source experiment.')
    if oldcfg['markers'][:-1]!=cfg['markers']: raise ValueError('Marker order mismatch.')
    for name in ['settings.json','splits.json','geometric_normalization/references.csv',
                 *[f'checkpoints/fold_{fold}_preprocessing.pkl' for fold in oldcfg['folds']]]:
        if oldcfg['artifact_sha256'][name] != cfg['dependencies'][name]:
            raise ValueError(f'Replacement and masking preprocessing differ: {name}')
    settings=cfg['model_settings']; markers=cfg['markers']; reference=training/'reference'
    device=device or ('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint_hashes={checkpoint_path(training,f,s,rate).name:_sha(checkpoint_path(training,f,s,rate))
                       for f in oldcfg['folds'] for s in oldcfg['seeds']}
    old_files=sorted((old/'tables').glob('*.csv.gz'))+sorted((old/'support').glob('*_cases.csv.gz'))
    checked_settings(out/'settings.json',dict(replacement_dir=str(old),masking_run=str(training),mask_rate=rate,
        folds=oldcfg['folds'],seeds=oldcfg['seeds'],counts=oldcfg['counts'],markers=oldcfg['markers'],
        code_sha256=_sha(__file__),model_code_sha256=_sha(Path(__file__).with_name('fate_masking.py')),
        training_settings_sha256=_sha(training/'settings.json'),reference_settings_sha256=_sha(old/'settings.json'),
        reference_files={str(p.relative_to(old)):_sha(p) for p in old_files},checkpoints=checkpoint_hashes))
    (out/'tables').mkdir(exist_ok=True);(out/'figures').mkdir(exist_ok=True)
    graphs=load_cohort(root/'training_data'/settings['DATASET_NAME'],settings,reference)
    refs=pd.read_csv(reference/'geometric_normalization/references.csv').set_index('fold')
    for split in json.loads((reference/'splits.json').read_text()):
        fold=split['fold']
        if fold not in oldcfg['folds']: continue
        with (reference/f'checkpoints/fold_{fold}_preprocessing.pkl').open('rb') as f: pre=pickle.load(f)
        val=fold_graphs(graphs,split,pre,settings)['val']
        manifest=pd.read_csv(old/'support'/f'fold_{fold}_cases.csv.gz')
        subs,cases=reconstruct_cases(val,manifest)
        for seed in oldcfg['seeds']:
            model=load_mask_checkpoint(training,fold,seed,rate,settings,markers)
            for n in [None,*oldcfg['counts']]:
                label='observed' if n is None else f'N{n:g}'
                filename=f'fold_{fold}_seed_{seed}_{label}.csv.gz';path=out/'tables'/filename
                if path.exists(): continue
                prior=pd.read_csv(old/'tables'/filename)
                selected=cases.set_index('case_id').loc[prior.case_id].reset_index()
                for col in ['organoid_str','orig_center','orig_source_node','source_identity','center_identity','hop','observed_n']:
                    if not np.array_equal(selected[col].to_numpy(),prior[col].to_numpy()):
                        raise ValueError(f'Case identity mismatch: {col}')
                k=len(markers)+1
                fixed=prior[[f'p_fixed_{j}' for j in range(k)]].to_numpy()
                dependent=prior[[f'p_dependent_{j}' for j in range(k)]].to_numpy()
                # Known-fate replacements never set the missingness flag.
                pred=evaluate_replacements(subs,selected,ObservedFateAdapter(model),pre['residual_transform'],
                    size_center=pre['size_center'],size_scale=pre['size_scale'],count=n,batch_size=batch_size,device=device)
                frame=contrast_table(selected,pred,fixed,dependent,count=n,fold=fold,seed=seed,reference=refs.loc[fold])
                missing=evaluate_single_mask(subs,selected,model,pre['residual_transform'],size_center=pre['size_center'],
                    size_scale=pre['size_scale'],count=n,batch_size=batch_size,device=device,include_zero=False)
                np.testing.assert_allclose(frame.base_mu,missing.intact_mu,atol=1e-7,rtol=1e-5)
                frame['masking_delta_mu']=missing.mask_delta_mu
                frame['masking_delta_z']=missing.mask_delta_z
                frame['masking_delta_relative']=missing.mask_delta_mu*frame.normalization_factor
                frame['mask_rate']=rate
                if n is None:
                    frame['masking_delta_mse_z']=missing.delta_mse_z
                    frame['masking_delta_mse_mu']=missing.delta_mse_mu
                _write_csv(frame,path)
                print(f'Masking ablation fold={fold} seed={seed}, {label}',flush=True)
    (out/'complete.json').write_text('{}')
    return out


def load_masking_comparison(output_dir, *, bootstrap_samples=500, seed=42):
    """Same-checkpoint four-method comparison on the original common cohort."""
    out=Path(output_dir)
    result=load_comparison(out,bootstrap_samples=bootstrap_samples,seed=seed)
    selected=result['paired_cases']; groups=['analysis','center_marker','source_marker_name','hop','coordinate']
    extra=[]
    for metric in ('delta_mu','delta_relative','delta_z'):
        part=selected.copy();part['effect']=part[f'masking_{metric}']
        summary=summarize_by_organoid(part,groups,'effect',bootstrap_samples=bootstrap_samples,seed=seed)
        if not summary.empty:
            pos=part.groupby(groups,observed=True).evaluated_n.median().rename('n').reset_index()
            summary=summary.merge(pos,on=groups,validate='one_to_one')
            summary['method']='masking';summary['metric']=metric;extra.append(summary)
    if extra: result['summary']=pd.concat([result['summary'],*extra],ignore_index=True)
    result['methods']=MASK_METHODS
    result['model_label']=f'Masking-trained FiLM ({result["config"]["mask_rate"]:.0%})'
    _write_csv(result['summary'],out/'comparison_with_masking.csv')
    return result
