"""Matched full-versus-exclusive ablation summaries and downstream reruns."""
import ast,json,shutil
from pathlib import Path
import numpy as np
import pandas as pd
from src.analysis.lgr5_film_summary import cluster_summary
from src.analysis.geometric_normalization import normalize_cases


def setup_matched_full(comparison):
    comparison=Path(comparison)
    config=json.loads((comparison/'comparison_settings.json').read_text())
    reference=Path(config['reference_run']);exclusive=comparison/'exclusive';out=comparison/'full_on_exclusive_cases'
    for folder in ['tables','checkpoints','geometric_normalization']: (out/folder).mkdir(parents=True,exist_ok=True)
    settings=json.loads((reference/'settings.json').read_text())
    settings.update(MODEL_NAMES=['gin_film_size'],MODEL_GLOBAL_FEATURES={'gin_film_size':['log_num_cells']},
        REFERENCE_RUN=str(reference),SAVE_DIR=str(out),MATCHED_ENCODING='exclusive_eligible_cases')
    (out/'settings.json').write_text(json.dumps(settings,indent=2))
    for path in ['splits.json','sweep_grid.json','tables/cohort.csv','tables/split_membership.csv','geometric_normalization/references.csv']:
        shutil.copy2(reference/path,out/path)
    for p in (reference/'checkpoints').glob('*'):
        if p.name.endswith('_preprocessing.pkl') or p.name.endswith('_gin_film_size.pt'):shutil.copy2(p,out/'checkpoints'/p.name)
    for p in (exclusive/'tables').glob('*_manifest.json'):shutil.copy2(p,out/'tables'/p.name)
    for p in (comparison/'matched_full').glob('*gin_film_size*.csv'):shutil.copy2(p,out/'tables'/p.name)
    return out


def run_diagnostics(comparison,root,device='cuda'):
    from src.analysis.lgr5_film import run_diagnostics as lgr5_run
    from src.analysis.lgr5_film_summary import summarize_diagnostics
    from src.analysis.niche_hypotheses import run as niche_run,validate_saved
    from src.analysis.niche_hypotheses_summary import summarize as niche_summary
    comparison,root=Path(comparison),Path(root)
    full=setup_matched_full(comparison);exclusive=comparison/'exclusive'
    exclusive_settings=json.loads((exclusive/'settings.json').read_text())
    for name,run in [('exclusive',exclusive),('full_matched',full)]:
        print('LGR5 diagnostics',name,flush=True)
        lgr5_run(run,root,device=device)
        summarize_diagnostics(run/'lgr5_film_diagnostics')
        print('Niche diagnostics',name,flush=True)
        sampling=None if name=='exclusive' else root/'training_data'/exclusive_settings['DATASET_NAME']
        niche_run(run,root,device=device,sampling_data_dir=sampling)
        niche_summary(run/'niche_hypotheses')
        validate_saved(run)
    # Prove common diagnostic cohorts, rather than merely relying on identical RNG seeds.
    for file in ['recipient_manifest.csv.gz','pair_manifest.csv.gz']:
        a=pd.read_csv(exclusive/'niche_hypotheses'/file);b=pd.read_csv(full/'niche_hypotheses'/file)
        pd.testing.assert_frame_equal(a,b)
    (comparison/'diagnostics_complete.json').write_text(json.dumps(dict(exclusive_checkpoints=15,matched_full_checkpoints=15,
        shared_niche_manifests=True,training_repeated_for_full=False)))


def summarize(comparison,draws=1000):
    comparison=Path(comparison);config=json.loads((comparison/'comparison_settings.json').read_text())
    exclusive=comparison/'exclusive';reference=Path(config['reference_run'])
    out=comparison/'comparison_tables';out.mkdir(exist_ok=True)
    refs=pd.read_csv(reference/'geometric_normalization/references.csv')
    cohort=pd.read_csv(reference/'tables/cohort.csv');counts=json.loads((reference/'sweep_grid.json').read_text())
    edges=np.unique(np.quantile(cohort.n_cells,np.linspace(0,1,6)));edges[0]=-np.inf;edges[-1]=np.inf
    cohort['size_bin']=pd.cut(cohort.n_cells,edges,labels=False).astype(int)
    centers=cohort.groupby('size_bin').n_cells.median()
    cohort.to_csv(out/'cohort_bins.csv',index=False)
    frames=[];seedframes=[];pairedframes=[];support=[]
    native=reference/'tables'
    for fold in range(5):
        for seed in [42,43,44]:
            for n in ['observed',*counts]:
                suffix='observed' if n=='observed' else f'N{n}_sweep'
                filename=f'fold_{fold}_seed_{seed}_gin_film_size_{suffix}.csv'
                loaded={}
                encodings=[('full_native',native),('full_matched',comparison/'matched_full'),('exclusive',exclusive/'tables')]
                if n!='observed' and (comparison/'full_fate_control'/filename).exists():
                    encodings.append(('full_fate_control',comparison/'full_fate_control'))
                for encoding,folder in encodings:
                    d=pd.read_csv(folder/filename)
                    d['center_marker_names']=d.center_marker_names.map(ast.literal_eval)
                    d=normalize_cases(d,refs).rename(columns={'delta_mu':'delta_raw','delta_mu_transformed':'delta_z'})
                    d['center_marker']=d.center_marker_names.map(lambda x:x or ['unmarked'])
                    d=d.explode('center_marker',ignore_index=True)
                    d['n']=pd.cut(d.observed_n,edges,labels=False).astype(int).map(centers) if n=='observed' else n
                    d['mode']='observed' if n=='observed' else 'sweep';d['encoding']=encoding
                    loaded[encoding]=d
                    keys=['encoding','mode','n','center_marker','source_marker_name','hop','organoid_str','fold','seed']
                    metrics=['delta_raw','delta_z','delta_relative']
                    org=d.groupby(keys,observed=True)[metrics].mean().reset_index()
                    frames.append(org)
                    if seed==42 and n in ['observed',counts[0]]:
                        support.append(d.groupby(['encoding','mode','center_marker','source_marker_name','hop']).agg(
                            n_cases=('case_id','size'),n_organoids=('organoid_str','nunique')).reset_index().assign(fold=fold))
                ids=['fold','seed','case_id','organoid_str','center_marker','source_marker_name','hop','mode','n']
                a=loaded['exclusive'];b=loaded['full_matched']
                match=a.merge(b,on=ids,suffixes=('_exclusive','_full'),validate='one_to_one')
                assert len(match)==len(a)==len(b)
                for m in metrics:match[m]=match[m+'_exclusive']-match[m+'_full']
                pairedframes.append(match.groupby([k for k in ids if k!='case_id'])[metrics].mean().reset_index())
            print(f'Comparison summary fold={fold} seed={seed}',flush=True)
    frame=pd.concat(frames,ignore_index=True);paired=pd.concat(pairedframes,ignore_index=True)
    frame.to_csv(out/'ablation_organoid.csv.gz',index=False);paired.to_csv(out/'paired_ablation_organoid.csv.gz',index=False)
    keys=['encoding','mode','n','center_marker','source_marker_name','hop'];metrics=['delta_raw','delta_z','delta_relative']
    cluster_summary(frame,keys,metrics,draws=draws).to_csv(out/'ablation_summary.csv',index=False)
    cluster_summary(paired,keys[1:],metrics,draws=draws).to_csv(out/'paired_ablation_summary.csv',index=False)
    frame.groupby([*keys,'seed'])[metrics].mean().to_csv(out/'ablation_seed_summary.csv')
    pd.concat(support).to_csv(out/'ablation_support.csv',index=False)
    ix=['encoding','center_marker','source_marker_name','hop','organoid_str','fold','seed']
    lo=frame[(frame['mode']=='sweep')&(frame.n==counts[0])].set_index(ix)[metrics]
    hi=frame[(frame['mode']=='sweep')&(frame.n==counts[-1])].set_index(ix)[metrics]
    endpoint=(hi-lo).dropna().reset_index()
    cluster_summary(endpoint,ix[:4],metrics,draws=draws).to_csv(out/'ablation_endpoint.csv',index=False)
    endpoint.groupby([*ix[:4],'seed'])[metrics].mean().to_csv(out/'ablation_endpoint_seeds.csv')
    commonkeys=ix[1:]
    contrast=(endpoint[endpoint.encoding=='exclusive'].set_index(commonkeys)[metrics]-
        endpoint[endpoint.encoding=='full_matched'].set_index(commonkeys)[metrics]).dropna().reset_index()
    cluster_summary(contrast,ix[1:4],metrics,draws=draws).to_csv(out/'ablation_endpoint_difference.csv',index=False)
    # Physical-unit validation MSE uses exactly the same held-out organoids and baseline.
    full=pd.read_csv(reference/'tables/validation_organoid_mse.csv').query("model == 'gin_film_size'").assign(encoding='full')
    exc=pd.read_csv(exclusive/'tables/validation_organoid_mse.csv').assign(encoding='exclusive')
    scores=pd.concat([full,exc],ignore_index=True).merge(cohort[['organoid_str','size_bin']],on='organoid_str',validate='many_to_one')
    cluster_summary(scores,['encoding'],['mse'],draws=draws).to_csv(out/'mse_overall.csv',index=False)
    summary=cluster_summary(scores,['encoding','size_bin'],['mse'],draws=draws);summary['n']=summary.size_bin.map(centers)
    summary.to_csv(out/'mse_by_size.csv',index=False)
    wide=scores.pivot(index=['organoid_str','fold','seed','size_bin'],columns='encoding',values='mse')
    delta=(wide.exclusive-wide.full).rename('delta_mse').reset_index().assign(comparison='exclusive minus full')
    cluster_summary(delta,['comparison'],['delta_mse'],draws=draws).to_csv(out/'mse_paired.csv',index=False)
    cluster_summary(delta,['size_bin'],['delta_mse'],draws=draws).to_csv(out/'mse_paired_by_size.csv',index=False)
    scores.to_csv(out/'mse_organoid.csv',index=False)
    return out


def summarize_diagnostic_comparison(comparison,draws=1000):
    comparison=Path(comparison);out=comparison/'comparison_tables'
    config=json.loads((comparison/'comparison_settings.json').read_text())
    runs={'full_native':Path(config['reference_run']),'full_matched':comparison/'full_on_exclusive_cases','exclusive':comparison/'exclusive'}
    for name in ['dominance','backup','pairs','specificity','crypt_paired','lineage','observed','timepoint']:
        summaries=[];frames={}
        for encoding,run in runs.items():
            p=run/'niche_hypotheses'
            summaries.append(pd.read_csv(p/f'{name}_summary.csv').assign(encoding=encoding))
            if encoding!='full_native':frames[encoding]=pd.read_csv(p/f'{name}_organoid.csv.gz')
        pd.concat(summaries,ignore_index=True).to_csv(out/f'niche_{name}_summary.csv',index=False)
        a,b=frames['exclusive'],frames['full_matched']
        metrics=[c for c in a if c.endswith(('_z','_raw','_relative'))]
        ids=[c for c in a if c not in metrics]
        match=a.merge(b,on=ids,suffixes=('_exclusive','_full'),validate='one_to_one')
        assert len(match)==len(a)==len(b)
        for m in metrics:match[m]=match[m+'_exclusive']-match[m+'_full']
        keys=[c for c in ids if c not in ['organoid_str','seed']]
        cluster_summary(match,keys,metrics,draws=draws).to_csv(out/f'niche_{name}_difference.csv',index=False)
        if 'n' in keys and (match.n==165).any():
            ix=[c for c in ids if c!='n']
            diff=(match[match.n==791].set_index(ix)[metrics]-match[match.n==165].set_index(ix)[metrics]).reset_index()
            cluster_summary(diff,[k for k in keys if k!='n'],metrics,draws=draws).to_csv(out/f'niche_{name}_endpoint_difference.csv',index=False)
    for name in ['effect_summary','interaction_summary','common_scaling','effect_endpoint_summary','effect_endpoint_seeds']:
        pd.concat([pd.read_csv(run/'lgr5_film_diagnostics'/f'{name}.csv').assign(encoding=encoding)
            for encoding,run in runs.items()],ignore_index=True).to_csv(out/f'lgr5_{name}.csv',index=False)
    for name in ['dominance_observed_n_bin','dominance_local_slope_summary','dominance_seed_summary','dominance_switch_summary']:
        pd.concat([pd.read_csv(run/'niche_hypotheses'/f'{name}.csv').assign(encoding=encoding)
            for encoding,run in runs.items()],ignore_index=True).to_csv(out/f'niche_{name}.csv',index=False)
    if (out/'mse_overall.csv').exists() and (out/'ablation_summary.csv').exists():
        (comparison/'comparison_complete.json').write_text(json.dumps(dict(bootstrap_draws=draws,all_summaries_complete=True)))
    return out


def summarize_pair_control(comparison,draws=1000):
    comparison=Path(comparison);out=comparison/'comparison_tables'
    a=pd.read_csv(comparison/'exclusive/niche_hypotheses/pairs_organoid.csv.gz')
    b=pd.read_csv(comparison/'full_fate_pairs/pairs_organoid.csv.gz')
    metrics=[c for c in a if c.endswith(('_z','_raw','_relative'))]
    ids=[c for c in a if c not in metrics]
    match=a.merge(b,on=ids,suffixes=('_exclusive','_control'),validate='one_to_one')
    assert len(match)==len(a)==len(b)
    for m in metrics:match[m]=match[m+'_exclusive']-match[m+'_control']
    keys=['selection','hop','n']
    cluster_summary(match,keys,metrics,draws=draws).to_csv(out/'pair_full_fate_control_difference.csv',index=False)
    ix=[c for c in ids if c!='n']
    endpoints=(match[match.n==791].set_index(ix)[metrics]-match[match.n==165].set_index(ix)[metrics]).reset_index()
    cluster_summary(endpoints,['selection','hop'],metrics,draws=draws).to_csv(out/'pair_full_fate_control_endpoint_difference.csv',index=False)
    # Departure from the line joining endpoints in log N, at the original
    # reference count. This distinguishes an interior dip from endpoint change.
    counts=json.loads((comparison/'exclusive/sweep_grid.json').read_text())
    low,reference,high=counts[0],counts[len(counts)//2],counts[-1]
    fraction=(np.log(reference)-np.log(low))/(np.log(high)-np.log(low))
    bends=[]
    full=pd.read_csv(comparison/'full_on_exclusive_cases/niche_hypotheses/pairs_organoid.csv.gz')
    for encoding,frame in [('exclusive',a),('full_fate_control',b),('full_matched',full)]:
        selected=['interaction_z','interaction_relative']
        values=(frame[frame.n==reference].set_index(ix)[selected]
            -(1-fraction)*frame[frame.n==low].set_index(ix)[selected]
            -fraction*frame[frame.n==high].set_index(ix)[selected])
        values=values.rename(columns={'interaction_z':'bend_z','interaction_relative':'bend_relative'})
        bends.append(values.reset_index().assign(encoding=encoding))
    bend=pd.concat(bends,ignore_index=True);bm=['bend_z','bend_relative']
    bend.to_csv(out/'pair_bending_organoid.csv.gz',index=False)
    cluster_summary(bend,['encoding','selection','hop'],bm,draws=draws).to_csv(out/'pair_bending_summary.csv',index=False)
    bend.groupby(['encoding','selection','hop','seed'])[bm].mean().to_csv(out/'pair_bending_seeds.csv')
    delta=(bend[bend.encoding=='exclusive'].set_index(ix)[bm]-bend[bend.encoding=='full_fate_control'].set_index(ix)[bm]).reset_index()
    cluster_summary(delta,['selection','hop'],bm,draws=draws).to_csv(out/'pair_bending_control_difference.csv',index=False)
