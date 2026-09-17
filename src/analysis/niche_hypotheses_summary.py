"""Paired, organoid-weighted summaries for snapshot niche hypotheses."""
from pathlib import Path
import itertools
import json
import numpy as np
import pandas as pd
from src.analysis.lgr5_film_summary import cluster_summary

METRICS=['delta_z','delta_raw','delta_relative']


def summarize(out, draws=1000):
    out=Path(out)
    sm=pd.read_csv(out/'recipient_manifest.csv.gz')
    pm=pd.read_csv(out/'pair_manifest.csv.gz')
    cells=pd.read_csv(out/'cell_census.csv.gz')
    config=json.loads((out/'settings.json').read_text()); counts=config['counts']
    # Sampling and descriptive prevalence refer to actual observed sizes only.
    orgs=cells[['organoid_str','observed_n','timepoint','dataset']].drop_duplicates()
    edges=np.unique(np.quantile(orgs.observed_n,[0,.25,.5,.75,1])); edges[0]-=1; edges[-1]+=1
    orgs['n_bin']=pd.cut(orgs.observed_n,edges,labels=False)
    nb=orgs.set_index('organoid_str').n_bin
    cells['n_bin']=cells.organoid_str.map(nb)
    sm['n_bin']=sm.organoid_str.map(nb)
    sm['recipient_class']=np.where(sm.recipient_lgr5==1,'LGR5+','LGR5−')
    sm['degree_bin']=(sm.recipient_degree//3).astype(int)
    sm['lgr5_bin']=np.minimum((sm.local_lgr5_fraction*4).astype(int),3)
    # Exact coarsened context strata. Same organoid controls N, timepoint and batch.
    context_keys=['organoid_str','source','hop','source_state','recipient_markers','recipient_region','degree_bin','lgr5_bin']
    context_meta=[]
    for marker, backup in [('Lysozyme','sero'),('Serotonin','lyso')]:
        m=sm[(sm.marker==marker)&(sm.recipient_lgr5==1)&(sm[f'source_{backup}']==0)].copy()
        m['backup_present']=(m[f'backup_{backup}']>0).astype(int)
        eligible=m.groupby(context_keys).backup_present.nunique()
        eligible=eligible[eligible==2].reset_index()[context_keys]
        m=m.merge(eligible,on=context_keys,validate='many_to_one')
        context_meta.append(m[['case_id','backup_present']])
    cm=pd.concat(context_meta,ignore_index=True)
    cm.to_csv(out/'backup_matched_cases.csv',index=False)
    metadata=dict(n_bin_edges=edges.tolist(),bootstrap_draws=draws,
        bootstrap_unit='organoid; fitted seeds averaged, not independent replicates',
        backup_matching=context_keys,backup_excludes_source_coexpressing_backup=True)
    (out/'summary_settings.json').write_text(json.dumps(metadata,indent=2))
    orgs.to_csv(out/'organoid_metadata.csv',index=False)
    # Cell fractions, measured curvature and marker-state availability: equal organoid weighting.
    census=[]
    for family,col in [('Lysozyme','lyso_state'),('Serotonin','sero_state')]:
        for state in ['precursor_only','double_positive','mature_only']:
            temp=cells.assign(fraction=(cells[col]==state).astype(float))
            g=temp.groupby(['organoid_str','timepoint','n_bin','region'],observed=True).agg(
                fraction=('fraction','mean'),n_cells=('fraction','size')).reset_index()
            census.append(g.assign(family=family,source_state=state))
    census=pd.concat(census,ignore_index=True)
    census.to_csv(out/'census_by_organoid.csv',index=False)
    cluster_summary(census,['timepoint','region','family','source_state'],['fraction'],draws=draws).to_csv(out/'prevalence_summary.csv',index=False)
    measured=[]
    for marker in ['LGR5','Agr2','Lysozyme','Chroma','Serotonin']:
        g=cells[cells[marker]==1].groupby(['organoid_str','timepoint','n_bin','region']).curvature.mean().reset_index()
        measured.append(g.assign(marker=marker))
    measured=pd.concat(measured,ignore_index=True)
    measured.to_csv(out/'measured_curvature_by_organoid.csv',index=False)
    cluster_summary(measured,['timepoint','region','marker'],['curvature'],draws=draws).to_csv(out/'measured_curvature_summary.csv',index=False)
    results={k:[] for k in ['field','specificity','lineage','dominance','backup','pairs','crypt','crypt_paired','timepoint','observed','recipient_types']}
    dominance_cases=[]
    support=[]
    def store(name, frame, keys, metrics):
        if len(frame):
            grouped=frame.groupby(['organoid_str',*keys],observed=True,dropna=False)[metrics].mean().reset_index()
            results[name].append(grouped)
    for fold in sorted(sm.fold.unique()):
        ms=sm[sm.fold==fold]; mp=pm[pm.fold==fold]
        for seed in config['seeds']:
            for n in counts+['observed']:
                suffix=f'fold_{fold}_seed_{seed}_N{n}'
                s=pd.read_csv(out/f'{suffix}_singles.csv.gz').merge(ms,on='case_id',validate='one_to_one')
                p=pd.read_csv(out/f'{suffix}_pairs.csv.gz').merge(mp,on='case_id',validate='one_to_one')
                assert len(s)==len(ms) and len(p)==len(mp)
                s['seed']=seed; s['n']=0 if n=='observed' else n
                p['seed']=seed; p['n']=s.n.iloc[0]
                basekeys=['seed','n']
                # Average recipients within each edited source first, then source cells within organoid.
                sourcekeys=['organoid_str','source','marker','source_state','hop','recipient_class',*basekeys]
                sf=s.groupby(sourcekeys,observed=True)[METRICS].mean().reset_index()
                store('field',sf,['marker','source_state','hop','recipient_class',*basekeys],METRICS)
                store('field',sf.assign(source_state='all_positive'),['marker','source_state','hop','recipient_class',*basekeys],METRICS)
                if n=='observed':
                    store('observed',s[s.recipient_lgr5==1],['marker','source_state','hop','n_bin',*basekeys],METRICS)
                    store('timepoint',s[s.recipient_lgr5==1],['marker','source_state','hop','timepoint',*basekeys],METRICS)
                    store('observed',s[s.recipient_lgr5==1].assign(source_state='all_positive'),['marker','source_state','hop','n_bin',*basekeys],METRICS)
                    store('timepoint',s[s.recipient_lgr5==1].assign(source_state='all_positive'),['marker','source_state','hop','timepoint',*basekeys],METRICS)
                # Paired LGR5+ minus LGR5-negative recipients around exactly the same source/hop.
                ix=['organoid_str','source','marker','source_state','hop',*basekeys]
                wide=sf.pivot(index=ix,columns='recipient_class',values=METRICS).dropna()
                contrast=pd.DataFrame({m:wide[(m,'LGR5+')]-wide[(m,'LGR5−')] for m in METRICS}).reset_index()
                for m in METRICS:
                    contrast['lgr5_'+m]=wide[(m,'LGR5+')].to_numpy()
                    contrast['other_'+m]=wide[(m,'LGR5−')].to_numpy()
                specificity_metrics=METRICS+['lgr5_'+m for m in METRICS]+['other_'+m for m in METRICS]
                store('specificity',contrast,['marker','source_state','hop',*basekeys],specificity_metrics)
                store('specificity',contrast.assign(source_state='all_positive'),['marker','source_state','hop',*basekeys],specificity_metrics)
                # Secondary recipient phenotypes overlap; LGR5+ recipients are excluded from all comparators.
                for phenotype in ['AldoB','KI67','Agr2','Chroma','Lysozyme','Serotonin']:
                    mask=(s.recipient_lgr5==0)&s.recipient_markers.fillna('').str.split('|').apply(lambda a: phenotype in a)
                    t=s[mask].groupby(['organoid_str','source','marker','hop',*basekeys])[METRICS].mean().reset_index()
                    store('recipient_types',t.assign(phenotype=phenotype),['marker','hop','phenotype',*basekeys],METRICS)
                l=s[s.recipient_lgr5==1].copy()
                store('lineage',l,['marker','source_state','hop',*basekeys],METRICS)
                store('crypt',l,['marker','source_state','hop','recipient_region',*basekeys],METRICS)
                store('lineage',l.assign(source_state='all_positive'),['marker','source_state','hop',*basekeys],METRICS)
                store('crypt',l.assign(source_state='all_positive'),['marker','source_state','hop','recipient_region',*basekeys],METRICS)
                cr=l.groupby(['organoid_str','source','marker','hop',*basekeys,'recipient_region'])[METRICS].mean().unstack('recipient_region')
                if all(region in cr.columns.levels[1] for region in ['detected_crypt','outside_detected_crypt']):
                    cr=cr.loc[:,pd.IndexSlice[:,['detected_crypt','outside_detected_crypt']]].dropna()
                    cc=pd.DataFrame({m:cr[(m,'detected_crypt')]-cr[(m,'outside_detected_crypt')] for m in METRICS}).reset_index()
                    store('crypt_paired',cc,['marker','hop',*basekeys],METRICS)
                # Mean absolute single-source sensitivity at a fixed recipient; excludes double-positive precursor edits.
                l=l[((l.marker=='Agr2')&(l.source_state=='precursor_only'))|
                    ((l.marker=='Chroma')&(l.source_state=='precursor_only'))|l.marker.isin(['Lysozyme','Serotonin'])].copy()
                for metric in METRICS: l['abs_'+metric]=l[metric].abs()
                idkeys=['organoid_str','recipient','hop',*basekeys]
                mm=l.groupby([*idkeys,'marker'])[METRICS+['abs_'+m for m in METRICS]].mean().unstack('marker')
                for a,b in itertools.combinations(['Agr2','Lysozyme','Chroma','Serotonin'],2):
                    if a not in mm.columns.levels[1] or b not in mm.columns.levels[1]: continue
                    subset=mm.loc[:,pd.IndexSlice[:,[a,b]]].dropna()
                    if not len(subset): continue
                    record=pd.DataFrame(index=subset.index)
                    for metric in METRICS:
                        av,bv=subset[('abs_'+metric,a)],subset[('abs_'+metric,b)]
                        record['a_'+metric]=subset[(metric,a)]; record['b_'+metric]=subset[(metric,b)]
                        record['strength_a_'+metric]=av; record['strength_b_'+metric]=bv
                        record['dominance_'+metric]=(av-bv)/(av+bv).where(av+bv>1e-10)
                    record=record.reset_index().assign(marker_a=a,marker_b=b)
                    columns=[c for c in record if c.endswith(tuple(METRICS))]
                    store('dominance',record,['marker_a','marker_b','hop',*basekeys],columns)
                    if n in [counts[0],counts[-1]]: dominance_cases.append(record)
                    if seed==config['seeds'][0] and n==counts[0]:
                        support.append(dict(analysis='dominance',marker_a=a,marker_b=b,hop=None,fold=fold,
                            n_recipients=len(record[['organoid_str','recipient','hop']].drop_duplicates()),n_organoids=record.organoid_str.nunique()))
                # Backup presence contrasts on matched within-organoid context strata.
                matched=s.merge(cm,on='case_id',validate='one_to_one')
                mk=[*context_keys,'marker',*basekeys]
                bm=matched.groupby([*mk,'backup_present'])[METRICS].mean().unstack('backup_present').dropna()
                if len(bm):
                    bc=pd.DataFrame({m:bm[(m,1)]-bm[(m,0)] for m in METRICS}).reset_index()
                    for m in METRICS:
                        bc['backup0_'+m]=bm[(m,0)].to_numpy(); bc['backup1_'+m]=bm[(m,1)].to_numpy()
                    store('backup',bc,['marker','hop',*basekeys],METRICS+['backup0_'+m for m in METRICS]+['backup1_'+m for m in METRICS])
                # Average source-pair draws inside recipient before organoid means.
                pmetrics=[c for c in p if c.endswith(('_z','_raw','_relative'))]
                for selection,mask in [('all',np.ones(len(p),dtype=bool)),
                    ('one_each_in_hop',(p.count_lyso==1)&(p.count_sero==1)&(p.cross_positive==0))]:
                    pp=p[mask].groupby(['organoid_str','recipient','hop',*basekeys])[pmetrics].mean().reset_index()
                    store('pairs',pp.assign(selection=selection),['selection','hop',*basekeys],pmetrics)
            print(f'Summarized fold={fold} seed={seed}',flush=True)
    pd.DataFrame(support).to_csv(out/'matched_support.csv',index=False)
    for name,parts in results.items():
        if not parts: continue
        frame=pd.concat(parts,ignore_index=True)
        frame.to_csv(out/f'{name}_organoid.csv.gz',index=False)
        keys=[c for c in frame if c not in ['organoid_str','seed'] and not c.endswith(('_z','_raw','_relative'))]
        metrics=[c for c in frame if c.endswith(('_z','_raw','_relative'))]
        cluster_summary(frame,keys,metrics,draws=draws).to_csv(out/f'{name}_summary.csv',index=False)
        frame.groupby([*keys,'seed'])[metrics].mean().reset_index().to_csv(out/f'{name}_seed_summary.csv',index=False)
        if 'n' in keys and frame.n.max()>0:
            ek=[k for k in keys if k!='n']
            # Endpoints are paired within organoid AND fitted seed before averaging seeds.
            low=frame[frame.n==counts[0]].set_index([*ek,'organoid_str','seed'])[metrics]
            high=frame[frame.n==counts[-1]].set_index([*ek,'organoid_str','seed'])[metrics]
            endpoint=(high-low).dropna(how='all').reset_index()
            cluster_summary(endpoint,ek,metrics,draws=draws).to_csv(out/f'{name}_endpoint.csv',index=False)
            endpoint.groupby([*ek,'seed'])[metrics].mean().reset_index().to_csv(out/f'{name}_endpoint_seeds.csv',index=False)
    dc=pd.concat(dominance_cases,ignore_index=True)
    dc.to_csv(out/'dominance_endpoint_cases.csv.gz',index=False)
    # A switch is an absolute-sensitivity rank change, not an additive share of the prediction.
    ix=['organoid_str','recipient','hop','marker_a','marker_b','seed']
    wide=dc.pivot(index=ix,columns='n',values='dominance_delta_z').dropna()
    switches=wide.reset_index()[ix].copy()
    switches['a_to_b']=((wide[counts[0]]>0)&(wide[counts[-1]]<0)).to_numpy().astype(float)
    switches['b_to_a']=((wide[counts[0]]<0)&(wide[counts[-1]]>0)).to_numpy().astype(float)
    switches.to_csv(out/'dominance_switch_cases.csv.gz',index=False)
    cluster_summary(switches,['hop','marker_a','marker_b'],['a_to_b','b_to_a'],draws=draws).to_csv(out/'dominance_switch_summary.csv',index=False)
    # Matching balance and support at the static case level; no model outcomes used for matching.
    balance=sm.merge(cm,on='case_id')
    balance.groupby(['marker','hop','backup_present']).agg(n_cases=('case_id','size'),n_organoids=('organoid_str','nunique'),
        mean_n=('observed_n','mean'),mean_degree=('recipient_degree','mean'),mean_lgr5_fraction=('local_lgr5_fraction','mean')).to_csv(out/'backup_balance.csv')
    covariates=['observed_n','recipient_degree','local_lgr5_fraction']
    weighted=balance.groupby([*context_keys,'marker','backup_present'])[covariates].mean().reset_index()
    weighted=weighted.groupby(['organoid_str','marker','hop','backup_present'])[covariates].mean().reset_index()
    weighted.groupby(['marker','hop','backup_present'])[covariates].mean().to_csv(out/'backup_balance_analysis_weighted.csv')
    dominance=pd.concat(results['dominance'],ignore_index=True)
    dm=['dominance_delta_z','dominance_delta_relative']
    observed=dominance[dominance.n==0].merge(orgs,on='organoid_str',validate='many_to_one')
    for stratum in ['timepoint','n_bin']:
        cluster_summary(observed,['marker_a','marker_b','hop',stratum],dm,draws=draws).to_csv(out/f'dominance_observed_{stratum}.csv',index=False)
    # Local sensitivity: nearest bracketing grid points to each graph's actual N.
    # This avoids making the full endpoint extrapolation the sole size test.
    local=[]
    actual=orgs.set_index('organoid_str').observed_n
    for keys,g in dominance[dominance.n>0].groupby(['organoid_str','marker_a','marker_b','hop','seed']):
        observed_n=actual.loc[keys[0]]
        if not counts[0] <= observed_n <= counts[-1]: continue
        g=g.sort_values('n').reset_index(drop=True)
        hi=min(max(int(np.searchsorted(g.n.to_numpy(),observed_n)),1),len(g)-1); lo=hi-1
        diff=(g.loc[hi,dm]-g.loc[lo,dm])/(np.log(g.loc[hi,'n'])-np.log(g.loc[lo,'n']))
        local.append(dict(zip(['organoid_str','marker_a','marker_b','hop','seed'],keys))|diff.to_dict())
    local=pd.DataFrame(local)
    local.to_csv(out/'dominance_local_slopes.csv',index=False)
    cluster_summary(local,['marker_a','marker_b','hop'],dm,draws=draws).to_csv(out/'dominance_local_slope_summary.csv',index=False)
    return out
