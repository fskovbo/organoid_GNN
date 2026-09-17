"""Qualify individual crypt necks from circumference, before interpreting s=1."""
from dataclasses import dataclass, asdict
from pathlib import Path
import json
import shutil
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter, find_peaks
from src.analysis.ki67_observed import ObservedConfig, bootstrap_summary, within_organoid_contrast


@dataclass(frozen=True)
class NeckConfig:
    minimum_crypt_cells: int = 10
    minimum_relative_depth: float = .05
    plateau_max_range: float = .10
    plateau_max_slope: float = .25
    plateau_exit_rise: float = .10
    minimum_window: tuple = (.8, 1.2)
    plateau_window: tuple = (.85, 1.15)
    smoothing_points: int = 7


def classify_profile(s, circumference, config=NeckConfig()):
    """Curvature/marker-blind shape rule; rejected shapes are not proven bulges.

    Normalize circumference by C(1). A trough must have two shoulders in
    [0.5,1.5]. A plateau must persist over [.85,1.15] and open out afterwards,
    so a flat dome top is not mistaken for a tubular segment.
    """
    s, c = np.asarray(s, float), np.asarray(circumference, float)
    result = dict(profile_class='unresolved', minimum_s=np.nan, relative_depth=np.nan,
                  plateau_range=np.nan, plateau_slope=np.nan, exit_rise=np.nan)
    if s.ndim != 1 or c.shape != s.shape or len(s)<15 or not np.all(np.diff(s)>0):
        return result
    window=(s>=.5)&(s<=1.5)
    if s[0]>.5 or s[-1]<1.5 or not np.isfinite(c[window]).all() or np.any(c[window]<=0):
        return result
    scale=np.interp(1.,s,c)
    # No missing-data interpolation: the entire stored profile must be finite.
    if not np.isfinite(c).all() or scale<=0:
        return result
    smooth=savgol_filter(c/scale,config.smoothing_points,2)
    idx=np.flatnonzero(window)
    minima=find_peaks(-smooth)[0]
    candidates=[]
    for i in minima:
        if not config.minimum_window[0]<=s[i]<=config.minimum_window[1]:continue
        left=idx[s[idx]<=s[i]-.1];right=idx[s[idx]>=s[i]+.1]
        if not len(left) or not len(right):continue
        depth=min(smooth[left].max(),smooth[right].max())-smooth[i]
        candidates.append((depth,float(s[i])))
    if candidates:
        depth, location=max(candidates)
        result.update(relative_depth=depth,minimum_s=location)
        if depth>=config.minimum_relative_depth:
            result['profile_class']='local_minimum'
            return result
    band=(s>=config.plateau_window[0])&(s<=config.plateau_window[1])
    if band.sum()<5:return result
    span=float(np.ptp(smooth[band]));slope=float(np.polyfit(s[band],smooth[band],1)[0])
    exit_band=(s>=1.3)&(s<=1.5)
    rise=float(np.median(smooth[exit_band])-np.median(smooth[band]))
    result.update(plateau_range=span,plateau_slope=slope,exit_rise=rise)
    result['profile_class']='flat_section' if (span<=config.plateau_max_range and
        abs(slope)<=config.plateau_max_slope and rise>=config.plateau_exit_rise) else 'no_neck_support'
    return result


def qualified_assignment(distances, passing):
    """Never assign a cell belonging to a rejected crypt to a farther valid one."""
    distances=np.asarray(distances)
    if not len(distances):return np.full(distances.shape[1],-1),np.full(distances.shape[1],np.nan)
    nearest=distances.argmin(0)
    axis=distances[nearest,np.arange(distances.shape[1])].astype(float)
    axis[~np.asarray(passing,bool)[nearest]]=np.nan
    return nearest,axis


def run(root, config=NeckConfig(), *, output_name='exclusive_v2_profile_necks'):
    root=Path(root)
    source=root/'results_experiments/ki67_observed_neighborhoods/exclusive_v1'
    out=source.parent/output_name;out.mkdir(exist_ok=True)
    original_settings=json.loads((source/'settings.json').read_text())
    settings=json.loads(json.dumps(dict(source=str(source),neck_config=asdict(config),original=original_settings)))
    if (out/'settings.json').exists() and json.loads((out/'settings.json').read_text())!=settings:
        raise ValueError('Neck settings changed; use a separate output version')
    (out/'settings.json').write_text(json.dumps(settings,indent=2))
    if (out/'complete.json').exists():return out
    data=Path(original_settings['dataset'])
    stats_config=ObservedConfig()
    nodes=pd.read_csv(source/'nodes.csv.gz')
    organs=pd.read_csv(source/'organoids.csv')
    registry=[];profile_rows=[];node_parts=[];variants=[]
    for counter,(org,frame) in enumerate(nodes.groupby('organoid_str',sort=True)):
        frame=frame.sort_values('node').copy();n=len(frame)
        meta=json.loads((data/f'{org}_aux.json').read_text())
        with np.load(data/f'{org}.npz',allow_pickle=False) as z:
            distances=z['d_crypts_graph'];projection=z['proj_vertex_ids']
        # Trusted locally produced segmentation object arrays. Only numerical arrays are used.
        with np.load(meta['segmentation_path'],allow_pickle=True) as z:
            s=z['d_discretized'];cc=z['circumference_crypts'];lengths=z['L_crypts']
            np.testing.assert_allclose(distances,z['d_crypts'][:,projection],rtol=2e-6,atol=2e-6)
        assert len(cc)==len(distances)==len(lengths)
        nearest,_=qualified_assignment(distances,np.ones(len(distances),bool))
        records=[]
        for k,c in enumerate(cc):
            diagnostics=classify_profile(s,c,config)
            interior_count=int(np.sum((nearest==k)&(distances[k]<=1)))
            passes_shape=diagnostics['profile_class'] in ['local_minimum','flat_section']
            row=dict(organoid_str=org,crypt_id=k,size_bin=int(frame.size_bin.iloc[0]),n=n,
                crypt_length=float(lengths[k]),crypt_cells=interior_count,shape_pass=passes_shape,
                eligible=passes_shape and interior_count>=config.minimum_crypt_cells,**diagnostics)
            row['status']='low_cell_support' if passes_shape and not row['eligible'] else row['profile_class']
            records.append(row);registry.append(row)
            for x,y in zip(s,c):profile_rows.append(dict(organoid_str=org,crypt_id=k,s=x,circumference=y))
        passing=[r['eligible'] for r in records]
        nearest,axis=qualified_assignment(distances,passing)
        # Missing positions include absent, unqualified and low-support crypts.
        frame['nearest_crypt_id']=nearest;frame['unqualified_axis']=frame.axis
        frame['axis']=axis;valid=np.isfinite(axis)
        frame['qualified_neck']=valid
        frame['region']=np.select([~valid,axis<.8,axis<=1.2],
            ['unqualified_crypt','crypt_body','neck_band'],default='beyond_neck')
        if not len(distances):frame['region']='no_detected_crypt'
        frame['crypt_profile_class']=[records[k]['status'] if k>=0 else 'no_detected_crypt' for k in nearest]
        for label in ['crypt_body','neck_band','beyond_neck']:
            value=np.where(valid,frame.region==label,np.nan)
            frame[f'fraction_{label}']=value
            frame[f'enrichment_{label}']=value-np.nanmean(value) if valid.any() else value
        node_parts.append(frame)
        # Sensitivity to cell support, keeping the same shape criterion and original assignment.
        for cell_threshold in [5,10,20]:
            _,ax=qualified_assignment(distances,[r['shape_pass'] and r['crypt_cells']>=cell_threshold for r in records])
            mask=np.isfinite(ax)
            if not mask.any():continue
            for width in [.1,.2,.3]:
                band=(ax>=1-width)&(ax<=1+width)
                for cohort,sel in [('KI67',(frame.marker=='KI67').to_numpy()),
                                   ('KI67_adjacent_LGR5',((frame.marker=='KI67')&(frame.ring1_fraction_LGR5>0)).to_numpy())]:
                    if not (mask&sel).any():continue
                    variants.append(dict(organoid_str=org,size_bin=int(frame.size_bin.iloc[0]),cohort=cohort,
                        minimum_cells=cell_threshold,half_width=width,fraction=band[mask&sel].mean(),
                        enrichment=band[mask&sel].mean()-band[mask].mean()))
        if (counter+1)%200==0:print(f'Profile audit: {counter+1}/{len(organs)} organoids',flush=True)
    revised=pd.concat(node_parts,ignore_index=True)
    crypts=pd.DataFrame(registry)
    crypts.to_csv(out/'crypt_profile_registry.csv',index=False)
    pd.DataFrame(profile_rows).to_csv(out/'circumference_profiles.csv.gz',index=False)
    revised.to_csv(out/'nodes.csv.gz',index=False)
    groups=['organoid_str','n','size_bin','timepoint','day','dataset','stratum_time','marker']
    metrics=['axis','fraction_crypt_body','fraction_neck_band','fraction_beyond_neck',
             'enrichment_crypt_body','enrichment_neck_band','enrichment_beyond_neck']
    profiles=revised.groupby(groups)[metrics].mean().reset_index()
    profiles.to_csv(out/'qualified_organoid_marker_profiles.csv',index=False)
    # Existing nonpositional measurements remain exactly as originally computed.
    for name in ['organoids.csv','overview_by_size.csv','lgr5_contrasts_by_time.csv']:
        shutil.copyfile(source/name,out/name)
    for name,grouping in [('profiles_by_size',['marker','size_bin']),('profiles_size_within_time',['marker','stratum_time','size_bin'])]:
        original=pd.read_csv(source/f'{name}.csv')
        replacement=bootstrap_summary(profiles,grouping,metrics,stats_config)
        pd.concat([original[~original.metric.isin(metrics)],replacement],ignore_index=True).to_csv(out/f'{name}.csv',index=False)
    relevant=revised[(revised.marker=='KI67')&(revised.ring1_fraction_LGR5>0)]
    conditional=relevant.groupby(['organoid_str','size_bin'])[metrics].mean().reset_index()
    bootstrap_summary(conditional,['size_bin'],metrics,stats_config).to_csv(out/'ki67_adjacent_lgr5_by_size.csv',index=False)
    ki=profiles[profiles.marker=='KI67'].set_index('organoid_str');lg=profiles[profiles.marker=='LGR5'].set_index('organoid_str')
    ids=ki.index.intersection(lg.index)
    paired=ki.loc[ids,['size_bin']].copy();paired[metrics]=ki.loc[ids,metrics]-lg.loc[ids,metrics]
    bootstrap_summary(paired.reset_index(),['size_bin'],metrics,stats_config).to_csv(out/'ki67_minus_lgr5_by_size.csv',index=False)
    contrasts=[]
    for org,frame in revised[revised.marker=='LGR5'].groupby('organoid_str'):
        common=dict(organoid_str=org,size_bin=int(frame.size_bin.iloc[0]))
        # "all" retains the unchanged curvature comparison; its axis is intentionally missing.
        for row in within_organoid_contrast(frame,stats_config.minimum_lgr5_per_group):
            if row['stratum']=='all':row['axis']=np.nan
            contrasts.append(common|row)
        qualifying=frame[frame.qualified_neck]
        for row in within_organoid_contrast(qualifying,stats_config.minimum_lgr5_per_group):
            if row['stratum']=='all':
                row['stratum']='qualified_all';contrasts.append(common|row)
    measures=['curvature','curvature_norm','curvature_rank','negative','axis','ring1_fraction_AldoB','ring1_fraction_LGR5']
    bootstrap_summary(pd.DataFrame(contrasts),['stratum','size_bin'],measures,stats_config).to_csv(out/'lgr5_contrasts_by_size.csv',index=False)
    bootstrap_summary(pd.DataFrame(variants),['cohort','minimum_cells','half_width','size_bin'],['fraction','enrichment'],stats_config).to_csv(out/'neck_sensitivity.csv',index=False)
    support=[]
    for b,g in organs.groupby('size_bin'):
        c=crypts[crypts.size_bin==b];f=revised[revised.size_bin==b]
        support.append(dict(size_bin=b,organoids=len(g),detected_organoids=int(g.detected_crypt.sum()),crypts=len(c),
            shape_pass=int(c.shape_pass.sum()),eligible_crypts=int(c.eligible.sum()),eligible_organoids=c[c.eligible].organoid_str.nunique(),
            ki67_organoids=f[(f.marker=='KI67')&f.qualified_neck].organoid_str.nunique()))
    pd.DataFrame(support).to_csv(out/'neck_support.csv',index=False)
    (out/'complete.json').write_text(json.dumps(dict(crypts=len(crypts),shape_counts=crypts.profile_class.value_counts().to_dict(),
        eligible_crypts=int(crypts.eligible.sum()),eligible_organoids=int(crypts[crypts.eligible].organoid_str.nunique())),indent=2))
    return out
