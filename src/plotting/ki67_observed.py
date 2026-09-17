"""Observed data displays: organoids, not cells, are the replicate unit."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.analysis.ki67_observed import BIN_LABELS, MARKERS

COLORS={'KI67':'#b35806','LGR5':'#2166ac','AldoB':'#359449'}


def read(out,name):
    return pd.read_csv(Path(out)/f'{name}.csv')


def axis(ax,ylabel):
    ax.set_xticks(range(6),BIN_LABELS,rotation=30)
    ax.set_xlabel('Observed cell count N (size bins)')
    ax.set_ylabel(ylabel)
    ax.grid(alpha=.15)


def curve(ax,table,metric,label,color,**kw):
    t=table[(table.metric==metric)&table.supported].sort_values('size_bin')
    # Reindex so unsupported intermediate bins are gaps rather than connected lines.
    t=t.set_index('size_bin').reindex(range(6))
    ax.plot(t.index,t['mean'],marker='o',color=color,label=label,**kw)
    ax.fill_between(t.index,t.low,t.high,color=color,alpha=.12)


def overview(out):
    org=read(out,'organoids');summary=read(out,'overview_by_size')
    fig,axs=plt.subplots(1,3,figsize=(12,3.8),layout='constrained')
    counts=pd.crosstab(org.size_bin,org.stratum_time).reindex(range(6),fill_value=0)
    counts.plot.bar(stacked=True,ax=axs[0],colormap='tab10',width=.8)
    axs[0].legend(fontsize=6,title='Collection / batch',title_fontsize=7)
    for metric,label,color in [('full_ki67_fraction','Before exclusion','.5'),('fraction_ki67','Exclusive KI67',COLORS['KI67'])]:
        curve(axs[1],summary,metric,label,color)
    curve(axs[2],summary,'detected_crypt','At least one detected crypt','#2166ac')
    for a,y in zip(axs,['Number of organoids','Mean fraction of all cells','Fraction of organoids']):axis(a,y)
    axs[1].legend(fontsize=8)
    for a,title in zip(axs,['A  Size and collection-time support','B  Which KI67 population is retained?','C  Crypt-detection support']):a.set_title(title)
    return fig


def measured_curvature(out):
    t=read(out,'profiles_by_size')
    fig,axs=plt.subplots(1,3,figsize=(12,3.8),layout='constrained')
    for marker,color in COLORS.items():
        for a,metric in zip(axs,['curvature_norm','curvature_rank','negative']):
            curve(a,t[t.marker==marker],metric,marker,color)
    for a,y,title in zip(axs,['Measured K × area / (4π)','Mean within-organoid curvature percentile','Fraction of cells with measured K < 0'],
                        ['A  Size-normalized measured curvature','B  Relative position in curvature distribution','C  Negative Gaussian curvature']):
        axis(a,y);a.set_title(title)
    axs[0].legend(fontsize=8)
    axs[1].axhline(.5,color='.5',ls=':',lw=1)
    return fig


def neighborhood_content(out):
    t=read(out,'profiles_by_size');t=t[t.marker=='KI67']
    fig,axs=plt.subplots(2,2,figsize=(12,7.5),layout='constrained')
    for i,hop in enumerate([1,2]):
        for j,kind in enumerate(['fraction','enrichment']):
            a=axs[i,j]
            matrix=[]
            for m in MARKERS:
                p=t[(t.metric==f'ring{hop}_{kind}_{m}')&t.supported].set_index('size_bin')
                matrix.append(p['mean'].reindex(range(6)).to_numpy())
            matrix=np.asarray(matrix)*100
            limit=max(1,float(np.nanmax(np.abs(matrix))))
            im=a.imshow(matrix,aspect='auto',cmap='Blues' if j==0 else 'RdBu_r',
                vmin=0 if j==0 else -limit,vmax=limit)
            for row in range(len(MARKERS)):
                for col in range(6):
                    v=matrix[row,col]
                    if np.isfinite(v):a.text(col,row,f'{v:.0f}',ha='center',va='center',fontsize=8,
                        color='white' if abs(v)>.65*limit else 'black')
            a.set_yticks(range(len(MARKERS)),MARKERS)
            a.set_xticks(range(6),BIN_LABELS,rotation=30)
            a.set_xlabel('Observed N')
            a.set_title(f'Hop {hop}: '+('neighbor marker fractions' if j==0 else 'excess over organoid composition'))
            fig.colorbar(im,ax=a,label='Percent' if j==0 else 'Percentage points')
    return fig


def crypt_position(out):
    t=read(out,'profiles_by_size');paired=read(out,'ki67_minus_lgr5_by_size')
    conditional=read(out,'ki67_adjacent_lgr5_by_size')
    fig,axs=plt.subplots(1,3,figsize=(12,4),layout='constrained')
    for marker,color in COLORS.items():
        curve(axs[0],t[t.marker==marker],'fraction_neck_band',marker,color)
        curve(axs[1],t[t.marker==marker],'enrichment_neck_band',marker,color)
    curve(axs[0],conditional,'fraction_neck_band','KI67 adjacent to LGR5',COLORS['KI67'],ls='--')
    curve(axs[1],conditional,'enrichment_neck_band','KI67 adjacent to LGR5',COLORS['KI67'],ls='--')
    curve(axs[2],paired,'axis','KI67 − LGR5 (same organoid)',COLORS['KI67'])
    for a,y,title in zip(axs,['Fraction in estimated neck band','Neck fraction minus all-cell neck fraction','Difference in mean normalized crypt distance'],
        ['A  Locations when a crypt is detected','B  Location enrichment within organoids','C  KI67 farther from crypt base than LGR5?']):
        axis(a,y);a.set_title(title)
    if (Path(out)/'neck_support.csv').exists():
        axs[0].set_title('A  Only circumference-qualified crypt territories')
        axs[1].set_ylabel('Neck fraction minus eligible-territory cell fraction')
    for a in axs[1:]:a.axhline(0,color='.5',ls=':',lw=1)
    axs[0].legend(fontsize=8)
    return fig


def lgr5_neighbors(out):
    t=read(out,'lgr5_contrasts_by_size')
    fig,axs=plt.subplots(1,3,figsize=(12,4.1),layout='constrained')
    for a,metric in zip(axs[:2],['curvature_norm','axis']):
        stratum='qualified_all' if metric=='axis' and 'qualified_all' in t.stratum.values else 'all'
        curve(a,t[t.stratum==stratum],metric,'Within the same organoid','#2166ac')
    for stratum,color in [('crypt_body','#2166ac'),('neck_band','#b35806'),('beyond_neck','#359449'),('no_detected_crypt','.5')]:
        curve(axs[2],t[t.stratum==stratum],'curvature_norm',stratum.replace('_',' '),color)
    for a,y,title in zip(axs,['Difference in measured K × area / (4π)','Difference in normalized crypt distance','Difference in measured K × area / (4π)'],
        ['A  LGR5 with KI67 nearby − without','B  Is that LGR5 cell nearer the neck?','C  Curvature contrast within position bands']):
        axis(a,y);a.set_title(title);a.axhline(0,color='.5',lw=.8)
    axs[2].legend(fontsize=7)
    if (Path(out)/'neck_support.csv').exists():
        axs[1].set_title('B  Position within qualified crypt territories')
        axs[2].set_title('C  Qualified bands; no-detected-crypt separate')
    return fig


def neck_profile_audit(out):
    registry=read(out,'crypt_profile_registry')
    profiles=pd.read_csv(Path(out)/'circumference_profiles.csv.gz')
    support=read(out,'neck_support')
    fig,axs=plt.subplots(2,2,figsize=(11,7),layout='constrained')
    for ax,kind,title in zip(axs.ravel()[:3],['local_minimum','flat_section','no_neck_support'],
                            ['Local-minimum profiles','Approximately flat sections','Profiles without neck support']):
        candidates=registry[registry.profile_class==kind]
        chosen=candidates.sample(min(12,len(candidates)),random_state=1841)
        for row in chosen.itertuples():
            part=profiles[(profiles.organoid_str==row.organoid_str)&(profiles.crypt_id==row.crypt_id)]
            ax.plot(part.s,part.circumference/np.interp(1,part.s,part.circumference),alpha=.5,lw=1)
        ax.axvspan(.8,1.2,color='.7',alpha=.2)
        ax.axvline(1,color='.3',ls=':',lw=1)
        ax.set_xlim(.4,1.6);ax.set_xlabel('Normalized crypt coordinate s')
        ax.set_ylabel('Circumference / C(1)')
        ax.set_title(f'{title} ({len(candidates):,} crypts)')
    a=axs[1,1]
    for column,label,color in [('detected_organoids','Any detected crypt','.6'),
                                ('eligible_organoids','Qualified crypt with ≥10 cells','#2166ac'),
                                ('ki67_organoids','KI67 in a qualified territory','#b35806')]:
        a.plot(support.size_bin,support[column],marker='o',label=label,color=color)
    axis(a,'Number of organoids');a.set_title('Support after individual-crypt qualification')
    a.legend(fontsize=8)
    return fig


def time_checks(out):
    t=read(out,'profiles_size_within_time');t=t[t.marker=='KI67']
    fig,axs=plt.subplots(1,3,figsize=(12,4),layout='constrained')
    for (stratum,p),color in zip(t.groupby('stratum_time'),plt.get_cmap('tab10').colors):
        for a,metric in zip(axs,['curvature_rank','axis','ring1_fraction_LGR5']):
            curve(a,p,metric,stratum,color)
    for a,y,title in zip(axs,['Mean within-organoid curvature percentile','Mean normalized crypt distance','Fraction of hop-1 neighbors that are LGR5'],
        ['A  KI67 curvature within collection / batch','B  KI67 position within collection / batch','C  KI67 neighborhood within collection / batch']):
        axis(a,y);a.set_title(title)
    axs[0].legend(fontsize=6)
    axs[1].axhline(1,color='.5',ls=':',lw=.8)
    return fig
