from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.analysis.ki67_observed import MARKERS, BIN_LABELS
from src.plotting.ki67_observed import axis, curve

COLORS={'Unmarked':'#7b3294','KI67':'#b35806','LGR5':'#2166ac','AldoB':'#359449','Agr2':'#d45a31','Mixed':'.4'}


def label(name):return 'Unassigned' if name=='Unmarked' else name


def read(out,name):return pd.read_csv(Path(out)/f'{name}.csv')


def population(out):
    fig,axs=plt.subplots(1,3,figsize=(12,3.8),layout='constrained')
    curve(axs[0],read(out,'prevalence_by_size'),'fraction_unassigned','Unassigned',COLORS['Unmarked'])
    profiles=read(out,'profiles_by_size')
    for marker in ['Unmarked','KI67','LGR5','AldoB']:
        part=profiles[profiles.marker==marker]
        curve(axs[1],part,'curvature_rank',label(marker),COLORS[marker])
        curve(axs[2],part,'negative',label(marker),COLORS[marker])
    for ax,y,t in zip(axs,['Fraction of cells','Mean within-organoid curvature percentile','Fraction with measured K < 0'],
        ['A  Unassigned-cell prevalence','B  Measured curvature rank','C  Negative Gaussian curvature']):axis(ax,y);ax.set_title(t)
    axs[1].legend(fontsize=8);axs[1].axhline(.5,color='.7',ls=':')
    return fig


def context(out):
    fig,axs=plt.subplots(1,3,figsize=(12,4.2),layout='constrained')
    qc=read(out,'curvature_qc_summary');qc=qc[qc.marker=='Unmarked']
    curve(axs[0],qc,'curvature_norm','Raw graph curvature','.45')
    curve(axs[0],qc,'clean_curvature_norm','Configured outlier interpolation',COLORS['Unmarked'])
    contexts=read(out,'unassigned_context_curves')
    for name in ['Unmarked','KI67','LGR5','AldoB','Mixed']:
        curve(axs[1],contexts[contexts.context==name],'curvature_rank',label(name),COLORS[name])
    profiles=read(out,'profiles_by_size')
    for marker in ['Unmarked','KI67','LGR5']:
        curve(axs[2],profiles[profiles.marker==marker],'fraction_neck_band',label(marker),COLORS[marker])
    for a,y,t in zip(axs,['Measured K × area / (4π)','Curvature percentile of unassigned center','Fraction in qualified neck band'],
        ['A  Outlier sensitivity of unassigned means','B  Unassigned cells, grouped by neighbors','C  Only circumference-qualified territories']):axis(a,y);a.set_title(t)
    axs[0].legend(fontsize=7);axs[1].legend(title='Strict majority of hop-1 neighbors',fontsize=7,title_fontsize=7)
    axs[2].legend(fontsize=8)
    return fig


def neighborhoods(out):
    t=read(out,'profiles_by_size');t=t[t.marker=='Unmarked']
    fig,axs=plt.subplots(2,2,figsize=(11,7.3),layout='constrained')
    for i,hop in enumerate([1,2]):
        for j,kind in enumerate(['fraction','enrichment']):
            a=axs[i,j];matrix=[]
            for marker in MARKERS:
                p=t[(t.metric==f'ring{hop}_{kind}_{marker}')&t.supported].set_index('size_bin')
                matrix.append(p['mean'].reindex(range(6)).to_numpy()*100)
            matrix=np.asarray(matrix);lim=np.nanmax(np.abs(matrix))
            im=a.imshow(matrix,aspect='auto',cmap='Blues' if j==0 else 'RdBu_r',vmin=0 if j==0 else -lim,vmax=lim)
            for row in range(8):
                for col in range(6):
                    v=matrix[row,col]
                    if np.isfinite(v):a.text(col,row,f'{v:.0f}',ha='center',va='center',fontsize=8,color='white' if abs(v)>.65*lim else 'black')
            a.set_yticks(range(8),[label(m) for m in MARKERS]);a.set_xticks(range(6),BIN_LABELS,rotation=30)
            a.set_xlabel('Observed N');a.set_title(f'Hop {hop}: '+('neighbor fractions' if j==0 else 'excess over organoid composition'))
            fig.colorbar(im,ax=a,label='Percent' if j==0 else 'Percentage points')
    return fig


def neighbor_associations(out):
    raw=read(out,'neighbor_summary_unmatched');matched=read(out,'neighbor_summary_degree_matched')
    fig,axs=plt.subplots(1,3,figsize=(12,4.8),layout='constrained')
    matrices=[]
    for table,hop in [(raw,1),(raw,2),(matched,1)]:
        p=table[(table.hop==hop)&(table.metric=='curvature_rank')&table.supported]
        matrices.append(p.pivot(index='marker',columns='size_bin',values='mean').reindex(index=MARKERS,columns=range(6)).to_numpy()*100)
    limit=max(np.nanmax(np.abs(m)) for m in matrices)
    for a,m,t in zip(axs,matrices,['A  Hop 1: with − without unassigned','B  Hop 2: with − without unassigned','C  Hop 1: also match exact degree']):
        im=a.imshow(np.ma.masked_invalid(m),aspect='auto',cmap='RdBu_r',vmin=-limit,vmax=limit)
        for i in range(8):
            for j in range(6):
                if np.isfinite(m[i,j]):a.text(j,i,f'{m[i,j]:.0f}',ha='center',va='center',fontsize=8,color='white' if abs(m[i,j])>.65*limit else 'black')
        a.set_yticks(range(8),[label(v) for v in MARKERS]);a.set_xticks(range(6),BIN_LABELS,rotation=45)
        a.set_xlabel('Observed N');a.set_title(t)
    axs[0].set_ylabel('Unchanged center fate')
    fig.colorbar(im,ax=list(axs),label='Difference in measured curvature percentile (percentage points)',shrink=.8)
    return fig


def prediction_checks(out):
    table=read(out,'curvature_qc_summary');neighbors=read(out,'neighbor_qc_summary')
    fig,axs=plt.subplots(1,3,figsize=(12,4.2),layout='constrained')
    for marker in ['Unmarked','KI67','LGR5','AldoB']:
        p=table[table.marker==marker]
        curve(axs[0],p,'clean_curvature_centered',label(marker)+' measured',COLORS[marker])
        curve(axs[0],p,'prediction_centered',label(marker)+' predicted',COLORS[marker],ls='--')
    p=table[table.marker=='Unmarked'].set_index(['metric','size_bin'])
    for metric,name,color in [('squared_error','Raw measured targets','.5'),('clean_squared_error','Interpolated targets',COLORS['Unmarked'])]:
        part=p.loc[metric].reindex(range(6));axs[1].plot(part.index,np.sqrt(part['mean']),marker='o',label=name,color=color)
    axs[1].set_yscale('log')
    for marker in ['LGR5','KI67','AldoB']:
        p=neighbors[(neighbors.marker==marker)&(neighbors.hop==1)]
        curve(axs[2],p,'clean_curvature_norm',label(marker)+' measured',COLORS[marker])
        curve(axs[2],p,'prediction_centered',label(marker)+' predicted',COLORS[marker],ls='--')
    for a,y,t in zip(axs,['Curvature relative to organoid mean × area / (4π)','Within-organoid RMSE (normalized units; log scale)',
        'With − without unassigned neighbor (normalized units)'],['A  Same observed cells: mean prediction','B  Unassigned-cell prediction error','C  Does the model reproduce neighbor associations?']):axis(a,y);a.set_title(t)
    axs[0].legend(fontsize=6,ncol=2);axs[1].legend(fontsize=7);axs[2].legend(fontsize=6,ncol=2)
    return fig


def ablation_reference(out):
    table=read(out,'ablation_matrix')
    fig,axs=plt.subplots(1,2,figsize=(12,5),layout='constrained')
    matrices=[]
    for hop in [1,2]:
        p=table[(table.hop==hop)&table.supported]
        matrices.append(p.pivot(index='recipient',columns='source_marker_name',values='mean').reindex(index=MARKERS,columns=MARKERS[:-1]).to_numpy())
    limit=max(np.nanmax(np.abs(m)) for m in matrices)
    for ax,m,hop in zip(axs,matrices,[1,2]):
        im=ax.imshow(np.ma.masked_invalid(m),aspect='auto',cmap='RdBu_r',vmin=-limit,vmax=limit)
        ax.set_xticks(range(7),MARKERS[:-1],rotation=45,ha='right');ax.set_yticks(range(8),[label(v) for v in MARKERS])
        ax.set_title(f'Hop {hop}: source marker → unassigned vector')
        ax.set_xlabel('Marker cleared from one source cell');ax.set_ylabel('Unchanged recipient fate')
        for i in range(8):
            for j in range(7):
                if np.isfinite(m[i,j]):ax.text(j,i,f'{m[i,j]:+.2f}',ha='center',va='center',fontsize=7,color='white' if abs(m[i,j])>.65*limit else 'black')
    fig.colorbar(im,ax=list(axs),label='Predicted K(after) − K(before), normalized by measured area',shrink=.8)
    return fig
