"""Figures from cached niche hypothesis summaries; never runs inference."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter

COLORS={'Agr2':'#d99a22','Lysozyme':'#2865ad','Chroma':'#9b51a0','Serotonin':'#c84c4c',
    'precursor_only':'#d99a22','double_positive':'#2865ad','mature_only':'#c84c4c'}


def curve(ax,data,label,color=None,style='-'):
    data=data[data.n>0].sort_values('n')
    if data.empty:return
    ax.plot(data.n,data['mean'],style,label=label,color=color,lw=2)
    ax.fill_between(data.n,data.ci_low,data.ci_high,color=color,alpha=.13)


def finish(ax,ylabel='Normalized curvature change',xlabel='Supplied cell count N'):
    ax.set_xscale('log'); ax.set_xticks([165,244,361,535,791],[165,244,361,535,791])
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.axhline(0,color='.5',lw=.7); ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.grid(alpha=.15); ax.legend(fontsize=8,frameon=False)


def figures(out):
    out=Path(out); figs={}
    def read(name):return pd.read_csv(out/f'{name}_summary.csv')
    unit='delta_relative'
    d=read('specificity');d=d[d.source_state=='all_positive']
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    for i,marker in enumerate(['Lysozyme','Serotonin']):
        for j,hop in enumerate([1,2]):
            ax=axes[i,j];g=d[(d.marker==marker)&(d.hop==hop)]
            for metric,label,color in [('lgr5_'+unit,'LGR5+ recipient','#2865ad'),('other_'+unit,'LGR5-negative recipient','#c84c4c')]:
                curve(ax,g[g.metric==metric],label,color)
            finish(ax);ax.set_title(f'{marker} source → hop {hop}; paired source cohort')
    fig.suptitle('Edit one source marker; compare unchanged recipients around the same source')
    figs['01_recipient_fields']=fig
    fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    for ax,hop in zip(axes,[1,2]):
        for marker in ['Lysozyme','Serotonin']:
            curve(ax,d[(d.marker==marker)&(d.hop==hop)&(d.metric==unit)],marker,COLORS[marker])
        finish(ax,'Effect at LGR5+ minus effect at LGR5-negative');ax.set_title(f'Paired recipient specificity, hop {hop}')
    figs['02_recipient_specificity']=fig
    d=read('lineage')
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    for row,(a,b) in enumerate([('Agr2','Lysozyme'),('Chroma','Serotonin')]):
        for col,hop in enumerate([1,2]):
            ax=axes[row,col]
            for marker,state,label,style in [(a,'precursor_only',f'{a}+ / {b}−: edit {a}','-'),
                (a,'double_positive',f'double+: edit {a}','--'),(b,'double_positive',f'double+: edit {b}','-'),
                (b,'mature_only',f'{a}− / {b}+: edit {b}',':')]:
                curve(ax,d[(d.marker==marker)&(d.source_state==state)&(d.hop==hop)&(d.metric==unit)],label,COLORS[marker],style)
            finish(ax);ax.set_title(f'{a} / {b} source states → LGR5+, hop {hop}')
    fig.suptitle('Separate marker states; these curves have different eligible cohorts')
    figs['03_source_states']=fig
    d=read('dominance')
    fig,axes=plt.subplots(2,3,figsize=(16,8),layout='constrained')
    pairs=[('Agr2','Lysozyme'),('Chroma','Serotonin'),('Lysozyme','Serotonin')]
    for col,(a,b) in enumerate(pairs):
        for row,hop in enumerate([1,2]):
            ax=axes[row,col];g=d[(d.marker_a==a)&(d.marker_b==b)&(d.hop==hop)]
            for metric,label,color in [('dominance_delta_relative','Geometrically normalized','#2865ad'),('dominance_delta_z','Transformed target','#c84c4c')]:
                curve(ax,g[g.metric==metric],label,color)
            finish(ax,f'D: positive favors {a}',xlabel='Supplied cell count N')
            ax.set_ylim(-1,1);support=g.n_organoids.max()
            ax.set_title(f'{a} vs {b}, hop {hop}; {support} organoids')
    fig.suptitle('Matched recipients: relative strength D = (mean |effect A| − mean |effect B|) / their sum')
    figs['04_matched_dominance']=fig
    fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    for ax,(a,b) in zip(axes,pairs[:2]):
        g=d[(d.marker_a==a)&(d.marker_b==b)&(d.hop==1)]
        for marker,prefix in [(a,'a_'),(b,'b_')]:
            curve(ax,g[g.metric==prefix+unit],marker,COLORS[marker])
        finish(ax);ax.set_title(f'{a} vs {b}: signed effects on the same LGR5 recipients')
    figs['05_matched_signed_effects']=fig
    d=read('backup');p=read('pairs')
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    for i,marker in enumerate(['Lysozyme','Serotonin']):
        ax=axes[0,i];g=d[(d.marker==marker)&(d.hop==1)]
        for prefix,label,color in [('backup0_','No other backup within 2 hops','#c84c4c'),('backup1_','Other backup within 2 hops','#2865ad')]:
            curve(ax,g[g.metric==prefix+unit],label,color)
        finish(ax);ax.set_title(f'Edit {marker}; matched context strata, hop 1')
    ax=axes[1,0];g=p[(p.selection=='all')&(p.hop==1)]
    for metric,label,color in [('lyso_relative','Remove Lysozyme','#2865ad'),('sero_relative','Remove Serotonin','#c84c4c'),('joint_relative','Remove both','#333333')]:
        curve(ax,g[g.metric==metric],label,color)
    finish(ax);ax.set_title('Distinct-source factorial experiment; unchanged LGR5 recipient')
    ax=axes[1,1]
    for selection,label,color in [('all','All selected pairs','#2865ad'),('one_each_in_hop','Exactly one of each in the tested hop','#c84c4c')]:
        curve(ax,p[(p.selection==selection)&(p.hop==1)&(p.metric=='interaction_relative')],label,color)
    finish(ax,'Joint − Lysozyme-only − Serotonin-only effect');ax.set_title('Nonadditivity of dual marker removal')
    figs['06_redundancy']=fig
    d=read('crypt');d=d[(d.source_state=='all_positive')&(d.metric==unit)]
    fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    for ax,marker in zip(axes,['Lysozyme','Serotonin']):
        for region,label,color in [('detected_crypt','In detected crypt','#2865ad'),('outside_detected_crypt','Outside detected crypt','#d99a22'),('no_crypt_detected','No crypt detected in organoid','#c84c4c')]:
            curve(ax,d[(d.marker==marker)&(d.hop==1)&(d.recipient_region==region)],label,color)
        finish(ax);ax.set_title(f'{marker} source → LGR5+ recipient, hop 1')
    fig.suptitle('Observed crypt labels held fixed throughout each hypothetical size sweep')
    figs['07_crypt_context']=fig
    d=pd.read_csv(out/'dominance_observed_n_bin.csv');orgs=pd.read_csv(out/'organoid_metadata.csv')
    centers=orgs.groupby('n_bin').observed_n.median()
    fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    for ax,(a,b) in zip(axes,pairs[:2]):
        g=d[(d.marker_a==a)&(d.marker_b==b)&(d.hop==1)]
        for metric,label,color in [('dominance_delta_relative','Geometrically normalized','#2865ad'),('dominance_delta_z','Transformed target','#c84c4c')]:
            q=g[g.metric==metric].sort_values('n_bin');xx=q.n_bin.map(centers)
            ax.errorbar(xx,q['mean'],yerr=[q['mean']-q.ci_low,q.ci_high-q['mean']],fmt='o-',label=label,color=color,capsize=3)
        ax.axhline(0,color='.5',lw=.7);ax.set_xlabel('Median observed N in cohort quartile');ax.set_ylabel(f'D: positive favors {a}')
        ax.set_title(f'{a} vs {b}: observed-size predictions');ax.legend(fontsize=8);ax.grid(alpha=.15)
    fig.suptitle('Cross-sectional comparison: different organoids occur in different size bins')
    figs['08_observed_dominance']=fig
    d=read('prevalence');days=['day3p5','day4','day4p5','day4p5-more']
    fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    for ax,family in zip(axes,['Lysozyme','Serotonin']):
        for state in ['precursor_only','double_positive','mature_only']:
            q=d[(d.family==family)&(d.source_state==state)&(d.region=='detected_crypt')].set_index('timepoint').reindex(days)
            ax.errorbar(range(4),q['mean'],yerr=[q['mean']-q.ci_low,q.ci_high-q['mean']],fmt='o-',label=state.replace('_',' '),color=COLORS[state],capsize=3)
        ax.set_xticks(range(4),days,rotation=15);ax.set_ylabel('Cell fraction within detected crypt regions')
        ax.set_title(f'{family} family: measured marker states');ax.legend(fontsize=8);ax.grid(alpha=.15)
    fig.suptitle('Destructive snapshots; day4p5 and day4p5-more are kept as separate acquisition groups')
    figs['09_state_prevalence']=fig
    d=read('measured_curvature')
    fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    for ax,region in zip(axes,['detected_crypt','outside_detected_crypt']):
        for marker in ['LGR5','Lysozyme','Serotonin']:
            q=d[(d.marker==marker)&(d.region==region)].set_index('timepoint').reindex(days)
            ax.errorbar(range(4),q['mean'],yerr=[q['mean']-q.ci_low,q.ci_high-q['mean']],fmt='o-',label=marker,capsize=3)
        ax.set_xticks(range(4),days,rotation=15);ax.set_ylabel('Measured Gaussian curvature (dataset units)');ax.set_title(region.replace('_',' '));ax.legend();ax.grid(alpha=.15)
    fig.suptitle('Measured cell-patch curvature; marker groups overlap and do not measure apical constriction')
    figs['10_measured_curvature']=fig
    d=pd.read_csv(out/'dominance_local_slope_summary.csv')
    fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    for ax,hop in zip(axes,[1,2]):
        q=d[(d.hop==hop)&(d.metric=='dominance_delta_z')].copy()
        labels=q.marker_a+' vs '+q.marker_b
        ax.errorbar(q['mean'],range(len(q)),xerr=[q['mean']-q.ci_low,q.ci_high-q['mean']],fmt='o',capsize=3)
        ax.set_yticks(range(len(q)),labels);ax.axvline(0,color='.5',lw=.7)
        ax.set_xlabel('Local change in D per unit log N');ax.set_title(f'Bracketing actual N, hop {hop}');ax.grid(alpha=.15)
    fig.suptitle('Local sweep check: only graphs with observed N inside the supplied grid')
    figs['11_local_size_sensitivity']=fig
    d=pd.read_csv(out/'dominance_observed_timepoint.csv')
    fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    for ax,(a,b) in zip(axes,pairs[:2]):
        for metric,label,color in [('dominance_delta_relative','Geometrically normalized','#2865ad'),('dominance_delta_z','Transformed target','#c84c4c')]:
            q=d[(d.marker_a==a)&(d.marker_b==b)&(d.hop==1)&(d.metric==metric)].set_index('timepoint').reindex(days)
            ax.errorbar(range(4),q['mean'],yerr=[q['mean']-q.ci_low,q.ci_high-q['mean']],fmt='o-',label=label,color=color,capsize=3)
        ax.set_xticks(range(4),days,rotation=15);ax.axhline(0,color='.5',lw=.7)
        ax.set_ylabel(f'D: positive favors {a}');ax.set_title(f'{a} vs {b}, observed size');ax.legend(fontsize=8);ax.grid(alpha=.15)
    fig.suptitle('Snapshot acquisition groups; these comparisons are not longitudinal or size-matched')
    figs['12_snapshot_timepoints']=fig
    d=read('crypt_paired')
    fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    for ax,hop in zip(axes,[1,2]):
        for marker in ['Lysozyme','Serotonin']:
            curve(ax,d[(d.marker==marker)&(d.hop==hop)&(d.metric==unit)],marker,COLORS[marker])
        finish(ax,'Effect inside minus outside detected crypt');ax.set_title(f'Same source; LGR5 recipients in both regions, hop {hop}')
    figs['13_paired_crypt_context']=fig
    d=read('recipient_types')
    fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    for ax,marker in zip(axes,['Lysozyme','Serotonin']):
        for phenotype,color in zip(['AldoB','KI67','Agr2','Chroma','Lysozyme','Serotonin'],plt.cm.tab10.colors):
            curve(ax,d[(d.marker==marker)&(d.hop==1)&(d.phenotype==phenotype)&(d.metric==unit)],phenotype,color)
        finish(ax);ax.set_title(f'{marker} source → LGR5-negative recipient phenotypes')
    fig.suptitle('Descriptive phenotype subsets overlap and have different source cohorts')
    figs['14_other_recipient_phenotypes']=fig
    p=read('pairs');b=read('backup')
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    for col,hop in enumerate([1,2]):
        for marker in ['Lysozyme','Serotonin']:
            curve(axes[0,col],b[(b.marker==marker)&(b.hop==hop)&(b.metric=='delta_z')],marker,COLORS[marker])
        finish(axes[0,col],'Backup present − absent effect (transformed)');axes[0,col].set_title(f'Matched backup context, hop {hop}')
        for selection,label,color in [('all','All selected pairs','#2865ad'),('one_each_in_hop','Exactly one of each in tested hop','#c84c4c')]:
            curve(axes[1,col],p[(p.selection==selection)&(p.hop==hop)&(p.metric=='interaction_z')],label,color)
        finish(axes[1,col],'Factorial interaction (transformed)');axes[1,col].set_title(f'Distinct-source nonadditivity, hop {hop}')
    figs['15_transformed_redundancy_checks']=fig
    return figs


def save_figures(out):
    out=Path(out); directory=out/'figures';directory.mkdir(exist_ok=True)
    items=figures(out)
    for name,fig in items.items():
        fig.savefig(directory/f'{name}.png',dpi=160,bbox_inches='tight')
        fig.savefig(directory/f'{name}.pdf',bbox_inches='tight')
    return items
