"""Comparison figures for full and mutually exclusive fate vectors."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
COLORS={'full_native':'#aaaaaa','full_matched':'#2865ad','exclusive':'#d05a32'}
LABELS={'full_native':'Full markers: original cohort','full_matched':'Full markers: matched cells','exclusive':'Exclusive markers: matched cells'}
MARKERS=['Agr2','AldoB','Chroma','KI67','LGR5','Lysozyme','Serotonin']


def line(ax,g,label,color,style='-'):
    g=g[g.n>0].sort_values('n')
    if g.empty:return
    ax.plot(g.n,g['mean'],style,color=color,lw=1.8,label=label)
    ax.fill_between(g.n,g.ci_low,g.ci_high,color=color,alpha=.11)


def axis(ax,ylabel='Normalized curvature change',xlabel='Supplied N'):
    ax.set_xscale('log');ax.set_xticks([165,244,361,535,791],[165,244,361,535,791]);ax.xaxis.set_minor_formatter(NullFormatter())
    ax.axhline(0,color='.5',lw=.7);ax.set_xlabel(xlabel);ax.set_ylabel(ylabel);ax.grid(alpha=.15)


def figures(comparison):
    comparison=Path(comparison);table=comparison/'comparison_tables';items={}
    def read(name):return pd.read_csv(table/f'{name}.csv')
    audit=pd.read_csv(comparison/'exclusive/tables/exclusivity_audit.csv').groupby('marker')[['full_positive','exclusive_positive']].sum().reindex(MARKERS)
    fig,ax=plt.subplots(figsize=(10,4),layout='constrained');xx=np.arange(7)
    ax.bar(xx-.2,audit.full_positive,width=.4,label='Full markers',color=COLORS['full_matched'])
    ax.bar(xx+.2,audit.exclusive_positive,width=.4,label='Exclusive markers',color=COLORS['exclusive'])
    ax.set_xticks(xx,MARKERS);ax.set_ylabel('Positive cells');ax.set_title('Same cells and seven feature columns; ordered suppression');ax.legend()
    items['01_encoding_audit']=fig
    overall=read('mse_overall');binned=read('mse_by_size')
    fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    for i,encoding in enumerate(['full','exclusive']):
        c=COLORS['full_matched' if encoding=='full' else encoding];g=overall[overall.encoding==encoding].iloc[0]
        axes[0].bar(i,g['mean'],color=c);axes[0].vlines(i,g.ci_low,g.ci_high,color='black')
        q=binned[binned.encoding==encoding];line(axes[1],q,encoding,c)
    axes[0].set_xticks([0,1],['Full','Exclusive']);axes[0].set_ylabel('Validation MSE (physical curvature²)')
    axes[0].set_title('Equal organoid weighting; same targets and folds')
    axis(axes[1],'Validation MSE (physical curvature²)','Observed N (bin median)');axes[1].legend()
    items['02_validation_mse']=fig
    d=read('ablation_summary');d=d[d.metric=='delta_relative']
    for mode in ['sweep','observed']:
        for hop in [1,2]:
            fig,axes=plt.subplots(8,7,figsize=(20,20),layout='constrained',sharex=True)
            for row,center in enumerate([*MARKERS,'unmarked']):
                for col,source in enumerate(MARKERS):
                    ax=axes[row,col];q=d[(d['mode']==mode)&(d.hop==hop)&(d.center_marker==center)&(d.source_marker_name==source)]
                    for encoding in ['full_native','full_matched','exclusive']:
                        g=q[(q.encoding==encoding)&(q.n_organoids>=5)]
                        line(ax,g,LABELS[encoding],COLORS[encoding],':' if encoding=='full_native' else '-')
                    axis(ax,center if col==0 else '',('Observed N' if mode=='observed' else 'Supplied N') if row==7 else '')
                    ax.tick_params(labelsize=7)
                    if row==0:ax.set_title(source,fontsize=10)
            handles,labels=axes[0,0].get_legend_handles_labels()
            fig.legend(handles,labels,loc='outside lower center',ncol=3,fontsize=10)
            fig.suptitle(f'{mode.title()} size, hop {hop}: rows = unchanged recipient marker; columns = edited source marker\nNormalized effects; panels mask fewer than 5 organoids',fontsize=14)
            items[f'03_{mode}_all_markers_hop{hop}']=fig
    fig,axes=plt.subplots(2,4,figsize=(16,8),layout='constrained')
    for row,hop in enumerate([1,2]):
        for col,source in enumerate(['Agr2','Lysozyme','Chroma','Serotonin']):
            ax=axes[row,col];q=d[(d['mode']=='sweep')&(d.hop==hop)&(d.center_marker=='LGR5')&(d.source_marker_name==source)]
            for encoding in ['full_native','full_matched','exclusive']:
                line(ax,q[(q.encoding==encoding)&(q.n_organoids>=5)],LABELS[encoding],COLORS[encoding],':' if encoding=='full_native' else '-')
            axis(ax);ax.set_title(f'{source} → LGR5, hop {hop}')
    handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncol=3)
    items['04_lgr5_sweep_comparison']=fig
    fig,axes=plt.subplots(2,4,figsize=(16,8),layout='constrained')
    z=read('ablation_summary')
    for row,encoding in enumerate(['full_matched','exclusive']):
        for col,source in enumerate(['Agr2','Lysozyme','Chroma','Serotonin']):
            ax=axes[row,col];q=z[(z.encoding==encoding)&(z.hop==1)&(z.center_marker=='LGR5')&(z.source_marker_name==source)&(z.metric=='delta_relative')]
            for mode,color in [('observed','#2865ad'),('sweep','#d05a32')]:line(ax,q[q['mode']==mode],mode,color)
            axis(ax,xlabel='Observed-bin or supplied N');ax.set_title(f'{encoding}: {source} → LGR5');ax.legend(fontsize=8)
    items['05_observed_and_sweep']=fig
    fig,axes=plt.subplots(1,4,figsize=(16,4),layout='constrained')
    for ax,source in zip(axes,['Agr2','Lysozyme','Chroma','Serotonin']):
        q=d[(d['mode']=='sweep')&(d.hop==1)&(d.center_marker=='LGR5')&(d.source_marker_name==source)]
        for encoding,label,color in [('full_matched','Full model: remove one marker','#2865ad'),
            ('full_fate_control','Full model: remove all source fate','#39925b'),('exclusive','Exclusive model: remove sole marker','#d05a32')]:
            line(ax,q[q.encoding==encoding],label,color)
        axis(ax);ax.set_title(f'{source} source → LGR5 recipient')
    fig.legend(*axes[0].get_legend_handles_labels(),loc='outside lower center',ncol=3)
    items['05b_full_source_fate_control']=fig
    # Matched niche panels are generated once all checkpoint diagnostics complete.
    if not (table/'niche_dominance_summary.csv').exists():return items
    d=read('niche_dominance_summary')
    pairs=[('Agr2','Lysozyme'),('Chroma','Serotonin'),('Lysozyme','Serotonin')]
    fig,axes=plt.subplots(2,3,figsize=(15,8),layout='constrained')
    for row,hop in enumerate([1,2]):
        for col,(a,b) in enumerate(pairs):
            ax=axes[row,col];q=d[(d.marker_a==a)&(d.marker_b==b)&(d.hop==hop)&(d.metric=='dominance_delta_relative')]
            for encoding in COLORS:line(ax,q[q.encoding==encoding],LABELS[encoding],COLORS[encoding],':' if encoding=='full_native' else '-')
            axis(ax,f'D: positive favors {a}');ax.set_ylim(-1,1);ax.set_title(f'{a} vs {b}, hop {hop}')
    fig.legend(*axes[0,0].get_legend_handles_labels(),loc='outside lower center',ncol=3)
    fig.suptitle('Relative single-source sensitivity at LGR5 recipients; common positive normalization cancels in D')
    items['06_niche_dominance']=fig
    fig,axes=plt.subplots(1,3,figsize=(15,4),layout='constrained')
    for ax,(a,b) in zip(axes,pairs):
        for encoding in ['full_matched','exclusive']:
            for metric,style in [('dominance_delta_relative','-'),('dominance_delta_z','--')]:
                q=d[(d.marker_a==a)&(d.marker_b==b)&(d.hop==1)&(d.metric==metric)&(d.encoding==encoding)]
                line(ax,q,encoding+(' normalized' if style=='-' else ' transformed'),COLORS[encoding],style)
        axis(ax,f'D: positive favors {a}');ax.set_title(f'{a} vs {b}');ax.legend(fontsize=7)
    items['07_dominance_output_scale_check']=fig
    obs=read('niche_dominance_observed_n_bin');local=read('niche_dominance_local_slope_summary')
    reference=Path(json.loads((comparison/'comparison_settings.json').read_text())['reference_run'])
    orgs=pd.read_csv(reference/'niche_hypotheses/organoid_metadata.csv')
    obscenters=orgs.groupby('n_bin').observed_n.median().to_dict()
    fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    for ax,(a,b) in zip(axes,pairs[:2]):
        for encoding in COLORS:
            q=obs[(obs.marker_a==a)&(obs.marker_b==b)&(obs.hop==1)&(obs.metric=='dominance_delta_relative')&(obs.encoding==encoding)].copy()
            q['n']=q.n_bin.map(obscenters);line(ax,q,LABELS[encoding],COLORS[encoding],':' if encoding=='full_native' else '-')
        axis(ax,f'D: positive favors {a}','Observed N (quartile median)');ax.set_title(f'{a} vs {b}');ax.legend(fontsize=7)
    items['08_observed_niche_dominance']=fig
    d=read('niche_pairs_summary');backup=read('niche_backup_summary')
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    for col,hop in enumerate([1,2]):
        for encoding in COLORS:
            q=d[(d.hop==hop)&(d.selection=='all')&(d.metric=='interaction_relative')&(d.encoding==encoding)]
            line(axes[0,col],q,LABELS[encoding],COLORS[encoding],':' if encoding=='full_native' else '-')
            q=d[(d.hop==hop)&(d.selection=='all')&(d.metric=='interaction_z')&(d.encoding==encoding)]
            line(axes[1,col],q,LABELS[encoding],COLORS[encoding],':' if encoding=='full_native' else '-')
        control=pd.read_csv(comparison/'full_fate_pairs/pairs_summary.csv')
        for row,metric in [(0,'interaction_relative'),(1,'interaction_z')]:
            q=control[(control.hop==hop)&(control.selection=='all')&(control.metric==metric)]
            line(axes[row,col],q,'Full model: remove whole source vectors','#39925b')
        axis(axes[0,col],'Factorial interaction, normalized');axis(axes[1,col],'Factorial interaction, transformed')
        axes[0,col].set_title(f'Lysozyme × Serotonin around unchanged LGR5, hop {hop}')
    fig.legend(*axes[0,0].get_legend_handles_labels(),loc='outside lower center',ncol=2)
    items['09_redundancy_comparison']=fig
    fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    for ax,marker in zip(axes,['Lysozyme','Serotonin']):
        for encoding in COLORS:
            q=backup[(backup.marker==marker)&(backup.hop==1)&(backup.metric=='delta_z')&(backup.encoding==encoding)]
            line(ax,q,LABELS[encoding],COLORS[encoding],':' if encoding=='full_native' else '-')
        axis(ax,'Backup present − absent effect, transformed');ax.set_title(f'Edit {marker}: same-source context matching');ax.legend(fontsize=7)
    items['10_backup_context']=fig
    for name,metric,ylabel,title in [('specificity','delta_relative','LGR5+ minus LGR5-negative effect','Matched recipient specificity'),
        ('crypt_paired','delta_relative','Inside minus outside detected crypt effect','Same-source detected-crypt context')]:
        d=read('niche_'+name+'_summary');fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
        if name=='specificity':d=d[d.source_state=='all_positive']
        for ax,marker in zip(axes,['Lysozyme','Serotonin']):
            for encoding in COLORS:
                q=d[(d.marker==marker)&(d.hop==1)&(d.metric==metric)&(d.encoding==encoding)]
                line(ax,q,LABELS[encoding],COLORS[encoding],':' if encoding=='full_native' else '-')
            axis(ax,ylabel);ax.set_title(f'{title}: {marker}, hop 1');ax.legend(fontsize=7)
        items['11_'+name]=fig
    d=read('lgr5_effect_summary')
    fig,axes=plt.subplots(2,4,figsize=(16,8),layout='constrained')
    for row,encoding in enumerate(['full_matched','exclusive']):
        for col,marker in enumerate(['Agr2','Lysozyme','Chroma','Serotonin']):
            ax=axes[row,col]
            for route,color in zip(['full','film_only','head_only','layer1_only','layer2_only'],plt.cm.tab10.colors):
                q=d[(d.encoding==encoding)&(d.source_marker_name==marker)&(d.hop==1)&(d.metric=='delta_z')&(d.route==route)]
                line(ax,q,route,color)
            axis(ax,'Transformed curvature change');ax.set_title(f'{encoding}: {marker} → LGR5');ax.legend(fontsize=7)
    items['12_conditioning_routes']=fig
    fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    for ax,hop in zip(axes,[1,2]):
        for encoding,style in [('full_matched','-'),('exclusive','--')]:
            for marker,color in zip(['Lysozyme','Serotonin'],['#2865ad','#d05a32']):
                q=d[(d.encoding==encoding)&(d.source_marker_name==marker)&(d.hop==hop)&(d.metric=='h2_cosine')&(d.route=='full')]
                line(ax,q,f'{encoding}: {marker}',color,style)
        axis(ax,'Cosine to own N=361 hidden-response direction');ax.set_title(f'Within-model representation change, hop {hop}');ax.legend(fontsize=7)
    items['13_hidden_direction']=fig
    common=read('lgr5_common_scaling')
    fig,axes=plt.subplots(1,3,figsize=(15,4),layout='constrained')
    for ax,(a,b) in zip(axes,[('LGR5','Lysozyme'),('Chroma','Serotonin'),('Lysozyme','Serotonin')]):
        for encoding in ['full_matched','exclusive']:
            q=common[(common.encoding==encoding)&(common.route=='full')&(common.hop==1)&
                (common.metric=='delta_z')&(common.marker_a==a)&(common.marker_b==b)].rename(columns={'mismatch':'mean','mismatch_low':'ci_low','mismatch_high':'ci_high'})
            line(ax,q,LABELS[encoding],COLORS[encoding])
        axis(ax,'Response norm not explained by common gain');ax.set_title(f'{a} / {b} profile, hop 1');ax.legend(fontsize=7)
    items['14_common_scaling_check']=fig
    interaction=read('lgr5_interaction_summary')
    fig,axes=plt.subplots(1,4,figsize=(16,4),layout='constrained')
    for ax,marker in zip(axes,['Agr2','Lysozyme','Chroma','Serotonin']):
        for encoding in ['full_matched','exclusive']:
            q=interaction[(interaction.encoding==encoding)&(interaction.kind=='center_neighbor')&
                (interaction.marker_b==marker)&(interaction.hop==1)&(interaction.metric=='interaction_z')]
            line(ax,q,LABELS[encoding],COLORS[encoding])
        axis(ax,'Factorial interaction, transformed');ax.set_title(f'Center LGR5 × neighbor {marker}');ax.legend(fontsize=7)
    fig.suptitle('Secondary diagnostic: center identity is removed here; perturbation severity differs across encodings')
    items['15_center_removal_secondary']=fig
    return items


def save_figures(comparison):
    comparison=Path(comparison);folder=comparison/'figures';folder.mkdir(exist_ok=True)
    items=figures(comparison)
    for name,fig in items.items():
        fig.savefig(folder/f'{name}.png',dpi=150,bbox_inches='tight');fig.savefig(folder/f'{name}.pdf',bbox_inches='tight')
    return items
