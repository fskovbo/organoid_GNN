"""Intact quality, single-neighbor masking quality and training robustness."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.analysis.pseudotime import summarize_by_organoid

COLORS = ['#555555','#3579b8','#d56a24','#6a9b42','#9063a6']
METRICS = ['mse_mu','mse_z','bias_z','nll_z','coverage_50','coverage_80','coverage_95',
           'standardized_error','standardized_error2']


def add_size_bins(frame):
    frame=frame.copy()
    frame['size_bin']=pd.cut(frame.observed_n,[0,150,250,350,500,800,np.inf],right=False,labels=False)
    return frame


def benchmark_tables(result, *, bootstrap_samples=500):
    """Intervals average seeds within organoids; seed spread is exported separately."""
    intact=add_size_bins(result['intact']); masked=add_size_bins(result['masked']) if not result['masked'].empty else result['masked']
    pairs=add_size_bins(result['pair_effects']) if not result['pair_effects'].empty else result['pair_effects']
    rows=[]
    for metric in METRICS:
        tab=summarize_by_organoid(intact,['mask_rate','center_marker','size_bin'],metric,
                                 bootstrap_samples=bootstrap_samples)
        pos=intact.groupby(['mask_rate','center_marker','size_bin'],observed=True).observed_n.median().rename('n').reset_index()
        tab=tab.merge(pos,on=['mask_rate','center_marker','size_bin']);tab['metric']=metric;rows.append(tab)
    intact_summary=pd.concat(rows,ignore_index=True)
    seed_quality=intact[intact.center_marker=='All'].groupby(['mask_rate','seed'])[METRICS].mean().reset_index()
    if masked.empty:
        return dict(intact=intact_summary,seed_quality=seed_quality,masked=pd.DataFrame(),
                    effects=pd.DataFrame(),seed_effects=pd.DataFrame(),pair_quality=pd.DataFrame())
    quality=[]
    for state in ['intact','mask']:
        for metric in METRICS:
            tab=summarize_by_organoid(masked,['mask_rate','hop','size_bin'],f'{state}_{metric}',
                                      bootstrap_samples=bootstrap_samples)
            pos=masked.groupby(['mask_rate','hop','size_bin'],observed=True).observed_n.median().rename('n').reset_index()
            tab=tab.merge(pos,on=['mask_rate','hop','size_bin']);tab['metric']=metric;tab['state']=state;quality.append(tab)
    for metric in ['delta_mse_z','delta_mse_mu']:
        tab=summarize_by_organoid(masked,['mask_rate','hop','size_bin'],metric,bootstrap_samples=bootstrap_samples)
        pos=masked.groupby(['mask_rate','hop','size_bin'],observed=True).observed_n.median().rename('n').reset_index()
        tab=tab.merge(pos,on=['mask_rate','hop','size_bin']);tab['metric']=metric;tab['state']='difference';quality.append(tab)
    keys=['mask_rate','seed','center_marker','source_marker_name','hop']
    # Equal organoid weight, even when an organoid has multiple eligible centers.
    seed_effects=pairs.groupby(keys+['organoid_str'],observed=True).agg(
        effect=('mask_delta_relative','mean'),delta_mse_z=('delta_mse_z','mean')).reset_index()
    seed_effects=seed_effects.groupby(keys,observed=True).agg(effect=('effect','mean'),
        delta_mse_z=('delta_mse_z','mean'),n_organoids=('organoid_str','nunique')).reset_index()
    effects=pairs.groupby(keys+['size_bin','organoid_str'],observed=True).mask_delta_relative.mean().reset_index()
    effects=effects.groupby(keys+['size_bin'],observed=True).agg(
        effect=('mask_delta_relative','mean'),n_organoids=('organoid_str','nunique')).reset_index()
    pair_rows=[]
    for state in ['intact','mask']:
        for metric in ['mse_z','bias_z','coverage_95']:
            tab=summarize_by_organoid(pairs,['mask_rate','center_marker','source_marker_name','hop','size_bin'],
                                      f'{state}_{metric}',bootstrap_samples=bootstrap_samples)
            tab['state']=state;tab['metric']=metric;pair_rows.append(tab)
    return dict(intact=intact_summary,masked=pd.concat(quality,ignore_index=True),seed_quality=seed_quality,
                seed_effects=seed_effects,effects=effects,pair_quality=pd.concat(pair_rows,ignore_index=True))


def _line(ax, part, color, label, *, style='-'):
    part=part.sort_values('n')
    ax.plot(part.n,part['mean'],color=color,label=label,ls=style,marker='.')
    ax.fill_between(part.n,part.ci_low,part.ci_high,color=color,alpha=.10)
    ax.set_xscale('log');ax.set_xlabel('Observed N (bin median)')


def plot_intact_quality(tables):
    frame=tables['intact'];frame=frame[frame.center_marker=='All']
    fig,axes=plt.subplots(1,3,figsize=(15,4))
    for rate,color in zip(sorted(frame.mask_rate.unique()),COLORS):
        for ax,metric,title in zip(axes,['mse_mu','mse_z','coverage_95'],
                                  ['Curvature MSE','Transformed-target MSE','95% predictive coverage']):
            _line(ax,frame[(frame.mask_rate==rate)&(frame.metric==metric)],color,f'{rate:.0%}')
            ax.set_title(title)
    axes[-1].axhline(.95,color='black',ls=':',lw=1);axes[-1].set_ylim(0,1)
    axes[0].legend(title='Training mask probability')
    fig.suptitle('Intact outer-fold predictions: matched unmasked control and masking models')
    fig.tight_layout();return fig


def plot_masked_quality(tables, *, hop=1):
    frame=tables['masked'];frame=frame[frame.hop==hop]
    fig,axes=plt.subplots(1,3,figsize=(16,4))
    for rate,color in zip(sorted(frame.mask_rate.unique()),COLORS[1:]):
        for state,style in [('intact','-'),('mask','--')]:
            part=frame[(frame.mask_rate==rate)&(frame.metric=='mse_z')&(frame.state==state)]
            _line(axes[0],part,color,f'{rate:.0%} {state}',style=style)
        _line(axes[1],frame[(frame.mask_rate==rate)&(frame.metric=='delta_mse_z')],color,f'{rate:.0%}')
        _line(axes[2],frame[(frame.mask_rate==rate)&(frame.metric=='coverage_95')&(frame.state=='mask')],color,f'{rate:.0%}')
    axes[0].set_title('Same recipients: intact vs one neighbor hidden');axes[0].set_ylabel('MSE, transformed target')
    axes[1].set_title('Masking-induced change in prediction error');axes[1].axhline(0,color='.5',lw=1)
    axes[1].set_ylabel('Masked MSE − intact MSE')
    axes[2].set_title('Masked 95% predictive coverage');axes[2].axhline(.95,color='black',ls=':',lw=1);axes[2].set_ylim(0,1)
    axes[0].legend(fontsize=8)
    fig.suptitle(f'Exactly one source hidden; recipient identity observed • hop {hop}')
    fig.tight_layout();return fig


def plot_training_robustness(result, tables):
    quality=tables['seed_quality'];masked=result['masked']
    fig,axes=plt.subplots(1,3,figsize=(15,4))
    for seed,frame in quality.groupby('seed'):
        frame=frame.sort_values('mask_rate')
        control=float(frame.loc[frame.mask_rate==0,'mse_z'].iloc[0])
        axes[0].plot(frame.mask_rate*100,frame.mse_z-control,marker='o',label=f'Seed {seed}')
    axes[0].axhline(0,color='.5',lw=1);axes[0].set_title('Intact MSE change vs matched 0% control')
    if not masked.empty:
        org=masked.groupby(['mask_rate','seed','organoid_str'])[['mask_mse_z','mask_nll_z']].mean().reset_index()
        means=org.groupby(['mask_rate','seed'])[['mask_mse_z','mask_nll_z']].mean().reset_index()
        for seed,frame in means.groupby('seed'):
            frame=frame.sort_values('mask_rate')
            axes[1].plot(frame.mask_rate*100,frame.mask_mse_z,marker='o',label=f'Seed {seed}')
            axes[2].plot(frame.mask_rate*100,frame.mask_nll_z,marker='o',label=f'Seed {seed}')
    axes[1].set_title('Single-neighbor masked MSE');axes[2].set_title('Single-neighbor masked Gaussian NLL')
    for ax in axes:ax.set_xlabel('Training masking probability (%)')
    axes[0].legend();fig.suptitle('Training robustness: one curve per seed, equal organoid weight')
    fig.tight_layout();return fig


def plot_effect_robustness(result,tables,*,min_organoids=5):
    frame=tables['seed_effects'];names=result['config']['markers']+['Unassigned']
    fig,axes=plt.subplots(1,2,figsize=(15,6))
    for ax,hop in zip(axes,[1,2]):
        part=frame[(frame.hop==hop)&(frame.n_organoids>=min_organoids)].copy()
        part['positive']=(part.effect>0).astype(float)
        support=part.groupby(['center_marker','source_marker_name']).size()
        fraction=part.groupby(['center_marker','source_marker_name']).positive.mean()
        expected=len(result['config']['seeds'])*sum(p>0 for p in result['config']['training']['rates'])
        fraction=fraction.where(support==expected).unstack().reindex(index=names,columns=names)
        img=ax.imshow(fraction.to_numpy(),vmin=0,vmax=1,cmap='coolwarm')
        ax.set_xticks(range(len(names)),names,rotation=45,ha='right');ax.set_yticks(range(len(names)),names)
        ax.set_xlabel('Original source identity');ax.set_ylabel('Unchanged center identity');ax.set_title(f'Hop {hop}')
    fig.subplots_adjust(right=.87,bottom=.23,wspace=.45,top=.86)
    cax=fig.add_axes([.91,.25,.015,.5]);fig.colorbar(img,cax=cax,label='Fraction of rate/seed means > 0')
    fig.suptitle('Pooled observed-size masking effects: sign consistency across rates and seeds\n'
                 'Descriptive only: weak effects can change sign; magnitudes and N-bin profiles are exported.')
    return fig
