"""Figures for checkpoint-based LGR5 FiLM diagnostics."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROUTE_LABELS = {'full':'FiLM + head vary', 'film_only':'FiLM varies; head fixed',
                'head_only':'Head varies; FiLM fixed', 'layer1_only':'Only FiLM layer 1 varies',
                'layer2_only':'Only FiLM layer 2 varies'}
COLORS = dict(zip(ROUTE_LABELS, ['#222222','#d45a31','#2878b5','#29945b','#9467bd']))


def axis_style(ax, counts):
    ax.set_xscale('log')
    ax.set_xticks([counts[0],counts[len(counts)//2],counts[-1]],
                  labels=[str(counts[0]),str(counts[len(counts)//2]),str(counts[-1])])
    ax.minorticks_off()
    ax.tick_params(labelsize=8)
    ax.spines[['top','right']].set_visible(False)


def route_curves(table, markers, counts, *, metric='delta_z', routes=None, minimum=10):
    routes = routes or ['full','film_only','head_only']
    labels={'delta_z':'Δ transformed curvature','delta_fixed_reference':'ΔK / Kref(361)',
            'delta_relative':'ΔK / Kref(N)'}
    fig,axes=plt.subplots(2,len(markers),squeeze=False,figsize=(3.1*len(markers),6.8),sharex=True)
    for row,hop in enumerate([1,2]):
        for col,marker in enumerate(markers):
            ax=axes[row,col];ax.axhline(0,color='0.7',lw=.7)
            for route in routes:
                group=table[(table.metric==metric)&(table.source_marker_name==marker)&(table.hop==hop)&(table.route==route)].sort_values('n')
                valid=group.n_organoids>=minimum
                ax.plot(group.n,group['mean'].where(valid),color=COLORS[route],label=ROUTE_LABELS[route],lw=1.5)
                ax.fill_between(group.n,group.ci_low.where(valid),group.ci_high.where(valid),color=COLORS[route],alpha=.1)
            axis_style(ax,counts)
            if row==0:ax.set_title(f'Ablate {marker}',fontsize=10)
            if col==0:ax.set_ylabel(f'Hop {hop}\n{labels.get(metric,metric)}')
            if row==1:ax.set_xlabel('Supplied N')
    handles=[Line2D([],[],color=COLORS[r],label=ROUTE_LABELS[r]) for r in routes]
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,.955),ncol=len(routes),fontsize=9)
    fig.suptitle('LGR5-positive centers: separating FiLM and head size inputs',y=.998)
    fig.tight_layout(rect=(0,0,1,.9))
    return fig


def hidden_curves(table, markers, counts, *, statistic='cosine', minimum=10):
    routes=['full','head_only','layer1_only','layer2_only']
    fig,axes=plt.subplots(3,len(markers),squeeze=False,figsize=(3.1*len(markers),8.8),sharex=True)
    for row,(layer,hop) in enumerate([('h1',1),('h2',1),('h2',2)]):
        for col,marker in enumerate(markers):
            ax=axes[row,col];ax.axhline(1,color='0.75',lw=.7)
            for route in routes:
                group=table[(table.metric==f'{layer}_{statistic}')&(table.hop==hop)&(table.source_marker_name==marker)&(table.route==route)].sort_values('n')
                valid=group.n_organoids>=minimum
                ax.plot(group.n,group['mean'].where(valid),color=COLORS[route],lw=1.5)
                ax.fill_between(group.n,group.ci_low.where(valid),group.ci_high.where(valid),color=COLORS[route],alpha=.09)
            axis_style(ax,counts)
            if statistic=='cosine':ax.set_ylim(-.05,1.05)
            if row==0:ax.set_title(marker,fontsize=10)
            if col==0:ax.set_ylabel(f'Layer {layer[-1]}, hop {hop}\n'+('Cosine to N=361 response' if statistic=='cosine' else 'Response norm / norm at N=361'))
            if row==2:ax.set_xlabel('Supplied N')
    fig.legend(handles=[Line2D([],[],color=COLORS[r],label=ROUTE_LABELS[r]) for r in routes],loc='upper center',bbox_to_anchor=(.5,.96),ncol=4,fontsize=9)
    fig.suptitle('Marker-induced hidden-state changes at LGR5 centers',y=.998)
    fig.tight_layout(rect=(0,0,1,.92))
    return fig


def scaling_heatmap(table, counts, *, hop=1, metric='delta_relative'):
    selected=table[(table.hop==hop)&(table.metric==metric)&((table.marker_a=='LGR5')|(table.marker_b=='LGR5'))].copy()
    selected['other']=np.where(selected.marker_a=='LGR5',selected.marker_b,selected.marker_a)
    markers=sorted(selected.other.unique())
    if not markers:raise ValueError('No supported common-center marker comparisons.')
    fig,axes=plt.subplots(1,3,figsize=(16, max(3.8,len(markers)*.55)),squeeze=False)
    for ax,route in zip(axes[0],['full','film_only','head_only']):
        group=selected[selected.route==route]
        matrix=group.pivot(index='other',columns='n',values='mismatch').reindex(index=markers,columns=counts)
        im=ax.imshow(matrix,aspect='auto',cmap='viridis',vmin=0,vmax=1)
        ax.set_xticks(range(len(counts)),counts,rotation=45)
        support=group.groupby('other').n_organoids.first()
        ax.set_yticks(range(len(markers)),[f'{m} vs LGR5 (n={support.get(m,0)})' for m in markers],fontsize=9)
        ax.set_title(ROUTE_LABELS[route],fontsize=11)
        ax.set_xlabel('Supplied N')
    fig.suptitle(f'Beyond common positive scaling: matched centers, hop {hop}, {metric}\nReference N=361; each row uses its own common-support cohort',y=1.02)
    fig.subplots_adjust(left=.12,right=.87,bottom=.22,top=.82,wspace=.75)
    color_axis=fig.add_axes([.92,.22,.012,.60])
    fig.colorbar(im,cax=color_axis,label='Unexplained response norm fraction')
    return fig


def center_interaction_curves(table,seeds,markers,counts,*,metric='interaction_z',minimum=10):
    fig,axes=plt.subplots(2,len(markers),squeeze=False,figsize=(3.1*len(markers),6.5),sharex=True)
    for row,hop in enumerate([1,2]):
        for col,marker in enumerate(markers):
            ax=axes[row,col];ax.axhline(0,color='0.6',lw=.7)
            group=table[(table.kind=='center_neighbor')&(table.marker_b==marker)&(table.hop==hop)&(table.metric==metric)].sort_values('n')
            valid=group.n_organoids>=minimum
            for seed,sg in seeds[(seeds.kind=='center_neighbor')&(seeds.marker_b==marker)&(seeds.hop==hop)].groupby('seed'):
                sg=sg.sort_values('n')
                if valid.any():ax.plot(sg.n,sg[metric],color='0.6',lw=.8,alpha=.65)
            ax.plot(group.n,group['mean'].where(valid),color='#9467bd',lw=1.7)
            ax.fill_between(group.n,group.ci_low.where(valid),group.ci_high.where(valid),color='#9467bd',alpha=.18)
            axis_style(ax,counts)
            if row==0:ax.set_title(f'Center LGR5 × {marker}',fontsize=10)
            if col==0:ax.set_ylabel(f'Hop {hop}\n'+('Interaction / Kref(N)' if metric=='interaction_relative' else 'Interaction in transformed curvature'))
            if row==1:ax.set_xlabel('Supplied N')
    fig.suptitle('Factorial center–neighbor interaction: both − center-only − neighbor-only + intact\nPurple: organoid mean and conditional 95% CI; gray: individual model seeds',y=.995)
    fig.tight_layout(rect=(0,0,1,.9))
    return fig


def neighbor_interaction_heatmap(endpoints,markers,*,minimum=10):
    fig,axes=plt.subplots(2,2,figsize=(13,11))
    for row,metric in enumerate(['interaction_z','interaction_relative']):
        eligible=endpoints[(endpoints.kind=='neighbor_neighbor')&(endpoints.metric==metric)&(endpoints.n_organoids>=minimum)]
        vmax=max(1e-8,float(eligible['mean'].abs().max()))
        for col,hop in enumerate([1,2]):
            matrix=np.full((len(markers),len(markers)),np.nan)
            support=np.zeros_like(matrix)
            for item in eligible[eligible.hop==hop].itertuples():
                i,j=markers.index(item.marker_a),markers.index(item.marker_b)
                matrix[i,j]=matrix[j,i]=item.mean;support[i,j]=support[j,i]=item.n_organoids
            ax=axes[row,col];im=ax.imshow(matrix,cmap='RdBu_r',vmin=-vmax,vmax=vmax)
            ax.set_xticks(range(len(markers)),markers,rotation=45,ha='right')
            ax.set_yticks(range(len(markers)),markers)
            for i,j in zip(*np.where(np.isfinite(matrix))):ax.text(j,i,f'{matrix[i,j]:.2g}\nn={int(support[i,j])}',ha='center',va='center',fontsize=7)
            ax.set_title(f'Hop {hop}: {metric}')
            fig.colorbar(im,ax=ax,shrink=.75)
    fig.suptitle('Change in neighbor–neighbor interaction: N=791 minus N=165\nDistinct source cells; LGR5-positive center retained; blank = insufficient support',y=.995)
    fig.tight_layout(rect=(0,0,1,.93))
    return fig


def hidden_interaction_heatmap(table,counts,*,kind='center_neighbor',minimum=10):
    selected=table[(table.kind==kind)&(table.metric=='h2_interaction_ratio')&(table.n_organoids>=minimum)].copy()
    selected['pair']=selected.marker_a+' × '+selected.marker_b
    pairs=sorted(selected.pair.unique())
    fig,axes=plt.subplots(1,2,figsize=(13,max(4,len(pairs)*.35)))
    vmax=max(.05,float(selected['mean'].max()))
    for ax,hop in zip(axes,[1,2]):
        matrix=selected[selected.hop==hop].pivot(index='pair',columns='n',values='mean').reindex(index=pairs,columns=counts)
        im=ax.imshow(matrix,aspect='auto',cmap='viridis',vmin=0,vmax=vmax)
        ax.set_xticks(range(len(counts)),counts,rotation=45);ax.set_yticks(range(len(pairs)),pairs,fontsize=8)
        ax.set_title(f'Layer 2, hop {hop}');ax.set_xlabel('Supplied N')
    fig.suptitle(f'Non-additivity before the prediction head: {kind}',y=1.01)
    fig.subplots_adjust(left=.2,right=.85,bottom=.17,top=.9,wspace=.85)
    color_axis=fig.add_axes([.92,.17,.012,.73])
    fig.colorbar(im,cax=color_axis,label='Relative hidden non-additivity')
    return fig


def neighbor_interaction_size_heatmap(table,counts,*,minimum=10):
    selected=table[(table.kind=='neighbor_neighbor')&(table.n_organoids>=minimum)].copy()
    selected['pair']=selected.marker_a+' × '+selected.marker_b
    pairs=sorted(selected.pair.unique())
    fig,axes=plt.subplots(2,2,figsize=(16,max(10,len(pairs)*.65)),layout='constrained')
    for row,metric in enumerate(['interaction_z','interaction_relative']):
        group=selected[selected.metric==metric]
        vmax=max(1e-8,float(group['mean'].abs().max()))
        for col,hop in enumerate([1,2]):
            panel=group[group.hop==hop]
            matrix=panel.pivot(index='pair',columns='n',values='mean').reindex(index=pairs,columns=counts)
            ax=axes[row,col];im=ax.imshow(matrix,aspect='auto',cmap='RdBu_r',vmin=-vmax,vmax=vmax)
            ax.set_xticks(range(len(counts)),counts,rotation=45)
            ax.set_yticks(range(len(pairs)),pairs,fontsize=8)
            ax.set_xlabel('Supplied N');ax.set_title(f'Hop {hop}: {metric}')
            fig.colorbar(im,ax=ax,shrink=.7)
    fig.suptitle('Neighbor–neighbor factorial interactions across size\nOriginally LGR5-positive centers; distinct source cells; blank = insufficient support')
    return fig
