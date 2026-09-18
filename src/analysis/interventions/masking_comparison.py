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

from src.analysis.interventions.masking import (
    ObservedFateAdapter, evaluate_single_mask, load_mask_checkpoint,
    checkpoint_path, checked_settings, _sha,
)
from src.analysis.size_conditioning.cohort_inputs import load_cohort, fold_graphs
from src.analysis.interventions.replacement import evaluate_replacements, contrast_table, _write_csv, load_comparison
from src.analysis.interventions.size_sweeps import summarize_by_organoid
from src.data.subgraphs import build_ego_subgraphs_for_graph

MASK_METHODS = ('zero', 'replacement_fixed', 'replacement_size_dependent', 'masking')


def reconstruct_cases(graphs, manifest, *, receptive_hops=2):
    """Reconstruct egos from saved original indices, never resample sources."""
    lookup={g.organoid_str:g for g in graphs}; subs=[]; mapping={}
    for org, frame in manifest.groupby('organoid_str',sort=True):
        if org not in lookup: raise ValueError(f'Case organoid is not in validation fold: {org}')
        centers=sorted(frame.orig_center.unique().astype(int).tolist())
        for sub in build_ego_subgraphs_for_graph(lookup[org],num_hops=receptive_hops,centers=centers):
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
