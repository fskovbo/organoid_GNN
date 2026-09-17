"""Ordered, destructive-in-the-copy exclusivity of binary fate markers."""
import numpy as np

EXCLUSIVITY_RULES = {
    'LGR5': ['Chroma','Mucin 2','AldoB','Glucagon','Agr2','Serotonin','Lysozyme'],
    'Chroma': ['Mucin 2','Glucagon','Serotonin','Lysozyme'],
    'Mucin 2': ['Chroma','Glucagon','Serotonin','Lysozyme'],
    'AldoB': ['Chroma','Mucin 2','Glucagon','Agr2','Serotonin','Lysozyme'],
    'Glucagon': ['Serotonin'],
    'Agr2': ['Chroma','Mucin 2','Glucagon','Serotonin','Lysozyme'],
    'Serotonin': [],
    'Lysozyme': ['Chroma','Glucagon','Serotonin'],
    'KI67': ['LGR5','Chroma','Mucin 2','AldoB','Glucagon','Agr2','Serotonin','Lysozyme'],
}


def exclusive_markers(x, marker_names, rules=None):
    """Clear each key if a currently positive exclusion exists, in dict order.

    Missing dataset markers are ignored. No winner is added to unmarked cells.
    The original array and feature order are preserved; unsupported unresolved
    combinations raise rather than silently choosing a further priority rule.
    """
    rules = EXCLUSIVITY_RULES if rules is None else rules
    result=np.asarray(x).copy()
    if result.ndim!=2 or result.shape[1]!=len(marker_names) or len(set(marker_names))!=len(marker_names):
        raise ValueError('Expected an N by marker matrix with unique marker names.')
    if not np.isin(result,[0,1]).all(): raise ValueError('Fate features must be binary.')
    lookup={name:i for i,name in enumerate(marker_names)}
    for marker, forbidden in rules.items():
        if marker not in lookup: continue
        columns=[lookup[m] for m in forbidden if m in lookup]
        if columns: result[np.any(result[:,columns]>0,axis=1),lookup[marker]]=0
    if np.any(result.sum(axis=1)>1): raise ValueError('The supplied rules leave some nodes with multiple markers.')
    return result
