"""Organoid-weighted evaluation of fields restored from observed graph moments."""
import numpy as np
import pandas as pd


def conditional_region_scores(archive, regions, moments, *, cohort, include_mean_reference=False):
    """Score a saved ragged prediction archive without pooling unequal cell counts.

    Return one row per organoid/region. ``total`` covers ordinary validation;
    spherical validation participates only in its combined regional category.
    ``regions`` maps organoid IDs to node-aligned labels. The caller averages
    organoids within each fold before averaging folds. Measured-mean reference
    rows are optional and have ``is_reference=True``.
    """
    ids=archive['organoid_ids'];offsets=archive['offsets']
    truth=np.asarray(archive['truth']);prediction=np.asarray(archive['prediction'])
    if (len(offsets)!=len(ids)+1 or offsets[0]!=0 or offsets[-1]!=len(truth)
            or len(truth)!=len(prediction) or np.any(np.diff(offsets)<=0)):
        raise ValueError('Invalid ragged prediction archive')
    if not np.isfinite(truth).all() or not np.isfinite(prediction).all():
        raise ValueError('Nonfinite curvature field')
    rows=[]
    for oid,start,end in zip(ids,offsets[:-1],offsets[1:]):
        oid=str(oid);actual=truth[start:end];labels=np.asarray(regions[oid]);moment=moments[oid]
        if len(labels)!=len(actual):raise ValueError(f'Region alignment differs: {oid}')
        fields=[(False,prediction[start:end])]
        if include_mean_reference:fields.append((True,np.full_like(actual,moment.mean)))
        for is_reference,field in fields:
            error=(field-actual)**2
            names=(['total'] if cohort=='val' else [])+list(np.unique(labels))
            for region in names:
                if region in ('excluded_boundary','boundary_without_qualified_neck','annotation_unavailable'):continue
                take=np.ones(len(actual),bool) if region=='total' else labels==region
                if not take.any():continue
                mse=float(error[take].mean())
                rows.append(dict(organoid_str=oid,cohort=cohort,region=region,cells=int(take.sum()),
                    mse=mse,standardized_mse=mse/moment.scale**2,is_reference=is_reference))
    return pd.DataFrame(rows)
