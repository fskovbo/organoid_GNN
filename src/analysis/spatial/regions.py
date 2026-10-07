"""Node-aligned anatomical regions from source crypt distances and profiles."""
from pathlib import Path
import numpy as np
import pandas as pd
from src.analysis.spatial.necks import NeckConfig, classify_profile


def graph_regions(graph, dataset_dir, *, crypt_max=.8, neck_max=1.2, neck_config=None):
    """Use the original nearest crypt; qualify its territory by circumference.

    A missing source or an undetected/unqualified crypt is never called villus.
    Villus is an operational distance proxy beyond a qualified crypt's neck.
    Source graph distances must match segmentation distances at projected vertices.
    """
    if not 0 < crypt_max < neck_max:
        raise ValueError('Require 0 < crypt_max < neck_max.')
    config = NeckConfig(**(neck_config or {}))
    n = len(graph.x)
    org = str(graph.organoid_str)
    source = Path(dataset_dir) / f'{org}.npz'
    frame = pd.DataFrame(dict(organoid_str=org, node=np.arange(n),
        region='annotation_unavailable', crypt_id=-1, crypt_distance=np.nan,
        profile_class='unavailable', annotation_status='missing_graph_source',
        graph_source=str(source), segmentation_source=''))
    if not source.is_file():
        return frame
    with np.load(source, allow_pickle=False) as data:
        if 'x' not in data or len(data['x']) != n:
            raise ValueError(f'{org}: source graph does not match saved node count.')
        if 'd_crypts_graph' not in data:
            frame['annotation_status'] = 'missing_crypt_distances'
            return frame
        distances = np.asarray(data['d_crypts_graph'], dtype=float)
        if distances.ndim != 2 or distances.shape[1] != n:
            raise ValueError(f'{org}: expected crypt distances shaped (crypts, {n}).')
        if not len(distances):
            frame['region'] = 'no_detected_crypt'
            frame['annotation_status'] = 'no_detected_crypt'
            return frame
        if 'proj_vertex_ids' not in data:
            frame['annotation_status'] = 'missing_vertex_projection'
            return frame
        projection = np.asarray(data['proj_vertex_ids'])
    if projection.shape != (n,) or not np.issubdtype(projection.dtype, np.integer) or np.any(projection < 0):
        raise ValueError(f'{org}: invalid node-to-vertex projection.')
    metadata = getattr(graph, 'meta', None) or {}
    path = metadata.get('segmentation_path')
    if not path:
        frame['annotation_status'] = 'missing_segmentation_path'
        return frame
    segmentation = Path(path).expanduser()
    if not segmentation.is_absolute():
        segmentation = Path(dataset_dir) / segmentation
    frame['segmentation_source'] = str(segmentation)
    if not segmentation.is_file():
        frame['annotation_status'] = 'missing_segmentation_file'
        return frame
    # Trusted source segmentation exports store circumference profiles as object arrays.
    with np.load(segmentation, allow_pickle=True) as data:
        required = {'d_crypts', 'd_discretized', 'circumference_crypts'}
        if not required <= set(data.files):
            frame['annotation_status'] = 'missing_segmentation_arrays'
            return frame
        mesh_distances = np.asarray(data['d_crypts'], dtype=float)
        profiles = data['circumference_crypts']
        axis = np.asarray(data['d_discretized'], dtype=float)
        if mesh_distances.ndim != 2 or mesh_distances.shape[0] != len(distances) or len(profiles) != len(distances):
            raise ValueError(f'{org}: crypt rows differ between graph and segmentation.')
        if np.any(projection >= mesh_distances.shape[1]):
            raise ValueError(f'{org}: projected vertices exceed segmentation dimensions.')
        if not np.allclose(distances, mesh_distances[:, projection], rtol=2e-6, atol=2e-6, equal_nan=True):
            raise ValueError(f'{org}: graph/segmentation crypt-distance alignment mismatch.')
    # A cell with an unknown distance to any crypt cannot be assigned reliably.
    valid = np.isfinite(distances).all(axis=0) & (distances >= 0).all(axis=0)
    nearest = np.where(np.isfinite(distances), distances, np.inf).argmin(axis=0)
    closest = distances[nearest, np.arange(n)]
    classes, eligible = [], []
    for k, profile in enumerate(profiles):
        kind = classify_profile(axis, profile, config)['profile_class']
        supported = np.sum(valid & (nearest == k) & (distances[k] <= 1)) >= config.minimum_crypt_cells
        shape = kind in ('local_minimum', 'flat_section')
        classes.append('low_cell_support' if shape and not supported else kind)
        eligible.append(shape and supported)
    qualified = valid & np.asarray(eligible)[nearest]
    frame['crypt_id'] = np.where(valid, nearest, -1)
    frame['crypt_distance'] = np.where(valid, closest, np.nan)
    frame['profile_class'] = np.where(valid, np.asarray(classes)[nearest], 'invalid_distance')
    frame['annotation_status'] = np.where(valid, 'available', 'invalid_distance')
    frame.loc[valid, 'region'] = 'unqualified_crypt'
    frame.loc[qualified & (closest < crypt_max), 'region'] = 'crypt'
    frame.loc[qualified & (closest >= crypt_max) & (closest <= neck_max), 'region'] = 'neck'
    frame.loc[qualified & (closest > neck_max), 'region'] = 'villus'
    return frame


def distance_regions(graph,dataset_dir,*,marker_names,crypt_marker='LGR5',crypt_max=.75,neck_max=1.1,neck_config=None):
    """Distance-first partition; only the boundary's neck label needs a profile.

    Interior membership uses the original nearest crypt, including qualification
    failures. An interior subtype belongs to the entire assigned crypt, based on
    any marker-positive cell at s<crypt_max in that territory. Missing profiles
    do not erase known interior/villus membership. No crypt-size filter is used
    unless explicitly supplied in neck_config. Old graph_regions is unchanged.
    """
    if not 0<crypt_max<neck_max:raise ValueError('Invalid region boundaries')
    if crypt_marker not in marker_names:raise ValueError(f'Missing marker: {crypt_marker}')
    config={'minimum_crypt_cells':0,**(neck_config or {})}
    frame=graph_regions(graph,dataset_dir,crypt_max=crypt_max,neck_max=neck_max,neck_config=config)
    source=Path(dataset_dir)/f'{graph.organoid_str}.npz'
    if not source.exists():return frame
    with np.load(source,allow_pickle=False) as data:
        if 'd_crypts_graph' not in data:return frame
        distances=np.asarray(data['d_crypts_graph'],float)
    if distances.ndim!=2 or distances.shape[1]!=len(graph.x):raise ValueError('Invalid source distances')
    frame['crypt_contains_marker']=False
    if not len(distances):return frame
    valid=np.isfinite(distances).all(0)&(distances>=0).all(0)
    nearest=np.where(np.isfinite(distances),distances,np.inf).argmin(0)
    closest=distances[nearest,np.arange(len(graph.x))]
    x=graph.x.detach().cpu().numpy() if hasattr(graph.x,'detach') else np.asarray(graph.x)
    positive=x[:,list(marker_names).index(crypt_marker)]>0
    contains=np.zeros(len(distances),bool)
    for k in range(len(distances)):contains[k]=np.any(valid&(nearest==k)&(closest<crypt_max)&positive)
    frame.loc[valid,'crypt_id']=nearest[valid];frame.loc[valid,'crypt_distance']=closest[valid]
    frame.loc[valid,'crypt_contains_marker']=contains[nearest[valid]]
    frame.loc[valid,'region']='boundary_without_qualified_neck'
    interior=valid&(closest<crypt_max)
    frame.loc[interior&contains[nearest],'region']=f'crypt_with_{crypt_marker}'
    frame.loc[interior&~contains[nearest],'region']=f'crypt_without_{crypt_marker}'
    frame.loc[valid&(closest>=neck_max),'region']='villus'
    qualified=frame.profile_class.isin(['local_minimum','flat_section']).to_numpy()
    frame.loc[valid&(closest>=crypt_max)&(closest<neck_max)&qualified,'region']='neck'
    return frame


def signed_neck_regions(graph, dataset_dir, *, crypt_max=.75, neck_max=1.1):
    """Reproduce the legacy distance masks and organoid-region budded labels.

    Crypt: s < crypt_max; neck: crypt_max <= s <= neck_max. No finite
    crypt distance is labelled villus, matching the legacy paper workflow.
    Signed neck shape is C(1)/max(C(s < crypt_max))-1 for the nearest crypt.
    The legacy figure classified the *whole organoid-region* as budded when
    its cell-weighted finite mean shape was negative, rather than selecting
    individual negative-shape crypts. ``budded`` repeats that exact policy.
    """
    n = len(graph.x)
    path = Path(dataset_dir)/f'{graph.organoid_str}.npz'
    with np.load(path, allow_pickle=False) as source:
        if len(source['x']) != n:
            raise ValueError('Source graph node count does not match saved graph.')
        distance = np.asarray(source['d_crypts_graph'], dtype=float)
    if distance.size == 0:
        distance = np.empty((0, n))
    if distance.ndim != 2 or distance.shape[1] != n:
        raise ValueError('Expected crypt distances of shape (crypts, nodes).')
    closest = np.full(n, np.nan)
    nearest = np.full(n, -1, dtype=int)
    if len(distance):
        finite = np.isfinite(distance).any(axis=0)
        filled = np.where(np.isfinite(distance), distance, np.inf)
        nearest[finite] = filled[:, finite].argmin(axis=0)
        closest[finite] = filled[nearest[finite], np.flatnonzero(finite)]
    shape = np.full(n, np.nan)
    segmentation = (getattr(graph, 'meta', {}) or {}).get('segmentation_path')
    if segmentation and Path(segmentation).is_file():
        with np.load(segmentation, allow_pickle=True) as source:
            s = np.asarray(source.get('d_discretized', []), float)
            profiles = np.asarray(source.get('circumference_crypts', []), float)
        if profiles.ndim == 1 and profiles.size:
            profiles = profiles[None, :]
        if profiles.ndim == 2 and s.ndim == 1 and profiles.shape[1] == len(s):
            for crypt, profile in enumerate(profiles):
                finite = np.isfinite(s) & np.isfinite(profile)
                reference = finite & (s < crypt_max)
                if not reference.any() or np.max(profile[reference]) <= 0:
                    continue
                order = np.argsort(s[finite])
                ss, cc = s[finite][order], profile[finite][order]
                neck = np.interp(1., ss, cc, left=np.nan, right=np.nan)
                if not np.isfinite(neck):
                    neck = cc[np.argmin(abs(ss-1.))]
                shape[nearest == crypt] = neck/np.max(profile[reference])-1.
    region = np.full(n, 'villus', dtype=object)
    region[np.isfinite(closest) & (closest < crypt_max)] = 'crypt'
    region[np.isfinite(closest) & (closest >= crypt_max) & (closest <= neck_max)] = 'neck'
    budded = np.zeros(n, bool)
    for name in ('crypt', 'neck'):
        mask = region == name
        values = shape[mask & np.isfinite(shape)]
        budded[mask] = bool(len(values) and values.mean() < 0)
    return pd.DataFrame(dict(organoid_str=str(graph.organoid_str), node=np.arange(n),
        region=region, budded=budded, nearest_crypt=nearest, crypt_distance=closest,
        signed_neck_shape=shape))
