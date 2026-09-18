"""Qualify individual crypt necks from circumference, before interpreting s=1."""
from dataclasses import dataclass, asdict
from pathlib import Path
import json
import shutil
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter, find_peaks
from src.analysis.spatial.neighborhoods import ObservedConfig, bootstrap_summary, within_organoid_contrast


@dataclass(frozen=True)
class NeckConfig:
    minimum_crypt_cells: int = 10
    minimum_relative_depth: float = .05
    plateau_max_range: float = .10
    plateau_max_slope: float = .25
    plateau_exit_rise: float = .10
    minimum_window: tuple = (.8, 1.2)
    plateau_window: tuple = (.85, 1.15)
    smoothing_points: int = 7


def classify_profile(s, circumference, config=NeckConfig()):
    """Curvature/marker-blind shape rule; rejected shapes are not proven bulges.

    Normalize circumference by C(1). A trough must have two shoulders in
    [0.5,1.5]. A plateau must persist over [.85,1.15] and open out afterwards,
    so a flat dome top is not mistaken for a tubular segment.
    """
    s, c = np.asarray(s, float), np.asarray(circumference, float)
    result = dict(profile_class='unresolved', minimum_s=np.nan, relative_depth=np.nan,
                  plateau_range=np.nan, plateau_slope=np.nan, exit_rise=np.nan)
    if s.ndim != 1 or c.shape != s.shape or len(s)<15 or not np.all(np.diff(s)>0):
        return result
    window=(s>=.5)&(s<=1.5)
    if s[0]>.5 or s[-1]<1.5 or not np.isfinite(c[window]).all() or np.any(c[window]<=0):
        return result
    scale=np.interp(1.,s,c)
    # No missing-data interpolation: the entire stored profile must be finite.
    if not np.isfinite(c).all() or scale<=0:
        return result
    smooth=savgol_filter(c/scale,config.smoothing_points,2)
    idx=np.flatnonzero(window)
    minima=find_peaks(-smooth)[0]
    candidates=[]
    for i in minima:
        if not config.minimum_window[0]<=s[i]<=config.minimum_window[1]:continue
        left=idx[s[idx]<=s[i]-.1];right=idx[s[idx]>=s[i]+.1]
        if not len(left) or not len(right):continue
        depth=min(smooth[left].max(),smooth[right].max())-smooth[i]
        candidates.append((depth,float(s[i])))
    if candidates:
        depth, location=max(candidates)
        result.update(relative_depth=depth,minimum_s=location)
        if depth>=config.minimum_relative_depth:
            result['profile_class']='local_minimum'
            return result
    band=(s>=config.plateau_window[0])&(s<=config.plateau_window[1])
    if band.sum()<5:return result
    span=float(np.ptp(smooth[band]));slope=float(np.polyfit(s[band],smooth[band],1)[0])
    exit_band=(s>=1.3)&(s<=1.5)
    rise=float(np.median(smooth[exit_band])-np.median(smooth[band]))
    result.update(plateau_range=span,plateau_slope=slope,exit_rise=rise)
    result['profile_class']='flat_section' if (span<=config.plateau_max_range and
        abs(slope)<=config.plateau_max_slope and rise>=config.plateau_exit_rise) else 'no_neck_support'
    return result


def qualified_assignment(distances, passing):
    """Never assign a cell belonging to a rejected crypt to a farther valid one."""
    distances=np.asarray(distances)
    if not len(distances):return np.full(distances.shape[1],-1),np.full(distances.shape[1],np.nan)
    nearest=distances.argmin(0)
    axis=distances[nearest,np.arange(distances.shape[1])].astype(float)
    axis[~np.asarray(passing,bool)[nearest]]=np.nan
    return nearest,axis


