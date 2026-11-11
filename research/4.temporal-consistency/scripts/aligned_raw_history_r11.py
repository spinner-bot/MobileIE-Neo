"""Causal raw-history gather with current-frame fallback at rejected pixels."""
import numpy as np

from video_metrics import warp_previous


def align_history(previous, current, flow, mask):
    if previous.shape!=current.shape or current.ndim!=3 or current.shape[2]!=3:
        raise ValueError('matching RGB images required')
    if flow.shape!=(*current.shape[:2],2) or mask.shape!=current.shape[:2]:
        raise ValueError('native-grid backward flow and mask required')
    if not all(np.isfinite(x).all() for x in [previous,current,flow]):
        raise ValueError('nonfinite history inputs')
    warped,bounds=warp_previous(previous,flow)
    valid=bounds & mask.astype(bool)
    return np.where(valid[...,None],warped,current).astype(np.float32),valid
