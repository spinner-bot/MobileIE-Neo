"""Exact integer-motion, GT-relative temporal control for the synthetic pair."""
import torch


def synthetic_residual_temporal(prediction, target, offset, margin=8):
    if prediction.shape != target.shape or prediction.ndim != 4 or prediction.shape[0] != 2:
        raise ValueError('one two-frame CHW pair required')
    dx, dy = offset
    error = prediction-target
    previous, current = error[0, :, margin:-margin, margin:-margin].detach(), error[1, :, margin:-margin, margin:-margin]
    h, w = previous.shape[-2:]
    if h <= abs(dy) or w <= abs(dx):
        raise ValueError('no valid motion overlap')
    px, py, cx, cy = max(dx, 0), max(dy, 0), max(-dx, 0), max(-dy, 0)
    previous = previous[:, py:py+h-abs(dy), px:px+w-abs(dx)]
    current = current[:, cy:cy+h-abs(dy), cx:cx+w-abs(dx)]
    return torch.sqrt((current-previous)**2+1e-6).mean()
