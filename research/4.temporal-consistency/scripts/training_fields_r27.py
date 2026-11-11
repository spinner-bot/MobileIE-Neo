"""Expose fixed teacher flow/acceptance while preserving the native RGB-history reference."""
import hashlib
import numpy as np
import torch

from aligned_raw_history_r11 import align_history
from explore_temporal_controls import inference
from half_motion_r4 import half_forward_backward


def pair_fields(pair, teacher, device):
    previous, current = pair.detach().permute(0, 2, 3, 1).cpu().numpy()*255.
    enhanced = inference(teacher, [previous, current], device, 1)[0]
    flow, mask = half_forward_backward(*enhanced)
    history, valid = align_history(previous, current, flow, mask)
    detail = {'valid_fraction': float(valid.mean()), 'mean_flow_norm': float(np.linalg.norm(flow, axis=2).mean()),
              'raw_pair_mae': float(np.abs(previous-current).mean()), 'aligned_pair_mae': float(np.abs(history-current).mean()),
              'history_array_sha256': hashlib.sha256(history.tobytes()).hexdigest(),
              'flow_sha256': hashlib.sha256(flow.tobytes()).hexdigest(),
              'FB_mask_sha256': hashlib.sha256(mask.tobytes()).hexdigest()}
    return history, flow, mask, detail


def task_fields(low, previous, teacher, device, scene, t):
    """Four tasks in original [real/synthetic, slot0/1] order; no ROI cache."""
    histories, fields, masks, details = [], [], [], []
    height, width = low.shape[-2:]
    for pair_index in range(2):
        for slot in range(2):
            current, prior = low[pair_index, slot], previous[pair_index, slot]
            fallback = pair_index == 0 and t-1+slot == 0
            if fallback:
                history = current.detach().permute(1, 2, 0).cpu().numpy()*255.
                flow = np.zeros((height, width, 2), np.float32)
                mask = np.zeros((height, width), bool)
                detail = {'valid_fraction': 1., 'mean_flow_norm': 0., 'raw_pair_mae': 0., 'aligned_pair_mae': 0.,
                          'history_array_sha256': hashlib.sha256(history.tobytes()).hexdigest(),
                          'flow_sha256': hashlib.sha256(flow.tobytes()).hexdigest(),
                          'FB_mask_sha256': hashlib.sha256(mask.tobytes()).hexdigest()}
            else:
                history, flow, mask, detail = pair_fields(torch.stack([prior, current]), teacher, device)
            histories.append(history)
            fields.append(flow)
            masks.append(mask)
            details.append({**detail, 'cache_hit': False, 'self_fallback': fallback,
                            'source': 'real' if pair_index == 0 else 'synthetic', 'pair_index': pair_index, 'task_slot': slot,
                            'scene': scene if pair_index == 0 else None,
                            'feature_FB_acceptance_fraction': float(mask.mean())})
    aligned = torch.from_numpy(np.stack(histories).transpose(0, 3, 1, 2).copy()/255.).to(device).reshape_as(low)
    return aligned, torch.from_numpy(np.stack(fields)).to(device), torch.from_numpy(np.stack(masks)).to(device), details
