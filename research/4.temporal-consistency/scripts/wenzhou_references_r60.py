"""Existing network inference, extended only in T; unpaired common-flow descriptors."""
import numpy as np
import torch

from aligned_raw_history_r11 import align_history
from evaluate_phone_references_r38 import statistics
from fastdvdnet_adapter_r12 import denoise
from multinex_reference_r55 import prediction
from task_fastdvd_r47 import SIGMA

PRIMARY = list(range(4, 124))


@torch.inference_mode()
def infer_sequence(name, model, low):
    assert low.ndim == 4 and low.dtype == np.uint8 and low.shape[-1] == 3
    inputs = torch.from_numpy(low.transpose(0, 3, 1, 2).copy()).float()/255
    if name.startswith('fastdvd'):
        result = denoise(model, inputs, SIGMA, 'window')
    else:
        result = torch.stack([(prediction(model, x[None]) if name.startswith('nano')
                               else model(x[None]).clamp(0, 1))[0] for x in inputs])
    assert torch.isfinite(result).all() and result.dtype == torch.float32
    output = result.permute(0, 2, 3, 1).contiguous().numpy()*255
    assert output.shape == low.shape
    return output


def describe(output, flows, masks):
    assert output.shape == (128, 360, 640, 3)
    assert flows.shape == (128, 360, 640, 2) and masks.shape == (128, 360, 640)
    frames = []
    for t, frame in enumerate(output):
        row = dict(frame=t, primary_target=t in PRIMARY, **statistics(frame),
                   DC_luma_change=None, shared_input_flow_valid_fraction=None,
                   shared_input_flow_valid_pixels=None, shared_input_flow_warp_luma_MAE_proxy=None)
        if t:
            import cv2
            moved, valid = align_history(output[t-1], frame, flows[t], masks[t])
            row['shared_input_flow_valid_pixels'] = int(valid.sum())
            row['shared_input_flow_valid_fraction'] = float(valid.mean())
            row['DC_luma_change'] = row['mean_luma_0to255']-frames[-1]['mean_luma_0to255']
            if valid.any():
                difference = cv2.cvtColor((frame-moved).astype(np.float32), cv2.COLOR_RGB2GRAY)
                row['shared_input_flow_warp_luma_MAE_proxy'] = float(np.abs(difference)[valid].mean())
        frames.append(row)
    summary = dict(frames=128, primary_frames=PRIMARY)
    keys = [key for key in frames[0] if key not in ['frame', 'primary_target']]
    for prefix, selected in [('', [frames[t] for t in PRIMARY]), ('full128_', frames)]:
        for key in keys:
            values = [row[key] for row in selected if row[key] is not None]
            summary[prefix+key] = float(np.mean(values)) if values else None
        summary[prefix+'warp_proxy_valid_frames'] = sum(row['shared_input_flow_warp_luma_MAE_proxy'] is not None for row in selected)
    return summary, frames
