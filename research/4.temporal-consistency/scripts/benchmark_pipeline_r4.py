"""Serial steady-state wall cost including fresh CPU motion and transfers."""
import argparse
import json
from pathlib import Path
import time

import cv2
import numpy as np
import torch

from calibrate_noise_gate_r4 import haar_noise_scale
from exploration_r2_data import ROOT
from explore_state_r2 import forward_backward
from explore_temporal_controls import inference, save_json
from prepare_cache_r4 import load_cache
from train_exploration_r2 import new_model
from video_metrics import warp_previous
from half_motion_r4 import half_forward_backward


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    parser.add_argument('--half-comparison', action='store_true')
    args = parser.parse_args()
    if Path(args.out).exists():
        raise FileExistsError(args.out)
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device('cuda')
    net = new_model(device)
    base = ROOT/'research/4.temporal-consistency/results'
    alpha = next(r['alpha'] for r in json.loads((base/'exploration-r4-alpha/selected.json').read_text())['configurations'] if r['motion'] == 'enhanced')
    photo = next(r['threshold'] for r in json.loads((base/'exploration-r4-reliability/selected.json').read_text())['configurations'] if r['motion'] == 'enhanced' and r['family'] == 'photo')
    noise = json.loads((base/'exploration-r4-noise-gate/selected.json').read_text())['threshold']
    photo, noise = [float('inf') if x is None else x for x in [photo, noise]]
    half_alpha = None
    if args.half_comparison:
        half_alpha = json.loads((base/'exploration-r4-half/selected.json').read_text())['configurations'][0]['alpha']
    frames = load_cache('test')['video_21']['low']
    names = ['raw', 'global_alpha', 'global_alpha_half'] if args.half_comparison else ['raw', 'global_alpha', 'photo_gate', 'noise_gate']
    records = {name: [] for name in names}
    for block in range(3):
        for name in names[block:]+names[:block]:
            for timed in [False, True]:
                previous = inference(net, frames[:1], device, 1)[0][0]
                history = previous.copy()
                previous_sigma = haar_noise_scale(previous) if name == 'noise_gate' else None
                for frame in frames[1:]:
                    start = time.perf_counter()
                    current = inference(net, [frame], device, 1)[0][0]
                    after_model = time.perf_counter()
                    motion_seconds = 0.
                    if name != 'raw':
                        motion = half_forward_backward if name == 'global_alpha_half' else forward_backward
                        flow, confidence = motion(previous, current)
                        after_motion = time.perf_counter()
                        motion_seconds = after_motion-after_model
                        accepted = confidence
                        if name in ['photo_gate', 'noise_gate']:
                            old_raw, valid = warp_previous(previous, flow)
                            difference = np.abs(current-old_raw).mean(-1)
                            threshold = photo
                            if name == 'noise_gate':
                                sigma = haar_noise_scale(current)
                                threshold = noise*max(1., np.hypot(previous_sigma, sigma))
                                previous_sigma = sigma
                            accepted = confidence & valid & (difference < threshold)
                        past, valid = warp_previous(history, flow)
                        coefficient = half_alpha if name == 'global_alpha_half' else alpha if name == 'global_alpha' else .7
                        weight = coefficient*(accepted & valid)[..., None]
                        history = (weight*past+(1-weight)*current).astype(np.float32)
                    previous = current
                    end = time.perf_counter()
                    if timed:
                        records[name].append({'model_transfer_sync_ms': (after_model-start)*1000,
                            'motion_ms': motion_seconds*1000, 'gate_fusion_ms': (end-after_model-motion_seconds)*1000,
                            'total_ms': (end-start)*1000})
    summary = {name: {key+'_median': float(np.median([r[key] for r in values]))
                      for key in values[0]} for name, values in records.items()}
    for name in names:
        summary[name]['total_ms_p95'] = float(np.quantile([r['total_ms'] for r in records[name]], .95))
    save_json(args.out, {'size': [480, 270], 'scene': 'video_21', 'blocks': 3, 'timed_frames_per_method': 45,
        'warmup': 'one full 16-frame sequence per block', 'opencv_threads': 1,
        'gpu': torch.cuda.get_device_name(), 'cudnn_tf32_allowed': torch.backends.cudnn.allow_tf32,
        'parameters': {'global_alpha': alpha, 'photo_threshold': photo, 'noise_normalized_threshold': noise, 'half_alpha': half_alpha},
        'scope': 'serial wall processing including current model transfers/sync and fresh motion; excludes decode, initialization and loading; not phone latency',
        'summary': summary, 'per_frame': records})


if __name__ == '__main__':
    main()
