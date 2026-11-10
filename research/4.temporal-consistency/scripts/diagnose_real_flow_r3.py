"""Fixed flow-input controls; reference-derived flow is diagnostic, not deployable."""
import argparse
import json
from pathlib import Path
import time

import cv2
import numpy as np
import torch

from exploration_r2_data import evaluate
from explore_real_pairs_r3 import load_pairs
from explore_state_r2 import forward_backward, fuse
from explore_temporal_controls import inference, save_json
from train_exploration_r2 import new_model
from video_metrics import warp_previous


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    device = torch.device('cuda')
    net = new_model(device)
    manifest = json.loads(Path(args.manifest).read_text())
    save_json(out/'config.json', {'alpha': .7, 'gamma_for_motion': .5, 'size': [480, 270],
        'motion_inputs': ['low', 'sqrt(low)', 'MobileIE(low)', 'reference diagnostic only', 'zero'],
        'reference_motion': 'estimated, not ground truth; cannot be used by deployable method'})
    rows, diagnostics = [], []
    start = time.perf_counter()
    for scene in ['example_1', 'example_2']:
        gt, low = load_pairs(manifest, scene)
        raw, _ = inference(net, low, device, 1)
        fields = {}
        for label, inputs in [('low', low), ('gamma', [np.sqrt(f/255)*255 for f in low]), ('enhanced', raw), ('reference', gt)]:
            pairs = [forward_backward(inputs[t-1], inputs[t]) for t in range(1, len(inputs))]
            fields[label] = ([p[0] for p in pairs], [p[1] for p in pairs])
        ref_flow, ref_mask = fields['reference']
        methods = {'raw': raw, **{name+'_fb': fuse(raw, flow, mask) for name, (flow, mask) in fields.items()},
                   'low_flow_no_fb': fuse(raw, fields['low'][0]),
                   'zero_motion_ema': fuse(raw, [np.zeros_like(f) for f in ref_flow])}
        for name, (flow, mask) in fields.items():
            epe, confident_epe, warp_error = [], [], []
            for t, (f, m, r, valid) in enumerate(zip(flow, mask, ref_flow, ref_mask)):
                delta = np.linalg.norm(f-r, axis=2)
                if valid.any():
                    epe.append(float(delta[valid].mean()))
                if (valid & m).any():
                    confident_epe.append(float(delta[valid & m].mean()))
                warped, bounds = warp_previous(gt[t], f)
                if (bounds & m).any():
                    warp_error.append(float(np.abs(warped-gt[t+1])[bounds & m].mean()))
            diagnostics.append({'scene': scene, 'motion_input': name,
                'reference_proxy_epe': float(np.mean(epe)),
                'confident_reference_proxy_epe': float(np.mean(confident_epe)),
                'fb_coverage': float(np.mean([m.mean() for m in mask])),
                'reference_rgb_warp_mae': float(np.mean(warp_error))})
        for method, outputs in methods.items():
            metrics, frames = evaluate(outputs, gt, ref_flow, ref_mask)
            rows.append({'scene': scene, 'method': method, **metrics})
            with (out/'per_frame.jsonl').open('a') as stream:
                for frame in frames:
                    stream.write(json.dumps({'scene': scene, 'method': method, **frame}, allow_nan=False)+'\n')
        print(json.dumps({'scene': scene, 'rows': len(rows)}), flush=True)
    save_json(out/'per_sequence.json', rows)
    save_json(out/'flow_diagnostics.json', diagnostics)
    save_json(out/'runtime.json', {'wall_seconds': time.perf_counter()-start})


if __name__ == '__main__':
    main()
