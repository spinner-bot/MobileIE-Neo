"""Strong frozen, synthetic-trained and traditional controls on shared R4 inputs."""
import argparse
import json
from pathlib import Path
import time

import cv2
import numpy as np
import torch

from exploration_r2_data import ROOT, evaluate
from explore_products_r3 import intervention
from explore_state_r2 import fuse
from explore_temporal_controls import inference, save_json, sha256
from prepare_cache_r4 import CACHE_INDEX, load_cache
from train_exploration_r2 import new_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device('cuda')
    pretrained = new_model(device)
    models, provenance = {}, {}
    for seed in [0, 1]:
        path = ROOT/'.cache/checkpoints/exploration-r2'/f'task-seed{seed}'/'last.pth'
        saved = torch.load(path, map_location=device, weights_only=False)
        net = new_model(device)
        net.load_state_dict(saved['model'])
        models[f'synthetic_task-seed{seed}'] = net
        provenance[f'synthetic_task-seed{seed}'] = {'path': str(path.relative_to(ROOT)), 'sha256': sha256(path)}
    save_json(out/'config.json', {'cache_index_sha256': sha256(CACHE_INDEX), 'models': provenance,
        'fusion_alpha': .7, 'input_nlm': {'h': 3, 'hcolor': 3, 'template': 7, 'search': 21},
        'output_gaussian_sigma': .6, 'feature_mask_erosion_radius': 8,
        'scope': 'same official enhanced-domain motion shared with trainable models; reference only for evaluation'})
    rows = []
    start = time.perf_counter()
    max_raw_difference = 0.
    for domain in ['test', 'legacy_synthetic']:
        for scene, data in load_cache(domain).items():
            raw = data['pretrained']
            recomputed, _ = inference(pretrained, data['low'], device, 1)
            max_raw_difference = max(max_raw_difference, float(np.abs(np.stack(recomputed)-raw).max()))
            flow, mask = data['enhanced_flow'], data['enhanced_mask']
            methods = {'pretrained_raw': raw,
                'pretrained_low_fb': fuse(raw, data['low_flow'], data['low_mask']),
                'pretrained_gamma_fb': fuse(raw, data['gamma_flow'], data['gamma_mask']),
                'pretrained_enhanced_fb': fuse(raw, flow, mask),
                'pretrained_zero_ema': fuse(raw, np.zeros_like(flow)),
                'output_gaussian': [cv2.GaussianBlur(f, (0, 0), .6) for f in raw]}
            denoised = [cv2.fastNlMeansDenoisingColored(np.clip(np.rint(frame), 0, 255).astype(np.uint8),
                         None, 3, 3, 7, 21).astype(np.float32) for frame in data['low']]
            methods['input_nlm'] = inference(pretrained, denoised, device, 1)[0]
            for label, net in models.items():
                output, _ = inference(net, data['low'], device, 1)
                methods[label+'__raw'] = output
                methods[label+'__enhanced_fb'] = fuse(output, flow, mask)
            feature_mask = [cv2.erode(m.astype(np.uint8), np.ones((17, 17), np.uint8),
                            borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool) for m in mask]
            for mode in ['cross', 'mean_square']:
                output = intervention(pretrained, data['low'], flow, feature_mask, 'body', mode, device)
                methods['frozen_body_'+mode+'__raw'] = output
                methods['frozen_body_'+mode+'__enhanced_fb'] = fuse(output, flow, mask)
            for method, output in methods.items():
                metrics, frames = evaluate(output, data['gt'], data['reference_flow'], data['reference_mask'])
                row = {'domain': domain, 'scene': scene, 'method': method, **metrics}
                rows.append(row)
                with (out/'per_sequence.jsonl').open('a') as stream:
                    stream.write(json.dumps(row, allow_nan=False)+'\n')
                with (out/'per_frame.jsonl').open('a') as stream:
                    for frame in frames:
                        stream.write(json.dumps({'domain': domain, 'scene': scene, 'method': method, **frame}, allow_nan=False)+'\n')
            print(json.dumps({'domain': domain, 'scene': scene, 'rows': len(rows)}), flush=True)
    save_json(out/'runtime.json', {'wall_seconds': time.perf_counter()-start, 'rows': len(rows),
        'raw_prediction_repeat_max_abs_difference': max_raw_difference,
        'raw_repeated_sequence_audits': 14})


if __name__ == '__main__':
    main()
