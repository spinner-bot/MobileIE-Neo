"""Prepare the registered six DAVIS low-light conditions on two CPU threads."""
import hashlib
import json
from pathlib import Path
import subprocess
import time

import cv2
import numpy as np
from PIL import Image
import torch

from exploration_r2_data import ROOT, degrade, evaluate
from explore_state_r2 import forward_backward
from explore_temporal_controls import inference, save_json, sha256
from half_motion_r4 import half_forward_backward
from train_exploration_r2 import new_model

BASE = ROOT/'research/4.temporal-consistency'


def main():
    source = BASE/'data/external-davis/manifest.json'
    manifest = json.loads(source.read_text())
    protocol = BASE/'docs/exploration-r30-protocol.md'
    assert sha256(protocol) == manifest['protocol_sha256']
    out = BASE/'results/exploration-r30-cache'
    out.mkdir(parents=True, exist_ok=False)
    cache = ROOT/'.cache/experiments/exploration-r30'
    assert cache.resolve().is_relative_to('/data/Users/zrh/data/MobileIE-Neo')
    cache.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    cv2.setNumThreads(1)
    torch.use_deterministic_algorithms(True)
    assert not torch.cuda.is_initialized()
    device = torch.device('cpu')
    net = new_model(device).eval()
    save_json(out/'config.json', {'device': 'cpu', 'torch_threads': 2, 'opencv_threads': 1,
              'source_manifest_sha256': sha256(source), 'protocol_sha256': sha256(protocol),
              'official_weights_sha256': sha256(ROOT/'MobileIE/pretrain/lolv1_best_slim.pkl'),
              'torch_version': torch.__version__, 'numpy_version': np.__version__, 'opencv_version': cv2.__version__,
              'revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
              'scope': 'No training or CUDA; fixed original 480p frames and six independent-noise conditions.'})
    start = time.perf_counter()
    rows, baselines = [], []
    for scene in ['bear', 'lucia', 'walking']:
        images = []
        members = [r for r in manifest['frames'] if '/'+scene+'/' in r['name']]
        assert len(members) == 64
        for record in members:
            path = ROOT/record['path']
            assert sha256(path) == record['sha256']
            with Image.open(path) as image:
                images.append(np.asarray(image.convert('RGB'), dtype=np.float32))
        gt = np.stack(images)
        assert len(gt) == 64 and gt.shape[1] == 480 and gt.shape[-1] == 3
        reference = [forward_backward(gt[t-1], gt[t]) for t in range(1, len(gt))]
        reference_flow = np.stack([r[0] for r in reference])
        reference_mask = np.stack([r[1] for r in reference])
        for photons in [20, 80]:
            seed = int.from_bytes(hashlib.sha256(f'R30:{scene}:{photons}:0'.encode()).digest()[:8], 'little')
            rng = np.random.default_rng(seed)
            low = np.stack([degrade(frame, rng, photons=photons, jpeg=False) for frame in gt])
            teacher = np.stack(inference(net, low, device, 1)[0])
            motion = [half_forward_backward(teacher[t-1], teacher[t]) for t in range(1, len(teacher))]
            values = {'low': low, 'gt': gt, 'teacher': teacher, 'reference_flow': reference_flow,
                      'reference_mask': reference_mask, 'half_flow': np.stack([r[0] for r in motion]),
                      'half_mask': np.stack([r[1] for r in motion])}
            label = f'{scene}_photons{photons}'
            path = cache/(label+'.npz')
            np.savez_compressed(path, **values)
            row = {'domain': 'davis_train_pilot', 'scene': label, 'source_video': scene, 'frames': len(gt),
                   'shape': list(gt.shape), 'photons': photons, 'input_rng_seed': seed,
                   'path': str(path.relative_to(ROOT)), 'sha256': sha256(path)}
            rows.append(row)
            save_json(out/(label+'.json'), row)
            metrics, per_frame = evaluate(teacher, gt, reference_flow, reference_mask)
            baselines.append({'domain': row['domain'], 'scene': label, 'method': 'official_mobileie_raw', **metrics})
            with (out/'baseline_per_frame.jsonl').open('a') as stream:
                for frame in per_frame:
                    stream.write(json.dumps({'scene': label, 'method': 'official_mobileie_raw', **frame}, allow_nan=False)+'\n')
            print(json.dumps({'scene': label, 'prepared_conditions': len(rows), 'shape': list(gt.shape)}), flush=True)
    assert len(rows) == 6 and not torch.cuda.is_initialized()
    save_json(out/'index.json', {'scenes': rows, 'source_manifest_sha256': sha256(source),
              'protocol_sha256': sha256(protocol), 'motion': 'official MobileIE teacher half-FB for deployment; GT-estimated full-FB only for evaluation',
              'source_video_count': 3, 'input_conditions': 6, 'validation_pixels_read': 0})
    save_json(out/'baseline_per_sequence.json', baselines)
    save_json(out/'runtime.json', {'completed_normally': True, 'conditions': 6, 'decoded_source_frames': 192,
              'degraded_frames': 384, 'cuda_initialized': False, 'device': 'cpu', 'wall_seconds': time.perf_counter()-start})


if __name__ == '__main__':
    main()
