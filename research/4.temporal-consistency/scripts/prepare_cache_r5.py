"""Freeze R4 inputs plus existing exploratory cut/light-step stress and cheap motion."""
import argparse
import json
from pathlib import Path
import subprocess
import time

import cv2
import numpy as np
import torch

from exploration_r2_data import MANIFEST, ROOT, evaluate, load_images, make_sequence
from explore_state_r2 import fuse
from explore_temporal_controls import inference, save_json, sha256
from half_motion_r4 import half_forward_backward
from prepare_cache_r4 import CACHE_INDEX as PREVIOUS_INDEX
from train_exploration_r2 import new_model

CACHE_INDEX = ROOT/'research/4.temporal-consistency/results/exploration-r5-cache/index.json'


def load_one(row):
    path = ROOT/row['path']
    if sha256(path) != row['sha256']:
        raise ValueError('cache checksum mismatch: '+str(path))
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def sources(net, device):
    for row in json.loads(PREVIOUS_INDEX.read_text())['scenes']:
        if row['split'] not in ('test', 'legacy_synthetic'):
            continue
        old = load_one(row)
        values = {k: old[k] for k in ('low', 'gt', 'reference_flow', 'reference_mask')}
        values['teacher'] = old['pretrained']
        yield row['split'], row['scene'], values, {'r4_cache_sha256': row['sha256']}
    images = load_images('exploratory_validation')
    names = list(images)
    for i, name in enumerate(names):
        for case in ('light_step', 'cut'):
            for seed in [0, 1]:
                gt, low, flows, masks = make_sequence(images[name], images[names[(i+1)%len(names)]], case, seed)
                teacher = inference(net, low, device, 1)[0]
                values = {k: np.stack(v) for k, v in [('gt', gt), ('low', low), ('teacher', teacher),
                            ('reference_flow', flows), ('reference_mask', masks)]}
                yield 'event_'+case, f'{name}_{case}_seed{seed}', values, {
                    'image': name, 'alternate': names[(i+1)%len(names)], 'case': case, 'input_seed': seed,
                    'event_frame': 12, 'motion': 'exact synthetic layered correspondence, cut frame invalid'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', required=True)
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    cache = ROOT/'.cache/experiments/exploration-r5'
    cache.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device('cuda')
    net = new_model(device)
    start = time.perf_counter()
    rows, baseline = [], []
    for domain, scene, data, origin in sources(net, device):
        teacher = data['teacher']
        pairs = [half_forward_backward(teacher[t-1], teacher[t]) for t in range(1, len(teacher))]
        data['half_flow'], data['half_mask'] = np.stack([p[0] for p in pairs]), np.stack([p[1] for p in pairs])
        path = cache/(scene+'.npz')
        np.savez_compressed(path, **data)
        row = {'domain': domain, 'scene': scene, 'frames': len(teacher), 'shape': list(data['low'].shape),
               'path': str(path.relative_to(ROOT)), 'sha256': sha256(path), 'origin': origin}
        rows.append(row)
        for method, output in [('pretrained__raw', teacher), ('pretrained__half01',
                fuse(teacher, data['half_flow'], data['half_mask'], alpha=.1))]:
            metrics, frames = evaluate(output, data['gt'], data['reference_flow'], data['reference_mask'])
            baseline.append({'domain': domain, 'scene': scene, 'method': method, **metrics})
            with (out/'baseline_per_frame.jsonl').open('a') as stream:
                for frame in frames:
                    stream.write(json.dumps({'domain': domain, 'scene': scene, 'method': method, **frame}, allow_nan=False)+'\n')
        print(json.dumps(row), flush=True)
    if len(rows) != 38:
        raise ValueError('expected 38 fixed input conditions')
    save_json(out/'index.json', {'scenes': rows, 'r4_index_sha256': sha256(PREVIOUS_INDEX),
        'kodak_manifest_sha256': sha256(MANIFEST), 'pretrained_sha256': sha256(ROOT/'MobileIE/pretrain/lolv1_best_slim.pkl'),
        'git_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'source_sha256': {str(p.relative_to(ROOT)): sha256(p) for p in Path(__file__).parent.glob('*.py')},
        'motion': 'shared half-resolution forward/backward estimated on official teacher; no candidate-dependent flow',
        'alpha': .1, 'reset': 'sequence boundary only', 'cudnn_deterministic': True,
        'cudnn_tf32_allowed': torch.backends.cudnn.allow_tf32})
    save_json(out/'baseline_per_sequence.json', baseline)
    save_json(out/'runtime.json', {'wall_seconds': time.perf_counter()-start})


if __name__ == '__main__':
    main()
