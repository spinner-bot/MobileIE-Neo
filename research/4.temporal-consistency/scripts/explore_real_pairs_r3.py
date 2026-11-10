"""Fixed author-sample real low-light pairs; no training or hyperparameter search."""
import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import torch

from exploration_r2_data import ROOT, evaluate
from explore_products_r3 import intervention, pair_blend
from explore_state_r2 import forward_backward, fuse
from explore_temporal_controls import inference, save_json, sha256
from paired_fst_r3 import InputPairNetwork, TrainablePairFST
from train_exploration_r2 import new_model
from train_products_r3 import streaming


def load_pairs(manifest, scene):
    pairs = {}
    for side in ['gt', 'input']:
        rows = sorted([r for r in manifest['files'] if r['source_path'].startswith(side+'/'+scene+'/')],
                      key=lambda r: r['source_path'])
        if len(rows) != 16:
            raise ValueError('expected exactly 16 frames per side')
        frames = []
        for row in rows:
            path = ROOT/row['path']
            if sha256(path) != row['sha256']:
                raise ValueError('paired file hash mismatch')
            with Image.open(path) as image:
                rgb = np.asarray(image.convert('RGB'))
            if rgb.shape[:2] != (1080, 1920):
                raise ValueError(f'unexpected native image shape: {rgb.shape}')
            frames.append(cv2.resize(rgb, (480, 270), interpolation=cv2.INTER_AREA).astype(np.float32))
        pairs[side] = frames
    return pairs['gt'], pairs['input']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    cv2.setNumThreads(1)
    torch.set_num_threads(2)
    device = torch.device('cuda')
    models, recurrent, provenance = {'pretrained': new_model(device)}, {}, {}
    for label, cache, variant in [('task', 'exploration-r2', 'task'), ('T1', 'exploration-r2-lambda10', 'task_temporal')]:
        for seed in [0, 1]:
            path = ROOT/'.cache/checkpoints'/cache/f'{variant}-seed{seed}'/'last.pth'
            saved = torch.load(path, map_location=device, weights_only=False)
            net = new_model(device)
            net.load_state_dict(saved['model'])
            name = f'{label}-seed{seed}'
            models[name] = net
            provenance[name] = {'path': str(path.relative_to(ROOT)), 'sha256': sha256(path)}
    for variant in ['cross', 'mean_square', 'input_mean']:
        for seed in [0, 1]:
            path = ROOT/'.cache/checkpoints/exploration-r3'/f'{variant}-seed{seed}'/'last.pth'
            saved = torch.load(path, map_location=device, weights_only=False)
            if saved['step'] != 400:
                raise ValueError('only fixed final checkpoint eligible')
            net = new_model(device)
            if variant == 'input_mean':
                net = InputPairNetwork(net)
            else:
                net.body = TrainablePairFST(net.body, variant)
            net.load_state_dict(saved['model'])
            recurrent[f'{variant}-seed{seed}'] = net
            provenance[f'{variant}-seed{seed}'] = {'path': str(path.relative_to(ROOT)), 'sha256': sha256(path)}
    manifest = json.loads(Path(args.manifest).read_text())
    save_json(out/'config.json', {'manifest_sha256': sha256(args.manifest), 'repo_revision': manifest['revision'],
        'models': provenance, 'size': [480, 270], 'resize': 'INTER_AREA', 'extra_degradation': False,
        'fusion_alpha': .7, 'feature_mask_erosion_radius': 8,
        'temporal_metric': 'estimated reference flow+FB, subtract reference frame residual; not GT flow',
        'scope': 'two author samples, 16 frames each; not full native-resolution benchmark'})
    start = time.perf_counter()
    count = 0
    with (out/'per_sequence.jsonl').open('w') as stream, (out/'per_frame.jsonl').open('w') as trace:
        for scene in ['example_1', 'example_2']:
            gt, low = load_pairs(manifest, scene)
            ref = [forward_backward(gt[t-1], gt[t]) for t in range(1, len(gt))]
            reference_flow, reference_mask = [r[0] for r in ref], [r[1] for r in ref]
            pairs = [forward_backward(low[t-1], low[t]) for t in range(1, len(low))]
            flows, full_mask = [p[0] for p in pairs], [p[1] for p in pairs]
            masks = [cv2.erode(m.astype(np.uint8), np.ones((17, 17), np.uint8),
                     borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool) for m in full_mask]
            methods = {}
            for name, model in models.items():
                raw, _ = inference(model, low, device, 1)
                methods[name+'__raw'] = raw
                methods[name+'__fb'] = fuse(raw, flows, full_mask)
            for name, net in recurrent.items():
                gamma = float(net.body.gamma.detach())
                for state in ['history', 'history_off']:
                    with torch.no_grad():
                        net.body.gamma.fill_(gamma if state == 'history' else 0.)
                    raw = streaming(net, low, flows, masks, device)
                    methods[name+'__'+state+'__raw'] = raw
                    methods[name+'__'+state+'__fb'] = fuse(raw, flows, full_mask)
                with torch.no_grad():
                    net.body.gamma.fill_(gamma)
            for mode in ['cross', 'mean_square']:
                raw = intervention(models['pretrained'], low, flows, masks, 'body', mode, device)
                methods['frozen_body_'+mode+'__raw'] = raw
                methods['frozen_body_'+mode+'__fb'] = fuse(raw, flows, full_mask)
            for name, raw in [('input_pair', inference(models['pretrained'], pair_blend(low, flows, masks), device, 1)[0]),
                              ('output_pair', pair_blend(methods['pretrained__raw'], flows, masks))]:
                methods[name+'__raw'] = raw
                methods[name+'__fb'] = fuse(raw, flows, full_mask)
            if len(methods) != 42:
                raise ValueError('incomplete preregistered method matrix')
            for method, output in methods.items():
                metrics, frames = evaluate(output, gt, reference_flow, reference_mask)
                row = {'scene': scene, 'method': method, **metrics,
                    'input_mean': float(np.mean([f.mean() for f in low])),
                    'reference_mean': float(np.mean([f.mean() for f in gt])),
                    'low_flow_coverage': float(np.mean([m.mean() for m in full_mask])),
                    'feature_history_coverage': float(np.mean([m.mean() for m in masks]))}
                stream.write(json.dumps(row, allow_nan=False)+'\n')
                for frame in frames:
                    trace.write(json.dumps({'scene': scene, 'method': method, **frame}, allow_nan=False)+'\n')
                count += 1
            print(json.dumps({'scene': scene, 'rows': count}), flush=True)
    save_json(out/'runtime.json', {'wall_seconds': time.perf_counter()-start, 'sequence_rows': count})


if __name__ == '__main__':
    main()
