"""Final-step evaluation with current-history and branch-off causal ablations."""
import argparse
import json
from pathlib import Path
import subprocess
import time

import cv2
import numpy as np
import torch

from causal_fst_r5 import CausalFST, MODES
from exploration_r2_data import ROOT, evaluate
from explore_state_r2 import fuse
from explore_temporal_controls import save_json, sha256
from prepare_cache_r5 import CACHE_INDEX, load_one
from train_exploration_r2 import new_model


def infer(net, images, device, history):
    net.reset(history)
    result = []
    with torch.inference_mode():
        for frame in images:
            x = torch.from_numpy((frame/255.).transpose(2, 0, 1).copy()).unsqueeze(0).to(device)
            pred = net(x).clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy()*255.
            result.append(pred.astype(np.float32))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--variants', required=True)
    p.add_argument('--out', required=True)
    args = p.parse_args()
    variants = args.variants.split(',')
    if any(v not in (*MODES, 'r2_task') for v in variants):
        p.error('known modes or r2_task required')
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device('cuda')
    metadata = json.loads(CACHE_INDEX.read_text())
    cache = [(row, load_one(row)) for row in metadata['scenes']]
    save_json(out/'config.json', {'variants': variants, 'checkpoint_step': 1000,
        'cache_index_sha256': sha256(CACHE_INDEX), 'alpha': .1,
        'git_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'source_sha256': {str(p.relative_to(ROOT)): sha256(p) for p in Path(__file__).parent.glob('*.py')},
        'cudnn_deterministic': True, 'cudnn_tf32_allowed': torch.backends.cudnn.allow_tf32,
        'r2_task_scope': '400-step 128-patch synthetic reference, not budget-matched'} )
    start, count = time.perf_counter(), 0
    checkpoints = []
    for variant in variants:
        for seed in [0, 1]:
            if variant == 'r2_task':
                path = ROOT/'.cache/checkpoints/exploration-r2'/f'task-seed{seed}'/'last.pth'
                net = CausalFST(new_model(device), 'plain').to(device)
                net.backbone.load_state_dict(torch.load(path, map_location=device, weights_only=False)['model'])
            else:
                path = ROOT/'.cache/checkpoints/exploration-r5'/f'{variant}-seed{seed}'/'step1000.pth'
                net = CausalFST(new_model(device), variant).to(device)
                net.load_state_dict(torch.load(path, map_location=device, weights_only=False)['model'])
            net.eval()
            checkpoints.append({'variant': variant, 'seed': seed, 'path': str(path.relative_to(ROOT)), 'sha256': sha256(path)})
            histories = ['causal']
            if variant.startswith('temporal_'):
                histories += ['current', 'off']
            elif variant.startswith('tsm_'):
                histories += ['current']
            for history in histories:
                for row, data in cache:
                    outputs = infer(net, data['low'], device, history)
                    for post, result in [('raw', outputs), ('half01', fuse(outputs, data['half_flow'], data['half_mask'], alpha=.1))]:
                        method = f'{variant}__{history}__{post}'
                        identity = {'domain': row['domain'], 'scene': row['scene'], 'variant': variant,
                                    'train_seed': seed, 'history': history, 'post': post, 'method': method}
                        metrics, frames = evaluate(result, data['gt'], data['reference_flow'], data['reference_mask'])
                        with (out/'per_sequence.jsonl').open('a') as stream:
                            stream.write(json.dumps({**identity, **metrics}, allow_nan=False)+'\n')
                        with (out/'per_frame.jsonl').open('a') as stream:
                            for frame in frames:
                                stream.write(json.dumps({**identity, **frame}, allow_nan=False)+'\n')
                        count += 1
                print(json.dumps({'variant': variant, 'seed': seed, 'history': history, 'completed_rows': count}), flush=True)
            del net
    save_json(out/'checkpoints.json', checkpoints)
    save_json(out/'runtime.json', {'wall_seconds': time.perf_counter()-start, 'rows': count})


if __name__ == '__main__':
    main()
