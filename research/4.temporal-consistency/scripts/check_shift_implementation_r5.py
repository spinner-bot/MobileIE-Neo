"""Numerical audit before using direct fixed-channel replacement timing."""
import argparse
import copy
import json

import numpy as np
import torch

from causal_fst_r5 import CausalFST
from exploration_r2_data import ROOT
from explore_temporal_controls import save_json, sha256
from prepare_cache_r5 import CACHE_INDEX, load_one
from train_exploration_r2 import new_model
from video_metrics import psnr


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', required=True)
    args = p.parse_args()
    torch.set_num_threads(2)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device('cuda')
    rows = []
    cache = [(r, load_one(r)) for r in json.loads(CACHE_INDEX.read_text())['scenes']]
    with torch.inference_mode():
        for seed in [0, 1]:
            path = ROOT/'.cache/checkpoints/exploration-r5'/f'tsm_fixed-seed{seed}'/'step1000.pth'
            original = CausalFST(new_model(device), 'tsm_fixed').to(device).eval()
            original.load_state_dict(torch.load(path, map_location=device, weights_only=False)['model'])
            direct = copy.deepcopy(original)
            direct.direct_fixed_shift = True
            for meta, data in cache:
                original.reset(); direct.reset()
                differences, clipped_differences, psnr_delta = [], [], []
                for frame, gt in zip(data['low'], data['gt']):
                    x = torch.from_numpy((frame/255).transpose(2, 0, 1).copy()).unsqueeze(0).to(device)
                    a, b = original(x), direct(x)
                    differences.append(float((a-b).abs().max()))
                    clipped_differences.append(float((a.clamp(0, 1)-b.clamp(0, 1)).abs().max()))
                    convert = lambda v: v.clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy()*255
                    psnr_delta.append(psnr(convert(b), gt)-psnr(convert(a), gt))
                row = {'scene': meta['scene'], 'domain': meta['domain'], 'seed': seed,
                    'checkpoint_sha256': sha256(path), 'unclipped_normalized_max_abs': max(differences),
                    'clipped_normalized_max_abs': max(clipped_differences),
                    'psnr_mean_delta_direct_minus_reference': float(np.mean(psnr_delta))}
                row['passes_prerun_thresholds'] = row['unclipped_normalized_max_abs'] <= 1e-4 and abs(row['psnr_mean_delta_direct_minus_reference']) <= 1e-3
                rows.append(row)
    save_json(args.out, {'rows': rows, 'cache_index_sha256': sha256(CACHE_INDEX),
        'all_pass_prerun_thresholds': all(r['passes_prerun_thresholds'] for r in rows),
        'scope': 'same trained weights, tested 38 cached conditions x2 seeds; floating-point tolerance check, not bitwise identity proof',
        'thresholds_prerun': {'unclipped_max_abs': 1e-4, 'absolute_sequence_mean_psnr_delta': .001},
        'reference': 'u+(previous-u)', 'direct': 'previous', 'primary_quality_evaluation': 'unchanged reference expression'})
    if not all(r['passes_prerun_thresholds'] for r in rows):
        raise SystemExit('predeclared numerical threshold failed; full diagnostics saved without relaxing it')


if __name__ == '__main__':
    main()
