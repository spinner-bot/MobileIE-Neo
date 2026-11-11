"""Project correlated FST and attention noise components to actual RGB output."""
import argparse
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import torch

from causal_fst_r5 import CausalFST
from exploration_r2_data import ROOT
from explore_temporal_controls import save_json, sha256
from noise_moments_r6 import forward_probe, rgb_statistics
from rgb_noise_components_r6c import body_components, covariance_summary
from train_exploration_r2 import new_model


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', required=True)
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    previous = ROOT/'research/4.temporal-consistency/results/exploration-r6-moments'
    original_config = json.loads((previous/'config.json').read_text())
    refs = [json.loads(line) for line in (previous/'rgb_per_condition.jsonl').read_text().splitlines()]
    ref = {(r['scene'], r['photons'], r['jpeg'], r['model'], r['clipped']): r for r in refs if r['intervention'] == 'raw'}
    torch.set_num_threads(2)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device('cuda')
    models = {}
    for name, info in original_config['models'].items():
        path = ROOT/info['path']
        if sha256(path) != info['sha256']:
            raise ValueError('checkpoint hash changed')
        base = new_model(device)
        if name == 'official':
            net = base
        else:
            mode = 'spatial_post' if name.endswith('spatial_post') else 'plain'
            net = CausalFST(base, mode).to(device).eval()
            net.load_state_dict(torch.load(path, map_location=device, weights_only=False)['model'])
        models[name] = net
    save_json(out/'config.json', {'r6_config_sha256': sha256(previous/'config.json'),
        'r6_cache_index_sha256': sha256(previous/'cache_index.json'), 'draws': 64, 'batch': 8,
        'protocol_sha256': sha256(ROOT/'research/4.temporal-consistency/docs/exploration-r6c-protocol.md'),
        'git_revision_at_startup': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'source_sha256_at_startup': {str(p.relative_to(ROOT)): sha256(p) for p in Path(__file__).parent.glob('*.py')},
        'cudnn_deterministic': True, 'cudnn_tf32_allowed': torch.backends.cudnn.allow_tf32,
        'rounding_variance_ratio_limit': 1e-4, 'scope': 'exact correlated empirical algebra, not causal removal or a deployable video algorithm'})
    start, rows, repeat = time.perf_counter(), [], []
    with torch.inference_mode():
        for meta in json.loads((previous/'cache_index.json').read_text())['conditions']:
            path = ROOT/meta['path']
            if sha256(path) != meta['sha256']:
                raise ValueError('input cache changed')
            with np.load(path, allow_pickle=False) as archive:
                inputs = torch.from_numpy(archive['evaluation']).to(device)
                target = torch.from_numpy(archive['target'].transpose(2, 0, 1).copy()).to(device)
            identity = {k: meta[k] for k in ['scene', 'photons', 'jpeg']}
            for name, net in models.items():
                pred, prebody, gates = [], [], []
                for offset in range(0, 64, 8):
                    y, f = forward_probe(net, inputs[offset:offset+8])
                    pred.append(y); prebody.append(f['body']); gates.append(f['gate'])
                pred, prebody, gates = torch.cat(pred), torch.cat(prebody), torch.cat(gates)
                _, coarse = body_components(net, prebody, gates, pred)
                _, fine = body_components(net, prebody, gates, pred, split_interaction=True)
                for clipped in [False, True]:
                    row = covariance_summary(coarse, pred, clipped)
                    detail = covariance_summary(fine, pred, clipped)
                    for item in [row, detail]:
                        residual_index = item['components'].index('rounding_residual')
                        if not clipped and item['component_variance_over_total_variance'][residual_index] > 1e-4:
                            raise AssertionError('rounding cannot explain reconstruction residual')
                        if item['centered_reconstruction_max_abs'] > 1e-9 or item['variance_closure_abs'] > 1e-9:
                            raise AssertionError('RGB covariance closure failed')
                    metrics = rgb_statistics(pred.clamp(0, 1) if clipped else pred, target)
                    reference = ref[(meta['scene'], meta['photons'], meta['jpeg'], name, clipped)]
                    delta = {k: abs(v-reference[k]) for k, v in metrics.items()}
                    if max(delta.values()) > 1e-6:
                        raise AssertionError('raw repeat changed')
                    repeat.append({**identity, 'model': name, 'clipped': clipped, 'max_abs_by_metric': delta})
                    rows.append({**identity, 'model': name, 'clipped': clipped, **row, 'split_interaction': detail})
                print(json.dumps({**identity, 'model': name, 'component_rows': len(rows)}), flush=True)
    if len(rows) != 144:
        raise ValueError('fixed decomposition budget mismatch')
    save_json(out/'per_condition.json', rows)
    save_json(out/'raw_repeat_audit.json', repeat)
    save_json(out/'runtime.json', {'wall_seconds': time.perf_counter()-start, 'rows': len(rows),
                                 'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated()})


if __name__ == '__main__':
    main()
