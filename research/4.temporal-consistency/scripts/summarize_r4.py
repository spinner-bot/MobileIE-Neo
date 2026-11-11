"""Audit real adaptation, retention, calibration and scale; no model selection."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from exploration_r2_data import ROOT
from explore_temporal_controls import save_json, sha256
from reliability_r4 import quality_feasible
from summarize_r3 import BASE, METRICS, grouped, lines


def training_rows(root, variants):
    rows, runs, streams = [], [], {}
    for variant in variants:
        for seed in [0, 1]:
            folder = root/f'{variant}-seed{seed}'
            history = lines(folder/'training.jsonl')
            assert [r['step'] for r in history] == list(range(1, 401))
            assert all(np.isfinite(r['task']) and np.isfinite(r['gradient_norm']) for r in history)
            runtime = json.loads((folder/'runtime.json').read_text())
            path = ROOT/runtime['checkpoint']
            assert sha256(path) == runtime['checkpoint_sha256']
            saved = torch.load(path, map_location='cpu', weights_only=False)
            assert saved['step'] == 400
            runs.append({'variant': variant, 'seed': seed, 'updates': 400,
                         'initial_task': history[0]['task'], 'checkpoint_sha256': sha256(path)})
            streams[(variant, seed)] = [r['sample_identity'] for r in history]
            for row in json.loads((folder/'validation.json').read_text()):
                rows.append({**row, 'model_seed': seed, 'method': variant+'__'+row['fusion']})
    return rows, runs, streams


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    parser.add_argument('--expanded', action='store_true')
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    baseline = lines(BASE/'results/exploration-r4-baselines/per_sequence.jsonl')
    assert len(baseline) == 210
    for row in baseline:
        row['method'] = row['method'].replace('-seed0', '').replace('-seed1', '')
    rows, runs, streams = training_rows(BASE/'results/exploration-r4-training', ['real_task', 'real_anchor', 'real_temporal'])
    initial = {}
    for seed in [0, 1]:
        assert streams[('real_task', seed)] == streams[('real_anchor', seed)] == streams[('real_temporal', seed)]
        initial[str(seed)] = max(r['initial_task'] for r in runs if r['seed'] == seed)-min(r['initial_task'] for r in runs if r['seed'] == seed)
    assert max(initial.values()) == 0.
    combined = baseline+rows
    if args.expanded:
        extra_rows, extra_runs, extra_streams = [], [], {}
        for name, variants in [('exploration-r4b-diversity', ['real12_only']),
                               ('exploration-r4b-mixture', ['real12_mix', 'real4_mix'])]:
            a, b, c = training_rows(BASE/'results'/name, variants)
            extra_rows.extend(a); extra_runs.extend(b); extra_streams.update(c)
        for seed in [0, 1]:
            for step in range(400):
                real = streams[('real_task', seed)][step][0]
                mixed = extra_streams[('real4_mix', seed)][step][0]
                assert [mixed['scene'], mixed['frame_pair']] == real
                assert extra_streams[('real12_only', seed)][step][0] == extra_streams[('real12_mix', seed)][step][0]
                assert extra_streams[('real12_mix', seed)][step][1] == extra_streams[('real4_mix', seed)][step][1]
        combined.extend(extra_rows); runs.extend(extra_runs)
    reliable = json.loads((BASE/'results/exploration-r4-reliability/validation.json').read_text())
    trials = json.loads((BASE/'results/exploration-r4-reliability/calibration_trials.json').read_text())
    assert len(reliable) == 56 and len(trials) == 120
    for row in reliable:
        row['method'] = 'gate_'+row['method']
    combined.extend(reliable)
    mapping = json.loads((BASE/'results/exploration-r4-mapping/validation.json').read_text())
    assert len(mapping) == 42
    for row in mapping:
        row['method'] = 'mapping_'+row['method']
    combined.extend(mapping)
    alpha = json.loads((BASE/'results/exploration-r4-alpha/validation.json').read_text())
    assert len(alpha) == 28
    combined.extend(alpha)
    summary = grouped(combined, ['domain', 'method'], METRICS)
    save_json(out/'domain_summary.json', summary)
    save_json(out/'by_scene.json', grouped(combined, ['domain', 'scene', 'method'], METRICS))
    save_json(out/'new_real_clips_only.json', grouped([r for r in combined if r['domain'] == 'test' and r['scene'] != 'video_21'], ['method'], METRICS))
    base_index = {(r['domain'], r['scene']): r for r in baseline if r['method'] == 'pretrained_raw'}
    violations = []
    for row in reliable+alpha:
        reference = base_index[(row['domain'], row['scene'])]
        violations.append({'domain': row['domain'], 'scene': row['scene'], 'method': row['method'],
            'quality_constraints_hold': quality_feasible(row, reference),
            **{key+'_delta': row[key]-reference[key] for key in METRICS},
            'coverage': row.get('coverage'), 'alpha': row.get('alpha')})
    save_json(out/'quality_constraint_transfer.json', violations)
    text = ['# R4探索：固定训练/测试清单子集', '', '六真实片含一个旧样例；合成域八条件来自两个已看过素材。', '',
            '| domain | method | PSNR | SSIM | temporal error | edge MAE |', '|---|---|---:|---:|---:|---:|']
    for row in summary:
        text.append(f"| {row['domain']} | {row['method']} | {row['psnr_rgb']:.4f} | {row['ssim_rgb']:.4f} | {row['temporal_error']:.4f} | {row['edge_mae']:.4f} |")
    (out/'tables.md').write_text('\n'.join(text)+'\n')
    transfer = lines(BASE/'results/exploration-r4-transfer/per_sequence.jsonl')
    assert len(transfer) == 180
    for row in transfer:
        row['size'] = 'x'.join(map(str, row['size']))
        row['method'] = row['model'].split('-seed')[0]
        row['brightness_abs_error'] = abs(float(np.mean(row['output_rgb_mean']))-float(np.mean(row['target_rgb_mean'])))
    fit = grouped(transfer, ['size', 'split', 'method'], METRICS+['brightness_abs_error'])
    save_json(out/'scale_fit_summary.json', fit)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    methods = ['pretrained', 'synthetic_task', 'real_task', 'real_anchor', 'real_temporal']
    for ax, size in zip(axes, ['320x180', '480x270']):
        index = {(r['split'], r['method']): r for r in fit if r['size'] == size}
        x = np.arange(len(methods))
        for split, delta in [('train', -.18), ('test', .18)]:
            ax.bar(x+delta, [index[(split, m)]['psnr_rgb'] for m in methods], width=.36, label=split)
        ax.set_xticks(x, methods, rotation=25, ha='right')
        ax.set(title=size, ylabel='PSNR (dB)')
        ax.legend(); ax.grid(axis='y', alpha=.2)
    fig.tight_layout(); fig.savefig(out/'fit_and_resolution.png', dpi=160); plt.close(fig)
    selected = ['pretrained_raw', 'pretrained_enhanced_fb', 'synthetic_task__raw', 'real_task__raw',
                'real_anchor__raw', 'real_temporal__raw', 'gate_enhanced__photo', 'global_alpha_enhanced']
    if args.expanded:
        selected += ['real12_only__raw', 'real12_mix__raw', 'real4_mix__raw']
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4))
    handles = []
    for ax, domain in zip(axes, ['test', 'legacy_synthetic']):
        index = {r['method']: r for r in summary if r['domain'] == domain}
        for i, method in enumerate(selected):
            row = index[method]
            point = ax.scatter(row['temporal_error'], row['psnr_rgb'], color=plt.get_cmap('tab20')(i), label=method)
            if domain == 'test':
                handles.append(point)
        ax.set(title=domain, xlabel='GT-relative temporal error (lower better)', ylabel='PSNR (dB)')
        ax.grid(alpha=.2)
    fig.subplots_adjust(right=.74, bottom=.17)
    fig.legend(handles, selected, loc='center right', fontsize=8)
    fig.savefig(out/'two_domain_tradeoff.png', dpi=160); plt.close(fig)
    save_json(out/'audit.json', {'training_runs': runs, 'updates': len(runs)*400,
        'evaluation_rows': len(combined), 'calibration_rows': 184, 'resolution_diagnostic_rows': 180,
        'matched_initial_task_difference': initial, 'same_real_streams_verified': True,
        'checkpoint_hashes_verified': True, 'expanded_mix_streams_verified': args.expanded,
        'real_test_clips': 6, 'new_real_test_clips': 5, 'legacy_conditions': 8, 'legacy_sources': 2,
        'note': 'clip identities may share physical scenes; no independent-sample or formal benchmark claim'})
    print(json.dumps({'runs': len(runs), 'updates': len(runs)*400, 'evaluation_rows': len(combined)}, indent=2))


if __name__ == '__main__':
    main()
