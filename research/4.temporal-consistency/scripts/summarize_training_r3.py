"""Matched training audits, scene-cluster uncertainty and optional motion results."""
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
from summarize_r3 import BASE, METRICS, grouped, lines


def paired_delta(rows, first, second, metric):
    index = {(r['scene'], r['case'], r['input_seed'], r['model_seed'], r['method']): r for r in rows}
    scene = defaultdict(list)
    for row in rows:
        if row['method'] == first:
            other = index[(row['scene'], row['case'], row['input_seed'], row['model_seed'], second)]
            scene[row['scene']].append(row[metric]-other[metric])
    values = np.asarray([np.mean(v) for v in scene.values()])
    rng = np.random.default_rng(20261111)
    boot = rng.choice(values, size=(10000, len(values)), replace=True).mean(1)
    return {'first': first, 'second': second, 'metric': metric,
            'mean_delta': float(values.mean()), 'scene_means': {k: float(np.mean(v)) for k, v in scene.items()},
            'positive_scenes': int((values > 0).sum()), 'scene_count': len(values),
            'scene_bootstrap_95': np.quantile(boot, [.025, .975]).tolist(),
            'interpretation': 'exploratory scene-cluster bootstrap, only six scenes and two training seeds; not a confirmatory significance claim'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    parser.add_argument('--motion', action='store_true')
    parser.add_argument('--real', action='store_true')
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows, runs = [], []
    for variant in ['task', 'cross', 'mean_square', 'input_mean']:
        name = 'exploration-r3-training-task' if variant == 'task' else 'exploration-r3-training-input' if variant == 'input_mean' else 'exploration-r3-training-products'
        folder = BASE/'results'/name
        for seed in [0, 1]:
            run = folder/f'{variant}-seed{seed}'
            history = lines(run/'training.jsonl')
            assert [h['step'] for h in history] == list(range(1, 401))
            config = json.loads((run/'config.json').read_text())
            runtime = json.loads((run/'runtime.json').read_text())
            checkpoint = ROOT/runtime['checkpoint']
            assert sha256(checkpoint) == runtime['checkpoint_sha256']
            saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
            assert saved['step'] == 400
            runs.append({'variant': variant, 'seed': seed, 'steps': len(history),
                         'initial_task': history[0]['task'], 'initial_gradient_norm': history[0]['gradient_norm'],
                         'final_gamma': history[-1]['gamma'], 'checkpoint_sha256': sha256(checkpoint)})
            for row in json.loads((run/'validation.json').read_text()):
                rows.append({**row, 'variant': variant, 'model_seed': seed,
                    'method': variant+'__'+row['state']+'__'+row['fusion']})
    initial = {str(seed): max(r['initial_task'] for r in runs if r['seed'] == seed)
               -min(r['initial_task'] for r in runs if r['seed'] == seed) for seed in [0, 1]}
    assert max(initial.values()) == 0.
    assert len(rows) == 1344
    save_json(out/'training_audit.json', {'runs': runs, 'total_updates': 3200,
        'validation_rows': len(rows), 'unique_input_conditions': 48,
        'initial_task_within_seed_max_abs_difference': initial, 'checkpoint_hashes_match': True})
    summary = grouped(rows, ['method'], METRICS)
    save_json(out/'training_summary.json', summary)
    save_json(out/'training_by_case.json', grouped(rows, ['method', 'case'], METRICS))
    save_json(out/'training_by_scene.json', grouped(rows, ['method', 'scene'], METRICS))
    save_json(out/'training_by_seed.json', grouped(rows, ['method', 'model_seed'], METRICS))
    deltas = [paired_delta(rows, first, second, metric) for first, second in [
        ('cross__history__fb', 'task__ordinary__fb'),
        ('mean_square__history__fb', 'task__ordinary__fb'),
        ('cross__history__fb', 'mean_square__history__fb'),
        ('cross__history__fb', 'input_mean__history__fb'),
        ('input_mean__history__fb', 'task__ordinary__fb'),
        ('cross__history__fb', 'cross__history_off__fb')]
        for metric in ['psnr_rgb', 'temporal_error']]
    save_json(out/'paired_scene_deltas.json', deltas)
    text = ['# 第三轮受控训练', '', '48合成输入条件，两个训练seed均完整400步。', '',
        '| method | PSNR | SSIM | temporal error | edge MAE |', '|---|---:|---:|---:|---:|']
    for row in summary:
        text.append(f"| {row['method']} | {row['psnr_rgb']:.4f} | {row['ssim_rgb']:.4f} | {row['temporal_error']:.4f} | {row['edge_mae']:.4f} |")
    (out/'training_tables.md').write_text('\n'.join(text)+'\n')
    fig, ax = plt.subplots(figsize=(11, 5))
    for i, row in enumerate(summary):
        ax.scatter(row['temporal_error'], row['psnr_rgb'], color=plt.get_cmap('tab10')(i),
                   marker='o' if row['method'].endswith('__fb') else '^', label=row['method'])
    ax.set(xlabel='GT-relative temporal error (lower better)', ylabel='PSNR (dB)', title='Matched training and history-off controls')
    ax.grid(alpha=.2)
    fig.subplots_adjust(right=.61)
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
    fig.savefig(out/'trained_product_tradeoff.png', dpi=160)
    plt.close(fig)
    if args.motion:
        motion = (lines(BASE/'results/exploration-r3-motion-baseline/per_sequence.jsonl')
                  +lines(BASE/'results/exploration-r3-motion-trained/per_sequence.jsonl')
                  +lines(BASE/'results/exploration-r3-motion-input/per_sequence.jsonl'))
        assert len(motion) == 8*44
        assert len({(r['scene'], r['read_rho'], r['input_seed'], r['method']) for r in motion}) == len(motion)
        hashes = defaultdict(set)
        for row in motion:
            hashes[(row['scene'], row['read_rho'], row['input_seed'])].add(row['input_sha256'])
            row['model_method'] = row['method']
            row['method'] = row['method'].replace('-seed0', '').replace('-seed1', '')
        assert len(hashes) == 8 and all(len(h) == 1 for h in hashes.values())
        save_json(out/'motion_by_model.json', grouped(motion, ['model_method'], METRICS))
        save_json(out/'motion_summary.json', grouped(motion, ['method'], METRICS))
        save_json(out/'motion_by_scene.json', grouped(motion, ['scene', 'method'], METRICS))
        save_json(out/'motion_by_noise.json', grouped(motion, ['read_rho', 'method'], METRICS))
        save_json(out/'motion_audit.json', {'rows': len(motion), 'unique_input_conditions': 8,
            'source_clips': 2, 'frames_per_clip': 24, 'reference_flow_is_estimated': True,
            'staged_input_hashes_match': True,
            'note': 'actual motion with synthetic low-light noise; not paired real low-light evaluation'})
    if args.real:
        real = lines(BASE/'results/exploration-r3-real-pairs/per_sequence.jsonl')
        assert len(real) == 84
        assert len({(r['scene'], r['method']) for r in real}) == 84
        for row in real:
            row['model_method'] = row['method']
            row['method'] = row['method'].replace('-seed0', '').replace('-seed1', '')
        real_summary = grouped(real, ['method'], METRICS)
        save_json(out/'real_summary.json', real_summary)
        save_json(out/'real_by_scene.json', grouped(real, ['scene', 'method'], METRICS))
        save_json(out/'real_audit.json', {'rows': 84, 'unique_input_sequences': 2, 'frames_per_sequence': 16,
            'resized_size': [480, 270], 'extra_degradation': False, 'note': 'author samples, not full native-resolution benchmark'})
        text = ['# 真实低光样例探索', '', '两段作者样例各16帧，480×270；不代表完整标准基准。', '',
                '| method | PSNR | SSIM | temporal error | edge MAE |', '|---|---:|---:|---:|---:|']
        for row in real_summary:
            text.append(f"| {row['method']} | {row['psnr_rgb']:.4f} | {row['ssim_rgb']:.4f} | {row['temporal_error']:.4f} | {row['edge_mae']:.4f} |")
        (out/'real_tables.md').write_text('\n'.join(text)+'\n')
        if args.motion:
            transfer = json.loads((out/'motion_summary.json').read_text())
            selections = ['task__fb', 'cross__history__fb', 'mean_square__history__fb', 'input_mean__history__fb',
                          'frozen_body_cross__fb', 'frozen_body_mean_square__fb']
            fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
            for ax, source, title in zip(axes, [transfer, real_summary], ['Actual motion + synthetic low light', 'Real low-light pairs (two samples)']):
                index = {r['method'].replace('pretrained_body_', 'frozen_body_'): r for r in source}
                baseline = index['pretrained__fb']['psnr_rgb']
                labels = ['Task', 'Cross', 'Mean square', 'Input mean', 'Frozen cross', 'Frozen mean']
                values = [index[m]['psnr_rgb']-baseline for m in selections]
                ax.bar(labels, values, color=['tab:blue']*4+['tab:orange']*2)
                ax.axhline(0, color='black', linewidth=1)
                ax.axhline(index['pretrained__raw']['psnr_rgb']-baseline, color='grey', linestyle='--',
                           linewidth=1, label='Pretrained without fusion')
                ax.legend(fontsize=8)
                ax.tick_params(axis='x', labelrotation=35, labelsize=9)
                ax.set(title=title, ylabel='PSNR delta vs pretrained + same FB fusion (dB)')
                ax.grid(axis='y', alpha=.2)
            fig.tight_layout()
            fig.savefig(out/'transfer_against_pretrained.png', dpi=160)
            plt.close(fig)
    print(json.dumps({'runs': len(runs), 'updates': 3200, 'validation_rows': len(rows), 'initial_difference': initial}, indent=2))


if __name__ == '__main__':
    main()
