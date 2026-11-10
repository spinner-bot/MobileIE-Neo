"""Audit R3 repeat controls and summarize scene-level mechanism diagnostics."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from exploration_r2_data import ROOT
from explore_temporal_controls import save_json

BASE = ROOT/'research/4.temporal-consistency'
METRICS = ['psnr_rgb', 'ssim_rgb', 'edge_mae', 'temporal_error']


def lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def grouped(rows, keys, metrics):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[k] for k in keys)].append(row)
    result = []
    for key, values in groups.items():
        row = {**dict(zip(keys, key)), 'n': len(values)}
        for metric in metrics:
            valid = [v[metric] for v in values if v[metric] is not None]
            row[metric] = float(np.mean(valid)) if valid else None
        result.append(row)
    return result


def compare(old, new, mapping):
    index = {(r['scene'], r['case'], r['input_seed'], r['method']): r for r in old}
    delta = {m: 0. for m in METRICS}
    n = 0
    for row in new:
        if row['method'] not in mapping:
            continue
        ref = index[(row['scene'], row['case'], row['input_seed'], mapping[row['method']])]
        for m in METRICS:
            delta[m] = max(delta[m], abs(row[m]-ref[m]))
        n += 1
    return {'compared_rows': n, 'max_absolute_difference': delta,
            'note': 'GPU numerical execution may differ; report differences, do not silently treat them as exact'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    feature = json.loads((BASE/'results/exploration-r3-features/per_channel.json').read_text())
    reset = lines(BASE/'results/exploration-r3-resets/per_sequence.jsonl')
    trace = lines(BASE/'results/exploration-r3-resets/per_frame.jsonl')
    product = lines(BASE/'results/exploration-r3-products/per_sequence.jsonl')
    assert len(feature) == 12*4*2*12
    assert len(reset) == len(product) == 720 and len(trace) == 720*24
    for rows in [reset, product]:
        assert len({(r['scene'], r['case'], r['input_seed'], r['method']) for r in rows}) == 720
    old = lines(BASE/'results/exploration-r2-state/per_sequence.jsonl')
    audit = {'feature_rows': len(feature), 'reset_rows': len(reset), 'reset_frame_rows': len(trace),
             'product_rows': len(product), 'unique_input_sequences': 48,
             'repeat_R2_state': compare(old, reset, {'raw__none': 'mobileie', 'ema__none': 'gate_ema',
                 'reset__none': 'gate_reset', 'raw__flow__keep': 'flow_ema', 'raw__fb__keep': 'fb_ema',
                 'reset__fb__reset': 'gate_reset_fb'}),
             'product_raw_vs_reset_raw': compare(reset, product, {'raw': 'raw__none'})}
    save_json(out/'audit.json', audit)
    fscene = grouped(feature, ['scene', 'split', 'layer', 'shared_read', 'jpeg'],
                     ['self_mse', 'cross_mse', 'mean_square_mse', 'feature_bias_mse', 'feature_variance'])
    for row in fscene:
        row['cross_over_self'] = row['cross_mse']/row['self_mse']
        row['mean_square_over_self'] = row['mean_square_mse']/row['self_mse']
        row['cross_over_mean_square'] = row['cross_mse']/row['mean_square_mse']
    fsummary = grouped(fscene, ['split', 'layer', 'shared_read', 'jpeg'],
                       ['cross_over_self', 'mean_square_over_self', 'cross_over_mean_square'])
    save_json(out/'feature_scene_summary.json', fscene)
    save_json(out/'feature_summary.json', fsummary)
    rsummary = grouped(reset, ['case', 'method'], METRICS)
    psummary = grouped(product, ['case', 'method'], METRICS+['coverage'])
    save_json(out/'reset_summary.json', rsummary)
    save_json(out/'product_summary.json', psummary)
    for row in trace:
        t = row['frame']
        row['window'] = 'before' if t < 12 else 'event' if t == 12 else 'recovery' if t < 16 else 'late'
    windows = grouped(trace, ['case', 'method', 'window'], METRICS)
    save_json(out/'reset_windows.json', windows)
    text = ['# 第三轮探索统计', '', '同一48输入条件的探索性评价；非真实低光、非独立盲测。', '',
            '## 冻结网络干预', '', '| method | PSNR | SSIM | temporal error | edge MAE |', '|---|---:|---:|---:|---:|']
    pglobal = grouped(product, ['method'], METRICS)
    for row in pglobal:
        text.append(f"| {row['method']} | {row['psnr_rgb']:.4f} | {row['ssim_rgb']:.4f} | {row['temporal_error']:.4f} | {row['edge_mae']:.4f} |")
    text += ['', '## 重置控制', '', '| method | PSNR | SSIM | temporal error |', '|---|---:|---:|---:|']
    for row in grouped(reset, ['method'], METRICS):
        text.append(f"| {row['method']} | {row['psnr_rgb']:.4f} | {row['ssim_rgb']:.4f} | {row['temporal_error']:.4f} |")
    (out/'tables.md').write_text('\n'.join(text)+'\n')
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, layer in zip(axes, ['head', 'body']):
        subset = [r for r in fscene if r['split'] == 'exploratory_validation' and r['layer'] == layer]
        for jpeg, marker in [(False, 'o'), (True, '^')]:
            s = [r for r in subset if r['jpeg'] == jpeg]
            ax.scatter([r['mean_square_over_self'] for r in s], [r['cross_over_self'] for r in s],
                       marker=marker, label='JPEG70' if jpeg else 'No JPEG', alpha=.7)
        ax.plot([0, 1], [0, 1], '--', color='grey', linewidth=1)
        ax.set(xlabel='Mean then square / self MSE', ylabel='Cross product / self MSE', title=layer,
               xlim=(0, 1), ylim=(0, 1))
        ax.legend()
        ax.grid(alpha=.2)
    fig.suptitle('Actual FST features: each point = one scene / read-noise condition')
    fig.tight_layout()
    fig.savefig(out/'feature_products.png', dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    selected = ['raw__fb__keep', 'raw__fb__reset', 'ema__fb__keep', 'ema__fb__reset',
                'reset__fb__keep', 'reset__fb__reset']
    for ax, case in zip(axes, ['light_step', 'cut']):
        for method in selected:
            data = [r for r in trace if r['case'] == case and r['method'] == method]
            curve = grouped(data, ['frame'], ['psnr_rgb'])
            ax.plot([r['frame'] for r in curve], [r['psnr_rgb'] for r in curve], label=method)
        ax.axvline(12, linestyle='--', color='grey')
        ax.set(title=case, xlabel='Frame', ylabel='PSNR (dB)')
        ax.grid(alpha=.2)
    axes[-1].legend(fontsize=8, loc='lower right')
    fig.tight_layout()
    fig.savefig(out/'separate_reset_response.png', dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 5))
    for index, row in enumerate(pglobal):
        ax.scatter(row['temporal_error'], row['psnr_rgb'], marker='o' if row['method'].startswith('fb') else '^')
        ax.annotate(str(index+1), (row['temporal_error'], row['psnr_rgb']), xytext=(3, 3), textcoords='offset points', fontsize=8)
    ax.set(xlabel='GT-relative temporal error (lower better)', ylabel='PSNR (dB)', title='Frozen FST intervention: local moments vs final RGB')
    ax.grid(alpha=.2)
    fig.subplots_adjust(right=.59)
    fig.text(.61, .85, '\n'.join(f"{i+1}. {r['method']}" for i, r in enumerate(pglobal)), fontsize=8, va='top')
    fig.savefig(out/'product_rgb_tradeoff.png', dpi=160)
    plt.close(fig)
    print(json.dumps(audit, indent=2))


if __name__ == '__main__':
    main()
