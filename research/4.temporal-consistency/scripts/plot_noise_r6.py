"""Standalone mechanism plots; covariance fractions are correlated, not causal gains."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from exploration_r2_data import ROOT


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--report', required=True)
    args = p.parse_args()
    report = Path(args.report)
    feature = json.loads((report/'feature_per_condition.json').read_text())
    component = json.loads((report/'component_summary.json').read_text())
    paired = json.loads((report/'paired_conditions.json').read_text())
    local_path = ROOT/'research/4.temporal-consistency/results/exploration-r6b-centered/layer_per_condition.jsonl'
    local = [json.loads(s) for s in local_path.read_text().splitlines()]
    models = [('official', 'Official', '#497fa8'), ('r5_plain', 'Task-trained', '#c78146'),
              ('r5_spatial_post', 'Spatial post', '#628d64')]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.2), constrained_layout=True)
    for i, (model, label, color) in enumerate(models):
        values = [r['quadratic_ratio'] for r in feature if r['model'] == model and r['layer'] == 'body' and r['draw_subset'] == 'all64']
        axes[0, 0].bar(i, np.mean(values), width=.6, color=color, alpha=.75)
        axes[0, 0].scatter(i+np.linspace(-.12, .12, len(values)), values, color='#333333', s=9, alpha=.5)
        r = [r for r in component if r['model'] == model and r['clipped']][0]
        keys = ['feature_linear_mean_gate', 'feature_quadratic_mean_gate', 'gate_mean_feature', 'gate_feature_interaction', 'clipping_residual']
        vals = [r['marginal_covariance_fraction'][key]['mean']*100 for key in keys]
        axes[0, 1].bar(np.arange(5)+(i-1)*.24, vals, width=.23, color=color, label=label)
        pairs = [r for r in paired if r['model'] == model and r['intervention'] == 'centered_body' and r['clipped']]
        for row in pairs:
            loc = [v for v in local if all(v[k] == row[k] for k in ['scene', 'photons', 'jpeg', 'model']) and v['layer'] == 'body'][0]
            axes[1, 0].scatter(loc['local_variance_ratio'], 1+row['variance_relative_delta'], color=color, alpha=.6, s=22)
        pairs = [r for r in paired if r['model'] == model and r['intervention'] == 'centered_head' and r['clipped']]
        axes[1, 1].scatter([r['psnr_delta'] for r in pairs], [100*r['noise_sd_relative_delta'] for r in pairs], color=color, label=label, alpha=.65, s=22)
    axes[0, 0].set(xticks=range(3), xticklabels=[m[1] for m in models],
                   ylabel='Q / Var(u squared), per condition', title='Large local body quadratic-noise energy')
    axes[0, 1].set(xticks=range(5), xticklabels=['Linear', 'Quadratic', 'Gate', 'Interaction', 'Clipping'],
                   ylabel='Mean Cov(component, RGB) / Var(RGB), %', title='Actual RGB: correlated decomposition')
    axes[0, 1].legend(fontsize=8)
    axes[0, 1].axhline(0, color='gray', lw=.7)
    axes[1, 0].set(xlabel='Local body variance ratio after oracle', ylabel='Clipped RGB variance ratio',
                   title='Removing local variance does not ensure RGB gain')
    axes[1, 0].axhline(1, color='gray', lw=.8, linestyle='--')
    axes[1, 1].set(xlabel='Mean-preserving head oracle: PSNR delta (dB)', ylabel='Noise SD change, % (lower is better)',
                   title='Quality gains and noise reduction are distinct')
    axes[1, 1].axhline(0, color='gray', lw=.8)
    axes[1, 1].axvline(0, color='gray', lw=.8)
    for ax in axes.flat:
        ax.grid(alpha=.15)
    fig.suptitle('Static repeated-noise diagnosis: 6 images x 4 noise settings; oracle controls, no video claim', fontsize=13)
    fig.savefig(report/'noise_mechanism.png', dpi=180)
    fig.savefig(report/'noise_mechanism.pdf')
    plt.close(fig)


if __name__ == '__main__':
    main()
