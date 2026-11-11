"""Standalone scientific figures for the complete exploratory mechanism controls."""
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
    deltas = json.loads((report/'paired_deltas.json').read_text())
    variants = ['temporal_bilinear', 'temporal_post', 'tsm_learned', 'tsm_fixed', 'temporal_bilinear_st', 'tsm_learned_st']
    labels = ['Bilinear', 'Post', 'TSM learned', 'TSM fixed', 'Bilinear + ST', 'TSM + ST']
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
    for col, domain in enumerate(['test', 'legacy_synthetic']):
        for row, metric in enumerate(['psnr_rgb', 'temporal_error']):
            ax = axes[row, col]
            for i, variant in enumerate(variants):
                target = [r for r in deltas if r['domain'] == domain and r['first_minus_second'] ==
                          [variant+'__causal__raw', variant+'__current__raw']][0]
                values = np.array([r[metric] for r in target['by_input_condition'].values()])
                ax.barh(i, values.mean(), color='#527caa' if row == 0 else '#ca8650', height=.65, alpha=.8)
                jitter = np.linspace(-.15, .15, len(values))
                ax.scatter(values, i+jitter, s=10, color='#222222', alpha=.55, zorder=3)
                ax.annotate(f'{values.mean():+.4f}', (values.mean(), i), xytext=(4 if values.mean() >= 0 else -4, 0),
                            textcoords='offset points', va='center', ha='left' if values.mean() >= 0 else 'right', fontsize=8)
            ax.axvline(0, color='#555555', lw=.8)
            ax.set_yticks(range(len(labels)), labels)
            ax.invert_yaxis()
            ax.set_xlabel('Causal - current-history PSNR (dB); higher is better' if row == 0 else
                          'Causal - current-history temporal error; lower is better')
            ax.set_title(('Real ND: 6 clips' if col == 0 else 'Legacy synthetic: 8 conditions / 2 source clips') if row == 0 else '')
            ax.grid(axis='x', alpha=.2)
            ax.margins(x=.3)
    fig.suptitle('Same weights, replace only the history: much of the coarse-branch gain is spatial', fontsize=13)
    fig.savefig(report/'history_mechanism.png', dpi=180)
    fig.savefig(report/'history_mechanism.pdf')
    plt.close(fig)
    frames = []
    base = ROOT/'research/4.temporal-consistency/results'
    for name in ['exploration-r5-eval-spatial', 'exploration-r5-eval-temporal', 'exploration-r5-eval-shift', 'exploration-r5c-eval-spatial-post']:
        frames.extend(json.loads(line) for line in (base/name/'per_frame.jsonl').read_text().splitlines())
    chosen = [('plain', 'Original architecture'), ('spatial_post', 'Spatial post'),
              ('temporal_post', 'Temporal post'), ('tsm_learned', 'TSM learned'), ('tsm_fixed', 'TSM fixed')]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), constrained_layout=True)
    for ax, domain in zip(axes, ['event_light_step', 'event_cut']):
        for variant, label in chosen:
            values = [r for r in frames if r['domain'] == domain and r['variant'] == variant and r['history'] == 'causal' and r['post'] == 'raw']
            curve = [np.mean([r['psnr_rgb'] for r in values if r['frame'] == t]) for t in range(24)]
            ax.plot(range(24), curve, label=label, lw=1.5, linestyle='--' if variant == 'temporal_post' else '-')
        ax.axvline(11.5, color='gray', linestyle=':', lw=1)
        ax.set_title('Target illumination step' if domain == 'event_light_step' else 'Scene cut')
        ax.set_xlabel('Frame; event begins at 12')
        ax.set_ylabel('Mean frame PSNR (dB)')
        ax.grid(alpha=.2)
    axes[1].legend(fontsize=8)
    fig.suptitle('Event response: 6 source images x 2 input seeds x 2 training seeds; exploratory')
    fig.savefig(report/'event_response.png', dpi=180)
    fig.savefig(report/'event_response.pdf')
    plt.close(fig)


if __name__ == '__main__':
    main()
