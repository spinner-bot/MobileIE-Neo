"""Frozen operator quality/stability relative to the same weights with finite post .5."""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from exploration_r2_data import ROOT

BASE = ROOT/'research/4.temporal-consistency/report/exploration-r27-frozen'


def main():
    rows = json.loads((BASE/'domain_summary.json').read_text())
    modes = ['post_transport', 'post_after_warp', 'pre_square', 'cross']
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.5))
    for ax, metric, title in zip(axes, ['psnr_rgb', 'ssim_rgb', 'temporal_error', 'edge_mae'],
                                 ['PSNR (higher better)', 'SSIM (higher better)', 'Temporal (lower better)', 'Edge (lower better)']):
        for i, objective in enumerate(['official', 'charb', 'mse']):
            delta = []
            for mode in modes:
                a = np.mean([r[metric] for r in rows if r['objective'] == objective and r['mode'] == mode and r['history'] == 'causal'])
                b = np.mean([r[metric] for r in rows if r['objective'] == objective and r['mode'] == 'copy' and r['history'] == 'causal' and r['fusion'] == 'postwarp05'])
                delta.append(a-b)
            ax.bar(np.arange(4)+(i-1)*.24, delta, width=.24, label=objective)
        ax.axhline(0, color='black', linewidth=.7)
        ax.set_xticks(range(4), ['post / W', 'W / post', 'pre square', 'cross'], rotation=30, ha='right')
        ax.set_title(title, fontsize=10)
        ax.grid(axis='y', alpha=.2)
    axes[0].legend()
    fig.suptitle('FROZEN operator minus SAME-WEIGHT finite output post .5\nFour native DEV clips; Charb/MSE average two seeds; not trained-method evidence', fontsize=12)
    fig.tight_layout()
    for suffix in ['png', 'pdf']:
        fig.savefig(BASE/('frozen_operator_contrasts.'+suffix), dpi=170)
    plt.close(fig)


if __name__ == '__main__':
    main()
