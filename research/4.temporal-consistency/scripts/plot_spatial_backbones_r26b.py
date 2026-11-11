"""Fixed end-point contrasts against each candidate's own spatial-plus-post control."""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from exploration_r2_data import ROOT

BASE = ROOT/'research/4.temporal-consistency/report'


def main():
    original = json.loads((BASE/'exploration-r26/domain_summary.json').read_text())
    spatial = json.loads((BASE/'exploration-r26b/domain_summary.json').read_text())
    domains = ['dev_native', 'dev', 'test', 'legacy_synthetic']
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.8))
    labels = ['pre/Charb', 'pre/MSE', 'post/Charb', 'post/MSE']
    for ax, metric, title in zip(axes, ['psnr_rgb', 'ssim_rgb', 'temporal_error', 'edge_mae'],
                                 ['PSNR delta (higher better)', 'SSIM delta (higher better)', 'Temporal delta (lower better)', 'Edge delta (lower better)']):
        for i, domain in enumerate(domains):
            values = []
            for architecture in ['pre_fixedhalf', 'post_fixedhalf']:
                for objective in ['charb', 'mse']:
                    def match(r):
                        return r['domain'] == domain and r['architecture'] == architecture and r['objective'] == objective and r['step'] == 4000 and not r['input_reset']
                    a = next(r for r in original if match(r) and r['history'] == 'causal' and r['fusion'] == 'raw')
                    b = next(r for r in spatial if match(r) and r['fusion'] == 'postwarp05')
                    values.append(a[metric]-b[metric])
            ax.bar(np.arange(4)+(i-1.5)*.18, values, width=.18, label=domain)
        ax.axhline(0, color='black', linewidth=.7)
        ax.set_xticks(range(4), labels, rotation=30, ha='right')
        ax.set_title(title, fontsize=10)
        ax.grid(axis='y', alpha=.2)
    axes[0].legend(fontsize=8)
    fig.suptitle('Actual history minus SAME-WEIGHT single-frame backbone + finite post .5\n4000 steps, two seeds; exploratory domain means, reset off', fontsize=12)
    fig.tight_layout()
    for suffix in ['png', 'pdf']:
        fig.savefig(BASE/'exploration-r26b'/('same_weight_post_contrasts.'+suffix), dpi=170)
    plt.close(fig)


if __name__ == '__main__':
    main()
