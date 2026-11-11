"""Separate foldable optimization gains from actual two-frame contributions."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--report', required=True)
    folder = Path(p.parse_args().report)
    rows = json.loads((folder/'domain_summary.json').read_text())
    def get(domain, variant, history='causal'):
        return next(r for r in rows if r['domain']==domain and r['variant']==variant
                    and r['history']==history and not r['input_reset'] and r['fusion']=='raw')
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    names = ['spatial_copy', 'spatial_detail', 'temporal_concat', 'temporal_delta', 'temporal_delta_aug']
    labels = ['Copy\n(foldable)', 'Detail', 'Past input', 'Past-current', 'Difference\n+ augmentation']
    for col, (domain, title) in enumerate([('test', '6 real ND clips'), ('legacy_synthetic', '8 conditions / 2 synthetic clips')]):
        base = get(domain, 'plain')
        ax = axes[0, col]
        ax.bar(range(5), [get(domain, v)['psnr_rgb']-base['psnr_rgb'] for v in names], color=['#c18b51']*2+['#518aab']*3)
        ax.set(xticks=range(5), xticklabels=labels, ylabel='PSNR change from plain, dB', title=title)
        ax = axes[1, col]
        for v, label, color in zip(names[2:], labels[2:], ['#518aab', '#648c62', '#b75b59']):
            a, b = get(domain, v), get(domain, v, 'current')
            ax.scatter(a['temporal_error']-b['temporal_error'], a['psnr_rgb']-b['psnr_rgb'], s=60, label=label.replace('\n',' '), color=color)
        ax.set(xlabel='Temporal error: causal minus self-history', ylabel='PSNR: causal minus self-history, dB', title='Same weights; only history source changes')
        ax.legend(fontsize=8)
    for ax in axes.flat:
        ax.axhline(0, color='black', lw=.7)
        ax.grid(alpha=.15)
    for ax in axes[1]:ax.axvline(0, color='black', lw=.7)
    fig.suptitle('Two-seed means, fixed 1000 updates; raw outputs, no reset\nFoldable copy controls fast-head optimization; no independent final-test claim', fontsize=11)
    fig.savefig(folder/'optimization_history.png', dpi=180)
    fig.savefig(folder/'optimization_history.pdf')
    plt.close(fig)


if __name__=='__main__':main()
