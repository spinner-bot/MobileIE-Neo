"""Export complete classic-control means and per-condition quality tradeoffs."""
from collections import defaultdict
import json
from pathlib import Path
from statistics import mean

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from dynamic_training_r34 import BASE, ROOT, digest


def main():
    folder = BASE/'report/exploration-r36-classical'
    audit = json.loads((folder/'audit.json').read_text())
    assert audit['passed'] and audit['rows'] == 48
    source = BASE/'results/exploration-r36-classical/per_sequence.jsonl'
    rows = list(map(json.loads, source.read_text().splitlines()))
    lookup = {(r['method'], r['scene'], r['train_seed']): r for r in rows}
    domain_path = folder/'domain_summary.json'
    domain = {r['method']: r for r in json.loads(domain_path.read_text())}
    names = {'raw': 'Original raw', 'finite_self05': 'Finite post .5',
             'recursive_self05': 'Recursive EMA .5', 'adaptive_recursive': 'Adaptive heuristic'}
    colors = {'raw': '#475569', 'finite_self05': '#2563eb',
              'recursive_self05': '#059669', 'adaptive_recursive': '#b45309'}
    scenes = ['bear_photons20', 'bear_photons80', 'lucia_photons20',
              'lucia_photons80', 'walking_photons20', 'walking_photons80']
    condition_rows = []
    for method in names:
        for scene in scenes:
            values = [lookup[(method, scene, seed)] for seed in [0, 1]]
            deltas = [r['psnr_rgb']-lookup[('raw', scene, r['train_seed'])]['psnr_rgb'] for r in values]
            condition_rows.append({'method': method, 'scene': scene,
                 'mean_delta_psnr_vs_raw': mean(deltas), 'min_seed_delta': min(deltas),
                 'max_seed_delta': max(deltas),
                 **{m: mean(r[m] for r in values) for m in ['psnr_rgb', 'ssim_rgb', 'temporal_error', 'edge_mae']}})
    for name in ['quality_condition.png', 'quality_condition.pdf', 'condition_summary.json', 'figure_sources.json']:
        assert not (folder/name).exists()
    fig, (left, right) = plt.subplots(1, 2, figsize=(13, 5.2), gridspec_kw={'width_ratios': [1, 1.35]})
    for method, label in names.items():
        row = domain[method]
        left.scatter(row['temporal_error'], row['psnr_rgb'], color=colors[method], s=95, label=label)
        if method == 'raw':
            continue
        values = [r for r in condition_rows if r['method'] == method]
        y = [r['mean_delta_psnr_vs_raw'] for r in values]
        right.errorbar(range(6), y, yerr=[[r['mean_delta_psnr_vs_raw']-r['min_seed_delta'] for r in values],
                                        [r['max_seed_delta']-r['mean_delta_psnr_vs_raw'] for r in values]],
                       color=colors[method], marker='o', capsize=3, label=label)
    left.set(xlabel='GT-residual temporal error (lower is better)', ylabel='RGB PSNR (dB, higher is better)',
             title='Complete DEV means')
    right.axhline(0, color=colors['raw'], linestyle='--', linewidth=1)
    right.set_xticks(range(6), ['bear 20', 'bear 80', 'lucia 20', 'lucia 80', 'walking 20', 'walking 80'], rotation=20)
    right.set(ylabel='PSNR change vs own raw (dB)', title='Per-condition gain and degradation')
    for ax in [left, right]:
        ax.grid(alpha=.2)
    handles, labels = left.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4, frameon=False, bbox_to_anchor=(.5, .01))
    fig.suptitle('Fixed classic controls; three DEV videos, two noise conditions, both training seeds')
    fig.text(.5, .11, 'Condition bars: min–max over two seed deltas, not confidence intervals. No phone or independent-test claim.',
             ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .15, 1, .94))
    fig.savefig(folder/'quality_condition.png', dpi=160)
    fig.savefig(folder/'quality_condition.pdf')
    plt.close(fig)
    (folder/'condition_summary.json').write_text(json.dumps(condition_rows, indent=2)+'\n')
    (folder/'figure_sources.json').write_text(json.dumps({
        'source_sha256': {str(p.relative_to(ROOT)): digest(p) for p in [source, domain_path, Path(__file__)]},
        'scope': 'All conditions retained; ranges are two seed deltas, not confidence intervals or mobile measurements.'}, indent=2)+'\n')
    print(json.dumps({'conditions': 6, 'rows': len(rows), 'out': str(folder.relative_to(ROOT))}))


if __name__ == '__main__':
    main()
