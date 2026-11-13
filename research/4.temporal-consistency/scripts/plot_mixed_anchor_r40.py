"""Plot the complete cross-domain anchor comparison, retaining both seeds."""
import json
from pathlib import Path
from statistics import mean

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from dynamic_training_r34 import BASE, ROOT, digest


def main():
    folder = BASE/'report/exploration-r40-mixed-anchor'
    audit = json.loads((folder/'audit.json').read_text())
    assert audit['passed'] and audit['new_rows'] == 60 and audit['total_rows'] == 156
    summary_path = folder/'domain_seed_summary.json'
    old_path = BASE/'report/external-baselines/domain_seed_summary.json'
    values = json.loads(summary_path.read_text())
    old = json.loads(old_path.read_text())
    lookup = {(r['domain'], r['anchor_family'], r['method'], r['seed']): r['psnr_rgb'] for r in values}
    synthetic, real = 'davis_train_pilot', 'paired_real_ND_DEV'
    labels = {'R26_plain_MSE': 'R26 earlier real/synthetic', 'R34_original': 'R34 dynamic synthetic',
              'R40_mixed': 'R40 fixed 50/50'}
    colors = {'R26_plain_MSE': '#475569', 'R34_original': '#2563eb', 'R40_mixed': '#059669'}
    for seed in [0, 1]:
        lookup[(synthetic, 'R26_plain_MSE', 'raw', seed)] = next(r['psnr_rgb'] for r in old if r['method'] == 'raw' and r['seed'] == seed)
    fig, (left, right) = plt.subplots(1, 2, figsize=(12.5, 5.3))
    coordinates = []
    for family, label in labels.items():
        x = [lookup[(synthetic, family, 'raw', seed)] for seed in [0, 1]]
        y = [lookup[(real, family, 'raw', seed)] for seed in [0, 1]]
        left.scatter(x, y, color=colors[family], alpha=.35, s=35)
        left.scatter(mean(x), mean(y), color=colors[family], s=105, label=label)
        coordinates.append({'anchor_family': family, 'seed_coordinates': list(zip(x, y)),
                            'mean_synthetic_raw_psnr': mean(x), 'mean_real_raw_psnr': mean(y)})
    left.set(xlabel='Synthetic DAVIS DEV raw PSNR (dB)', ylabel='Paired real ND DEV raw PSNR (dB)',
             title='Spatial quality across domains')
    comparisons = [(domain, method) for domain in [synthetic, real]
                   for method in ['raw', 'finite_self05', 'recursive_self05']]
    for seed, marker in [(0, 'o'), (1, 's')]:
        deltas = [lookup[(domain, 'R40_mixed', method, seed)]
                  -lookup[(domain, 'R34_original' if domain == synthetic else 'R26_plain_MSE', method, seed)]
                  for domain, method in comparisons]
        right.plot(range(6), deltas, marker=marker, label=f'seed {seed}')
    right.axhline(0, color='#475569', linestyle='--', linewidth=1)
    right.set_xticks(range(6), ['SYN raw', 'SYN finite', 'SYN rec.', 'ND raw', 'ND finite', 'ND rec.'], rotation=23)
    right.set(ylabel='R40 PSNR change vs domain reference (dB)', title='All routes, both seeds')
    right.legend(frameon=False)
    for ax in [left, right]:
        ax.grid(alpha=.2)
    handles, names = left.get_legend_handles_labels()
    fig.legend(handles, names, loc='lower center', bbox_to_anchor=(.5, .01), ncol=3, frameon=False)
    fig.suptitle('Original backbone, fixed training schemes; no novel temporal method claim')
    fig.text(.5, .11, 'Left: faint = seeds, bold = means. Right references: R34 for synthetic, R26 for paired ND. No phone/final-test claim.',
             ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .15, 1, .94))
    for name in ['cross_domain.png', 'cross_domain.pdf', 'figure_sources.json']:
        assert not (folder/name).exists()
    fig.savefig(folder/'cross_domain.png', dpi=160)
    fig.savefig(folder/'cross_domain.pdf')
    plt.close(fig)
    (folder/'figure_sources.json').write_text(json.dumps({
        'source_sha256': {str(p.relative_to(ROOT)): digest(p) for p in [summary_path, old_path, Path(__file__)]},
        'raw_coordinates': coordinates,
        'scope': 'Both seeds, all three post routes. Training/data/optimization differ for the earlier R26 scheme; no isolated causal attribution.'}, indent=2)+'\n')
    print(json.dumps({'out': str(folder.relative_to(ROOT)), 'schemes': 3, 'routes': 3}))


if __name__ == '__main__':
    main()
