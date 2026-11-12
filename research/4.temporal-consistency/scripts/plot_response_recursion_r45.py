"""Plot complete fixed-alpha frontiers and learned controls, with both seeds visible."""
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from dynamic_training_r34 import BASE, digest


def main():
    folder = BASE/'report/exploration-r45-response-recursion'
    audit = json.loads((folder/'audit.json').read_text())
    assert audit['passed'] and audit['rows'] == 180 and audit['new_neural_sequences'] == 20
    source, seed_source = folder/'domain_summary.json', folder/'domain_seed_summary.json'
    lookup = {(r['domain'], r['method']): r for r in json.loads(source.read_text())}
    seeds = {(r['domain'], r['method'], r['seed']): r for r in json.loads(seed_source.read_text())}
    fixed = [('fixed_01', '.1'), ('fixed_025', '.25'), ('fixed_05', '.5'), ('fixed_075', '.75'), ('fixed_09', '.9')]
    domains = [('davis_train_pilot', 'Synthetic DEV: 3 sources / 6 conditions'),
               ('paired_real_ND_DEV', 'Paired ND DEV: 4 sources')]
    styles = [('raw', 'Raw spatial base', '#111827', '*'), ('scalar', 'Learned global alpha', '#2563eb', 'o'),
              ('rgb', 'RGB gate (41 params)', '#059669', 's'), ('response', 'Internal feature cue (41 params)', '#9333ea', 'D')]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.7))
    for ax, (domain, title) in zip(axes, domains):
        curve = [lookup[(domain, method)] for method, _ in fixed]
        ax.plot([r['temporal_error'] for r in curve], [r['psnr_rgb'] for r in curve], 'x--',
                color='#6b7280', label='Fixed recursive alpha frontier', linewidth=1)
        for r, (_, alpha) in zip(curve, fixed):
            ax.annotate(alpha, (r['temporal_error'], r['psnr_rgb']), xytext=(4, 4),
                        textcoords='offset points', fontsize=8)
        for method, label, color, marker in styles:
            values = [seeds[(domain, method, seed)] for seed in [0, 1]]
            ax.scatter([r['temporal_error'] for r in values], [r['psnr_rgb'] for r in values],
                       color=color, marker=marker, alpha=.3, s=35)
            r = lookup[(domain, method)]
            ax.scatter(r['temporal_error'], r['psnr_rgb'], color=color, marker=marker, s=80, label=label)
        ax.set(title=title, xlabel='Reference-flow temporal error (lower is better)', ylabel='RGB PSNR (dB)')
        ax.grid(alpha=.2)
    fig.suptitle('R45 complete fixed comparison: identical frozen spatial base; all five alpha points retained\n'
                 'Pale: individual seeds. Solid: mean. Stateful outputs require motion and RGB history.', fontsize=11)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, fontsize=8)
    fig.tight_layout(rect=(0, .13, 1, 1))
    for extension in ['png', 'pdf']:
        fig.savefig(folder/('frontier.'+extension), dpi=160, bbox_inches='tight')
    plt.close(fig)
    alpha_source = folder/'alpha_domain_seed_summary.json'
    alpha = {(r['domain'], r['seed'], r['method']): r for r in json.loads(alpha_source.read_text())}
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for row, (domain, title) in enumerate(domains):
        for seed in [0, 1]:
            ax = axes[row, seed]
            for method, label, color, _ in styles[1:]:
                counts = np.asarray(alpha[(domain, seed, method)]['histogram_20_bins_0_to_1'], dtype=float)
                ax.step((np.arange(20)+.5)/20, counts/max(counts.sum(), 1), where='mid',
                        label=label, color=color)
            ax.set(title=title+f' / seed {seed}', xlim=(0, 1), xlabel='History weight alpha',
                   ylabel='Accepted-pixel fraction')
            ax.grid(alpha=.2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, fontsize=8)
    fig.suptitle('R45 actual alpha distributions: 20 bins, pixel-weighted; not uncertainty calibration', fontsize=11)
    fig.tight_layout(rect=(0, .045, 1, .97))
    for extension in ['png', 'pdf']:
        fig.savefig(folder/('alpha_distribution.'+extension), dpi=160, bbox_inches='tight')
    plt.close(fig)
    (folder/'plot_sources.json').write_text(json.dumps({
        'source_sha256': {p.name: digest(p) for p in [source, seed_source, alpha_source, folder/'audit.json']},
        'scope': 'All fixed alpha points and both seeds retained. DEV only, not CI, calibrated uncertainty, phone cost or final test.'
    }, indent=2)+'\n')


if __name__ == '__main__':
    main()
