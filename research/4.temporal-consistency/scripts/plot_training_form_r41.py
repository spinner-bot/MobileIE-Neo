"""Plot complete R41 matched-form quality and stability, with both seeds visible."""
import json
from statistics import mean

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from dynamic_training_r34 import BASE, digest


def main():
    folder = BASE/'report/exploration-r41-training-form'
    audit = json.loads((folder/'audit.json').read_text())
    assert audit['passed'] and audit['new_rows'] == 120
    path = folder/'domain_seed_summary.json'
    values = json.loads(path.read_text())
    lookup = {(r['domain'], r['anchor_family'], r['method'], r['seed']): r for r in values}
    domains = [('davis_train_pilot', 'Synthetic DEV: 3 sources / 6 conditions'),
               ('paired_real_ND_DEV', 'Paired real ND DEV: 4 sources')]
    forms = [('R41_slim_fp32', 'Matched Slim FP32', '#2563eb'),
             ('R41_expanded_fp32', 'Expanded FP32, folded', '#059669')]
    methods = [('raw', 'raw', 'o'), ('finite_self05', 'finite .5', 's'),
               ('recursive_self05', 'recursive .5', '^')]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    coordinates = []
    for ax, (domain, title) in zip(axes, domains):
        for family, label, color in forms:
            for method, short, marker in methods:
                rows = [lookup[(domain, family, method, seed)] for seed in [0, 1]]
                x = [r['temporal_error'] for r in rows]
                y = [r['psnr_rgb'] for r in rows]
                ax.scatter(x, y, color=color, marker=marker, alpha=.35, s=35)
                ax.scatter(mean(x), mean(y), color=color, marker=marker, s=85,
                           label=label+' / '+short)
                coordinates.append({'domain': domain, 'anchor_family': family, 'method': method,
                                    'temporal_seeds': x, 'psnr_seeds': y})
        ax.set(title=title, xlabel='Reference-flow temporal error (lower is better)',
               ylabel='RGB PSNR (dB, higher is better)')
        ax.grid(alpha=.2)
        ax.legend(fontsize=7, loc='best')
    fig.suptitle('R41 matched FP32 forms: same data / updates, 4075-parameter deployment\n'
                 'Pale points: individual seeds; solid points: mean. Post variants retain history and flow.',
                 fontsize=11)
    fig.tight_layout()
    for extension in ['png', 'pdf']:
        fig.savefig(folder/('matched_forms.'+extension), dpi=160, bbox_inches='tight')
    plt.close(fig)
    (folder/'plot_sources.json').write_text(json.dumps({
        'summary_sha256': digest(path), 'audit_sha256': digest(folder/'audit.json'),
        'coordinates': coordinates,
        'scope': 'Complete DEV only; reference-flow error is not a perceptual flicker user study. '
                 'Two seeds are displayed, not a confidence interval. Equal update budgets do not imply equal training FLOPs.'
    }, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
