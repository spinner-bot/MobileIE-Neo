"""Show all fixed R44 arms and seeds after complete evaluation only."""
import json
from statistics import mean

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from dynamic_training_r34 import BASE, digest


def main():
    folder = BASE/'report/exploration-r44-bounded-video'
    audit = json.loads((folder/'audit.json').read_text())
    assert audit['passed'] and audit['new_rows'] == 240 and audit['new_neural_sequences'] == 80
    source = folder/'domain_seed_summary.json'
    rows = json.loads(source.read_text())
    assert len(rows) == 48
    lookup = {(r['domain'], r['model'], r['method'], r['seed']): r for r in rows}
    domains = [('davis_train_pilot', 'Synthetic DEV: 3 sources / 6 conditions'),
               ('paired_real_ND_DEV', 'Paired ND DEV: 4 sources')]
    models = [('square_mse', 'Square / MSE', '#2563eb', 'o'),
              ('square_temporal', 'Square / temporal', '#059669', 's'),
              ('bounded_mse', 'Bounded / MSE', '#ea580c', '^'),
              ('bounded_temporal', 'Bounded / temporal', '#9333ea', 'D')]
    outputs = [('raw', 'Raw: single-frame inference'),
               ('recursive_self05', 'Recursive .5: motion + RGB history')]
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for row_index, (method, output_label) in enumerate(outputs):
        for column_index, (domain, domain_label) in enumerate(domains):
            ax = axes[row_index, column_index]
            for model, label, color, marker in models:
                values = [lookup[(domain, model, method, seed)] for seed in [0, 1]]
                x = [r['temporal_error'] for r in values]
                y = [r['psnr_rgb'] for r in values]
                ax.scatter(x, y, color=color, marker=marker, alpha=.35, s=40)
                ax.scatter(mean(x), mean(y), color=color, marker=marker, s=90, label=label)
            ax.set(title=domain_label+'\n'+output_label,
                   xlabel='Reference-flow temporal error (lower is better)',
                   ylabel='RGB PSNR (dB, higher is better)')
            ax.grid(alpha=.2)
    fig.suptitle('R44 complete fixed 2x2: inherited 4000 + new 4000 updates per model/seed\n'
                 'Pale: both individual seeds. Solid: mean. Rational inference retains its denominator.', fontsize=11)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4, fontsize=9)
    fig.tight_layout(rect=(0, .055, 1, .97))
    for extension in ['png', 'pdf']:
        fig.savefig(folder/('factorial.'+extension), dpi=160, bbox_inches='tight')
    plt.close(fig)
    (folder/'plot_sources.json').write_text(json.dumps({
        'summary_sha256': digest(source), 'audit_sha256': digest(folder/'audit.json'),
        'plotted_outputs': [v[0] for v in outputs],
        'scope': 'Complete development evidence only. Finite outputs remain in full tables. '
                 'Two seeds are not confidence intervals; no phone, final-test or novelty claim.'
    }, indent=2)+'\n')


if __name__ == '__main__':
    main()
