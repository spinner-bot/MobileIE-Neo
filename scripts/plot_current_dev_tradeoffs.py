"""Display completed repeated-DEV evidence; no new images, metrics or model calls."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'research/4.temporal-consistency'


def main():
    r82 = BASE/'report/exploration-r82-joint/domain_seed_means.json'
    r86 = BASE/'report/exploration-r86-joint/domain_seed_means.json'
    assert json.loads((r86.parent/'audit.json').read_text())['completed']
    a, b = json.loads(r82.read_text()), json.loads(r86.read_text())
    rows = [v for v in a if v['label'] in ('MobileIE_raw', 'MobileIE_recursive_self05', 'centered')]
    rows += [v for v in b if v['label'] == 'DIS_recursive_self05']
    assert len(rows) == 16
    styles = [('MobileIE_raw', 'F0 raw', '#696969', 'o'),
        ('MobileIE_recursive_self05', 'F0 + half-FB rec.', '#2375b7', 's'),
        ('DIS_recursive_self05', 'F0 + native DIS rec.', '#d57a15', '^'),
        ('centered', 'R82 centered', '#8c4fa4', 'D')]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.6))
    for col, (domain, title) in enumerate([('davis_train_pilot', 'SYN: six conditions / three sources'),
                                          ('paired_real_ND_DEV', 'ND: four paired-real DEV sources')]):
        for row, (metric, ylabel) in enumerate([('psnr_rgb', 'PSNR (dB), higher better'),
                                               ('JOD', 'ColorVideoVDP JOD, higher better')]):
            ax = axes[row, col]
            for label, _, color, marker in styles:
                for seed in (0, 1):
                    selected = [v for v in rows if (v['domain'], v['seed'], v['label']) == (domain, seed, label)]
                    assert len(selected) == 1
                    v = selected[0]
                    ax.scatter(v['temporal_error'], v[metric], s=73, marker=marker,
                        facecolors=color if seed == 0 else 'none', edgecolors=color, linewidths=1.4)
            ax.set_title(title if row == 0 else 'Same domain means, perceptual view', fontsize=11)
            ax.set_xlabel('Temporal error, lower better'); ax.set_ylabel(ylabel)
            ax.grid(alpha=.22); ax.margins(x=.12, y=.15)
    legend = [Line2D([], [], color=c, marker=m, linestyle='None', label=t) for _, t, c, m in styles]
    legend += [Line2D([], [], color='black', marker='o', linestyle='None', label='seed0 filled'),
               Line2D([], [], color='black', marker='o', markerfacecolor='none', linestyle='None', label='seed1 open')]
    fig.suptitle('Current quality / temporal trade-offs on repeated development data', fontsize=14)
    fig.legend(handles=legend, loc='lower center', ncol=3, bbox_to_anchor=(.5, .037), fontsize=9)
    fig.text(.5, .012, 'Each marker is a domain mean, not an independent benchmark or confidence interval. '
        'Per-source losses and registered failures remain in the full tables.', ha='center', fontsize=8)
    fig.tight_layout(rect=(0, .14, 1, .95))
    out = r86.parent
    for extension in ('png', 'pdf'):
        path = out/f'current-dev-tradeoffs.{extension}'
        if path.exists():
            raise FileExistsError(path)
        fig.savefig(path, dpi=180)
    plt.close(fig)
    sources = [r82, r86, Path(__file__).resolve()]
    (out/'tradeoff-figure-sources.json').write_text(json.dumps(dict(
        sources={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        plotted_means=16, seeds=[0, 1], new_CNN_calls=0, new_quality_metrics=0,
        choice='Post-result descriptive illustration of current F0 baselines and latest learned R82 reference; not new selection or Pareto-optimality claim.',
        actual_assistant_viewed=False, compiled_into_manuscript=False), indent=2)+'\n')


if __name__ == '__main__':
    main()
