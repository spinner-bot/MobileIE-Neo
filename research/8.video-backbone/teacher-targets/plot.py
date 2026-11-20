"""All 48 group means, exported without additional inference or score selection."""
import csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from study_io import ROOT, BASE, read, write, stamp, digest, verify


def main():
    verify()
    source = BASE / 'report/joint/domain_seed_means.json'
    rows = read(source)
    assert len(rows) == 48
    styles = {
        'MobileIE_raw': ('F0 raw', '#333333', 'o'),
        'MobileIE_recursive_self05': ('F0 + FB', '#333333', '^'),
        'DIS_recursive_self05': ('F0 + DIS', '#333333', 's')}
    for arm, color in [('gt', '#0072B2'), ('spatial', '#E69F00'), ('temporal', '#009E73')]:
        for tail, marker in [('raw', 'o'), ('FB05', '^'), ('DIS05', 's')]:
            styles[f't1_{arm}_{tail}'] = (arm + ' ' + tail.replace('05', ''), color, marker)
    assert set(styles) == {r['label'] for r in rows}
    out = BASE / 'report/plots'
    out.mkdir(parents=True, exist_ok=False)
    fields = ('domain', 'seed', 'label', 'psnr_rgb', 'ssim_rgb', 'temporal_error', 'JOD')
    with (out / 'points.csv').open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator='\n')
        writer.writeheader()
        writer.writerows({k: r[k] for k in fields} for r in rows)
    images = []
    for metric, title in [('psnr_rgb', 'PSNR (dB)'), ('JOD', 'ColorVideoVDP JOD')]:
        fig, axes = plt.subplots(2, 2, figsize=(12.5, 7.8), constrained_layout=False)
        groups = [('davis_train_pilot', 0), ('davis_train_pilot', 1),
                  ('paired_real_ND_DEV', 0), ('paired_real_ND_DEV', 1)]
        for ax, (domain, seed) in zip(axes.flat, groups):
            chosen = {r['label']: r for r in rows if (r['domain'], r['seed']) == (domain, seed)}
            for labels in [('MobileIE_raw', 'MobileIE_recursive_self05', 'DIS_recursive_self05'),
                           *((f't1_{arm}_raw', f't1_{arm}_FB05', f't1_{arm}_DIS05') for arm in ('gt', 'spatial', 'temporal'))]:
                for post in labels[1:]:
                    ax.plot([chosen[labels[0]]['temporal_error'], chosen[post]['temporal_error']],
                            [chosen[labels[0]][metric], chosen[post][metric]],
                            color=styles[labels[0]][1], alpha=.25, linewidth=.9, zorder=1)
            for label, (name, color, marker) in styles.items():
                row = chosen[label]
                ax.scatter(row['temporal_error'], row[metric], label=name, c=color, marker=marker,
                           s=48, edgecolors='white', linewidths=.5, zorder=3)
            ax.set_title(('Synthetic DEV' if domain == 'davis_train_pilot' else 'Paired real ND DEV') + f' / seed {seed}')
            ax.set_xlabel('Temporal error (lower is better)')
            ax.set_ylabel(title + ' (higher is better)')
            ax.grid(alpha=.2)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', ncol=4, frameon=False)
        fig.suptitle('Same-F0 fixed-budget supervision controls: all domain/seed means')
        fig.subplots_adjust(top=.91, bottom=.19, hspace=.36, wspace=.25)
        for ext in ('png', 'pdf'):
            path = out / f'{metric}-vs-temporal.{ext}'
            fig.savefig(path, dpi=150, bbox_inches='tight')
            images.append(dict(path=str(path.relative_to(ROOT)), sha256=digest(path)))
        plt.close(fig)
    write(out / 'render.json', dict(at=stamp(), completed=True, all_group_points=48,
          source=str(source.relative_to(ROOT)), source_sha256=digest(source), plotter_sha256=digest(__file__),
          images=images, actual_viewed=False, new_NN_images=0, new_metric_scores=0,
          scope='Complete repeated DEV means; lines join each raw to own classical posts, not measured interpolation or phone speed.'))


if __name__ == '__main__':
    main()
