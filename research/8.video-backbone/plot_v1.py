"""Plot every completed V1 domain/seed mean; no model or metric execution."""
import csv
import hashlib
import json
import math
from datetime import datetime
from pathlib import Path


BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]
FAMILIES = {
    'F0': ('#555555', 'MobileIE F0'),
    'spatial': ('#0072B2', 'Independent packed spatial'),
    'fixed': ('#D55E00', 'Fixed-fusion weights'),
    'learned': ('#009E73', 'Learned-fusion weights'),
}
FORMS = {
    'raw': ('o', 'Raw / single-frame preview'),
    'FB': ('^', 'Own-preview + half-FB 0.5'),
    'DIS': ('v', 'Own-preview + DIS 0.5'),
    'causal': ('*', 'Causal input fusion'),
}
LABELS = {
    'MobileIE_raw': ('F0', 'raw'),
    'MobileIE_recursive_self05': ('F0', 'FB'),
    'DIS_recursive_self05': ('F0', 'DIS'),
    'topic8_spatial_raw': ('spatial', 'raw'),
    'topic8_spatial_raw_FB05': ('spatial', 'FB'),
    'topic8_spatial_raw_DIS05': ('spatial', 'DIS'),
    'topic8_fixed_preview': ('fixed', 'raw'),
    'topic8_fixed_preview_FB05': ('fixed', 'FB'),
    'topic8_fixed_preview_DIS05': ('fixed', 'DIS'),
    'topic8_fixed_causal': ('fixed', 'causal'),
    'topic8_learned_preview': ('learned', 'raw'),
    'topic8_learned_preview_FB05': ('learned', 'FB'),
    'topic8_learned_preview_DIS05': ('learned', 'DIS'),
    'topic8_learned_causal': ('learned', 'causal'),
}
DOMAINS = ('davis_train_pilot', 'paired_real_ND_DEV')
METRICS = ('psnr_rgb', 'ssim_rgb', 'temporal_error', 'JOD')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    source = BASE / 'report/v1-joint'
    audit = json.loads((source / 'audit.json').read_text())
    assert audit['completed'] and audit['conditions'] == 280 and audit['means'] == 56
    means = json.loads((source / 'domain_seed_means.json').read_text())
    conditions = [json.loads(line) for line in (source / 'all_conditions.jsonl').read_text().splitlines()]
    assert len(conditions) == len({(r['seed'], r['scene'], r['label']) for r in conditions}) == 280
    assert len(means) == len({(r['domain'], r['seed'], r['label']) for r in means}) == 56
    for domain in DOMAINS:
        for seed in (0, 1):
            selected = [r for r in means if (r['domain'], r['seed']) == (domain, seed)]
            assert {r['label'] for r in selected} == set(LABELS)
            for row in selected:
                assert all(math.isfinite(row[m]) for m in METRICS)
                assert row['temporal_error'] > 0
                raw = [r for r in conditions if (r['domain'], r['seed'], r['label']) ==
                       (domain, seed, row['label'])]
                assert len(raw) == (6 if domain == 'davis_train_pilot' else 4)
                for metric in METRICS:
                    assert math.isclose(row[metric], sum(r[metric] for r in raw) / len(raw),
                                        rel_tol=1e-12, abs_tol=1e-12)

    # Import plotting only after confirming the complete joint result exists.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'pdf.fonttype': 42, 'ps.fonttype': 42})
    out = BASE / 'report/v1-plots'
    out.mkdir(parents=True, exist_ok=False)
    fieldnames = ('domain', 'seed', 'label', 'family', 'output_form', *METRICS)
    with (out / 'points.csv').open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in means:
            family, form = LABELS[row['label']]
            writer.writerow({**row, 'family': family, 'output_form': form})
    files = [out / 'points.csv']
    family_legend = [Line2D([], [], color=color, marker='s', linestyle='', label=label)
                     for color, label in FAMILIES.values()]
    form_legend = [Line2D([], [], color='black', marker=marker, linestyle='', label=label)
                   for marker, label in FORMS.values()]
    for metric, ylabel in (('psnr_rgb', 'PSNR (dB; higher is better)'),
                            ('JOD', 'ColorVideoVDP JOD (higher is better)')):
        fig, axes = plt.subplots(2, 2, figsize=(11, 9), sharex='row', sharey='row', squeeze=False)
        for domain_index, domain in enumerate(DOMAINS):
            for seed in (0, 1):
                ax = axes[domain_index, seed]
                selected = [r for r in means if (r['domain'], r['seed']) == (domain, seed)]
                for row in selected:
                    family, form = LABELS[row['label']]
                    ax.scatter(row['temporal_error'], row[metric], color=FAMILIES[family][0],
                               marker=FORMS[form][0], s=110 if form == 'causal' else 55,
                               edgecolors='white', linewidths=.5,
                               zorder=4 if form == 'causal' else 3)
                domain_label = 'SYN: 6 conditions / 3 source videos' if domain_index == 0 else 'ND: 4 source videos'
                ax.set_title(f'{domain_label}; training seed {seed}')
                ax.grid(alpha=.25)
                ax.set_xlabel('Temporal error (lower is better)')
                ax.set_ylabel(ylabel if seed == 0 else '')
                ax.ticklabel_format(axis='x', style='plain', useOffset=False)
        fig.suptitle('Repeated DEV: all 14 outputs per domain and seed', fontsize=13)
        fig.legend(handles=family_legend + form_legend, loc='lower center', ncol=2,
                   bbox_to_anchor=(.5, .035), frameon=False, fontsize=9)
        fig.text(.5, .012, 'Means over existing conditions; no confidence intervals or independent-test claim.',
                 ha='center', fontsize=9)
        fig.tight_layout(rect=(0, .16, 1, .96))
        for extension in ('pdf', 'png'):
            path = out / f'{metric}-versus-temporal.{extension}'
            fig.savefig(path, dpi=180, bbox_inches='tight')
            files.append(path)
        plt.close(fig)
    record = dict(at=datetime.now().astimezone().isoformat(), completed=True,
                  sources={str(p.relative_to(ROOT)): digest(p) for p in
                           (source / 'audit.json', source / 'domain_seed_means.json',
                            source / 'all_conditions.jsonl', Path(__file__))},
                  files={str(p.relative_to(ROOT)): digest(p) for p in files},
                  points=56, outputs_per_group=14, new_model_forwards=0, new_metric_calls=0,
                  selected_conditions=False, actual_viewed=False,
                  scope='All completed repeated DEV means; no latency, phone, novelty or independent-final evidence.')
    (out / 'render.json').write_text(json.dumps(record, indent=2, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    main()
