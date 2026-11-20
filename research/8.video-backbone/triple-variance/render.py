"""Complete N1 calibration plots from all 48 finished scalar cases; no new NN."""
from pathlib import Path
import sys
import json
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[2]
sys.path.insert(0, str(ROOT / 'research/4.temporal-consistency/scripts'))
from causal_reference_io_r75 import write, digest, stamp


def main():
    source = BASE / 'results/run001/per_condition.jsonl'
    rows = [json.loads(l) for l in source.read_text().splitlines()]
    assert len(rows) == 48
    out = BASE / 'report'
    out.mkdir(exist_ok=False)
    fields = ('key', 'seed', 'texture', 'photons', 'common_fraction', 'main_independent_group',
        'reference_global_variance', 'signed_global_estimate', 'signed_global_reference_ratio',
        'per_triplet_local_relative_L1_median', 'mean_negative_tile_fraction',
        'reference_mean_bias_over_variance', 'probe_covariance_bias_over_variance', 'all_gates_passed')
    with (out / 'points.csv').open('x', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator='\n')
        writer.writeheader(); writer.writerows({k: r[k] for k in fields} for r in rows)
    patterns = ([20, 20, 20], [20, 80, 80], [80, 20, 20])
    styles = (('20 / 20 / 20', 'o', '#0072B2'), ('20 / 80 / 80', '^', '#E69F00'), ('80 / 20 / 20', 'D', '#CC79A7'))
    fig, axes = plt.subplots(2, 2, figsize=(11, 10))
    for index, fraction in enumerate((0, .5)):
        for seed in (0, 1):
            ax = axes[index, seed]
            chosen = [r for r in rows if r['seed'] == seed and r['common_fraction'] == fraction]
            maximum = max(max(r['reference_global_variance'], r['signed_global_estimate']) for r in chosen) * 255 ** 2 * 1.15
            coordinates = np.array([0, maximum])
            ax.plot(coordinates, coordinates, color='gray', linestyle='--', linewidth=1, label='Equality to total V0')
            if fraction == 0:
                ax.fill_between(coordinates, .85 * coordinates, 1.15 * coordinates, color='gray', alpha=.10, label='Registered ±15% band')
            for photons, (name, marker, color) in zip(patterns, styles):
                selected = [r for r in chosen if r['photons'] == photons]
                x = [r['reference_global_variance'] * 255 ** 2 for r in selected]
                y = [r['signed_global_estimate'] * 255 ** 2 for r in selected]
                ax.scatter(x, y, marker=marker, color=color, label=name, s=55)
                for xx, yy, row in zip(x, y, selected):
                    ax.annotate(str(row['texture']), (xx, yy), xytext=(4, 4), textcoords='offset points', fontsize=8)
            ax.set_title(f'F0 seed {seed} / common fraction {fraction}' + (' / diagnostic' if fraction else ' / primary'))
            ax.set_xlabel('Independent MC reference V0 (gray-level squared)')
            ax.set_ylabel('Signed three-difference estimate (gray-level squared)')
            ax.set_xlim(0, maximum); ax.set_ylim(0, maximum)
            ax.grid(alpha=.2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3)
    fig.suptitle('N1 all 48 groups; labels are training texture IDs; correlated equality is not expected')
    fig.tight_layout(rect=(0, .08, 1, .96))
    for ext in ('png', 'pdf'):
        fig.savefig(out / ('global-calibration.' + ext), dpi=140)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for seed, ax in enumerate(axes):
        for photons, (label, marker, color) in zip(patterns, styles):
            selected = [r for r in rows if r['seed'] == seed and r['main_independent_group'] and r['photons'] == photons]
            x = [r['reference_mean_bias_over_variance'] for r in selected]
            y = [r['per_triplet_local_relative_L1_median'] for r in selected]
            ax.scatter(x, y, marker=marker, color=color, label=label, s=55)
            for xx, yy, row in zip(x, y, selected):
                ax.annotate(str(row['texture']), (xx, yy), xytext=(4, 4), textcoords='offset points', fontsize=8)
        ax.axhline(.35, color='gray', linestyle='--', label='Registered local-error limit')
        ax.set_title(f'All independent cases / F0 seed {seed}')
        ax.set_xlabel('Reference conditional-mean term / V0')
        ax.set_ylabel('Median per-triplet tile relative L1 error')
        ax.grid(alpha=.2)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=2)
    fig.suptitle('Local estimation with fixed 16-pixel tiles; all 24 independent cases')
    fig.tight_layout(rect=(0, .14, 1, .94))
    for ext in ('png', 'pdf'):
        fig.savefig(out / ('local-error-and-mean-term.' + ext), dpi=140)
    plt.close(fig)
    groups = []
    for fraction in (0, .5):
        for photons in patterns:
            chosen = [r for r in rows if r['common_fraction'] == fraction and r['photons'] == photons]
            groups.append(dict(common_fraction=fraction, photons=photons, conditions=len(chosen),
                full_gate_passes=sum(r['all_gates_passed'] for r in chosen), primary=fraction == 0,
                ranges={k: [min(r[k] for r in chosen), max(r[k] for r in chosen)] for k in
                    ('signed_global_reference_ratio', 'per_triplet_local_relative_L1_median',
                     'reference_mean_bias_over_variance', 'probe_covariance_bias_over_variance')}))
    write(out / 'descriptive-groups.json', groups)
    write(out / 'render.json', dict(at=stamp(), source_sha256=digest(source), script_sha256=digest(__file__),
        actual_viewed=False, new_NN_images=0, new_metric_scores=0, gate_changes=0,
        scalar_cases=48, artifacts=[dict(path=str(p.relative_to(ROOT)), sha256=digest(p)) for p in sorted(out.iterdir())],
        scope='Static training textures and finite MC; not video quality, sensor calibration, phone speed or independent final'))


if __name__ == '__main__':
    main()
