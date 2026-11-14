"""Plot complete DEV quality and fixed-workload CPU costs, with both seeds shown."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from dynamic_training_r34 import BASE, ROOT, digest

CASES = [('original', 'current', 'raw', 'Original'),
         ('original', 'current', 'self_motion_post05', 'Original + self-post'),
         ('spatial', 'current', 'raw', 'Spatial'),
         ('spatial', 'current', 'self_motion_post05', 'Spatial + self-post'),
         ('causal', 'current', 'raw', 'Causal/current'),
         ('causal', 'current', 'self_motion_post05', 'Causal/current + self-post'),
         ('causal', 'causal', 'raw', 'Causal/actual')]


def main():
    paths = {name: BASE/path for name, path in {
        'quality_audit': 'report/exploration-r34/audit.json',
        'quality': 'report/exploration-r34/domain_summary.json',
        'seeds': 'report/exploration-r34/domain_seed_summary.json',
        'cost_audit': 'report/exploration-r34-runtime/audit.json',
        'cost': 'report/exploration-r34-runtime/cost.json'}.items()}
    data = {name: json.loads(path.read_text()) for name, path in paths.items()}
    assert data['quality_audit']['passed'] and data['quality_audit']['rows'] == 120
    assert data['cost_audit']['passed'] and data['cost_audit']['runs'] == 42
    key = lambda row: (row['model'], row['history'], row['fusion'])
    quality = {key(row): row for row in data['quality']}
    cost = {key(row): row for row in data['cost']}
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.3))
    colors = ['#4c566a', '#111827', '#2980b9', '#1f618d', '#e67e22', '#a04000', '#c0392b']
    markers = ['o', 's', 'o', 's', 'o', 's', '*']
    for case, color, marker in zip(CASES, colors, markers):
        identity, label = case[:3], case[3]
        q, c = quality[identity], cost[identity]
        seed_rows = [r for r in data['seeds'] if key(r) == identity]
        assert len(seed_rows) == 2 and {r['seed'] for r in seed_rows} == {0, 1}
        axes[0].scatter([r['temporal_error'] for r in seed_rows], [r['psnr_rgb'] for r in seed_rows],
                        s=20, alpha=.35, color=color, marker=marker)
        axes[0].scatter(q['temporal_error'], q['psnr_rgb'], s=95 if marker == '*' else 55,
                        color=color, marker=marker, label=label, zorder=3)
        middle = c['median_pipeline_ms_per_frame']
        axes[1].errorbar(middle, q['psnr_rgb'],
                        xerr=[[middle-c['min_pipeline_ms_per_frame']], [c['max_pipeline_ms_per_frame']-middle]],
                        color=color, marker=marker, markersize=10 if marker == '*' else 6, capsize=2, linestyle='none')
    current, actual = quality[('causal', 'current', 'raw')], quality[('causal', 'causal', 'raw')]
    axes[0].annotate('', xy=(actual['temporal_error'], actual['psnr_rgb']),
                     xytext=(current['temporal_error'], current['psnr_rgb']),
                     arrowprops={'arrowstyle': '->', 'color': '#c0392b', 'lw': 1})
    axes[0].set(xlabel='GT-residual temporal error (lower is better)', ylabel='RGB PSNR (dB, higher is better)',
                title='DEV quality: three videos, both seeds')
    axes[1].set(xlabel='Resident CPU pipeline ms/frame (lower is better)', ylabel='RGB PSNR (dB)',
                title='CPU cost: fixed 480 x 854 workload')
    for ax in axes:
        ax.grid(alpha=.2)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4, frameon=False, fontsize=8)
    fig.text(.5, .11, 'Faint quality points: individual seeds; bold: mean. CPU bars: all six run averages, not confidence intervals.',
             ha='center', fontsize=8)
    fig.suptitle('Exploratory DEV and shared-server CPU; no phone or independent-test result', fontsize=11)
    fig.tight_layout(rect=[0, .15, 1, .95])
    out = BASE/'report/exploration-r34'
    for extension in ['png', 'pdf']:
        path = out/f'quality_cpu_cost.{extension}'
        assert not path.exists()
        fig.savefig(path, dpi=180)
    plt.close(fig)
    (out/'figure_sources.json').write_text(json.dumps({str(path.relative_to(ROOT)): digest(path) for path in paths.values()}, indent=2)+'\n')
    print(json.dumps({'plotted_mappings': 7, 'quality_rows': 120, 'cost_runs': 42}))


if __name__ == '__main__':
    main()
