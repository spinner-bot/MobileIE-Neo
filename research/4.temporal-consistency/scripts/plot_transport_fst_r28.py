"""Show registered native-domain quality/stability points, including both seeds."""
import json
from pathlib import Path
from statistics import mean

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT/'research/4.temporal-consistency'


def main():
    folder = BASE/'report/exploration-r28'
    audit = json.loads((folder/'audit.json').read_text())
    assert audit['passed'] and audit['rows'] == 4416
    rows = json.loads((folder/'domain_seed_summary.json').read_text())
    specifications = [
        ('Plain raw', 'copy', 'same', 'causal', 'raw', '#666666', 'o'),
        ('Plain + post .5', 'copy', 'same', 'causal', 'postwarp05', '#0072B2', 's'),
        ('R26 pre/MSE current + post .5', 'r26_pre_mse_spatial', 'mse', 'causal', 'postwarp05', '#009E73', '^'),
        ('R28 current raw (single frame)', 'post_transport', 'same', 'current', 'raw', '#E69F00', 'D'),
        ('R28 current + post .5', 'post_transport', 'same', 'current', 'postwarp05', '#CC79A7', 'P'),
        ('R28 causal raw (flow + state)', 'post_transport', 'same', 'causal', 'raw', '#D55E00', 'X'),
    ]
    figure, axes = plt.subplots(1, 2, figsize=(11, 5.5), constrained_layout=False)
    for axis, objective in zip(axes, ['charb', 'mse']):
        points = {}
        for label, mode, target, history, fusion, color, marker in specifications:
            selected = [r for r in rows if r['domain'] == 'dev_native' and r['objective'] == (objective if target == 'same' else target)
                        and r['mode'] == mode and r['history'] == history and r['fusion'] == fusion and not r['input_reset']]
            assert len(selected) == 2 and {r['train_seed'] for r in selected} == {0, 1}
            x, y = mean(r['temporal_error'] for r in selected), mean(r['psnr_rgb'] for r in selected)
            points[label] = x, y
            axis.scatter([r['temporal_error'] for r in selected], [r['psnr_rgb'] for r in selected],
                         s=40, marker=marker, facecolors='none', edgecolors=color, linewidths=1)
            axis.scatter([x], [y], s=85, marker=marker, color=color, label=label, zorder=4)
        axis.annotate('', xy=points['R28 causal raw (flow + state)'], xytext=points['R28 current raw (single frame)'],
                      arrowprops={'arrowstyle': '->', 'color': '#333333', 'linewidth': .8}, zorder=2)
        axis.set_title(objective.upper()+' adaptation')
        axis.set_xlabel('Temporal residual error (0-255; lower is better)')
        axis.set_ylabel('RGB PSNR (dB; higher is better)')
        axis.grid(alpha=.2)
        axis.margins(.12)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc='lower center', ncol=2, fontsize=8, frameon=False, bbox_to_anchor=(.5, .01))
    figure.suptitle('R28 fixed4000 native DEV: four clips, two training seeds', fontsize=12)
    figure.text(.5, .16, 'Small hollow: each seed. Large filled: two-seed mean. Arrow: same-weight current to causal.\n'
                'Flow/state cost is additional; phone latency is unmeasured. Per-seed screening is reported separately.',
                ha='center', fontsize=8)
    figure.subplots_adjust(bottom=.29, top=.88, left=.075, right=.985, wspace=.25)
    figure.savefig(folder/'native_quality_stability.png', dpi=180)
    figure.savefig(folder/'native_quality_stability.pdf')
    plt.close(figure)


if __name__ == '__main__':
    main()
