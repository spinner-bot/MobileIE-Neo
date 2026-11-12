"""Plot the entire preregistered oracle alpha range, without a new metric."""
import csv
import hashlib
import json
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from coordinate_state_r46 import BASE, ROOT, ALPHAS


def main():
    source = BASE/'results/exploration-r46-oracle001/sequence_metrics.csv'
    audit = json.loads((source.parent/'audit.json').read_text())
    assert audit['processing_complete'] and audit['rows'] == 150
    out = BASE/'report/exploration-r46-coordinate-state'
    assert not (out/'oracle_frontier.png').exists()
    groups = defaultdict(list)
    with source.open() as stream:
        for row in csv.DictReader(stream):
            if row['velocity'] == '2':
                groups[float(row['sigma']), float(row['alpha']), row['method']].append(float(row['late_mse']))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, sigma in zip(axes, [0., .03]):
        for method, label, color in [('classical','Frame-recursive RGB','tab:blue'),
                                     ('anchor8','Anchor-coordinate RGB / K=8','tab:red')]:
            ax.plot(ALPHAS, [np.mean(groups[sigma,a,method]) for a in ALPHAS],
                    'o-', label=label, color=color)
        ax.set_xlabel('History alpha')
        ax.set_ylabel('Late-16 central-ROI MSE')
        ax.set_title('Clean / deterministic' if sigma == 0 else 'Gaussian sigma=.03 / four-seed mean')
        ax.grid(alpha=.25)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=2, frameon=False)
    fig.suptitle('Known fractional motion only: no model or real-video evidence')
    fig.tight_layout(rect=[0,.10,1,.93])
    for extension in ['png','pdf']:
        fig.savefig(out/f'oracle_frontier.{extension}', dpi=150)
    plt.close(fig)
    (out/'oracle_plot_sources.json').write_text(json.dumps({
        'source': str(source.relative_to(ROOT)), 'sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'scope': 'All five registered alphas; only fractional-motion condition; MSE not a new metric.'}, indent=2)+'\n')


if __name__ == '__main__':
    main()
