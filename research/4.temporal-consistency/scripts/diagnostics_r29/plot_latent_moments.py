"""Display mean-shift/variance and total latent bias, without additive RMS attribution."""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from exploration_r2_data import ROOT

BASE = ROOT/'research/4.temporal-consistency/report/exploration-r29'


def main():
    rows = json.loads((BASE/'group_summary.json').read_text())
    fig, axes = plt.subplots(3, 2, figsize=(12, 12))
    conditions = [(0, False), (.9, False), (0, True), (.9, True)]
    for i, objective in enumerate(['official', 'charb', 'mse']):
        for j, stage in enumerate(['head', 'body']):
            ax = axes[i, j]
            for k, (label, key) in enumerate([('Mean-shift term', 'self_mean_shift_term'), ('Variance term', 'self_variance_term'),
                                             ('Self latent bias', 'self'), ('Cross latent bias', 'cross')]):
                values = []
                for rho, jpeg in conditions:
                    selected = [r for r in rows if r['objective'] == objective and r['stage'] == stage and r['rho'] == rho and r['jpeg'] == jpeg]
                    assert len(selected) == 2
                    values.append(np.mean([(r[key] if key.endswith('_term') else r['latent_bias'][key])['rms_over_reference_q_rms'] for r in selected]))
                assert min(values) > 0
                ax.bar(np.arange(4)+(k-1.5)*.2, values, width=.2, label=label)
            ax.set_yscale('log')
            ax.set_xticks(range(4), ['rho0 / no JPEG', 'rho.9 / no JPEG', 'rho0 / JPEG', 'rho.9 / JPEG'], rotation=20, ha='right')
            ax.set_title(objective+' / '+stage+' FST', fontsize=11)
            ax.grid(axis='y', alpha=.2)
            ax.set_ylabel('RMS / noiseless-LOW q RMS')
    axes[0, 0].legend(fontsize=8)
    fig.suptitle('Frozen latent noise perturbation: mean shift remains after cross multiplication\nTwo TRAIN images, photons20/80; terms are signed fields, component RMS DOES NOT ADD\nCharb/MSE average two seeds; reference is encoded noiseless LOW input, no RGB quality claim', fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, .94])
    for suffix in ['png', 'pdf']:
        fig.savefig(BASE/('latent_bias_components.'+suffix), dpi=170)
    plt.close(fig)


if __name__ == '__main__':
    main()
