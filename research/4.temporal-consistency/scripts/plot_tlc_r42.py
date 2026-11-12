"""Show all fixed-TLC raw condition deltas and domain/seed temporal ratios."""
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from dynamic_training_r34 import BASE, digest


def main():
    folder = BASE/'report/exploration-r42-tlc'
    audit = json.loads((folder/'audit.json').read_text())
    assert audit['passed'] and audit['new_rows'] == 60 and audit['total_rows'] == 120
    pair_path, effect_path = folder/'paired_by_condition.json', folder/'raw_effects.json'
    pairs = [r for r in json.loads(pair_path.read_text()) if r['method'] == 'raw']
    effects = json.loads(effect_path.read_text())
    assert len(pairs) == 20 and len(effects) == 4
    conditions = sorted({(r['domain'], r['scene']) for r in pairs})
    assert len(conditions) == 10
    fig, (left, right) = plt.subplots(1, 2, figsize=(12.8, 5.1), gridspec_kw={'width_ratios': [1.5, 1]})
    for seed, marker in [(0, 'o'), (1, 's')]:
        y = [next(r['delta_psnr_rgb'] for r in pairs if r['seed'] == seed and
                  (r['domain'], r['scene']) == condition) for condition in conditions]
        left.plot(range(10), y, marker=marker, linewidth=1, label=f'seed {seed}')
    left.axhline(0, color='#475569', linewidth=1)
    left.axvline(5.5, color='#94a3b8', linestyle=':', linewidth=1)
    left.set_xticks(range(10), [scene.replace('_photons', ' p').replace('video_', 'ND ')
                               for _, scene in conditions], rotation=40, ha='right', fontsize=8)
    left.set(ylabel='TLC − same-weight global raw PSNR (dB)', title='All 20 seed × condition pairs')
    left.legend()
    labels = []
    for x, row in enumerate(effects):
        right.scatter(x, row['temporal_ratio'], marker='o' if row['seed'] == 0 else 's', s=70)
        labels.append(('SYN' if row['domain'] == 'davis_train_pilot' else 'ND')+f" s{row['seed']}")
    right.axhline(1., color='#475569', linewidth=1, label='Equal temporal error')
    right.axhline(1.02, color='#dc2626', linestyle='--', linewidth=1, label='Registered upper bound')
    right.set_xticks(range(4), labels)
    right.set(ylabel='Raw temporal error: TLC / global', title='Every domain and seed')
    right.legend(fontsize=8)
    for ax in [left, right]:
        ax.grid(alpha=.2)
    fig.suptitle('R42 known spatial TLC, fixed window192, no training / unchanged weights\n'
                 'Existing DEV only; local operations add cost. Condition points are not independent videos.', fontsize=11)
    fig.tight_layout()
    for extension in ['png', 'pdf']:
        fig.savefig(folder/('fixed_tlc.'+extension), dpi=160, bbox_inches='tight')
    plt.close(fig)
    (folder/'plot_sources.json').write_text(json.dumps({
        'paired_sha256': digest(pair_path), 'effects_sha256': digest(effect_path),
        'condition_order': conditions, 'scope': 'Complete raw conditions, no selection or new inference.'
    }, indent=2)+'\n')


if __name__ == '__main__':
    main()
