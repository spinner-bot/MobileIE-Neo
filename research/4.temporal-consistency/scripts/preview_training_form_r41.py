"""Inspect all four paired-real DEV sources at fixed frames, without new CNN calls."""
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from dynamic_training_r34 import BASE, ROOT, digest
from prepare_cache_r5 import load_one


def main():
    folder = BASE/'report/exploration-r41-training-form'
    audit = json.loads((folder/'audit.json').read_text())
    assert audit['passed'] and audit['new_rows'] == 120
    index = BASE/'results/exploration-r10-cache/index.json'
    metas = [m for m in json.loads(index.read_text())['scenes'] if m['domain'] == 'native_crop']
    assert len(metas) == 4
    manifests = {}
    for seed in [0, 1]:
        path = BASE/f'results/exploration-r41-stage002-eval-seed{seed}/prediction_manifest.jsonl'
        for row in map(json.loads, path.read_text().splitlines()):
            manifests[(row['scene'], row['model'], seed)] = row
    records = []
    for meta in metas:
        data = load_one(meta)
        arrays = [('Input', data['low']), ('GT', data['gt'])]
        for seed in [0, 1]:
            for model, label in [('slim_fp32', 'Slim'), ('expanded_fp32', 'Expanded')]:
                record = manifests[(meta['scene'], model, seed)]
                path = ROOT/record['path']
                assert digest(path) == record['sha256']
                value = np.load(path, allow_pickle=False)
                assert value.shape == data['gt'].shape and len(value) == 16
                arrays.append((f'{label} seed{seed}', value))
        fig, axes = plt.subplots(3, 6, figsize=(15, 5.8))
        for row, frame in enumerate([0, 8, 15]):
            for col, (label, values) in enumerate(arrays):
                axes[row, col].imshow(np.clip(values[frame], 0, 255).astype(np.uint8))
                axes[row, col].set(xticks=[], yticks=[])
                if row == 0:
                    axes[row, col].set_title(label, fontsize=10)
                if col == 0:
                    axes[row, col].set_ylabel(f'frame {frame}', fontsize=10)
        fig.suptitle(meta['scene']+': all matched raw deployments, fixed 0/8/15 frames\n'
                     'Common display conversion only; metrics use the complete float sequences.', fontsize=11)
        fig.tight_layout()
        fig.savefig(folder/(meta['scene']+'-raw.png'), dpi=120, bbox_inches='tight')
        plt.close(fig)
        records.append({'scene': meta['scene'], 'frames': [0, 8, 15],
                        'whole_sequence_RGB_component_statistics': [
                            {'output': label, 'mean': float(value.mean()),
                             'fraction_ge_254': float((value >= 254).mean()),
                             'fraction_le_1': float((value <= 1).mean())}
                            for label, value in arrays]})
    (folder/'preview_sources.json').write_text(json.dumps({
        'new_neural_inferences': 0, 'source_index_sha256': digest(index),
        'scenes': records,
        'scope': 'All four already-opened paired-real DEV sources, not phone captures or a user study. '
                 'Fixed frames for display; component clipping fractions are not pixel fractions or a quality metric.'
    }, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
