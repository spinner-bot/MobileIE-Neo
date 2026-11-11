"""Check singleton noiseless references against the batch-eight feature extraction path."""
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
import torch
from continuation_r14 import create_model
from correlated_observations_r19 import encode
from exploration_r2_data import ROOT
from explore_temporal_controls import save_json, sha256
from probe_latent_moments import features


@torch.inference_mode()
def main():
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    cv2.setNumThreads(1)
    base = ROOT/'research/4.temporal-consistency'
    weights = json.loads((base/'results/exploration-r27-frozen-cpu/checkpoints.json').read_text())
    conditions = json.loads((base/'results/exploration-r19/cache_index.json').read_text())['conditions']
    clean_inputs = {}
    for r in conditions:
        key = (r['scene'], r['jpeg'])
        if key in clean_inputs:
            continue
        assert sha256(ROOT/r['path']) == r['sha256']
        with np.load(ROOT/r['path'], allow_pickle=False) as arrays:
            clean = encode((arrays['target'].astype(np.float64)**2.2*.35)[None], r['jpeg'])
        clean_inputs[key] = torch.from_numpy(clean.transpose(0, 3, 1, 2).copy())
    rows = []
    for weight in weights:
        assert sha256(ROOT/weight['path']) == weight['sha256']
        net = create_model('plain', 'cpu').eval()
        if weight['label'] != 'official':
            net.load_state_dict(torch.load(ROOT/weight['path'], map_location='cpu', weights_only=False)['model'], strict=True)
        for (scene, jpeg), clean in clean_inputs.items():
            singleton = features(net.backbone, clean)
            repeated = features(net.backbone, clean.repeat(8, 1, 1, 1))
            for stage in ['head', 'body']:
                delta = float((singleton[stage]-repeated[stage][:1]).abs().max())
                within = float((repeated[stage]-repeated[stage][:1]).abs().max())
                rows.append({'weight': weight['label'], 'scene': scene, 'jpeg': jpeg, 'stage': stage,
                             'singleton_vs_batch8_max_abs': delta, 'within_repeated_batch8_max_abs': within})
    assert len(rows) == 40
    save_json(base/'process/exploration-r29/reference-batching-audit.json', {'checks': rows,
              'all_singleton_batch8_exact': all(r['singleton_vs_batch8_max_abs'] == 0 for r in rows),
              'max_abs_feature_difference': max(r['singleton_vs_batch8_max_abs'] for r in rows),
              'scope': 'Reference numerical-scope audit, not quality or new-data evaluation.'})


if __name__ == '__main__':
    main()
