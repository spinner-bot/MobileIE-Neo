"""One fixed coordinate-state evaluation; reused final spatial caches, zero CNN."""
import argparse
import hashlib
import json
import time
import traceback
from datetime import datetime

import cv2
import numpy as np
import torch

from coordinate_state_r46 import ROOT, BASE
from coordinate_motion_r46 import bidirectional, predict
from exploration_r2_data import evaluate
from prepare_cache_r5 import load_one


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            value.update(block)
    return value.hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def append(path, value):
    with path.open('a') as stream:
        stream.write(json.dumps(value, allow_nan=False)+'\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', required=True, type=int, choices=[0, 1])
    args = parser.parse_args()
    cv2.setNumThreads(1); torch.set_num_threads(2)
    assert not torch.cuda.is_initialized()
    gate_path = BASE/'results/exploration-r46-motion-gate001/gate.json'
    gate = json.loads(gate_path.read_text())
    assert gate['passed'] and len(gate['cases']) == 40
    oracle = json.loads((BASE/'results/exploration-r46-oracle001/audit.json').read_text())
    assert oracle['mechanism_gate_passed']
    out = BASE/f'results/exploration-r46-stage002-eval-seed{args.seed}'
    out.mkdir(exist_ok=False)
    indices = [BASE/f'results/{n}/index.json' for n in
               ['exploration-r30-cache', 'exploration-r10-cache']]
    metas = json.loads(indices[0].read_text())['scenes']
    metas += [{**r, 'domain': 'paired_real_ND_DEV', 'source_video': r['scene']}
              for r in json.loads(indices[1].read_text())['scenes'] if r['domain'] == 'native_crop']
    assert len(metas) == 10
    manifest = BASE/f'results/exploration-r44-stage001-eval-seed{args.seed}/prediction_manifest.jsonl'
    originals = {r['scene']: r for r in map(json.loads, manifest.read_text().splitlines())
                 if r['model'] == 'square_mse'}
    old_path = BASE/f'results/exploration-r45-stage001-eval-seed{args.seed}/per_sequence.jsonl'
    reference = {r['scene']: r for r in map(json.loads, old_path.read_text().splitlines())
                 if r['method'] == 'fixed_05'}
    names = ['coordinate_motion_r46.py', 'evaluate_coordinate_motion_r46.py',
             'aligned_raw_history_r11.py', 'half_motion_r4.py', 'video_metrics.py',
             'exploration_r2_data.py', 'prepare_cache_r5.py']
    write(out/'config.json', {'started': datetime.now().astimezone().isoformat(),
        'seed': args.seed, 'alpha': .5, 'period': 8, 'neural_sequences': 0,
        'methods': ['ordinary', 'same_mask', 'anchor8'], 'opencv_threads': 1,
        'torch_threads': 2, 'protocol_sha256': digest(BASE/'docs/exploration-r46-real-motion-protocol.md'),
        'pre_DEV_gate_sha256': digest(gate_path),
        'source_sha256': {n: digest(BASE/'scripts'/n) for n in names},
        'raw_manifest_sha256': digest(manifest), 'R45_reference_sha256': digest(old_path),
        'cache_indices': {str(p.relative_to(ROOT)): digest(p) for p in indices},
        'scope': 'Fixed DEV quality screen; native coordinate state costs +75% history, no deployment timing.'})
    start, count, frame_count, complete, max_replay = time.perf_counter(), 0, 0, False, 0.
    try:
        for meta in metas:
            data = load_one(meta)
            original = originals[meta['scene']]
            path = ROOT/original['path']
            assert path.resolve().is_relative_to('/data/Users/zrh/data/MobileIE-Neo')
            assert digest(path) == original['sha256']
            raw = np.load(path, mmap_mode='r', allow_pickle=False)
            assert list(raw.shape) == original['shape']
            fields = [bidirectional(raw[t-1], raw[t]) for t in range(1, len(raw))]
            values, stats = predict(raw, fields, .5, 8)
            identity = {'domain': meta['domain'], 'scene': meta['scene'],
                        'source_video': meta['source_video'], 'train_seed': args.seed}
            append(out/'raw_sources.jsonl', {**identity, **original})
            for row in stats:
                append(out/'fusion_stats.jsonl', {**identity, **row})
            for method, output in values.items():
                assert np.array_equal(output[0], raw[0]) and np.isfinite(output).all()
                metrics, frames = evaluate(output, data['gt'], data['reference_flow'], data['reference_mask'])
                if method == 'ordinary':
                    deltas = {k: abs(metrics[k]-reference[meta['scene']][k])
                              for k in ['psnr_rgb', 'ssim_rgb', 'edge_mae', 'temporal_error']}
                    assert max(deltas.values()) <= 1e-5, deltas
                    max_replay = max(max_replay, max(deltas.values()))
                    append(out/'reference_replay.jsonl', {**identity, 'metric_deltas': deltas})
                h, w = raw.shape[1:3]
                append(out/'per_sequence.jsonl', {**identity, 'method': method, 'parameters': 4075,
                       'history_storage_bytes': h*w*(42 if method == 'anchor8' else 24),
                       'same_mask_scope': 'offline rejection-matched diagnostic, coordinate-mask cost excluded'
                           if method == 'same_mask' else None, **metrics})
                for frame in frames:
                    append(out/'per_frame.jsonl', {**identity, 'method': method, **frame})
                count += 1; frame_count += len(frames)
            print(json.dumps({**identity, 'rows': count}), flush=True)
            del data, raw, fields, values, output
        assert count == 30 and frame_count == 1344 and not torch.cuda.is_initialized()
        complete = True
    except BaseException:
        (out/'failure.txt').write_text(traceback.format_exc())
        raise
    finally:
        write(out/'runtime.json', {'completed_normally': complete, 'rows': count,
            'per_frame_rows': frame_count, 'neural_sequences': 0, 'cuda_initialized': torch.cuda.is_initialized(),
            'max_reference_metric_difference': max_replay, 'seconds': time.perf_counter()-start,
            'ended': datetime.now().astimezone().isoformat()})


if __name__ == '__main__':
    main()
