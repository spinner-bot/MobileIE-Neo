"""Final R37 weights on common CPU, retaining all completed strong references."""
import argparse
import json
from pathlib import Path
import subprocess
import time
import traceback

import cv2
import numpy as np
import torch

from dynamic_training_r34 import BASE, ROOT, digest
from exploration_r2_data import evaluate
from prepare_cache_r5 import load_one
from training_preserved_r37 import models


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def append(path, value):
    with path.open('a') as stream:
        stream.write(json.dumps(value, allow_nan=False)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, choices=[0, 1], required=True)
    args = parser.parse_args()
    seed = args.seed
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    torch.use_deterministic_algorithms(True)
    assert not torch.cuda.is_initialized()
    folder = BASE/f'results/exploration-r37-seed{seed}'
    runtime = json.loads((folder/'runtime.json').read_text())
    assert runtime['completed_normally'] and runtime['total_updates'] == 8000
    assert set(runtime['model_updates'].values()) == {4000}
    nets, anchor = models(seed, torch.device('cpu'))
    markers = {}
    for name, net in nets.items():
        marker = json.loads((folder/f'{name}-step4000.json').read_text())
        assert digest(ROOT/marker['path']) == marker['sha256']
        saved = torch.load(ROOT/marker['path'], map_location='cpu', weights_only=False)
        assert saved['step'] == 4000 and saved['seed'] == seed and saved['model'] == name
        # A bounded invariant check, not an optimizer/bit-replay experiment.
        assert all(torch.equal(value, saved['state_dict']['backbone.'+key])
                   for key, value in net.backbone.state_dict().items())
        net.load_state_dict(saved['state_dict'], strict=True)
        net.eval()
        assert torch.isfinite(net.basis()).all()
        markers[name] = marker
    reference = BASE/'results/exploration-r36-classical'
    prior_runtime = json.loads((reference/'runtime.json').read_text())
    assert prior_runtime['completed_normally'] and prior_runtime['rows'] == 48
    prior_config = json.loads((reference/'config.json').read_text())
    assert prior_config['checkpoints'][seed]['sha256'] == anchor['sha256']
    reference_rows = [r for r in map(json.loads, (reference/'per_sequence.jsonl').read_text().splitlines())
                      if r['train_seed'] == seed]
    reference_frames = [r for r in map(json.loads, (reference/'per_frame.jsonl').read_text().splitlines())
                        if r['train_seed'] == seed]
    manifest = [r for r in map(json.loads, (reference/'prediction_manifest.jsonl').read_text().splitlines())
                if r['train_seed'] == seed]
    predictions = {r['scene']: r for r in manifest}
    out = BASE/f'results/exploration-r37-eval-seed{seed}'
    out.mkdir(parents=True, exist_ok=False)
    index = BASE/'results/exploration-r30-cache/index.json'
    metas = json.loads(index.read_text())['scenes']
    assert len(metas) == 6 and len(reference_rows) == 24 and len(manifest) == 6
    write(out/'config.json', {'seed': seed, 'device': 'cpu', 'torch_threads': 2, 'opencv_threads': 1,
          'checkpoints': markers, 'anchor': anchor, 'weight_selection': 'final4000', 'torch_version': torch.__version__,
          'protocol_sha256': digest(BASE/'docs/exploration-r37-preserved-state-protocol.md'),
          'cache_index_sha256': digest(index),
          'reference_source_sha256': {str((reference/name).relative_to(ROOT)): digest(reference/name) for name in
                                     ['per_sequence.jsonl', 'per_frame.jsonl', 'prediction_manifest.jsonl', 'config.json', 'runtime.json']},
          'source_sha256': {name: digest(Path(__file__).parent/name) for name in
                           ['evaluate_preserved_r37.py', 'training_preserved_r37.py', 'preserved_state_r37.py',
                            'budget_causal_r33.py', 'exploration_r2_data.py', 'video_metrics.py']},
          'revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
          'scope': 'Common CPU DEV; four strong references reused exactly from R36 with the same frozen anchor. Current explicitly returns the immutable anchor, not a new CNN evaluation. No phone or independent-test claim.'})
    started, count, neural, complete = time.perf_counter(), 0, 0, False
    try:
        for meta in metas:
            data = load_one(meta)
            cached = predictions[meta['scene']]
            assert digest(ROOT/cached['path']) == cached['sha256']
            raw = np.load(ROOT/cached['path'], mmap_mode='r', allow_pickle=False)
            assert raw.shape == data['low'].shape and cached['checkpoint_sha256'] == anchor['sha256']
            for row in reference_rows:
                if row['scene'] != meta['scene']:
                    continue
                append(out/'per_sequence.jsonl', {**row, 'history': 'current' if row['method'] == 'raw' else 'post',
                       'reference_reused_from': str((reference/'per_sequence.jsonl').relative_to(ROOT)),
                       'current_branch_equals_frozen_anchor': row['method'] == 'raw'})
                count += 1
            for row in reference_frames:
                if row['scene'] == meta['scene']:
                    append(out/'per_frame.jsonl', {**row, 'reference_reused_from': 'R36'})
            inputs = torch.from_numpy((data['low']/255).transpose(0, 3, 1, 2).copy())
            for name, net in nets.items():
                net.reset()
                begin = time.perf_counter()
                with torch.inference_mode():
                    values = []
                    for frame in inputs:
                        value = net.step(frame[None])[0]
                        assert torch.isfinite(value).all() and torch.isfinite(net._state).all()
                        values.append(value)
                    predicted = torch.stack(values)
                    current_first, _ = net(inputs[:1], history='current')
                seconds = time.perf_counter()-begin
                output = predicted.clamp(0, 1).permute(0, 2, 3, 1).contiguous().numpy()*255
                current_first = current_first[0].clamp(0, 1).permute(1, 2, 0).contiguous().numpy()*255
                first_delta = float(np.abs(output[0]-raw[0]).max())
                current_delta = float(np.abs(current_first-raw[0]).max())
                assert first_delta <= 1e-4 and current_delta <= 1e-4
                identity = {'domain': meta['domain'], 'scene': meta['scene'], 'source_video': meta['source_video'],
                            'train_seed': seed, 'model': name, 'method': name, 'history': 'causal', 'input_reset': False}
                metrics, frames = evaluate(output, data['gt'], data['reference_flow'], data['reference_mask'])
                append(out/'per_sequence.jsonl', {**identity, 'online_history_storage_bytes': net.state_storage_bytes(),
                                                  'parameters': 4587, **metrics})
                for row in frames:
                    append(out/'per_frame.jsonl', {**identity, **row})
                append(out/'first_frame.jsonl', {**identity, 'first_frame_max_abs_0to255': first_delta,
                                                'current_first_max_abs_0to255': current_delta})
                append(out/'stage_cost.jsonl', {**identity, 'forward_seconds': seconds,
                       'frames': len(inputs), 'online_history_storage_bytes': net.state_storage_bytes(),
                       'learned_amplitude': float(net.amplitude.detach()),
                       'scope': 'CPU neural/state stage with finite checks; not an independently warmed complete pipeline or phone latency.'})
                neural += 1
                count += 1
                print(json.dumps({**identity, 'rows': count, **metrics}), flush=True)
        assert count == 36 and neural == 12 and not torch.cuda.is_initialized()
        complete = True
    except Exception:
        (out/'failure.txt').write_text(traceback.format_exc())
        raise
    finally:
        write(out/'runtime.json', {'completed_normally': complete, 'rows': count, 'neural_sequences': neural,
                                  'cuda_initialized': torch.cuda.is_initialized(), 'wall_seconds': time.perf_counter()-started})


if __name__ == '__main__':
    main()
