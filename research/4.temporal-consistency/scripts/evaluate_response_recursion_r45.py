"""Complete CPU evaluation of final controllers and the full fixed-alpha frontier."""
import argparse
from datetime import datetime
import json
import time
import traceback

import cv2
import numpy as np
import torch

from aligned_raw_history_r11 import align_history
from dynamic_training_r34 import BASE, ROOT, digest
from exploration_r2_data import evaluate
from half_motion_r4 import half_forward_backward
from prepare_cache_r5 import load_one
from response_cue_r45 import FrozenResponses, RecursiveControl, load_frozen

ALPHAS = {'fixed_01': .1, 'fixed_025': .25, 'fixed_05': .5, 'fixed_075': .75, 'fixed_09': .9}
METHODS = ['raw', *ALPHAS, 'scalar', 'rgb', 'response']


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def append(path, value):
    with path.open('a') as stream:
        stream.write(json.dumps(value, allow_nan=False)+'\n')


def predict(method, controls, raw, cues, fields):
    outputs = [raw[0]]
    histogram = np.zeros(20, dtype=np.int64)
    total, alpha_sum = 0, 0.
    previous = torch.from_numpy(raw[0].transpose(2, 0, 1).copy()/255)[None]
    for t, (flow, mask) in enumerate(fields, 1):
        if method in ALPHAS:
            moved, valid = align_history(outputs[-1], raw[t], flow, mask)
            outputs.append(raw[t]+ALPHAS[method]*(moved-raw[t]))
            values = np.full(int(valid.sum()), ALPHAS[method], dtype=np.float32)
        else:
            current = torch.from_numpy(raw[t].transpose(2, 0, 1).copy()/255)[None]
            field = torch.from_numpy(flow.transpose(2, 0, 1).copy())[None]
            accepted = torch.from_numpy(mask.copy())[None, None].bool()
            with torch.inference_mode():
                previous, alpha, valid = controls[method](current, previous, field, accepted,
                                                          torch.from_numpy(cues[t])[None])
            outputs.append(previous[0].permute(1, 2, 0).contiguous().numpy()*255)
            values = alpha[valid].numpy()
        histogram += np.histogram(values, bins=np.linspace(0, 1, 21))[0]
        total += len(values);alpha_sum += float(values.astype(np.float64).sum())
    stats = {'accepted_pixels': total, 'mean_accepted_alpha': alpha_sum/total if total else None,
             'alpha_histogram_20_bins_0_to_1': histogram.tolist()}
    return np.stack(outputs), stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, choices=[0, 1], required=True)
    args = parser.parse_args()
    torch.set_num_threads(2);cv2.setNumThreads(1)
    torch.use_deterministic_algorithms(True)
    assert not torch.cuda.is_initialized()
    training = BASE/f'results/exploration-r45-stage001-seed{args.seed}'
    runtime = json.loads((training/'runtime.json').read_text())
    assert runtime['completed_normally'] and runtime['total_new_controller_updates'] == 12000
    assert runtime['model_updates'] == {n: 4000 for n in ['scalar', 'rgb', 'response']}
    assert runtime['frozen_base_unchanged']
    backbone, base_marker = load_frozen(args.seed)
    encoder = FrozenResponses(backbone).eval()
    controls, markers = {}, {}
    for name in ['scalar', 'rgb', 'response']:
        marker = json.loads((training/f'{name}-step4000.json').read_text())
        assert marker['new_step'] == 4000 and marker['base_checkpoint_sha256'] == base_marker['sha256']
        assert digest(ROOT/marker['path']) == marker['sha256']
        saved = torch.load(ROOT/marker['path'], map_location='cpu', weights_only=False)
        assert saved['base_checkpoint_sha256'] == base_marker['sha256'] and saved['control'] == name
        control = RecursiveControl(name, args.seed).eval()
        control.load_state_dict(saved['state_dict'], strict=True)
        controls[name], markers[name] = control, marker
    synthetic, real = [BASE/f'results/{n}/index.json' for n in ['exploration-r30-cache', 'exploration-r10-cache']]
    metas = json.loads(synthetic.read_text())['scenes']
    metas += [{**r, 'domain': 'paired_real_ND_DEV', 'source_video': r['scene']}
              for r in json.loads(real.read_text())['scenes'] if r['domain'] == 'native_crop']
    assert len(metas) == 10
    old_manifest = BASE/f'results/exploration-r44-stage001-eval-seed{args.seed}/prediction_manifest.jsonl'
    originals = {r['scene']: r for r in map(json.loads, old_manifest.read_text().splitlines()) if r['model'] == 'square_mse'}
    assert len(originals) == 10
    out = BASE/f'results/exploration-r45-stage001-eval-seed{args.seed}'
    cache = ROOT/f'.cache/experiments/exploration-r45/stage001/seed{args.seed}'
    out.mkdir(parents=True, exist_ok=False);cache.mkdir(parents=True, exist_ok=False)
    assert cache.resolve().is_relative_to('/data/Users/zrh/data/MobileIE-Neo')
    names = ['evaluate_response_recursion_r45.py', 'response_cue_r45.py', 'budget_causal_r33.py',
             'half_motion_r4.py', 'aligned_raw_history_r11.py', 'aggregation_controls_r20.py',
             'exploration_r2_data.py', 'video_metrics.py']
    write(out/'config.json', {'started_at': datetime.now().astimezone().isoformat(), 'seed': args.seed,
          'device': 'cpu', 'torch_threads': 2, 'opencv_threads': 1, 'methods': METHODS,
          'checkpoints': markers, 'base_checkpoint': base_marker,
          'protocol_sha256': digest(BASE/'docs/exploration-r45-controlled-recursion-training-protocol.md'),
          'cache_indices': {str(p.relative_to(ROOT)): digest(p) for p in [synthetic, real]},
          'source_sha256': {n: digest(BASE/'scripts'/n) for n in names},
          'original_manifest_sha256': digest(old_manifest),
          'scope': 'Final4000 controllers on inherited frozen8000 spatial base. Shared raw/cues/own-raw flow for quality only, not deployment timings. Fixed alpha frontier, no phone or final test.'})
    started, count, neural, complete, replay_max = time.perf_counter(), 0, 0, False, 0.
    try:
        for meta in metas:
            data = load_one(meta)
            inputs = torch.from_numpy((data['low']/255).transpose(0, 3, 1, 2).copy())
            raw_frames, cue_frames = [], []
            with torch.inference_mode():
                for image in inputs:
                    value, cue = encoder(image[None])
                    assert bool(torch.isfinite(value).all()) and bool(torch.isfinite(cue).all())
                    raw_frames.append(value[0].clamp(0, 1).permute(1, 2, 0).contiguous().numpy()*255)
                    cue_frames.append(cue[0].contiguous().numpy())
            raw, cues = np.stack(raw_frames), np.stack(cue_frames)
            neural += 1
            previous = np.load(ROOT/originals[meta['scene']]['path'], mmap_mode='r', allow_pickle=False)
            delta = float(np.max(np.abs(raw-previous)))/255
            assert delta <= 1e-5
            replay_max = max(replay_max, delta)
            cue_path = cache/(meta['scene']+'--cues.npy')
            np.save(cue_path, cues, allow_pickle=False)
            fields = [half_forward_backward(raw[t-1], raw[t]) for t in range(1, len(raw))]
            identity = {'domain': meta['domain'], 'scene': meta['scene'], 'source_video': meta['source_video'],
                        'train_seed': args.seed, 'anchor_family': 'R45_frozen_square_mse'}
            append(out/'prediction_manifest.jsonl', {**identity, 'raw_source': originals[meta['scene']]['path'],
                   'raw_replay_max_abs_normalized': delta, 'cue_path': str(cue_path.relative_to(ROOT)),
                   'cue_sha256': digest(cue_path), 'cue_shape': list(cues.shape),
                   'scope': 'One new CNN sequence capturing internal cues; raw matches prior R44 cache, not independent observations.'})
            for method in METHODS:
                output, stats = (raw, None) if method == 'raw' else predict(method, controls, raw, cues, fields)
                assert np.array_equal(output[0], raw[0]) and np.isfinite(output).all()
                metrics, frames = evaluate(output, data['gt'], data['reference_flow'], data['reference_mask'])
                parameters = 4076 if method == 'scalar' else 4116 if method in ['rgb', 'response'] else 4075
                append(out/'per_sequence.jsonl', {**identity, 'method': method, 'parameters': parameters,
                       'history_storage_bytes': 0 if method == 'raw' else 2*raw[0].nbytes, **metrics})
                for frame in frames:
                    append(out/'per_frame.jsonl', {**identity, 'method': method, **frame})
                if stats is not None:
                    append(out/'fusion_stats.jsonl', {**identity, 'method': method, **stats})
                count += 1
            print(json.dumps({**identity, 'rows': count}), flush=True)
        assert count == 90 and neural == 10 and not torch.cuda.is_initialized()
        complete = True
    except Exception:
        (out/'failure.txt').write_text(traceback.format_exc());raise
    finally:
        write(out/'runtime.json', {'completed_normally': complete, 'rows': count, 'neural_sequences': neural,
              'raw_replay_max_abs_normalized': replay_max, 'cuda_initialized': torch.cuda.is_initialized(),
              'wall_seconds': time.perf_counter()-started})


if __name__ == '__main__':
    main()
