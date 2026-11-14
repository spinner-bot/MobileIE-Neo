"""Complete CPU developer evaluation of fixed-weight training-only BN controls."""
import argparse
import json
import time
import traceback
from datetime import datetime

import cv2
import numpy as np
import torch

from aggregation_controls_r20 import blend
from aligned_raw_history_r11 import align_history
from bn_calibration_r50 import MODELS
from dynamic_training_r34 import BASE, ROOT, digest
from exploration_r2_data import evaluate
from half_motion_r4 import half_forward_backward
from prepare_cache_r5 import load_one
from train_exploration_r2 import new_model


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def append(path, value):
    with path.open('a') as f: f.write(json.dumps(value, allow_nan=False)+'\n')


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--seed', type=int, choices=[0, 1], required=True)
    args = parser.parse_args()
    torch.set_num_threads(2); cv2.setNumThreads(1)
    torch.use_deterministic_algorithms(True)
    assert not torch.cuda.is_initialized()
    folder = BASE/'results/exploration-r50-stage001-calibration'
    runtime = json.loads((folder/'runtime.json').read_text())
    assert runtime['completed_normally'] and runtime['models_completed'] == 6 and runtime['SGD_updates'] == 0
    nets, markers = {}, {}
    for name in MODELS:
        marker = json.loads((folder/f'seed{args.seed}-{name}.json').read_text())
        assert marker['windows'] == 128 and marker['fold_max_abs_delta'] <= 1e-4
        assert marker['check']['learned_parameters_unchanged'] and marker['SGD_updates'] == 0
        assert digest(ROOT/marker['path']) == marker['sha256']
        saved = torch.load(ROOT/marker['path'], map_location='cpu', weights_only=False)
        assert saved['seed'] == args.seed and saved['model'] == name and saved['SGD_updates'] == 0
        net = new_model(torch.device('cpu')); net.load_state_dict(saved['state_dict'], strict=True)
        assert sum(p.numel() for p in net.parameters()) == 4075
        nets[name], markers[name] = net.eval(), marker
    indices = [BASE/f'results/{n}/index.json' for n in ['exploration-r30-cache', 'exploration-r10-cache']]
    metas = json.loads(indices[0].read_text())['scenes']
    metas += [{**r, 'domain': 'paired_real_ND_DEV', 'source_video': r['scene']}
              for r in json.loads(indices[1].read_text())['scenes'] if r['domain'] == 'native_crop']
    assert len(metas) == 10
    out = BASE/f'results/exploration-r50-stage001-eval-seed{args.seed}'
    cache = ROOT/f'.cache/experiments/exploration-r50/stage001/seed{args.seed}'
    assert cache.resolve().is_relative_to('/data/Users/zrh/data/MobileIE-Neo')
    out.mkdir(exist_ok=False); cache.mkdir(parents=True, exist_ok=False)
    for name in MODELS: (cache/name).mkdir()
    names = ['evaluate_bn_calibration_r50.py', 'bn_calibration_r50.py', 'half_motion_r4.py',
             'aligned_raw_history_r11.py', 'aggregation_controls_r20.py', 'exploration_r2_data.py', 'video_metrics.py']
    write(out/'config.json', {'at': datetime.now().astimezone().isoformat(), 'seed': args.seed,
          'device': 'cpu', 'torch_threads': 2, 'opencv_threads': 1, 'checkpoints': markers,
          'methods': ['raw', 'recursive_self05'], 'neural_sequences_expected': 30,
          'protocol_sha256': digest(BASE/'docs/exploration-r50-bn-calibration-protocol.md'),
          'source_sha256': {n: digest(BASE/'scripts'/n) for n in names},
          'cache_indices': {str(p.relative_to(ROOT)): digest(p) for p in indices},
          'scope': 'Training-only BN buffers changed; no corrected output, SGD, DEV adaptation or new temporal method.'})
    start, complete, rows, neural, frame_count = time.perf_counter(), False, 0, 0, 0
    try:
        for meta in metas:
            data = load_one(meta)
            inputs = torch.from_numpy((data['low']/255).transpose(0, 3, 1, 2).copy())
            for name, net in nets.items():
                with torch.inference_mode():
                    value = torch.stack([net(image[None])[0] for image in inputs])
                    assert torch.isfinite(value).all()
                    raw = value.clamp(0, 1).permute(0, 2, 3, 1).contiguous().numpy()*255
                neural += 1
                identity = {'domain': meta['domain'], 'scene': meta['scene'], 'source_video': meta['source_video'],
                            'train_seed': args.seed, 'model': name, 'parameters': 4075}
                path = cache/name/(meta['scene']+'--raw.npy'); np.save(path, raw, allow_pickle=False)
                append(out/'prediction_manifest.jsonl', {**identity, 'path': str(path.relative_to(ROOT)),
                       'sha256': digest(path), 'shape': list(raw.shape), 'checkpoint_sha256': markers[name]['sha256']})
                for t, (image, gt) in enumerate(zip(raw, data['gt'])):
                    error = image.astype(np.float64)-gt.astype(np.float64)
                    bias = error.mean(axis=(0, 1)); mse = float(np.mean(error**2))
                    dc = float(np.mean(bias**2)); centered = float(np.mean((error-bias)**2))
                    assert abs(mse-dc-centered) <= 1e-10*max(1., mse)
                    append(out/'raw_bias_per_frame.jsonl', {**identity, 'frame': t, 'MSE': mse,
                           'DC_MSE': dc, 'centered_MSE': centered, 'mean_RGB_error_0_to_255': bias.tolist()})
                recursive, accepted = [raw[0]], []
                for t in range(1, len(raw)):
                    flow, mask = half_forward_backward(raw[t-1], raw[t]); accepted.append(float(mask.mean()))
                    recursive.append(blend(raw[t], align_history(recursive[-1], raw[t], flow, mask)[0], .5))
                append(out/'motion_acceptance.jsonl', {**identity, 'pair_fractions': accepted,
                       'mean_accepted_fraction': float(np.mean(accepted))})
                for method, output in [('raw', raw), ('recursive_self05', np.stack(recursive))]:
                    assert np.array_equal(output[0], raw[0])
                    metrics, frames = evaluate(output, data['gt'], data['reference_flow'], data['reference_mask'])
                    append(out/'per_sequence.jsonl', {**identity, 'method': method, **metrics})
                    for frame in frames: append(out/'per_frame.jsonl', {**identity, 'method': method, **frame})
                    rows += 1; frame_count += len(frames)
                print(json.dumps({**identity, 'rows': rows, 'CNN_sequences': neural}), flush=True)
        assert rows == 60 and neural == 30 and frame_count == 2688 and not torch.cuda.is_initialized()
        complete = True
    except BaseException:
        (out/'failure.txt').write_text(traceback.format_exc()); raise
    finally:
        write(out/'runtime.json', {'completed_normally': complete, 'rows': rows, 'neural_sequences': neural,
              'per_frame_rows': frame_count, 'cuda_initialized': torch.cuda.is_initialized(),
              'seconds': time.perf_counter()-start, 'ended': datetime.now().astimezone().isoformat()})


if __name__ == '__main__':
    main()
