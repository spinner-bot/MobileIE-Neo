"""Common-CPU quality for the registered single-window known TLC spatial control."""
import argparse
import json
from pathlib import Path
import subprocess
import time
import traceback

import cv2
import numpy as np
import torch

from aggregation_controls_r20 import blend
from aligned_raw_history_r11 import align_history
from dynamic_training_r34 import BASE, ROOT, digest
from exploration_r2_data import evaluate
from half_motion_r4 import half_forward_backward
from prepare_cache_r5 import load_one
from tlc_pool_r42 import convert, load_anchor


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def append(path, value):
    with path.open('a') as stream:
        stream.write(json.dumps(value, allow_nan=False)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, choices=[0, 1], required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    torch.use_deterministic_algorithms(True)
    assert not torch.cuda.is_initialized()
    folder = BASE/f'results/exploration-r41-stage002-seed{args.seed}'
    gate = BASE/'process/exploration-r42/cpu-gate.json'
    assert json.loads(gate.read_text())['passed']
    reference = BASE/f'results/exploration-r41-stage002-eval-seed{args.seed}'
    prior = json.loads((reference/'runtime.json').read_text())
    assert prior['completed_normally'] and prior['rows'] == 60 and prior['neural_sequences'] == 20
    net, marker = load_anchor(args.seed)
    nets, markers = {'tlc192': convert(net)}, {'tlc192': marker}
    synthetic = BASE/'results/exploration-r30-cache/index.json'
    real = BASE/'results/exploration-r10-cache/index.json'
    metas = json.loads(synthetic.read_text())['scenes']
    metas += [{**r, 'domain': 'paired_real_ND_DEV', 'source_video': r['scene']}
              for r in json.loads(real.read_text())['scenes'] if r['domain'] == 'native_crop']
    assert len(metas) == 10
    out = BASE/f'results/exploration-r42-seed{args.seed}'
    cache = ROOT/f'.cache/experiments/exploration-r42/seed{args.seed}'
    assert cache.resolve().is_relative_to('/data/Users/zrh/data/MobileIE-Neo')
    out.mkdir(parents=True, exist_ok=False)
    cache.mkdir(parents=True, exist_ok=False)
    for model in nets:
        (cache/model).mkdir()
    write(out/'config.json', {'seed': args.seed, 'device': 'cpu', 'torch_threads': 2, 'opencv_threads': 1,
          'checkpoints': markers, 'deployment_parameters_each': 4075, 'torch_version': torch.__version__,
          'training_config_sha256': digest(folder/'config.json'),
          'fixed_TLC_window': 192, 'correctness_gate_sha256': digest(gate),
          'reference_rows_path': str((reference/'per_sequence.jsonl').relative_to(ROOT)),
          'reference_rows_sha256': digest(reference/'per_sequence.jsonl'),
          'cache_indices': {str(p.relative_to(ROOT)): digest(p) for p in [synthetic, real]},
          'protocol_sha256': digest(BASE/'docs/exploration-r42-tlc-protocol.md'),
          'source_sha256': {name: digest(Path(__file__).parent/name) for name in
                           ['evaluate_tlc_r42.py', 'tlc_pool_r42.py', 'half_motion_r4.py', 'aligned_raw_history_r11.py',
                            'aggregation_controls_r20.py', 'exploration_r2_data.py', 'video_metrics.py']},
          'revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
          'scope': 'Fixed known TLC spatial conversion of R41 expanded final4000; unchanged weights/4075 parameters, added local operations and memory; no raw history, new training, novelty or phone-cost claim.'})
    started, count, neural, complete = time.perf_counter(), 0, 0, False
    try:
        for meta in metas:
            data = load_one(meta)
            inputs = torch.from_numpy((data['low']/255).transpose(0, 3, 1, 2).copy())
            for model, net in nets.items():
                with torch.inference_mode():
                    value = torch.stack([net(image[None])[0] for image in inputs])
                    assert torch.isfinite(value).all()
                    raw = value.clamp(0, 1).permute(0, 2, 3, 1).contiguous().numpy()*255
                neural += 1
                identity = {'domain': meta['domain'], 'scene': meta['scene'], 'source_video': meta['source_video'],
                            'anchor_family': 'R42_TLC192', 'anchor': f'R42_TLC192_seed{args.seed}',
                            'base_family': 'R41_expanded_fp32',
                            'train_seed': args.seed, 'model': model}
                path = cache/model/(meta['scene']+'--raw.npy')
                np.save(path, raw, allow_pickle=False)
                append(out/'prediction_manifest.jsonl', {**identity, 'path': str(path.relative_to(ROOT)),
                       'sha256': digest(path), 'shape': list(raw.shape), 'dtype': str(raw.dtype),
                       'checkpoint_sha256': markers[model]['sha256']})
                finite, recursive = [raw[0]], [raw[0]]
                for t in range(1, len(raw)):
                    flow, mask = half_forward_backward(raw[t-1], raw[t])
                    finite.append(blend(raw[t], align_history(raw[t-1], raw[t], flow, mask)[0], .5))
                    recursive.append(blend(raw[t], align_history(recursive[-1], raw[t], flow, mask)[0], .5))
                for method, prediction in [('raw', raw), ('finite_self05', np.stack(finite)),
                                           ('recursive_self05', np.stack(recursive))]:
                    assert np.array_equal(prediction[0], raw[0])
                    metrics, frames = evaluate(prediction, data['gt'], data['reference_flow'], data['reference_mask'])
                    append(out/'per_sequence.jsonl', {**identity, 'method': method, **metrics})
                    for frame in frames:
                        append(out/'per_frame.jsonl', {**identity, 'method': method, **frame})
                    count += 1
                print(json.dumps({**identity, 'rows': count}), flush=True)
        assert count == 30 and neural == 10 and not torch.cuda.is_initialized()
        complete = True
    except Exception:
        (out/'failure.txt').write_text(traceback.format_exc())
        raise
    finally:
        write(out/'runtime.json', {'completed_normally': complete, 'rows': count, 'neural_sequences': neural,
              'cuda_initialized': torch.cuda.is_initialized(), 'wall_seconds': time.perf_counter()-started})


if __name__ == '__main__':
    main()
