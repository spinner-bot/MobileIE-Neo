"""Complete fixed R36 controls on the strong R34 original backbone, CPU only."""
import json
from pathlib import Path
import subprocess
import time
import traceback

import cv2
import numpy as np
import torch

from classical_recursion_r36 import METHODS, Recursion
from dynamic_training_r34 import BASE, ROOT, digest
from exploration_r2_data import evaluate
from half_motion_r4 import half_forward_backward
from prepare_cache_r5 import load_one
from train_exploration_r2 import new_model


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def append(path, value):
    with path.open('a') as stream:
        stream.write(json.dumps(value, allow_nan=False)+'\n')


def main():
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    torch.use_deterministic_algorithms(True)
    assert not torch.cuda.is_initialized()
    out = BASE/'results/exploration-r36-classical'
    cache = ROOT/'.cache/experiments/exploration-r36'
    assert cache.resolve().is_relative_to('/data/Users/zrh/data/MobileIE-Neo')
    out.mkdir(parents=True, exist_ok=False)
    cache.mkdir(parents=True, exist_ok=False)
    index = BASE/'results/exploration-r30-cache/index.json'
    metas = json.loads(index.read_text())['scenes']
    assert len(metas) == 6
    nets, markers, references, reference_sources = {}, [], {}, {}
    for seed in [0, 1]:
        folder = BASE/f'results/exploration-r34-seed{seed}'
        runtime = json.loads((folder/'runtime.json').read_text())
        assert runtime['completed_normally'] and runtime['model_updates']['original'] == 4000
        marker = json.loads((folder/'original-step4000.json').read_text())
        assert digest(ROOT/marker['path']) == marker['sha256']
        saved = torch.load(ROOT/marker['path'], map_location='cpu', weights_only=False)
        assert saved['seed'] == seed and saved['step'] == 4000 and saved['model'] == 'original'
        net = new_model(torch.device('cpu'))
        net.load_state_dict(saved['state_dict'], strict=True)
        nets[seed] = net.eval()
        markers.append(marker)
        source = BASE/f'results/exploration-r34-eval-seed{seed}/per_sequence.jsonl'
        reference_sources[str(source.relative_to(ROOT))] = digest(source)
        for row in map(json.loads, source.read_text().splitlines()):
            if row['model'] == 'original' and row['history'] == 'current':
                references[(row['scene'], seed, row['fusion'])] = row
    sources = ['evaluate_classical_r36.py', 'classical_recursion_r36.py',
               'half_motion_r4.py', 'aligned_raw_history_r11.py', 'aggregation_controls_r20.py',
               'exploration_r2_data.py', 'video_metrics.py']
    write(out/'config.json', {'methods': METHODS, 'seeds': [0, 1], 'frames': 64,
          'device': 'cpu', 'torch_threads': 2, 'opencv_threads': 1, 'torch_version': torch.__version__,
          'checkpoints': markers, 'parameters': 4075, 'future_frames': 0,
          'cache_index_sha256': digest(index), 'reference_sources': reference_sources,
          'protocol_sha256': digest(BASE/'docs/exploration-r36-classical-recursion-protocol.md'),
          'source_sha256': {name: digest(Path(__file__).parent/name) for name in sources},
          'revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
          'scope': 'Three DEV sources; shared motion for fair quality only. Variance is approximate. Timings are shared stages, not standalone/phone speed.'})
    started, count, neural, complete = time.perf_counter(), 0, 0, False
    try:
        for meta in metas:
            data = load_one(meta)
            assert data['low'].shape[0] == 64
            for seed in [0, 1]:
                state = Recursion()
                outputs = {method: [] for method in METHODS}
                context, previous, timings = [], None, []
                net = nets[seed]
                with torch.inference_mode():
                    image = torch.from_numpy((data['low'][0]/255).transpose(2, 0, 1).copy())[None]
                    for _ in range(3):
                        net(image)
                    for t, frame in enumerate(data['low']):
                        begin = time.perf_counter()
                        image = torch.from_numpy((frame/255).transpose(2, 0, 1).copy())[None]
                        value = net(image)[0]
                        assert bool(torch.isfinite(value).all())
                        raw = value.clamp(0, 1).permute(1, 2, 0).contiguous().numpy()*255
                        cnn_end = time.perf_counter()
                        flow, mask = half_forward_backward(previous, raw) if previous is not None else (None, None)
                        motion_end = time.perf_counter()
                        result, stats = state.step(raw, previous, flow, mask)
                        previous = raw
                        post_end = time.perf_counter()
                        for method, output in result.items():
                            assert output.shape == raw.shape and np.isfinite(output).all()
                            if t == 0:
                                assert np.array_equal(output, raw)
                            outputs[method].append(output)
                        context.append({'frame': t, **stats})
                        timings.append({'cnn_layout_seconds': cnn_end-begin,
                                        'motion_seconds': motion_end-cnn_end,
                                        'all_three_controls_seconds': post_end-motion_end})
                neural += 1
                identity = {'scene': meta['scene'], 'source_video': meta['source_video'],
                            'domain': meta['domain'], 'train_seed': seed, 'model': 'original',
                            'weight_source': 'R34/final4000', 'input_reset': False}
                history = state.history_bytes(previous)
                raw_path = cache/f"{meta['scene']}-seed{seed}-raw.npy"
                for method, values in outputs.items():
                    output = np.stack(values)
                    if method == 'raw':
                        np.save(raw_path, output, allow_pickle=False)
                        append(out/'prediction_manifest.jsonl', {**identity, 'path': str(raw_path.relative_to(ROOT)),
                               'sha256': digest(raw_path), 'shape': list(output.shape), 'dtype': str(output.dtype),
                               'checkpoint_sha256': markers[seed]['sha256']})
                    metrics, frames = evaluate(output, data['gt'], data['reference_flow'], data['reference_mask'])
                    if method in ['raw', 'finite_self05']:
                        ref = references[(meta['scene'], seed, 'raw' if method == 'raw' else 'self_motion_post05')]
                        delta = {name: abs(metrics[name]-ref[name]) for name in
                                 ['psnr_rgb', 'ssim_rgb', 'temporal_error', 'edge_mae']}
                        assert max(delta.values()) <= 1e-6, delta
                        append(out/'reference_checks.jsonl', {**identity, 'method': method, 'metric_abs_delta': delta})
                    append(out/'per_sequence.jsonl', {**identity, 'method': method,
                           'online_history_storage_bytes': history[method], **metrics})
                    for row in frames:
                        append(out/'per_frame.jsonl', {**identity, 'method': method, **row})
                    count += 1
                    print(json.dumps({**identity, 'method': method, 'rows': count, **metrics}), flush=True)
                append(out/'stage_cost.jsonl', {**identity, 'per_frame': timings,
                       'note': 'One CNN and shared motion, all controls together. Not standalone pipeline timing.'})
                for row in context:
                    append(out/'variance_context.jsonl', {**identity, **row})
        assert count == 48 and neural == 12 and not torch.cuda.is_initialized()
        complete = True
    except Exception:
        (out/'failure.txt').write_text(traceback.format_exc())
        raise
    finally:
        write(out/'runtime.json', {'completed_normally': complete, 'rows': count,
                                  'neural_sequences': neural, 'cuda_initialized': torch.cuda.is_initialized(),
                                  'wall_seconds': time.perf_counter()-started})


if __name__ == '__main__':
    main()
