"""Matched independent CPU pipelines for the strong classic reference."""
from collections import defaultdict
import json
from pathlib import Path
import subprocess
from statistics import mean, median
import time
import traceback

import cv2
import numpy as np
import torch

from aggregation_controls_r20 import blend
from aligned_raw_history_r11 import align_history
from dynamic_training_r34 import BASE, ROOT, digest
from half_motion_r4 import half_forward_backward
from train_exploration_r2 import new_model


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


@torch.inference_mode()
def pipeline(net, method, frames):
    previous_raw, previous_fused, outputs, tensors, stages = None, None, [], [], []
    started = time.perf_counter()
    for frame in frames:
        begin = time.perf_counter()
        image = torch.from_numpy((frame/255).transpose(2, 0, 1).copy())[None]
        value = net(image)[0]
        raw = value.clamp(0, 1).permute(1, 2, 0).contiguous().numpy()*255
        cnn_end = time.perf_counter()
        output, motion_end = raw, cnn_end
        if method != 'raw':
            if previous_raw is not None:
                flow, mask = half_forward_backward(previous_raw, raw)
                motion_end = time.perf_counter()
                history = previous_fused if method == 'recursive_self05' else previous_raw
                aligned, _ = align_history(history, raw, flow, mask)
                output = blend(raw, aligned, .5)
            previous_raw = raw
            if method == 'recursive_self05':
                previous_fused = output
        end = time.perf_counter()
        stages.append({'cnn_layout_seconds': cnn_end-begin, 'motion_seconds': motion_end-cnn_end,
                       'warp_blend_seconds': end-motion_end, 'pipeline_seconds': end-begin})
        outputs.append(output)
        tensors.append(value)
    seconds = time.perf_counter()-started
    assert all(bool(torch.isfinite(value).all()) for value in tensors)
    assert np.isfinite(np.stack(outputs)).all()
    buffers = [value for value in [previous_raw, previous_fused] if value is not None]
    assert all(value.flags.owndata and value.base is None for value in buffers)
    if method == 'recursive_self05':
        assert not np.shares_memory(previous_raw, previous_fused)
    return {'frames': len(frames), 'wall_seconds': seconds, 'per_frame': stages,
            'online_history_storage_bytes': sum(value.nbytes for value in buffers)}


def main():
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    torch.use_deterministic_algorithms(True)
    assert not torch.cuda.is_initialized()
    audit = json.loads((BASE/'report/exploration-r36-classical/audit.json').read_text())
    assert audit['passed'] and audit['rows'] == 48
    nets, markers = {}, []
    for seed in [0, 1]:
        marker = json.loads((BASE/f'results/exploration-r34-seed{seed}/original-step4000.json').read_text())
        assert digest(ROOT/marker['path']) == marker['sha256']
        saved = torch.load(ROOT/marker['path'], map_location='cpu', weights_only=False)
        net = new_model(torch.device('cpu'))
        net.load_state_dict(saved['state_dict'], strict=True)
        nets[seed] = net.eval()
        markers.append(marker)
    index = BASE/'results/exploration-r30-cache/index.json'
    meta = next(r for r in json.loads(index.read_text())['scenes'] if r['scene'] == 'bear_photons20')
    assert digest(ROOT/meta['path']) == meta['sha256']
    with np.load(ROOT/meta['path'], allow_pickle=False) as data:
        frames = data['low'][:16].copy()
    assert frames.shape == (16, 480, 854, 3)
    out = BASE/'results/exploration-r36-runtime'
    out.mkdir(parents=True, exist_ok=False)
    methods = ['raw', 'finite_self05', 'recursive_self05']
    write(out/'config.json', {'device': 'cpu', 'torch_threads': 2, 'opencv_threads': 1,
          'methods': methods, 'checkpoints': markers, 'workload': meta, 'frames': list(range(16)),
          'warmup_frames': 4, 'repetitions': 3, 'torch_version': torch.__version__,
          'protocol_sha256': digest(BASE/'docs/exploration-r36-runtime-protocol.md'),
          'source_sha256': {name: digest(Path(__file__).parent/name) for name in
                           ['benchmark_classical_r36.py', 'half_motion_r4.py', 'aligned_raw_history_r11.py', 'aggregation_controls_r20.py']},
          'revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
          'scope': 'Shared-server CPU, independent complete resident-input pipelines. No phone, energy, peak-memory or camera-to-display claim.'})
    cases = [(seed, method) for seed in [0, 1] for method in methods]
    rows, orders, complete, started = [], [], False, time.perf_counter()
    try:
        for repetition in range(3):
            shift = (5*repetition) % len(cases)
            order = cases[shift:]+cases[:shift]
            if repetition % 2:
                order = list(reversed(order))
            orders.append({'repetition': repetition, 'order': order})
            for position, (seed, method) in enumerate(order):
                pipeline(nets[seed], method, frames[:4])
                value = pipeline(nets[seed], method, frames)
                row = {'seed': seed, 'method': method, 'repetition': repetition, 'position': position, **value}
                with (out/'per_run.jsonl').open('a') as stream:
                    stream.write(json.dumps(row, allow_nan=False)+'\n')
                rows.append(row)
                print(json.dumps({'runs': len(rows), 'seed': seed, 'method': method,
                                  'wall_seconds': value['wall_seconds']}), flush=True)
        assert len(rows) == 18 and not torch.cuda.is_initialized()
        write(out/'execution_order.json', orders)
        complete = True
    except Exception:
        (out/'failure.txt').write_text(traceback.format_exc())
        raise
    finally:
        write(out/'runtime.json', {'completed_normally': complete, 'runs': len(rows),
                                  'cuda_initialized': torch.cuda.is_initialized(),
                                  'wall_seconds': time.perf_counter()-started})
    groups, summary = defaultdict(list), []
    for row in rows:
        groups[row['method']].append(row)
    for method in methods:
        values = groups[method]
        assert len(values) == 6 and len({r['online_history_storage_bytes'] for r in values}) == 1
        times = [r['wall_seconds']/r['frames']*1000 for r in values]
        summary.append({'method': method, 'median_pipeline_ms_per_frame': median(times),
                        'mean_pipeline_ms_per_frame': mean(times), 'min_pipeline_ms_per_frame': min(times),
                        'max_pipeline_ms_per_frame': max(times),
                        'online_history_storage_bytes': values[0]['online_history_storage_bytes'],
                        'per_seed_median_ms': {seed: median(r['wall_seconds']/r['frames']*1000 for r in values if r['seed'] == seed)
                                               for seed in [0, 1]}})
    report = BASE/'report/exploration-r36-runtime'
    report.mkdir(parents=True, exist_ok=False)
    write(report/'cost.json', summary)
    write(report/'audit.json', {'passed': True, 'runs': 18,
          'source_sha256': {str((out/name).relative_to(ROOT)): digest(out/name) for name in
                           ['config.json', 'per_run.jsonl', 'runtime.json', 'execution_order.json']},
          'scope': 'Median of six run-average frame costs per method; ranges are not confidence intervals, no phone measurement.'})
    lines = ['# R36 classic standalone CPU pipelines', '',
             'Same 480×854 bear_photons20 first16 workload, two Torch/one OpenCV threads, two seeds × three rotated/reversed rounds. Median of six run-average frame costs; min–max is not a confidence interval. Current server load, not phone/energy/peak-memory evidence.', '',
             '|Method|Median ms/frame|Min–max|History MiB|', '|---|---:|---:|---:|']
    for row in summary:
        lines.append(f"|{row['method']}|{row['median_pipeline_ms_per_frame']:.3f}|{row['min_pipeline_ms_per_frame']:.3f}–{row['max_pipeline_ms_per_frame']:.3f}|{row['online_history_storage_bytes']/2**20:.3f}|")
    lines += ['', 'Raw/finite were remeasured in this same round; do not mix R34 times with this table. Each pipeline includes resident tensor preparation, enhancement/layout, its own required motion/warp/blend and history. Output references remain in the loop; final finite checks/stack/IO/model loading are excluded. Recursive retains two independent RGB allocations; no shared extra teacher CNN.']
    (report/'tables.md').write_text('\n'.join(lines)+'\n')


if __name__ == '__main__':
    main()
