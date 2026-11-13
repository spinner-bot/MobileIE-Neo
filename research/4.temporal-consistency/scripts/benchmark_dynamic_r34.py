"""Matched final-weight CPU pipelines on a fixed resident-input DEV workload."""
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
from dynamic_training_r34 import BASE, ROOT, digest, models
from half_motion_r4 import half_forward_backward


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


@torch.inference_mode()
def pipeline(net, model, history, fusion, frames):
    if hasattr(net, 'reset'):
        net.reset()
    previous, outputs, unclipped, stages = None, [], [], []
    started = time.perf_counter()
    for frame in frames:
        begin = time.perf_counter()
        image = torch.from_numpy((frame/255).transpose(2, 0, 1).copy())[None]
        if model == 'original':
            value = net(image)[0]
        elif history == 'causal':
            value = net.step(image)[0]
        else:
            value = net(image, history='current')[0][0]
        raw = (value.clamp(0, 1).permute(1, 2, 0).contiguous().numpy()*255)
        assert raw.flags.owndata and raw.flags.c_contiguous
        forward_end = time.perf_counter()
        output, motion_end = raw, forward_end
        if fusion == 'self_motion_post05' and previous is not None:
            flow, mask = half_forward_backward(previous, raw)
            motion_end = time.perf_counter()
            aligned, _ = align_history(previous, raw, flow, mask)
            output = blend(raw, aligned, .5)
        if fusion == 'self_motion_post05':
            previous = raw  # Standalone allocation, no extra history copy/view.
        end = time.perf_counter()
        stages.append({'forward_layout_seconds': forward_end-begin, 'motion_seconds': motion_end-forward_end,
                       'warp_blend_seconds': end-motion_end, 'pipeline_seconds': end-begin})
        outputs.append(output)
        unclipped.append(value)
    wall_seconds = time.perf_counter()-started
    assert all(bool(torch.isfinite(value).all()) for value in unclipped)
    output = np.stack(outputs)
    assert output.shape == frames.shape and np.isfinite(output).all()
    history_bytes = (net.state_storage_bytes() if history == 'causal' else
                     previous.nbytes if previous is not None else 0)
    return {'frames': len(frames), 'wall_seconds': wall_seconds, 'per_frame': stages,
            'online_history_storage_bytes': history_bytes, 'output_shape': list(output.shape),
            'finite_before_and_after_clipping': True}


def main():
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    torch.use_deterministic_algorithms(True)
    assert not torch.cuda.is_initialized()
    nets, markers = {}, []
    for seed in [0, 1]:
        folder = BASE/f'results/exploration-r34-seed{seed}'
        runtime = json.loads((folder/'runtime.json').read_text())
        assert runtime['completed_normally'] and runtime['total_updates'] == 12000
        assert set(runtime['model_updates'].values()) == {4000}
        group, _ = models(seed, torch.device('cpu'))
        for name, net in group.items():
            marker = json.loads((folder/f'{name}-step4000.json').read_text())
            assert digest(ROOT/marker['path']) == marker['sha256']
            saved = torch.load(ROOT/marker['path'], map_location='cpu', weights_only=False)
            assert saved['step'] == 4000 and saved['seed'] == seed and saved['model'] == name
            net.load_state_dict(saved['state_dict'], strict=True)
            nets[(seed, name)] = net.eval()
            markers.append(marker)
    index_path = BASE/'results/exploration-r30-cache/index.json'
    workload = next(r for r in json.loads(index_path.read_text())['scenes'] if r['scene'] == 'bear_photons20')
    assert digest(ROOT/workload['path']) == workload['sha256']
    with np.load(ROOT/workload['path'], allow_pickle=False) as data:
        frames = data['low'][:16].copy()
    assert frames.shape == (16, 480, 854, 3)
    cases = [(seed, name, history, fusion) for seed in [0, 1] for name, history, fusion in [
             ('original', 'current', 'raw'), ('original', 'current', 'self_motion_post05'),
             ('spatial', 'current', 'raw'), ('spatial', 'current', 'self_motion_post05'),
             ('causal', 'current', 'raw'), ('causal', 'current', 'self_motion_post05'),
             ('causal', 'causal', 'raw')]]
    out = BASE/'results/exploration-r34-runtime'
    out.mkdir(parents=True, exist_ok=False)
    write(out/'config.json', {'device': 'cpu', 'torch_threads': 2, 'opencv_threads': 1, 'dtype': 'float32',
          'torch_version': torch.__version__, 'workload': workload, 'workload_frames': list(range(16)),
          'repetitions': 3, 'warmup_frames_per_case_per_round': 4, 'checkpoints': markers,
          'parameters': {f'{seed}/{name}': sum(p.numel() for p in net.parameters()) for (seed, name), net in nets.items()},
          'protocol_sha256': digest(BASE/'docs/exploration-r34-runtime-protocol.md'),
          'source_sha256': {name: digest(Path(__file__).parent/name) for name in
                           ['benchmark_dynamic_r34.py', 'budget_causal_r33.py', 'dynamic_training_r34.py', 'half_motion_r4.py', 'aligned_raw_history_r11.py']},
          'revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
          'scope': 'Shared-server CPU, resident input, rotated/reversed order. Loop includes output reference retention; final finite checks/stack/IO excluded. Historical state is not full peak memory. No phone, energy or camera-to-display claim.'})
    started, count, complete, orders = time.perf_counter(), 0, False, []
    try:
        for repetition in range(3):
            shift = (5*repetition) % len(cases)
            order = cases[shift:]+cases[:shift]
            if repetition % 2:
                order = list(reversed(order))
            orders.append({'repetition': repetition, 'order': order})
            for position, (seed, model, history, fusion) in enumerate(order):
                net = nets[(seed, model)]
                pipeline(net, model, history, fusion, frames[:4])
                row = pipeline(net, model, history, fusion, frames)
                identity = {'seed': seed, 'model': model, 'history': history, 'fusion': fusion,
                            'repetition': repetition, 'order_position': position}
                with (out/'per_run.jsonl').open('a') as stream:
                    stream.write(json.dumps({**identity, **row}, allow_nan=False)+'\n')
                count += 1
                print(json.dumps({**identity, 'completed_runs': count, 'wall_seconds': row['wall_seconds']}), flush=True)
        assert count == 42 and not torch.cuda.is_initialized()
        write(out/'execution_order.json', orders)
        complete = True
    except Exception:
        (out/'failure.txt').write_text(traceback.format_exc())
        raise
    finally:
        write(out/'runtime.json', {'completed_normally': complete, 'runs': count,
                                  'cuda_initialized': torch.cuda.is_initialized(), 'wall_seconds': time.perf_counter()-started})


if __name__ == '__main__':
    main()
