"""All final R73 paired CPU outputs; both real-history and current replacements."""
import argparse
from datetime import datetime
import json
import time
import traceback

import cv2
import numpy as np
import torch

from native_moment_core_r73 import BASE, ROOT, MODES, PARAMETERS, NativeMomentCore, PROTOCOL
from check_native_moment_r73 import write
from dynamic_training_r34 import digest
from exploration_r2_data import evaluate
from prepare_cache_r5 import load_one
from render_dense_history_r64 import recursive
from train_exploration_r2 import new_model


def append(path, value):
    with path.open('a') as stream: stream.write(json.dumps(value, allow_nan=False)+'\n')


def metas():
    indices = [BASE/f'results/{name}/index.json' for name in ['exploration-r30-cache', 'exploration-r10-cache']]
    rows = json.loads(indices[0].read_text())['scenes']
    rows += [{**r, 'domain': 'paired_real_ND_DEV', 'source_video': r['scene']}
             for r in json.loads(indices[1].read_text())['scenes'] if r['domain'] == 'native_crop']
    assert len(rows) == 10
    return rows, indices


def load_final(seed):
    folder = BASE/f'results/exploration-r73-stage001-seed{seed}'
    terminal = json.loads((folder/'runtime.json').read_text())
    assert terminal['completed_normally'] and terminal['model_updates'] == {n: 4000 for n in MODES}
    nets, markers = {}, {}
    for name in MODES:
        marker = json.loads((folder/f'{name}-step4000.json').read_text())
        assert marker['new_step'] == 4000 and marker['inherited_updates'] == 8000
        assert marker['final_fold_passed'] and marker['final_fold_delta'] <= 1e-4
        path = ROOT/marker['path']; assert path.resolve().is_relative_to('/data/Users/zrh/data/MobileIE-Neo')
        assert digest(path) == marker['sha256']
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        assert checkpoint['model'] == name and checkpoint['seed'] == seed and checkpoint['new_step'] == 4000
        net = NativeMomentCore(new_model(torch.device('cpu')), name)
        net.load_state_dict(checkpoint['state_dict'], strict=True)
        assert sum(p.numel() for p in net.parameters()) == PARAMETERS[name]
        assert all(torch.isfinite(p).all() for p in net.parameters())
        nets[name], markers[name] = net.eval().requires_grad_(False), marker
    return nets, markers


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--seed', type=int, choices=[0, 1], required=True)
    args = parser.parse_args(); torch.set_num_threads(2); cv2.setNumThreads(1)
    torch.use_deterministic_algorithms(True); assert not torch.cuda.is_initialized()
    nets, markers = load_final(args.seed); scenes, indices = metas()
    out = BASE/f'results/exploration-r73-eval-seed{args.seed}'; out.mkdir(exist_ok=False)
    cache = ROOT/f'.cache/experiments/exploration-r73/eval001/seed{args.seed}'
    assert cache.resolve().is_relative_to('/data/Users/zrh/data/MobileIE-Neo')
    cache.mkdir(parents=True, exist_ok=False)
    write(out/'config.json', dict(at=datetime.now().astimezone().isoformat(), seed=args.seed, checkpoints=markers,
          protocol_sha256=digest(PROTOCOL), indices={str(p.relative_to(ROOT)): digest(p) for p in indices},
          expected_rows=100, expected_frame_rows=4480, expected_CNN_sequences=50, expected_CNN_centers=2240,
          scope='Final12000 paired reused DEV. Current replacements and own-output recursive references; no new final data or phone claim.'))
    start, complete, count, frames, neural, centers = time.perf_counter(), False, 0, 0, 0, 0
    try:
        for meta in scenes:
            data = load_one(meta); first = {}
            inputs = torch.from_numpy((data['low']/255).transpose(0, 3, 1, 2).copy())
            for name, route in [('spatial', 'current'), ('mean', 'causal'), ('mean', 'current'),
                                ('moment', 'causal'), ('moment', 'current')]:
                net = nets[name]; net.reset(); began = time.perf_counter(); values = []
                with torch.inference_mode():
                    for frame in inputs:
                        value = net.step(frame[None], history=route)
                        assert torch.isfinite(value).all()
                        assert all(v is None or torch.isfinite(v).all() for v in net._state)
                        values.append(value[0])
                    raw = torch.stack(values).clamp(0, 1).permute(0, 2, 3, 1).contiguous().numpy()*255
                neural += 1; centers += len(raw)
                identity = dict(domain=meta['domain'], scene=meta['scene'], source_video=meta['source_video'],
                                seed=args.seed, model=name, history=route, parameters=PARAMETERS[name], project_updates=12000)
                if route == 'causal': first[name] = raw[0].copy()
                elif name != 'spatial':
                    delta = float(np.abs(raw[0]-first[name]).max()); assert delta <= 1e-4
                    append(out/'first_frame.jsonl', dict(**identity, max_abs_delta_8bit=delta))
                append(out/'stage_cost.jsonl', dict(**identity, neural_seconds=time.perf_counter()-began,
                       frames=len(raw), state_bytes=net.state_storage_bytes() if route == 'causal' else 0,
                       scope='Descriptive whole neural stage, no isolated handset/full-pipeline speed.'))
                began = time.perf_counter(); rec = recursive(raw)
                append(out/'post_cost.jsonl', dict(**identity, seconds=time.perf_counter()-began, frames=len(raw)))
                for method, output in [('raw', raw), ('recursive_self05', rec)]:
                    assert np.isfinite(output).all() and np.array_equal(output[0], raw[0])
                    path = cache/f'{meta["scene"]}--{name}--{route}--{method}.npy'
                    np.save(path, output, allow_pickle=False)
                    append(out/'prediction_manifest.jsonl', dict(**identity, method=method, path=str(path.relative_to(ROOT)),
                           sha256=digest(path), shape=list(output.shape), checkpoint_sha256=markers[name]['sha256']))
                    metrics, per_frame = evaluate(output, data['gt'], data['reference_flow'], data['reference_mask'])
                    append(out/'per_sequence.jsonl', dict(**identity, method=method, **metrics))
                    for row in per_frame: append(out/'per_frame.jsonl', dict(**identity, method=method, **row))
                    count += 1; frames += len(per_frame)
                print(json.dumps(dict(**identity, quality_rows=count)), flush=True)
            del data, inputs
        assert (count, frames, neural, centers) == (100, 4480, 50, 2240)
        complete = True
    except BaseException:
        (out/'failure.txt').write_text(traceback.format_exc()); raise
    finally:
        write(out/'runtime.json', dict(completed_normally=complete, quality_rows=count, frame_rows=frames,
              CNN_sequences=neural, CNN_centers=centers, CUDA_initialized=torch.cuda.is_initialized(),
              seconds=time.perf_counter()-start, ended=datetime.now().astimezone().isoformat()))


if __name__ == '__main__': main()
