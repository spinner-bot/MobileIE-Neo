"""All140 cached reference comparisons, original ColorVideoVDP, zero new enhancement CNN."""
import json
import subprocess
import time
import traceback
from datetime import datetime

import cv2
import numpy as np
import torch

from cvvdp_adapter_r66 import BASE, ROOT, load_metric, predict, qualify
from dynamic_training_r34 import digest
from prepare_cache_r5 import load_one
from render_dense_history_r64 import recursive
from training_form_r41 import configure_precision, cuda_headroom


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def append(path, value):
    with path.open('a') as stream: stream.write(json.dumps(value, allow_nan=False)+'\n')


def lines(path):
    return [json.loads(row) for row in path.read_text().splitlines()]


def cached(entry):
    path = ROOT/entry['path']
    assert path.resolve().is_relative_to('/data/Users/zrh/data/MobileIE-Neo') and digest(path) == entry['sha256']
    values = np.load(path, mmap_mode='r', allow_pickle=False)
    assert list(values.shape) == entry['shape']
    return values


def reference_manifests():
    records, sources = {}, {}
    for seed in [0, 1]:
        for label, stage in [('MobileIE', 44), ('Nano', 56), ('FastDVDnet', 53)]:
            run = BASE/f'results/exploration-r{stage}-stage001-eval-seed{seed}'
            assert json.loads((run/'runtime.json').read_text())['completed_normally']
            path = run/'prediction_manifest.jsonl'; rows = lines(path)
            if label == 'MobileIE': rows = [r for r in rows if r['model'] == 'square_mse']
            if label == 'Nano': rows = [r for r in rows if r['paired']]
            assert len(rows) == (20 if label == 'Nano' else 10)
            records[label, seed] = rows; sources[str(path.relative_to(ROOT))] = digest(path)
    run = BASE/'results/exploration-r62-stablellve-common-attempt001'
    assert json.loads((run/'runtime.json').read_text())['completed_normally']
    path = run/'prediction_manifest.jsonl'; records['StableLLVE_clean', None] = lines(path)
    assert len(records['StableLLVE_clean', None]) == 20
    sources[str(path.relative_to(ROOT))] = digest(path)
    return records, sources


def conditions(meta, data, manifests):
    yield 'Input', None, 'input', data['low'], dict(input_cache=meta), 0, 0, 0
    yield 'GT', None, 'identity', data['gt'], dict(input_cache=meta), 0, 0, 0
    for label, parameters in [('MobileIE', 4075), ('Nano', 696), ('FastDVDnet', 2479096), ('StableLLVE_clean', 4316259)]:
        for seed in ([None] if label == 'StableLLVE_clean' else [0, 1]):
            entries = [r for r in manifests[label, seed] if r['scene'] == meta['scene']]
            assert len(entries) == (2 if label in ('Nano', 'StableLLVE_clean') else 1)
            for entry in entries:
                values = cached(entry)
                method = 'raw' if label == 'MobileIE' else entry['method']
                updates = 0 if label == 'StableLLVE_clean' else 8000
                yield label, seed, method, values, entry, parameters, updates, 2 if label == 'FastDVDnet' else 0
                if label == 'MobileIE':
                    yield label, seed, 'recursive_self05', recursive(values), dict(**entry,
                        recomputed='Original raw half-FB/warp recursive .5; no new CNN'), parameters, updates, 0


def main():
    torch.set_num_threads(2); cv2.setNumThreads(1); torch.use_deterministic_algorithms(True)
    configure_precision(); assert not torch.cuda.is_initialized()
    gate_path = BASE/'process/exploration-r66/cpu-gate001/gate.json'
    gate = json.loads(gate_path.read_text()); assert gate['passed']
    out = BASE/'results/exploration-r66-reference-attempt001'; out.mkdir(exist_ok=False)
    cache = ROOT/'.cache/experiments/exploration-r66/reference/attempt001'
    assert cache.resolve().is_relative_to('/data/Users/zrh/data/MobileIE-Neo')
    cache.mkdir(parents=True, exist_ok=False)
    start, complete, rows, frames = time.perf_counter(), False, 0, 0
    try:
        indices = [BASE/f'results/{name}/index.json' for name in ['exploration-r30-cache', 'exploration-r10-cache']]
        metas = json.loads(indices[0].read_text())['scenes']
        metas += [{**r, 'domain': 'paired_real_ND_DEV', 'source_video': r['scene']}
                  for r in json.loads(indices[1].read_text())['scenes'] if r['domain'] == 'native_crop']
        assert len(metas) == 10
        manifests, sources = reference_manifests()
        first = load_one(metas[0])  # Actual CPU source preparation before fresh internal GPU check.
        admission = cuda_headroom(); write(out/'pre_cuda_guard.json', admission); assert admission['allowed']
        cuda_gate = qualify('cuda', gate['scores']); write(out/'cuda-gate.json', cuda_gate)
        metric, source = load_metric('cuda'); write(out/'metric-source.json', source)
        names = ['evaluate_cvvdp_references_r66.py', 'cvvdp_adapter_r66.py', 'render_dense_history_r64.py',
                 'half_motion_r4.py', 'aligned_raw_history_r11.py', 'aggregation_controls_r20.py', 'prepare_cache_r5.py']
        frozen = {name: digest(BASE/'scripts'/name) for name in names}
        write(out/'config.json', dict(at=datetime.now().astimezone().isoformat(), expected_rows=140,
              expected_comparison_frames=6272, unique_observations=448, enhancement_CNN=0, SGD=0,
              new_GPU_metric_job=1, CPU_gate_sha256=digest(gate_path), source_sha256=frozen,
              protocol_sha256=digest(BASE/'docs/exploration-r66-perceptual-metric-protocol.md'),
              source_manifests=sources, cache_indices={str(p.relative_to(ROOT)): digest(p) for p in indices},
              revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
              scope='All cached common references under nominal FHD/30fps. No R65/five_current arrays, new enhancement inference, human/phone/final conclusion.'))
        torch.cuda.reset_peak_memory_stats()
        for ordinal, meta in enumerate(metas):
            data = first if ordinal == 0 else load_one(meta)
            target = data['gt'].astype(np.float32)/255
            count = 0
            for family, seed, method, values, entry, parameters, updates, future in conditions(meta, data, manifests):
                assert values.shape == data['gt'].shape
                began = time.perf_counter()
                score, statistics = predict(metric, np.asarray(values, dtype=np.float32)/255, target)
                identity = dict(domain=meta['domain'], scene=meta['scene'], source_video=meta['source_video'],
                    family=family, seed=seed, method=method, parameters=parameters, project_updates=updates,
                    future_wait_max=future, frames=len(values))
                append(out/'per_sequence.jsonl', dict(**identity, JOD=score, seconds=time.perf_counter()-began,
                    scope='Metric preparation/prediction only, not enhancer/phone latency; nominal display/time.'))
                features = cache/f'{family}--seed{seed}--{meta["scene"]}--{method}.npz'
                np.savez_compressed(features, Q_per_ch=statistics['Q_per_ch'], rho_band=np.asarray(statistics['rho_band']))
                append(out/'feature_manifest.jsonl', dict(**identity, source=entry,
                    path=str(features.relative_to(ROOT)), sha256=digest(features),
                    Q_per_ch_shape=list(statistics['Q_per_ch'].shape),
                    note='Original intermediate perceptual features; not separately calibrated per-frame JOD/flicker.'))
                rows += 1; frames += len(values); count += 1
                print(json.dumps(dict(**identity, JOD=score, completed_rows=rows)), flush=True)
            assert count == 14
            del data, target
        assert (rows, frames) == (140, 6272)
        assert all(digest(BASE/'scripts'/name) == sha for name, sha in frozen.items()); complete = True
    except BaseException:
        (out/'failure.txt').write_text(traceback.format_exc()); raise
    finally:
        write(out/'runtime.json', dict(completed_normally=complete, quality_rows=rows, comparison_frames=frames,
              new_enhancement_CNN=0, new_SGD=0, seconds=time.perf_counter()-start,
              peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else None,
              ended=datetime.now().astimezone().isoformat()))


if __name__ == '__main__': main()
