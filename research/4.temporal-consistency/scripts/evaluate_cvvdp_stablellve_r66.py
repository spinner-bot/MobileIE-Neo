"""All80 final R65 comparisons only after its complete original evaluation chain."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

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
from evaluate_cvvdp_references_r66 import append, cached, lines, write
from prepare_cache_r5 import load_one
from training_form_r41 import configure_precision, cuda_headroom


def main():
    torch.set_num_threads(2); cv2.setNumThreads(1); torch.use_deterministic_algorithms(True)
    configure_precision(); assert not torch.cuda.is_initialized()
    finalizer = BASE/'process/exploration-r65/finalize001/runtime.json'
    assert json.loads(finalizer.read_text())['completed_normally']
    manifests, source_paths = {}, [finalizer]
    for seed in [0, 1]:
        train = BASE/f'results/exploration-r65-stage001-seed{seed}'
        run = BASE/f'results/exploration-r65-stage001-eval-seed{seed}'
        training, evaluation = json.loads((train/'runtime.json').read_text()), json.loads((run/'runtime.json').read_text())
        assert training['completed_normally'] and set(training['updates'].values()) == {8000}
        assert evaluation['completed_normally'] and evaluation['quality_rows'] == 40 and evaluation['frame_rows'] == 1792
        path = run/'prediction_manifest.jsonl'; rows = lines(path)
        assert len(rows) == 40 and len({(r['scene'], r['arm'], r['method']) for r in rows}) == 40
        manifests[seed] = rows
        source_paths += [train/'runtime.json', run/'runtime.json', path]
    gate_path = BASE/'process/exploration-r66/cpu-gate001/gate.json'
    gate = json.loads(gate_path.read_text()); assert gate['passed']
    out = BASE/'results/exploration-r66-stablellve-attempt001'; out.mkdir(exist_ok=False)
    cache = ROOT/'.cache/experiments/exploration-r66/stablellve/attempt001'
    assert cache.resolve().is_relative_to('/data/Users/zrh/data/MobileIE-Neo')
    cache.mkdir(parents=True, exist_ok=False)
    start, complete, count, frames = time.perf_counter(), False, 0, 0
    try:
        indices = [BASE/f'results/{name}/index.json' for name in ['exploration-r30-cache', 'exploration-r10-cache']]
        metas = json.loads(indices[0].read_text())['scenes']
        metas += [{**r, 'domain': 'paired_real_ND_DEV', 'source_video': r['scene']}
                  for r in json.loads(indices[1].read_text())['scenes'] if r['domain'] == 'native_crop']
        assert len(metas) == 10
        first = load_one(metas[0])
        admission = cuda_headroom(); write(out/'pre_cuda_guard.json', admission); assert admission['allowed']
        cuda_gate = qualify('cuda', gate['scores']); write(out/'cuda-gate.json', cuda_gate)
        metric, source = load_metric('cuda'); write(out/'metric-source.json', source)
        names = ['evaluate_cvvdp_stablellve_r66.py', 'evaluate_cvvdp_references_r66.py', 'cvvdp_adapter_r66.py', 'prepare_cache_r5.py']
        frozen = {name: digest(BASE/'scripts'/name) for name in names}
        write(out/'config.json', dict(at=datetime.now().astimezone().isoformat(), expected_rows=80,
              expected_comparison_frames=3584, unique_observations=448, enhancement_CNN=0, SGD=0,
              new_GPU_metric_job=1, CPU_gate_sha256=digest(gate_path), source_sha256=frozen,
              protocol_sha256=digest(BASE/'docs/exploration-r66-perceptual-metric-protocol.md'),
              completed_R65_sources={str(p.relative_to(ROOT)): digest(p) for p in source_paths},
              cache_indices={str(p.relative_to(ROOT)): digest(p) for p in indices},
              revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
              scope='All normal final8000 R65 outputs. No new enhancement inference/training, selection, human/phone/final conclusion.'))
        torch.cuda.reset_peak_memory_stats()
        for ordinal, meta in enumerate(metas):
            data = first if ordinal == 0 else load_one(meta); target = data['gt'].astype(np.float32)/255
            for seed in [0, 1]:
                entries = [r for r in manifests[seed] if r['scene'] == meta['scene']]; assert len(entries) == 4
                for entry in entries:
                    values = cached(entry); assert values.shape == data['gt'].shape
                    began = time.perf_counter(); score, statistics = predict(metric, np.asarray(values, dtype=np.float32)/255, target)
                    identity = dict(domain=meta['domain'], scene=meta['scene'], source_video=meta['source_video'],
                        family='StableLLVE_'+entry['arm'], seed=seed, method=entry['method'], parameters=4316259,
                        project_updates=8000, future_wait_max=0, frames=len(values))
                    append(out/'per_sequence.jsonl', dict(**identity, JOD=score, seconds=time.perf_counter()-began,
                        scope='Metric preparation/prediction only, not enhancer/phone latency; nominal display/time.'))
                    path = cache/f'{entry["arm"]}--seed{seed}--{meta["scene"]}--{entry["method"]}.npz'
                    np.savez_compressed(path, Q_per_ch=statistics['Q_per_ch'], rho_band=np.asarray(statistics['rho_band']))
                    append(out/'feature_manifest.jsonl', dict(**identity, source=entry, path=str(path.relative_to(ROOT)),
                        sha256=digest(path), Q_per_ch_shape=list(statistics['Q_per_ch'].shape),
                        note='Original intermediate features, not separately calibrated per-frame JOD/flicker.'))
                    count += 1; frames += len(values)
                    print(json.dumps(dict(**identity, JOD=score, completed_rows=count)), flush=True)
            del data, target
        assert (count, frames) == (80, 3584)
        assert all(digest(BASE/'scripts'/name) == sha for name, sha in frozen.items()); complete = True
    except BaseException:
        (out/'failure.txt').write_text(traceback.format_exc()); raise
    finally:
        write(out/'runtime.json', dict(completed_normally=complete, quality_rows=count, comparison_frames=frames,
              new_enhancement_CNN=0, new_SGD=0, seconds=time.perf_counter()-start,
              peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else None,
              ended=datetime.now().astimezone().isoformat()))


if __name__ == '__main__': main()
