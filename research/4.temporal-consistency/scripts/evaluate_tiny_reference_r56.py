"""One final8000 adapted Nano seed, all paired DEV and full phone inputs, raw and classic."""
import argparse
import json
import subprocess
import time
import traceback
from datetime import datetime

import cv2
import numpy as np
import torch

from aligned_raw_history_r11 import align_history
from dynamic_training_r34 import BASE, ROOT, digest
from evaluate_phone_references_r38 import statistics
from exploration_r2_data import evaluate
from half_motion_r4 import half_forward_backward
from multinex_reference_r55 import prediction
from tiny_adapted_reference_r56 import load_adapted
from prepare_cache_r5 import load_one

METHODS = ['raw', 'recursive_self05']


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def append(path, value):
    with path.open('a') as stream:
        stream.write(json.dumps(value, allow_nan=False)+'\n')


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--seed", type=int, choices=[0, 1], required=True)
    args = parser.parse_args()
    torch.set_num_threads(2); cv2.setNumThreads(1); torch.use_deterministic_algorithms(True)
    assert not torch.cuda.is_initialized()
    gate = BASE/'process/exploration-r56/cpu-gate001/gate.json'
    assert json.loads(gate.read_text())['passed']
    out = BASE/f'results/exploration-r56-stage001-eval-seed{args.seed}'; out.mkdir(exist_ok=False)
    cache = ROOT/f'.cache/experiments/exploration-r56/stage001/seed{args.seed}'
    assert cache.resolve().is_relative_to('/data/Users/zrh/data/MobileIE-Neo')
    cache.mkdir(parents=True, exist_ok=False)
    start, complete, neural, neural_frames, quality_rows, phone_rows = time.perf_counter(), False, 0, 0, 0, 0
    try:
        model, source = load_adapted(args.seed)
        indices = [BASE/f'results/{name}/index.json' for name in
                   ['exploration-r30-cache', 'exploration-r10-cache', 'exploration-r49-full-phone-attempt001']]
        metas = json.loads(indices[0].read_text())['scenes']
        metas += [{**m, 'domain': 'paired_real_ND_DEV', 'source_video': m['scene']}
                  for m in json.loads(indices[1].read_text())['scenes'] if m['domain'] == 'native_crop']
        assert len(metas) == 10
        phones = json.loads(indices[2].read_text())['scenes']; assert [m['shape'][0] for m in phones] == [329, 327, 296]
        names = ['evaluate_tiny_reference_r56.py', 'tiny_adapted_reference_r56.py', 'multinex_reference_r55.py',
                 'aligned_raw_history_r11.py', 'half_motion_r4.py', 'prepare_cache_r5.py',
                 'evaluate_phone_references_r38.py', 'exploration_r2_data.py', 'video_metrics.py']
        write(out/'config.json', dict(at=datetime.now().astimezone().isoformat(), source=source, methods=METHODS,
              device='cpu', torch_threads=2, opencv_threads=1, adaptation_updates=8000, unique_checkpoints=1,
              CPU_gate_sha256=digest(gate), protocol_sha256=digest(BASE/'docs/exploration-r56-tiny-task-adaptation-protocol.md'),
              input_indices={str(p.relative_to(ROOT)): digest(p) for p in indices},
              source_sha256={n: digest(BASE/'scripts'/n) for n in names},
              revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
              scope='R56 fixed-budget spatial reference, unequal training parameterization/compute; not independent final, new method, full convergence or phone cost.'))
        for meta, paired in [(m, True) for m in metas]+[(m, False) for m in phones]:
            if paired:
                data = load_one(meta); low = data['low']
            else:
                assert digest(ROOT/meta['path']) == meta['sha256']
                with np.load(ROOT/meta['path'], allow_pickle=False) as archive: low = archive['low'].copy()
                assert list(low.shape) == meta['shape']
            identity = dict(domain=meta['domain'], scene=meta['scene'], source_video=meta.get('source_video', meta['scene']),
                            checkpoint=f'R56_final8000_seed{args.seed}', train_seed=args.seed, parameters=source['parameters'])
            raw_frames = []; began = time.perf_counter()
            with torch.inference_mode():
                for t, frame in enumerate(low):
                    image = torch.from_numpy(frame.transpose(2, 0, 1).copy()).float()[None]/255
                    value = prediction(model, image, clamp=False)[0]; assert torch.isfinite(value).all()
                    raw_frames.append(value.clamp(0, 1).permute(1, 2, 0).contiguous().numpy()*255)
                    neural_frames += 1
                    append(out/'preclip_statistics.jsonl', {**identity, 'frame': t,
                           'below_zero_fraction': float((value < 0).float().mean()),
                           'above_one_fraction': float((value > 1).float().mean())})
            raw = np.stack(raw_frames); del raw_frames
            append(out/'neural_stage_seconds.jsonl', {**identity, 'seconds': time.perf_counter()-began,
                   'scope': 'CPU2 neural stage plus preclip logging, excludes motion/metrics; not phone or independent production timing.'})
            neural += 1
            fields = [half_forward_backward(raw[t-1], raw[t]) for t in range(1, len(raw))]
            fused = [raw[0]]
            for t, (flow, mask) in enumerate(fields, 1):
                moved, _ = align_history(fused[-1], raw[t], flow, mask)
                fused.append(raw[t]+.5*(moved-raw[t]))
            recursive = np.stack(fused); del fused
            assert np.array_equal(recursive[0], raw[0]) and np.isfinite(recursive).all()
            if not paired:
                for t, frame in enumerate(low):
                    append(out/'phone_input_statistics.jsonl', {**identity, 'frame': t, **statistics(frame)})
            for method, output in [('raw', raw), ('recursive_self05', recursive)]:
                row_id = dict(**identity, method=method)
                if paired:
                    metrics, frames = evaluate(output, data['gt'], data['reference_flow'], data['reference_mask'])
                    append(out/'per_sequence.jsonl', {**row_id, **metrics})
                    for frame in frames: append(out/'per_frame.jsonl', {**row_id, **frame})
                    quality_rows += 1
                else:
                    rows = []
                    for t, frame in enumerate(output):
                        row = statistics(frame)
                        row['DC_luma_change'] = row['mean_luma_0to255']-rows[-1]['mean_luma_0to255'] if t else None
                        row['raw_flow_valid_fraction'] = row['shared_raw_flow_warp_luma_MAE_proxy'] = None
                        if t:
                            flow, mask = fields[t-1]
                            moved, valid = align_history(output[t-1], frame, flow, mask)
                            row['raw_flow_valid_fraction'] = float(valid.mean())
                            if valid.any():
                                delta = cv2.cvtColor((frame-moved).astype(np.float32), cv2.COLOR_RGB2GRAY)
                                row['shared_raw_flow_warp_luma_MAE_proxy'] = float(np.abs(delta)[valid].mean())
                        rows.append(row)
                        append(out/'phone_per_frame.jsonl', {**row_id, 'frame': t, **row})
                    means = {k: float(np.mean([r[k] for r in rows if r[k] is not None]))
                             if any(r[k] is not None for r in rows) else None for k in rows[0]}
                    append(out/'phone_per_sequence.jsonl', {**row_id, 'frames': len(output), **means,
                           'scope': 'Unpaired descriptive statistics only; raw-flow differs from other anchors; no quality/flicker rank.'})
                    phone_rows += 1
                path = cache/f"{'paired' if paired else 'phone'}--{meta['scene']}--{method}.npy"
                np.save(path, output, allow_pickle=False)
                append(out/'prediction_manifest.jsonl', {**row_id, 'paired': paired,
                       'path': str(path.relative_to(ROOT)), 'sha256': digest(path), 'shape': list(output.shape),
                       'bytes': path.stat().st_size, 'dtype': 'float32', 'input_cache': meta.get('path'),
                       'checkpoint_sha256': source['adapted_checkpoint']['sha256'],
                       'online_history_bytes': 0 if method == 'raw' else 2*raw[0].nbytes})
            print(json.dumps({**identity, 'paired': paired, 'quality_rows': quality_rows,
                              'phone_rows': phone_rows, 'neural_frames': neural_frames}), flush=True)
            del raw, recursive, output, low, fields
            if paired: del data
        assert quality_rows == 20 and phone_rows == 6 and neural == 13 and neural_frames == 1400
        assert not torch.cuda.is_initialized(); complete = True
    except BaseException:
        (out/'failure.txt').write_text(traceback.format_exc()); raise
    finally:
        write(out/'runtime.json', dict(completed_normally=complete, seed=args.seed, quality_rows=quality_rows, phone_rows=phone_rows,
              neural_sequences=neural, neural_frames=neural_frames, cuda_initialized=torch.cuda.is_initialized(),
              seconds=time.perf_counter()-start, ended=datetime.now().astimezone().isoformat()))


if __name__ == '__main__':
    main()
