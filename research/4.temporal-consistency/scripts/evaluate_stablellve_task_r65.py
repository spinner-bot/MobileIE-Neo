"""Both final published-loss arms on unchanged paired DEV, raw and recursive .5."""
import argparse
import json
import subprocess
import time
import traceback
from datetime import datetime

import cv2
import numpy as np
import torch

from check_stablellve_task_r65 import write
from drv_references_r59 import postprocess
from dynamic_training_r34 import BASE, ROOT, digest
from prepare_cache_r5 import load_one
from exploration_r2_data import evaluate
from stablellve_adapter_r15 import infer, input_tensor, output_rgb
from stablellve_adapted_r65 import load_adapted
from stablellve_task_r65 import ARMS


def append(path, value):
    with path.open('a') as stream: stream.write(json.dumps(value, allow_nan=False)+'\n')


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--seed', type=int, choices=[0, 1], required=True)
    args = parser.parse_args()
    torch.set_num_threads(2); cv2.setNumThreads(1); torch.use_deterministic_algorithms(True)
    assert not torch.cuda.is_initialized()
    indices = [BASE/f'results/{name}/index.json' for name in ['exploration-r30-cache', 'exploration-r10-cache']]
    metas = json.loads(indices[0].read_text())['scenes']
    metas += [{**r, 'domain': 'paired_real_ND_DEV', 'source_video': r['scene']}
              for r in json.loads(indices[1].read_text())['scenes'] if r['domain'] == 'native_crop']
    assert len(metas) == 10
    out = BASE/f'results/exploration-r65-stage001-eval-seed{args.seed}'; out.mkdir(exist_ok=False)
    cache = ROOT/f'.cache/experiments/exploration-r65/stage001/seed{args.seed}'
    assert cache.resolve().is_relative_to('/data/Users/zrh/data/MobileIE-Neo')
    cache.mkdir(parents=True, exist_ok=False)
    names = ['evaluate_stablellve_task_r65.py', 'stablellve_adapted_r65.py', 'stablellve_task_r65.py',
             'stablellve_adapter_r15.py', 'drv_references_r59.py', 'half_motion_r4.py',
             'aligned_raw_history_r11.py', 'prepare_cache_r5.py', 'exploration_r2_data.py', 'video_metrics.py']
    frozen = {name: digest(BASE/'scripts'/name) for name in names}
    write(out/'config.json', dict(at=datetime.now().astimezone().isoformat(), seed=args.seed, arms=ARMS,
          methods=['raw', 'recursive_self05'], parameters_per_arm=4316259, project_updates_per_arm=8000,
          device='cpu', torch_threads=2, opencv_threads=1, expected_quality_rows=40, expected_frame_rows=1792,
          expected_CNN_sequences=20, expected_CNN_centers=896, future_wait_max=0,
          protocol_sha256=digest(BASE/'docs/exploration-r65-stablellve-task-adaptation-protocol.md'),
          input_indices={str(p.relative_to(ROOT)): digest(p) for p in indices}, source_sha256=frozen,
          revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
          scope='Only reused paired DEV; no phone/Night/DRV/final data, no model selection or full-convergence claim.'))
    start, complete, count, frames, neural, centers = time.perf_counter(), False, 0, 0, 0, 0
    try:
        for arm in ARMS:
            model, source = load_adapted(args.seed, arm); write(out/f'{arm}-source.json', source)
            for meta in metas:
                data = load_one(meta); began = time.perf_counter()
                inputs = input_tensor(data['low'], torch.device('cpu'))
                values = infer(model, inputs); assert torch.isfinite(values).all()
                raw = output_rgb(values); neural += 1; centers += len(raw)
                identity = dict(domain=meta['domain'], scene=meta['scene'], source_video=meta['source_video'],
                                arm=arm, train_seed=args.seed, reference=f'R65_{arm}_seed{args.seed}',
                                project_updates=8000, parameters=4316259, future_wait_max=0)
                append(out/'neural_stage_seconds.jsonl', dict(**identity, seconds=time.perf_counter()-began,
                    scope='CPU2 preparation/neural/RGB only; excludes motion/metrics/IO, not phone/full-pipeline timing.'))
                del inputs, values
                outputs, statistics = postprocess(raw, control=None)
                assert set(outputs) == {'raw', 'recursive_self05'}
                for method, output in outputs.items():
                    assert output.shape == data['low'].shape and output.dtype == np.float32 and np.isfinite(output).all()
                    assert np.array_equal(output[0], raw[0])
                    scores, per_frame = evaluate(output, data['gt'], data['reference_flow'], data['reference_mask'])
                    row_id = dict(**identity, method=method)
                    append(out/'per_sequence.jsonl', dict(**row_id, **scores, **statistics))
                    for frame in per_frame: append(out/'per_frame.jsonl', dict(**row_id, **frame))
                    path = cache/f'{arm}--{meta["scene"]}--{method}.npy'; np.save(path, output, allow_pickle=False)
                    append(out/'prediction_manifest.jsonl', dict(**row_id, path=str(path.relative_to(ROOT)),
                        sha256=digest(path), shape=list(output.shape), dtype=str(output.dtype),
                        input_cache=meta['path'], input_cache_sha256=digest(ROOT/meta['path']),
                        checkpoint_sha256=source['adapted_checkpoint']['sha256'],
                        online_RGB_history_bytes=0 if method == 'raw' else 2*raw[0].nbytes,
                        note='Classic post additionally uses flow/workspace; inference training conditions differ from other architectures.'))
                    count += 1; frames += len(per_frame)
                print(json.dumps(dict(**identity, quality_rows=count, CNN_sequences=neural, CNN_centers=centers)), flush=True)
                del data, raw, outputs, output
            del model
        assert (count, frames, neural, centers) == (40, 1792, 20, 896)
        assert not torch.cuda.is_initialized()
        assert all(digest(BASE/'scripts'/name) == sha for name, sha in frozen.items()); complete = True
    except BaseException:
        (out/'failure.txt').write_text(traceback.format_exc()); raise
    finally:
        write(out/'runtime.json', dict(completed_normally=complete, seed=args.seed, quality_rows=count, frame_rows=frames,
              CNN_sequences=neural, CNN_centers=centers, unique_observations=448,
              cuda_initialized=torch.cuda.is_initialized(), seconds=time.perf_counter()-start,
              ended=datetime.now().astimezone().isoformat()))


if __name__ == '__main__': main()
