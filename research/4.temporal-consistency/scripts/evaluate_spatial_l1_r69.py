"""Final8000 original MobileIE, same paired CPU references and recursive .5."""
import argparse
from datetime import datetime
import json
import subprocess
import time
import traceback

import cv2
import numpy as np
import torch

from check_spatial_l1_r69 import write
from drv_references_r59 import postprocess
from dynamic_training_r34 import BASE, ROOT, digest
from exploration_r2_data import evaluate
from prepare_cache_r5 import load_one
from train_exploration_r2 import new_model


def append(path, value):
    with path.open('a') as stream: stream.write(json.dumps(value, allow_nan=False)+'\n')


def load_final(seed):
    folder = BASE/f'results/exploration-r69-stage001-seed{seed}'
    runtime = json.loads((folder/'runtime.json').read_text())
    marker = json.loads((folder/'step8000.json').read_text())
    assert runtime['completed_normally'] and runtime['updates'] == 8000
    assert marker['seed'] == seed and marker['step'] == 8000 and marker['parameters'] == 4075
    assert marker['final_quality_eligible'] and marker['final_fold_gate_passed']
    assert marker['final_eval_fold_max_abs_delta'] <= 1e-4
    for path_key, sha_key in [('path', 'sha256'), ('training_path', 'training_sha256')]:
        assert (ROOT/marker[path_key]).resolve().is_relative_to('/data/Users/zrh/data/MobileIE-Neo')
        assert digest(ROOT/marker[path_key]) == marker[sha_key]
    saved = torch.load(ROOT/marker['path'], map_location='cpu', weights_only=False)
    assert saved['task'] == 'R69 MobileIE spatial L1' and saved['seed'] == seed
    assert saved['step'] == 8000 and saved['data_step'] == 7999
    assert saved['training_sha256'] == marker['training_sha256']
    model = new_model(torch.device('cpu')); model.load_state_dict(saved['state_dict'], strict=True)
    assert all(torch.isfinite(v).all() for v in model.state_dict().values())
    return model.eval().requires_grad_(False), dict(checkpoint=marker,
              training_config_sha256=digest(folder/'config.json'), project_updates=8000,
              parameters=4075, future_wait_max=0, objective='spatial L1')


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--seed', type=int, choices=[0, 1], required=True)
    args = parser.parse_args()
    torch.set_num_threads(2); cv2.setNumThreads(1); torch.use_deterministic_algorithms(True)
    assert not torch.cuda.is_initialized()
    model, source = load_final(args.seed)
    indices = [BASE/f'results/{name}/index.json' for name in ['exploration-r30-cache', 'exploration-r10-cache']]
    metas = json.loads(indices[0].read_text())['scenes']
    metas += [{**r, 'domain': 'paired_real_ND_DEV', 'source_video': r['scene']}
              for r in json.loads(indices[1].read_text())['scenes'] if r['domain'] == 'native_crop']
    assert len(metas) == 10
    out = BASE/f'results/exploration-r69-stage001-eval-seed{args.seed}'; out.mkdir(exist_ok=False)
    cache = ROOT/f'.cache/experiments/exploration-r69/stage001/seed{args.seed}'
    assert cache.resolve().is_relative_to('/data/Users/zrh/data/MobileIE-Neo')
    cache.mkdir(parents=True, exist_ok=False)
    names = ['evaluate_spatial_l1_r69.py', 'train_exploration_r2.py', 'drv_references_r59.py',
             'half_motion_r4.py', 'aligned_raw_history_r11.py', 'prepare_cache_r5.py',
             'exploration_r2_data.py', 'video_metrics.py']
    frozen = {name: digest(BASE/'scripts'/name) for name in names}
    write(out/'source.json', source)
    write(out/'config.json', dict(at=datetime.now().astimezone().isoformat(), seed=args.seed,
          device='cpu', torch_threads=2, opencv_threads=1, expected_quality_rows=20,
          expected_frame_rows=896, expected_CNN_sequences=10, expected_CNN_centers=448,
          methods=['raw', 'recursive_self05'], source_sha256=frozen,
          input_indices={str(p.relative_to(ROOT)): digest(p) for p in indices},
          protocol_sha256=digest(BASE/'docs/exploration-r69-matched-spatial-L1-protocol.md'),
          revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
          scope='Same reused DEV; no GT deployment motion, future input, phone timing or independent final test.'))
    start, complete, count, frames, neural, centers = time.perf_counter(), False, 0, 0, 0, 0
    try:
        for meta in metas:
            data = load_one(meta); began = time.perf_counter()
            inputs = torch.from_numpy((data['low']/255).transpose(0, 3, 1, 2).copy())
            with torch.inference_mode():
                value = torch.stack([model(image[None])[0] for image in inputs])
                assert torch.isfinite(value).all()
                raw = value.clamp(0, 1).permute(0, 2, 3, 1).contiguous().numpy()*255
            neural += 1; centers += len(raw)
            identity = dict(domain=meta['domain'], scene=meta['scene'], source_video=meta['source_video'],
                            family='MobileIE_L1', seed=args.seed, parameters=4075, project_updates=8000, future_wait_max=0)
            append(out/'neural_stage_seconds.jsonl', dict(**identity, seconds=time.perf_counter()-began,
                   scope='CPU preparation/neural/RGB only; excludes motion/metrics/IO; not phone/full-pipeline timing.'))
            del inputs, value
            outputs, statistics = postprocess(raw, control=None)
            for method, output in outputs.items():
                assert output.dtype == np.float32 and np.isfinite(output).all() and np.array_equal(output[0], raw[0])
                scores, per_frame = evaluate(output, data['gt'], data['reference_flow'], data['reference_mask'])
                row_id = dict(**identity, method=method)
                append(out/'per_sequence.jsonl', dict(**row_id, **scores, **statistics))
                for frame in per_frame: append(out/'per_frame.jsonl', dict(**row_id, **frame))
                path = cache/f'{meta["scene"]}--{method}.npy'; np.save(path, output, allow_pickle=False)
                append(out/'prediction_manifest.jsonl', dict(**row_id, path=str(path.relative_to(ROOT)),
                       sha256=digest(path), shape=list(output.shape), dtype=str(output.dtype),
                       input_cache=meta['path'], input_cache_sha256=digest(ROOT/meta['path']),
                       checkpoint_sha256=source['checkpoint']['sha256'],
                       online_RGB_history_bytes=0 if method == 'raw' else 2*raw[0].nbytes,
                       note='Recursive output additionally uses flow/warp/workspace; raw graph unchanged.'))
                count += 1; frames += len(per_frame)
            print(json.dumps(dict(**identity, quality_rows=count, CNN_sequences=neural, CNN_centers=centers)), flush=True)
            del data, raw, outputs, output
        assert (count, frames, neural, centers) == (20, 896, 10, 448)
        assert not torch.cuda.is_initialized()
        assert all(digest(BASE/'scripts'/name) == sha for name, sha in frozen.items()); complete = True
    except BaseException:
        (out/'failure.txt').write_text(traceback.format_exc()); raise
    finally:
        write(out/'runtime.json', dict(completed_normally=complete, seed=args.seed, quality_rows=count,
              frame_rows=frames, CNN_sequences=neural, CNN_centers=centers, cuda_initialized=torch.cuda.is_initialized(),
              seconds=time.perf_counter()-start, ended=datetime.now().astimezone().isoformat()))


if __name__ == '__main__': main()
