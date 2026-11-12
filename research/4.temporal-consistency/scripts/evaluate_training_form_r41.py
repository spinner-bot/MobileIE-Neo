"""Common-CPU final deployment evaluation of the two matched FP32 forms."""
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
from train_exploration_r2 import new_model


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
    folder = BASE/f'results/exploration-r41-seed{args.seed}'
    runtime = json.loads((folder/'runtime.json').read_text())
    assert runtime['completed_normally'] and runtime['total_updates'] == 8000
    assert set(runtime['model_updates'].values()) == {4000}
    training = json.loads((folder/'config.json').read_text())
    assert not training['cudnn_tf32_allowed'] and not training['matmul_tf32_allowed']
    nets, markers = {}, {}
    for model in ['slim_fp32', 'expanded_fp32']:
        marker = json.loads((folder/f'{model}-step4000.json').read_text())
        assert marker['final_fold_gate_passed'] and marker['final_eval_fold_max_abs_delta'] <= 1e-4
        assert digest(ROOT/marker['path']) == marker['sha256']
        assert digest(ROOT/marker['training_path']) == marker['training_sha256']
        saved = torch.load(ROOT/marker['path'], map_location='cpu', weights_only=False)
        assert saved['model'] == model and saved['seed'] == args.seed and saved['step'] == 4000
        assert saved['training_sha256'] == marker['training_sha256']
        net = new_model(torch.device('cpu'))
        net.load_state_dict(saved['state_dict'], strict=True)
        nets[model] = net.eval()
        markers[model] = marker
    synthetic = BASE/'results/exploration-r30-cache/index.json'
    real = BASE/'results/exploration-r10-cache/index.json'
    metas = json.loads(synthetic.read_text())['scenes']
    metas += [{**r, 'domain': 'paired_real_ND_DEV', 'source_video': r['scene']}
              for r in json.loads(real.read_text())['scenes'] if r['domain'] == 'native_crop']
    assert len(metas) == 10
    out = BASE/f'results/exploration-r41-eval-seed{args.seed}'
    cache = ROOT/f'.cache/experiments/exploration-r41/seed{args.seed}'
    assert cache.resolve().is_relative_to('/data/Users/zrh/data/MobileIE-Neo')
    out.mkdir(parents=True, exist_ok=False)
    cache.mkdir(parents=True, exist_ok=False)
    for model in nets:
        (cache/model).mkdir()
    write(out/'config.json', {'seed': args.seed, 'device': 'cpu', 'torch_threads': 2, 'opencv_threads': 1,
          'checkpoints': markers, 'deployment_parameters_each': 4075, 'torch_version': torch.__version__,
          'training_config_sha256': digest(folder/'config.json'),
          'cache_indices': {str(p.relative_to(ROOT)): digest(p) for p in [synthetic, real]},
          'protocol_sha256': digest(BASE/'docs/exploration-r41-training-form-protocol.md'),
          'source_sha256': {name: digest(Path(__file__).parent/name) for name in
                           ['evaluate_training_form_r41.py', 'half_motion_r4.py', 'aligned_raw_history_r11.py',
                            'aggregation_controls_r20.py', 'exploration_r2_data.py', 'video_metrics.py']},
          'revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
          'scope': 'Final folded deployment weights, two common-FP32 training forms; CPU DEV, no online BN/history in raw or new inference capacity.'})
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
                            'anchor_family': 'R41_'+model, 'anchor': f'R41_{model}_seed{args.seed}',
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
        assert count == 60 and neural == 20 and not torch.cuda.is_initialized()
        complete = True
    except Exception:
        (out/'failure.txt').write_text(traceback.format_exc())
        raise
    finally:
        write(out/'runtime.json', {'completed_normally': complete, 'rows': count, 'neural_sequences': neural,
              'cuda_initialized': torch.cuda.is_initialized(), 'wall_seconds': time.perf_counter()-started})


if __name__ == '__main__':
    main()
