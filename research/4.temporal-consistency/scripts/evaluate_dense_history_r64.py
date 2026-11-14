"""Final-only common CPU quality, both strong post and genuine-history controls."""
import argparse
from datetime import datetime
import json
import time
import traceback

import cv2
import numpy as np
import torch

from aggregation_controls_r20 import blend
from aligned_raw_history_r11 import align_history
from dense_joint_state_r63 import DenseJointState
from dynamic_training_r34 import BASE, ROOT, digest
from exploration_r2_data import evaluate
from half_motion_r4 import half_forward_backward
from prepare_cache_r5 import load_one
from train_exploration_r2 import new_model
from train_dense_history_r64 import MODELS, write


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
    train = BASE/f'results/exploration-r64-stage001-seed{args.seed}'
    training = json.loads((train/'runtime.json').read_text())
    assert training['completed_normally'] and training['total_new_updates'] == 12000
    assert training['model_updates'] == {name: 4000 for name in MODELS}
    nets, markers = {}, {}
    for name in MODELS:
        marker = json.loads((train/f'{name}-step4000.json').read_text())
        assert marker['new_step'] == 4000 and marker['inherited_updates'] == 8000
        assert marker['final_fold_passed'] and marker['final_fold_delta'] <= 1e-4
        assert digest(ROOT/marker['path']) == marker['sha256']
        saved = torch.load(ROOT/marker['path'], map_location='cpu', weights_only=True)
        assert saved['model'] == name and saved['seed'] == args.seed and saved['new_step'] == 4000
        net = new_model(torch.device('cpu'))
        if name != 'plain':
            net = DenseJointState(net, temporal=name == 'causal')
        net.load_state_dict(saved['state_dict'], strict=True)
        assert sum(p.numel() for p in net.parameters()) == (4075 if name == 'plain' else 4618)
        nets[name], markers[name] = net.eval(), marker
    indices = [BASE/f'results/{name}/index.json' for name in ['exploration-r30-cache', 'exploration-r10-cache']]
    metas = json.loads(indices[0].read_text())['scenes']
    metas += [{**r, 'domain': 'paired_real_ND_DEV', 'source_video': r['scene']}
              for r in json.loads(indices[1].read_text())['scenes'] if r['domain'] == 'native_crop']
    assert len(metas) == 10
    out = BASE/f'results/exploration-r64-stage001-eval-seed{args.seed}'
    cache = ROOT/f'.cache/experiments/exploration-r64/stage001/seed{args.seed}'
    assert cache.resolve().is_relative_to('/data/Users/zrh/data/MobileIE-Neo')
    out.mkdir(exist_ok=False)
    cache.mkdir(parents=True, exist_ok=False)
    names = ['evaluate_dense_history_r64.py', 'dense_joint_state_r63.py', 'budget_causal_r33.py',
             'half_motion_r4.py', 'aligned_raw_history_r11.py', 'aggregation_controls_r20.py',
             'exploration_r2_data.py', 'video_metrics.py']
    write(out/'config.json', {'at': datetime.now().astimezone().isoformat(), 'seed': args.seed,
        'checkpoints': markers, 'device': 'cpu', 'torch_threads': 2, 'opencv_threads': 1,
        'cache_indices': {str(p.relative_to(ROOT)): digest(p) for p in indices},
        'protocol_sha256': digest(BASE/'docs/exploration-r64-dense-history-training-protocol.md'),
        'source_sha256': {n: digest(BASE/'scripts'/n) for n in names},
        'expected_quality_rows': 70, 'expected_CNN_sequences': 40, 'expected_center_outputs': 1792,
        'scope': 'Final12000 only, common GT evaluation flow; post uses own raw flow. Reused DEV, no final or phone.'})
    started, complete, count, neural, centers, frames_count = time.perf_counter(), False, 0, 0, 0, 0
    try:
        for meta in metas:
            data = load_one(meta)
            inputs = torch.from_numpy((data['low']/255).transpose(0, 3, 1, 2).copy())
            causal_first = None
            for name, route in [('plain', 'current'), ('spatial', 'current'),
                                ('causal', 'causal'), ('causal', 'current')]:
                net = nets[name]
                if name != 'plain':
                    net.reset()
                began = time.perf_counter()
                with torch.inference_mode():
                    values = []
                    for frame in inputs:
                        if name == 'plain':
                            value = net(frame[None])
                        elif route == 'causal':
                            value = net.step(frame[None])
                            assert torch.isfinite(net._state).all()
                        else:
                            value, _ = net(frame[None], history='current')
                        assert torch.isfinite(value).all()
                        values.append(value[0])
                    raw = torch.stack(values).clamp(0, 1).permute(0, 2, 3, 1).contiguous().numpy()*255
                seconds = time.perf_counter()-began
                neural += 1
                centers += len(raw)
                identity = {'domain': meta['domain'], 'scene': meta['scene'], 'source_video': meta['source_video'],
                            'train_seed': args.seed, 'model': name, 'history': route,
                            'parameters': markers[name]['deployment_parameters']}
                if route == 'causal':
                    causal_first = raw[0].copy()
                elif name == 'causal':
                    first_delta = float(np.abs(causal_first-raw[0]).max())
                    append(out/'first_frame.jsonl', {**identity, 'max_abs_delta_0to255': first_delta})
                    assert first_delta <= 1e-4
                path = cache/(meta['scene']+'--'+name+'--'+route+'.npy')
                np.save(path, raw, allow_pickle=False)
                append(out/'prediction_manifest.jsonl', {**identity, 'path': str(path.relative_to(ROOT)),
                    'shape': list(raw.shape), 'sha256': digest(path), 'checkpoint_sha256': markers[name]['sha256']})
                append(out/'stage_cost.jsonl', {**identity, 'neural_stage_seconds': seconds, 'frames': len(raw),
                    'persistent_state_bytes': net.state_storage_bytes() if route == 'causal' else 0,
                    'scope': 'Unwarmed descriptive neural stage including checks, not complete-pipeline or phone timing.'})
                outputs = [('current_raw' if name == 'causal' and route == 'current' else 'raw', raw)]
                if not (name == 'causal' and route == 'current'):
                    recursive, accepted = [raw[0]], []
                    for t in range(1, len(raw)):
                        flow, mask = half_forward_backward(raw[t-1], raw[t])
                        accepted.append(float(mask.mean()))
                        recursive.append(blend(raw[t], align_history(recursive[-1], raw[t], flow, mask)[0], .5))
                    outputs.append(('recursive_self05', np.stack(recursive)))
                    append(out/'motion_acceptance.jsonl', {**identity, 'pair_fractions': accepted,
                        'mean_accepted_fraction': float(np.mean(accepted))})
                for method, output in outputs:
                    assert np.isfinite(output).all()
                    metrics, frames = evaluate(output, data['gt'], data['reference_flow'], data['reference_mask'])
                    append(out/'per_sequence.jsonl', {**identity, 'method': method, **metrics})
                    for frame in frames:
                        append(out/'per_frame.jsonl', {**identity, 'method': method, **frame})
                    count += 1
                    frames_count += len(frames)
                print(json.dumps({**identity, 'rows': count, 'CNN_sequences': neural}), flush=True)
        assert (count, neural, centers, frames_count) == (70, 40, 1792, 3136)
        complete = True
    except BaseException:
        (out/'failure.txt').write_text(traceback.format_exc())
        raise
    finally:
        write(out/'runtime.json', {'completed_normally': complete, 'rows': count, 'neural_sequences': neural,
              'center_outputs': centers, 'per_frame_rows': frames_count, 'cuda_initialized': torch.cuda.is_initialized(),
              'seconds': time.perf_counter()-started, 'ended_at': datetime.now().astimezone().isoformat()})


if __name__ == '__main__':
    main()
