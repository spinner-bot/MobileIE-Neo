"""CPU evaluation of final R34 weights with own-current and classic-post controls."""
import argparse
import json
import subprocess
import time
import traceback

import cv2
import numpy as np
import torch

from aggregation_controls_r20 import blend
from aligned_raw_history_r11 import align_history
from dynamic_training_r34 import BASE, ROOT, digest, models
from exploration_r2_data import evaluate
from prepare_cache_r5 import load_one


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, choices=[0, 1], required=True)
    args = parser.parse_args()
    source = BASE/f'results/exploration-r34-seed{args.seed}'
    runtime = json.loads((source/'runtime.json').read_text())
    assert runtime['completed_normally'] and runtime['total_updates'] == 12000
    assert set(runtime['model_updates'].values()) == {4000}
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    torch.use_deterministic_algorithms(True)
    assert not torch.cuda.is_initialized()
    nets, _ = models(args.seed, torch.device('cpu'))
    markers = {}
    for name, net in nets.items():
        marker = json.loads((source/f'{name}-step4000.json').read_text())
        assert digest(ROOT/marker['path']) == marker['sha256']
        saved = torch.load(ROOT/marker['path'], map_location='cpu', weights_only=False)
        assert saved['step'] == 4000 and saved['seed'] == args.seed and saved['model'] == name
        net.load_state_dict(saved['state_dict'], strict=True)
        net.eval()
        markers[name] = marker
    out = BASE/f'results/exploration-r34-eval-seed{args.seed}'
    out.mkdir(parents=True, exist_ok=False)
    index_path = BASE/'results/exploration-r30-cache/index.json'
    metas = json.loads(index_path.read_text())['scenes']
    assert len(metas) == 6
    write(out/'config.json', {'seed': args.seed, 'device': 'cpu', 'torch_threads': 2, 'opencv_threads': 1,
          'torch_version': torch.__version__, 'checkpoints': markers, 'cache_index_sha256': digest(index_path),
          'training_config_sha256': digest(source/'config.json'), 'input_reset': False,
          'protocol_sha256': digest(BASE/'docs/exploration-r34-training-protocol.md'),
          'revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
          'scope': 'Custom external DEV, fixed final4000, same CPU and cached quality-evaluation motion; not original DAVIS LLVE benchmark or phone speed. Post deployment may estimate motion from its own enhanced output.'})
    started, count, inferences, complete, first = time.perf_counter(), 0, 0, False, []
    try:
        for meta in metas:
            data = load_one(meta)
            inputs = torch.from_numpy((data['low']/255).transpose(0, 3, 1, 2).copy())
            own_current = None
            for name, route in [('original', 'current'), ('spatial', 'current'), ('causal', 'current'), ('causal', 'causal')]:
                net = nets[name]
                if name == 'causal':
                    net.reset()
                begin = time.perf_counter()
                with torch.inference_mode():
                    values = []
                    for frame in inputs:
                        image = frame[None]
                        if name == 'original':
                            output = net(image)
                        elif route == 'causal':
                            output = net.step(image)
                        else:
                            # Includes learned low-resolution spatial mapping, not bare backbone.
                            output = net(image, history='current')[0]
                        values.append(output[0])
                    values = torch.stack(values)
                forward_seconds = time.perf_counter()-begin
                assert torch.isfinite(values).all()
                raw = values.clamp(0, 1).permute(0, 2, 3, 1).contiguous().numpy()*255
                inferences += 1
                storage = net.state_storage_bytes() if route == 'causal' else 0
                if name == 'causal' and route == 'current':
                    own_current = raw[0].copy()
                elif route == 'causal':
                    delta = float(np.abs(raw[0]-own_current).max())
                    assert delta <= 1e-4
                    first.append({'scene': meta['scene'], 'first_frame_max_abs_0to255': delta})
                fusions = ['raw', 'postwarp05'] if route == 'current' else ['raw']
                for fusion in fusions:
                    begin = time.perf_counter()
                    output = raw
                    if fusion == 'postwarp05':
                        warped = np.stack([raw[0], *[align_history(raw[t-1], raw[t], data['half_flow'][t-1],
                                                                 data['half_mask'][t-1])[0] for t in range(1, len(raw))]])
                        output = blend(raw, warped, .5)
                    post_seconds = time.perf_counter()-begin if fusion != 'raw' else 0.
                    metrics, frames = evaluate(output, data['gt'], data['reference_flow'], data['reference_mask'])
                    identity = {'domain': meta['domain'], 'scene': meta['scene'], 'source_video': meta['source_video'],
                                'train_seed': args.seed, 'model': name, 'history': route, 'fusion': fusion, 'input_reset': False}
                    with (out/'per_sequence.jsonl').open('a') as stream:
                        stream.write(json.dumps({**identity, **metrics}, allow_nan=False)+'\n')
                    with (out/'per_frame.jsonl').open('a') as stream:
                        for frame in frames:
                            stream.write(json.dumps({**identity, **frame}, allow_nan=False)+'\n')
                    with (out/'stage_cost.jsonl').open('a') as stream:
                        stream.write(json.dumps({**identity, 'forward_seconds': forward_seconds, 'post_seconds_excluding_motion': post_seconds,
                                                 'state_storage_bytes': storage, 'scope': 'CPU stage only; cached flow, no phone or complete deployed-pipeline timing.'})+'\n')
                    count += 1
                print(json.dumps({'scene': meta['scene'], 'model': name, 'history': route, 'rows': count}), flush=True)
        assert count == 42 and inferences == 24 and not torch.cuda.is_initialized()
        write(out/'first_frame.json', first)
        complete = True
    except Exception:
        (out/'failure.txt').write_text(traceback.format_exc())
        raise
    finally:
        write(out/'runtime.json', {'completed_normally': complete, 'rows': count, 'neural_sequences': inferences,
                                  'cuda_initialized': torch.cuda.is_initialized(), 'wall_seconds': time.perf_counter()-started})


if __name__ == '__main__':
    main()
