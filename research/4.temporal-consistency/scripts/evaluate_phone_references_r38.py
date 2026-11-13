"""Fixed real-phone reference outputs; no full-reference quality or phone timing."""
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
from half_motion_r4 import half_forward_backward
from train_exploration_r2 import new_model


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def append(path, value):
    with path.open('a') as stream:
        stream.write(json.dumps(value, allow_nan=False)+'\n')


def statistics(frame):
    gray = cv2.cvtColor(frame.astype(np.float32), cv2.COLOR_RGB2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
    return {'mean_rgb_0to255': float(frame.mean()), 'mean_luma_0to255': float(gray.mean()),
            'low_endpoint_fraction': float((frame <= .5).mean()),
            'high_endpoint_fraction': float((frame >= 254.5).mean()),
            'sobel_energy_proxy': float(np.hypot(gx, gy).mean())}


def main():
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    torch.use_deterministic_algorithms(True)
    assert not torch.cuda.is_initialized()
    official = new_model(torch.device('cpu')).eval()
    nets, markers = {'official_LOL': official}, {'official_LOL': {
        'path': 'MobileIE/pretrain/lolv1_best_slim.pkl',
        'sha256': digest(ROOT/'MobileIE/pretrain/lolv1_best_slim.pkl')}}
    for seed in [0, 1]:
        marker = json.loads((BASE/f'results/exploration-r34-seed{seed}/original-step4000.json').read_text())
        assert digest(ROOT/marker['path']) == marker['sha256']
        saved = torch.load(ROOT/marker['path'], map_location='cpu', weights_only=False)
        assert saved['step'] == 4000 and saved['model'] == 'original' and saved['seed'] == seed
        net = new_model(torch.device('cpu'))
        net.load_state_dict(saved['state_dict'], strict=True)
        nets[f'R34_seed{seed}'] = net.eval()
        markers[f'R34_seed{seed}'] = marker
    index = BASE/'results/exploration-r31-data/index.json'
    metadata = json.loads(index.read_text())
    assert not metadata['ground_truth_available'] and len(metadata['scenes']) == 3
    out = BASE/'results/exploration-r38-phone'
    cache = ROOT/'.cache/experiments/exploration-r38'
    assert cache.resolve().is_relative_to('/data/Users/zrh/data/MobileIE-Neo')
    out.mkdir(parents=True, exist_ok=False)
    cache.mkdir(parents=True, exist_ok=False)
    write(out/'config.json', {'anchors': markers, 'methods': ['raw', 'finite_self05', 'recursive_self05'],
          'device': 'cpu', 'torch_threads': 2, 'opencv_threads': 1, 'torch_version': torch.__version__,
          'input_index_sha256': digest(index), 'extra_noise': False, 'brightness_transform': None,
          'protocol_sha256': digest(BASE/'docs/exploration-r38-real-phone-reference-protocol.md'),
          'source_sha256': {name: digest(Path(__file__).parent/name) for name in
                           ['evaluate_phone_references_r38.py', 'half_motion_r4.py', 'aligned_raw_history_r11.py',
                            'aggregation_controls_r20.py', 'train_exploration_r2.py']},
          'revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
          'scope': 'Fixed three phone DEV inputs. No GT, PSNR/SSIM, flicker truth, best-seed selection, new method or mobile latency.'})
    started, count, neural, complete = time.perf_counter(), 0, 0, False
    try:
        for meta in metadata['scenes']:
            assert digest(ROOT/meta['path']) == meta['sha256']
            with np.load(ROOT/meta['path'], allow_pickle=False) as archive:
                low = archive['low'].copy()
            assert low.shape == (64, 360, 640, 3) and low.dtype == np.uint8
            for t, frame in enumerate(low):
                append(out/'input_statistics.jsonl', {'scene': meta['scene'], 'frame': t, **statistics(frame)})
            for anchor, net in nets.items():
                previous_raw = previous_fused = None
                outputs = {name: [] for name in ['raw', 'finite_self05', 'recursive_self05']}
                with torch.inference_mode():
                    for t, frame in enumerate(low):
                        image = torch.from_numpy(frame.transpose(2, 0, 1).copy()).float()[None]/255
                        value = net(image)[0]
                        assert torch.isfinite(value).all()
                        raw = value.clamp(0, 1).permute(1, 2, 0).contiguous().numpy()*255
                        append(out/'preclip_statistics.jsonl', {'scene': meta['scene'], 'anchor': anchor, 'frame': t,
                              'below_zero_fraction': float((value < 0).float().mean()),
                              'above_one_fraction': float((value > 1).float().mean())})
                        finite = recursive = raw
                        if previous_raw is not None:
                            flow, mask = half_forward_backward(previous_raw, raw)
                            finite = blend(raw, align_history(previous_raw, raw, flow, mask)[0], .5)
                            recursive = blend(raw, align_history(previous_fused, raw, flow, mask)[0], .5)
                        previous_raw, previous_fused = raw, recursive
                        for method, output in [('raw', raw), ('finite_self05', finite), ('recursive_self05', recursive)]:
                            assert np.isfinite(output).all()
                            if t == 0:
                                assert np.array_equal(raw, output)
                            outputs[method].append(output)
                neural += 1
                for method, values in outputs.items():
                    prediction = np.stack(values)
                    path = cache/f"{meta['scene']}--{anchor}--{method}.npy"
                    np.save(path, prediction, allow_pickle=False)
                    identity = {'scene': meta['scene'], 'phone': meta['phone'], 'anchor': anchor, 'method': method}
                    stats = [statistics(frame) for frame in prediction]
                    for t, row in enumerate(stats):
                        append(out/'per_frame.jsonl', {**identity, 'frame': t, **row})
                    history = 0 if method == 'raw' else previous_raw.nbytes*(2 if method == 'recursive_self05' else 1)
                    append(out/'per_sequence.jsonl', {**identity, 'frames': 64, 'online_history_storage_bytes': history,
                           **{key: float(np.mean([r[key] for r in stats])) for key in stats[0]},
                           'scope': 'Exposure/endpoint/edge diagnostics only; not truth-referenced image quality.'})
                    append(out/'prediction_manifest.jsonl', {**identity, 'path': str(path.relative_to(ROOT)),
                           'sha256': digest(path), 'shape': list(prediction.shape), 'dtype': str(prediction.dtype),
                           'anchor_sha256': markers[anchor]['sha256'], 'input_sha256': meta['sha256']})
                    count += 1
                    print(json.dumps({**identity, 'rows': count}), flush=True)
        assert count == 27 and neural == 9 and not torch.cuda.is_initialized()
        complete = True
    except Exception:
        (out/'failure.txt').write_text(traceback.format_exc())
        raise
    finally:
        write(out/'runtime.json', {'completed_normally': complete, 'rows': count, 'neural_sequences': neural,
                                  'cuda_initialized': torch.cuda.is_initialized(), 'wall_seconds': time.perf_counter()-started,
                                  'scope': 'Fixed real-phone input references; server runtime is not phone latency.'})


if __name__ == '__main__':
    main()
