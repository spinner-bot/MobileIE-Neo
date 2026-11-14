"""Single fixed training-only BN recalibration of six immutable source models."""
import hashlib
import inspect
import json
import os
import subprocess
import time
import traceback
from datetime import datetime

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import cv2
import torch

from bn_calibration_r50 import MODELS, WINDOWS, calibrate, source_model, verify_change
from dynamic_training_r34 import BASE, ROOT, digest
from mixed_anchor_r40 import MixedPool, prediction
from training_form_r41 import configure_precision, cuda_headroom


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def main():
    gate = BASE/'process/exploration-r50/cpu-gate001/gate.json'
    assert json.loads(gate.read_text())['passed']
    torch.set_num_threads(2); cv2.setNumThreads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    configure_precision()
    assert not torch.cuda.is_initialized()
    out = BASE/'results/exploration-r50-stage001-calibration'
    out.mkdir(exist_ok=False)
    cache = ROOT/'.cache/checkpoints/exploration-r50/stage001'
    assert cache.resolve().is_relative_to('/data/Users/zrh/data/MobileIE-Neo')
    start, complete, finals = time.perf_counter(), False, []
    try:
        pool = MixedPool()
        first_low, _, first_context = pool.batch(50, 0)
        # Source preparation remains on CPU before fresh internal headroom checks.
        sources = {(seed, name): source_model(seed, name) for seed in [0, 1] for name in MODELS}
        admission = cuda_headroom(); write(out/'pre_cuda_guard.json', admission)
        assert admission['allowed']
        cache.mkdir(parents=True, exist_ok=False)
        names = ['run_bn_calibration_r50.py', 'bn_calibration_r50.py', 'mixed_anchor_r40.py',
                 'training_form_r41.py', 'dynamic_training_r34.py']
        write(out/'config.json', {'at': datetime.now().astimezone().isoformat(), 'seed_order': [0, 1],
              'models': MODELS, 'calibration_seed': 50, 'data_steps': [0, 127], 'windows_each': WINDOWS,
              'batch': 4, 'frames': 4, 'crop': 128, 'SGD_updates': 0, 'GT_used_for_calibration': False,
              'method': 'official update_bn; equal B4 batches, four sequential frame calls',
              'torch': torch.__version__, 'FP32': True, 'TF32': False, 'data': pool.info,
              'update_bn_source_sha256': hashlib.sha256(inspect.getsource(torch.optim.swa_utils.update_bn).encode()).hexdigest(),
              'protocol_sha256': digest(BASE/'docs/exploration-r50-bn-calibration-protocol.md'),
              'source_sha256': {n: digest(BASE/'scripts'/n) for n in names},
              'revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()})
        common_digest = None
        with (out/'windows.jsonl').open('x') as windows:
            for (seed, name), (net, source) in sources.items():
                net = net.cuda().eval()
                inputs_digest = hashlib.sha256()
                count = 0
                def loader():
                    nonlocal count
                    for step in range(WINDOWS):
                        low, context = ((first_low, first_context) if step == 0 else
                                        (lambda x: (x[0], x[2]))(pool.batch(50, step)))
                        inputs_digest.update(low.numpy().tobytes())
                        windows.write(json.dumps({'seed': seed, 'model': name, 'data_step': step,
                                                  'windows': context})+'\n')
                        yield low
                        count += 1
                        if count % 32 == 0:
                            windows.flush()
                            print(json.dumps({'seed': seed, 'model': name, 'completed_windows': count}), flush=True)
                before = calibrate(net, loader(), 'cuda')
                assert count == WINDOWS
                check = verify_change(net, before, 4*WINDOWS)
                input_hash = inputs_digest.hexdigest()
                if common_digest is None: common_digest = input_hash
                assert input_hash == common_digest
                with torch.no_grad():
                    slim = net.slim().cuda().eval()
                    assert sum(p.numel() for p in slim.parameters()) == 4075
                    delta = float((prediction(net, first_low.cuda())-prediction(slim, first_low.cuda())).abs().max())
                    assert delta <= 1e-4
                full_path = cache/f'seed{seed}-{name}-training.pt'
                path = cache/f'seed{seed}-{name}-deploy.pt'
                torch.save({'state_dict': net.cpu().state_dict(), 'seed': seed, 'model': name,
                            'SGD_updates': 0, 'calibration_windows': WINDOWS}, full_path)
                torch.save({'state_dict': slim.cpu().state_dict(), 'seed': seed, 'model': name,
                            'SGD_updates': 0, 'calibration_windows': WINDOWS}, path)
                marker = {'seed': seed, 'model': name, 'source': source, 'check': check,
                          'same_calibration_input_sha256': input_hash, 'fold_max_abs_delta': delta,
                          'deployment_parameters': 4075, 'path': str(path.relative_to(ROOT)),
                          'sha256': digest(path), 'training_path': str(full_path.relative_to(ROOT)),
                          'training_sha256': digest(full_path), 'SGD_updates': 0, 'windows': count}
                write(out/f'seed{seed}-{name}.json', marker); finals.append(marker)
                del slim, net
        assert len(finals) == 6
        complete = True
    except BaseException:
        (out/'failure.txt').write_text(traceback.format_exc()); raise
    finally:
        write(out/'runtime.json', {'completed_normally': complete, 'models_completed': len(finals),
              'final_checkpoints': finals, 'SGD_updates': 0, 'calibration_windows_completed': 128*len(finals),
              'seconds': time.perf_counter()-start, 'ended': datetime.now().astimezone().isoformat()})


if __name__ == '__main__':
    main()
