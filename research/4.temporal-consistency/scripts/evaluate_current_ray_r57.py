"""One fixed 64-frame ND spatial bound; GT oracle, no model or temporal claim."""
from datetime import datetime
import json
import subprocess
import time
import traceback

import numpy as np
import torch

from current_ray_r57 import optimum
from dynamic_training_r34 import BASE, ROOT, digest


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def psnr(mse):
    return float(-10*np.log10(mse)) if mse else None  # None denotes +infinity, explicitly labelled.


def main():
    assert not torch.cuda.is_initialized()
    gate = BASE/'process/exploration-r57/cpu-gate001/gate.json'
    qualification = json.loads(gate.read_text())
    assert qualification['passed']
    assert qualification['source_sha256']['current_ray_r57.py'] == digest(BASE/'scripts/current_ray_r57.py')
    index = BASE/'results/exploration-r10-cache/index.json'
    metas = [m for m in json.loads(index.read_text())['scenes'] if m['domain'] == 'native_crop']
    assert [m['scene'] for m in metas] == ['video_15', 'video_47', 'video_91', 'video_121']
    out = BASE/'results/exploration-r57-current-ray-attempt001'; out.mkdir(exist_ok=False)
    write(out/'config.json', dict(at=datetime.now().astimezone().isoformat(), index_sha256=digest(index),
          gate_sha256=digest(gate), protocol_sha256=digest(BASE/'docs/exploration-r57-current-ray-feasibility-protocol.md'),
          source_sha256={n: digest(BASE/'scripts'/n) for n in ['current_ray_r57.py', 'evaluate_current_ray_r57.py']},
          revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
          frames=64, pixel_domain='encoded RGB [0,1]', spatial_crop_border=0, dtype='float64',
          scope='GT-dependent spatial upper bound for one strict output class; no CNN/SGD/flow, not deployable.'))
    started, complete, count, sequences = time.perf_counter(), False, 0, []
    try:
        for meta in metas:
            path = ROOT/meta['path']
            assert path.resolve().is_relative_to('/data/Users/zrh/data/MobileIE-Neo')
            assert digest(path) == meta['sha256']
            with np.load(path, allow_pickle=False) as source:
                low, target = source['low'], source['gt']
            assert list(low.shape) == meta['shape'] and len(low) == 16
            frames = []
            for t, (x, y) in enumerate(zip(low, target)):
                fit = optimum(x.astype(np.float64)/255, y.astype(np.float64)/255)
                mse = float(fit['mse'].mean()); ls = float(fit['least_squares_then_clip_mse'].mean())
                row = dict(scene=meta['scene'], frame=t, mse_rgb_0to1=mse, psnr_rgb=psnr(mse),
                           PSNR_infinite=mse == 0, least_squares_then_clip_psnr=psnr(ls))
                with (out/'per_frame.jsonl').open('a') as stream:
                    stream.write(json.dumps(row, allow_nan=False)+'\n')
                frames.append(row); count += 1
            value = None if any(r['PSNR_infinite'] for r in frames) else float(np.mean([r['psnr_rgb'] for r in frames]))
            sequences.append(dict(scene=meta['scene'], frames=16, psnr_rgb=value,
                                  PSNR_infinite=value is None,
                                  mean_mse_rgb_0to1=float(np.mean([r['mse_rgb_0to1'] for r in frames]))))
            print(json.dumps(sequences[-1]), flush=True)
        write(out/'per_sequence.json', sequences)
        assert count == 64 and not torch.cuda.is_initialized()
        complete = True
    except BaseException:
        (out/'failure.txt').write_text(traceback.format_exc()); raise
    finally:
        write(out/'runtime.json', dict(completed_normally=complete, frames=count, scenes=len(sequences),
              CNN=0, SGD=0, cuda_initialized=torch.cuda.is_initialized(),
              seconds=time.perf_counter()-started, ended=datetime.now().astimezone().isoformat()))


if __name__ == '__main__': main()
