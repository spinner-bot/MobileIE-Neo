"""Finite-kernel algebra/gradient gate and known periodic sampling control."""
import json
import time
from datetime import datetime

import numpy as np
from scipy.signal import resample
import torch

from dynamic_training_r34 import BASE
from phase_quadratic_r51 import blur, direct_quadratic, quadratic, translate


def ideal_square(value):
    h, w = value.shape[-2:]
    high = resample(resample(value, 2*h, axis=-2), 2*w, axis=-1)
    return resample(resample(high**2, h, axis=-2), w, axis=-1)


def main():
    torch.set_num_threads(2); torch.manual_seed(51)
    out = BASE/'process/exploration-r51/cpu-gate001'; out.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    x = torch.randn(2, 3, 17, 19, dtype=torch.float64, requires_grad=True)
    weight = torch.randn_like(x)
    q, direct = quadratic(x), direct_quadratic(x)
    delta = float((q-direct).abs().max().detach())
    g1, = torch.autograd.grad((q*weight).sum(), x)
    g2, = torch.autograd.grad((direct*weight).sum(), x)
    gradient_delta = float((g1-g2).abs().max())
    lower = float((q-blur(x).square()).min().detach())
    upper = float((blur(x.square())-q).min().detach())
    constant = torch.full((1, 2, 13, 15), 2.3, dtype=torch.float64)
    constant_delta = float((quadratic(constant)-constant.square()).abs().max())
    assert max(delta, gradient_delta, constant_delta) <= 1e-10
    assert lower >= -1e-10 and upper >= -1e-10
    yy, xx = np.mgrid[:63, :63]
    ops = {'square': lambda u: u**2, 'ideal_fft': ideal_square,
           'quadrature': lambda u: quadratic(torch.from_numpy(u[None,None])).numpy()[0,0],
           'preblur': lambda u: blur(torch.from_numpy(u[None,None])).numpy()[0,0]**2,
           'postblur': lambda u: blur(torch.from_numpy((u**2)[None,None])).numpy()[0,0]}
    rows = []
    for kx, ky in [(5,0), (19,0), (19,13)]:
        base = .5+.25*np.cos(2*np.pi*(kx*xx+ky*yy)/63)
        for name, op in ops.items():
            ref = op(base)
            for t in range(16):
                dx, dy = t/16, t/32
                value = op(translate(base, dx, dy))
                error = value-translate(ref, dx, dy)
                rms = float(np.sqrt(np.mean(error**2)))
                if name == 'ideal_fft': assert rms <= 1e-10
                rows.append({'kx': kx, 'ky': ky, 'mode': name, 'phase': t, 'RMS': rms})
            assert float(np.max(np.abs(op(translate(base, 1., 0.))-translate(ref, 1., 0.)))) <= 1e-10
    assert len(rows) == 240
    assert max(r['RMS'] for r in rows if r['mode'] == 'square' and r['kx'] == 19) > .001
    (out/'periodic_oracle.json').write_text(json.dumps(rows, indent=2)+'\n')
    result = {'at': datetime.now().astimezone().isoformat(), 'passed': True,
        'direct_vs_fused_max_abs': delta, 'gradient_max_abs_delta': gradient_delta,
        'jensen_lower_min': lower, 'jensen_upper_min': upper, 'constant_max_abs_delta': constant_delta,
        'periodic_cases': 240, 'ideal_oracle_max_RMS': max(r['RMS'] for r in rows if r['mode'] == 'ideal_fft'),
        'SGD_updates': 0, 'new_CNN': 0, 'cuda_initialized': torch.cuda.is_initialized(),
        'seconds': time.perf_counter()-start,
        'scope': 'Known synthetic sampling control. Finite bilinear quadrature is not ideal alias-free or an established video method.'}
    assert not result['cuda_initialized']
    (out/'gate.json').write_text(json.dumps(result, indent=2)+'\n'); print(json.dumps(result))


if __name__ == '__main__': main()
