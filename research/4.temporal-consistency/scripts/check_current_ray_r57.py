"""Bounded numerical qualification before any paired real image is opened."""
from datetime import datetime
import json

import numpy as np

from current_ray_r57 import optimum
from dynamic_training_r34 import BASE, digest


def main():
    out = BASE/'process/exploration-r57/cpu-gate001'; out.mkdir(parents=True, exist_ok=False)
    x = np.array([[0, 0, 0], [.1, .9, .1], [.1, .2, .3], [0, .3, .6], [0, 0, 0]])
    y = np.array([[.3, .4, .5], [1, .9, 1], [.2, .4, .6], [0, .3, .6], [0, 0, 0]])
    result = optimum(x, y)
    assert np.allclose(result['mse'], [(.09+.16+.25)/3, .01/3, 0, 0, 0], atol=1e-14)
    assert np.allclose(result['gain'][[1, 2, 3]], [10, 2, 1])
    random = np.random.default_rng(57001)
    rx, ry = random.random((64, 3)), random.random((64, 3))
    exact = optimum(rx, ry)['mse']
    grid = np.linspace(0, 1, 10001)[None, :]/rx.min(axis=1)[:, None]
    errors = np.mean((np.clip(grid[:, :, None]*rx[:, None, :], 0, 1)-ry[:, None, :])**2, axis=2)
    dense = errors.min(axis=1)
    assert np.all(exact <= dense+1e-12)
    sources = {n: digest(BASE/'scripts'/n) for n in ['current_ray_r57.py', 'check_current_ray_r57.py']}
    record = dict(at=datetime.now().astimezone().isoformat(), passed=True, manual_cases=5,
                  random_RGB_pairs=64, grid_points_per_pair=10001,
                  max_exact_minus_grid=float(np.max(exact-dense)), source_sha256=sources,
                  real_images_opened=0, CNN=0, SGD=0, GPU=False,
                  scope='Piecewise RGB oracle numerical qualification, not model quality or new theory.')
    (out/'gate.json').write_text(json.dumps(record, indent=2, allow_nan=False)+'\n')
    print(json.dumps(record))


if __name__ == '__main__': main()
