"""Synthetic radiometric qualification before opening any RAW pixel data."""
from datetime import datetime
import json

import cv2
import numpy as np
import rawpy

from drv_render_r58 import encode_srgb, make_view, parameters
from dynamic_training_r34 import BASE, digest


def main():
    cv2.setNumThreads(1)
    out = BASE/'process/external-drv/render-gate001'; out.mkdir(exist_ok=False)
    values = encode_srgb(np.array([0, .0031308, .18, 1.]))
    assert np.allclose(values, [0, .040449936, .46135612950044164, 1], atol=1e-12)
    linear_side, power_side = 12.92*.0031308, 1.055*.0031308**(1/2.4)-.055
    assert abs(linear_side-power_side) <= 5e-8
    assert np.all(np.diff(encode_srgb(np.linspace(0, 1, 65537))) >= 0)
    black, _ = make_view(np.zeros((16, 24, 3), np.uint16), 1)
    white, info = make_view(np.full((16, 24, 3), 65535, np.uint16), 2)
    assert black.shape == white.shape == (427, 640, 3)
    assert not black.any() and np.all(white == 255) and info['pre_resize_above_one_fraction'] == 1
    rawpy.Params(**parameters([2, 1, 1.5, 1]))
    record = dict(at=datetime.now().astimezone().isoformat(), passed=True,
          srgb_endpoint_values=values.tolist(), srgb_boundary_discrepancy=abs(linear_side-power_side),
          monotonic_grid_points=65537, fixed_resize_shape=list(black.shape),
          rawpy=rawpy.__version__, libraw_version=rawpy.libraw_version, real_RAW_opened=0, CNN=0, GPU=False,
          source_sha256={n: digest(BASE/'scripts'/n) for n in ['drv_render_r58.py', 'check_drv_render_r58.py']},
          scope='Synthetic encoding/resize/clip and installed parameter qualification only; no real sensor/ISP accuracy claim.')
    (out/'gate.json').write_text(json.dumps(record, indent=2, allow_nan=False)+'\n')
    print(json.dumps(record))


if __name__ == '__main__': main()
