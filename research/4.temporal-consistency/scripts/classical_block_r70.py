"""Fixed first-frame blind sigma and an unchanged official VBM numerical core."""
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np

from prepare_vbm3d_r70 import BASE, ROOT, digest, write

MODES = {'spatial_only': 0, 'video_default': 4}


def first_frame_sigma(images):
    assert images.ndim == 4 and images.shape[-1] == 3 and np.isfinite(images).all()
    first = images[0].astype(np.float64)
    h, w = first.shape[0]//2*2, first.shape[1]//2*2
    hh = (first[:h:2, :w:2]-first[:h:2, 1:w:2]-first[1:h:2, :w:2]+first[1:h:2, 1:w:2])/2
    coefficients = hh.reshape(-1, 3)
    center = np.median(coefficients, axis=0)
    per_channel = np.median(np.abs(coefficients-center), axis=0)/.6744897501960817
    sigma = float(np.median(per_channel)); assert np.isfinite(sigma) and sigma >= 0
    return sigma, dict(source='first input frame only', per_channel_sigma=per_channel.tolist(),
                       estimator='nonoverlapping Haar HH centered MAD / .6744897501960817; median RGB',
                       calibrated_sensor_noise=False, target_or_known_degradation_used=False)


def load_tool():
    path = BASE/'process/exploration-r70/build001/completed.json'
    info = json.loads(path.read_text()); assert info['completed'] and info['original_sources_unmodified']
    for key, sha_key in [('binary', 'binary_sha256'), ('FFTW_library', 'FFTW_library_sha256')]:
        target = ROOT/info[key]
        assert target.resolve().is_relative_to('/data/Users/zrh/data/MobileIE-Neo') and digest(target) == info[sha_key]
    assert digest(BASE/'scripts/vbm3d_array_r70.cpp') == info['adapter_source_sha256']
    return info


def filter_video(images, sigma, mode, work, tool=None):
    assert mode in MODES and images.dtype == np.float32 and images.ndim == 4 and images.shape[-1] == 3
    assert np.isfinite(images).all() and images.min() >= 0 and images.max() <= 255
    assert np.isfinite(sigma) and sigma >= 0 and np.little_endian
    work = Path(work); assert work.resolve().is_relative_to('/data/Users/zrh/data/MobileIE-Neo')
    work.mkdir(parents=True, exist_ok=False)
    tool = load_tool() if tool is None else tool
    start = time.perf_counter()
    if sigma == 0:
        result = images.copy(); command, native_seconds = None, 0.
    else:
        input_path, output_path = work/'input-tchw.f32', work/'native-output-tchw.f32'
        images.transpose(0, 3, 1, 2).copy().tofile(input_path)
        t, h, w, _ = images.shape
        command = [str(ROOT/tool['binary']), str(input_path), str(output_path), str(w), str(h), str(t), format(sigma, '.9g'), str(MODES[mode])]
        env = os.environ.copy(); env.update(OMP_NUM_THREADS='2', OMP_DYNAMIC='FALSE')
        began = time.perf_counter()
        with (work/'native.log').open('xb') as log:
            subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        native_seconds = time.perf_counter()-began
        values = np.fromfile(output_path, dtype=np.float32)
        assert values.size == images.size
        result = values.reshape(t, 3, h, w).transpose(0, 2, 3, 1).copy()
    assert result.shape == images.shape and result.dtype == np.float32 and np.isfinite(result).all()
    record = dict(mode=mode, sigma_8bit=sigma, shape=list(images.shape), Nf=MODES[mode],
                  command=command, bypassed_zero_sigma=sigma == 0, native_filter_seconds=native_seconds,
                  full_filter_stage_seconds=time.perf_counter()-start, binary_sha256=tool['binary_sha256'],
                  raw_output_min=float(result.min()), raw_output_max=float(result.max()),
                  scheduler='offline full provided clip' if mode == 'video_default' else 'same core, spatial search only with fixed first-frame calibration',
                  scope='CPU2 plus array I/O. Not exact dependency bound, phone timing or calibrated AWGN/ISP reproduction.')
    write(work/'run.json', record)
    return result, record
