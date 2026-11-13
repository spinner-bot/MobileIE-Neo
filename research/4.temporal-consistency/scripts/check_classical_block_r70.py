"""Small artificial array/color/filter/support qualification before real pixels."""
from datetime import datetime
import json
import time
import traceback

import numpy as np

from classical_block_r70 import MODES, filter_video, first_frame_sigma, load_tool
from prepare_vbm3d_r70 import BASE, ROOT, digest, write


def main():
    out = BASE/'process/exploration-r70/cpu-gate001'; out.mkdir(exist_ok=False)
    cache = ROOT/'.cache/experiments/exploration-r70/cpu-gate001'
    assert cache.resolve().is_relative_to('/data/Users/zrh/data/MobileIE-Neo')
    cache.mkdir(parents=True, exist_ok=False)
    start, complete, calls = time.perf_counter(), False, 0
    try:
        tool = load_tool()
        names = ['check_classical_block_r70.py', 'classical_block_r70.py', 'prepare_vbm3d_r70.py', 'vbm3d_array_r70.cpp']
        frozen = {name: digest(BASE/'scripts'/name) for name in names}
        write(out/'config.json', dict(at=datetime.now().astimezone().isoformat(), shape=[5, 64, 64, 3],
              source_sha256=frozen, original_core_binary_sha256=tool['binary_sha256'],
              protocol_sha256=digest(BASE/'docs/exploration-r70-classical-block-reference-protocol.md'),
              scope='Artificial tensors only; not DEV/phone/full-paper reproduction or model ranking.'))
        constant = np.empty((5, 64, 64, 3), np.float32); constant[:] = [40, 100, 180]
        constant_sigma, _ = first_frame_sigma(constant); assert constant_sigma == 0
        constants, noise_checks, predictions = {}, {}, {}
        for mode in MODES:
            output, record = filter_video(constant, 10., mode, cache/('constant-'+mode), tool)
            calls += 1; error = float(np.abs(output-constant).max())
            assert error <= .05
            constants[mode] = dict(passed=True, constant_RGB_max_abs_error_8bit=error, run=record)
        y, x = np.mgrid[:64, :64].astype(np.float32)
        image = np.stack([64+.8*x, 96+.4*y, 128+.2*(x+y)], -1)
        clean = np.repeat(image[None], 5, axis=0)
        noisy = np.clip(clean+np.random.default_rng(7001).normal(0, 20, clean.shape), 0, 255).astype(np.float32)
        before = noisy.copy(); noisy_mse = float(np.mean((noisy.astype(np.float64)-clean)**2))
        sigma, sigma_info = first_frame_sigma(noisy); assert np.isfinite(sigma) and sigma > 0
        for mode in MODES:
            output, record = filter_video(noisy, 20., mode, cache/('noisy-'+mode), tool)
            calls += 1; mse = float(np.mean((output.astype(np.float64)-clean)**2))
            assert mse < noisy_mse and np.array_equal(before, noisy)
            noise_checks[mode] = dict(passed=True, noisy_MSE=noisy_mse, filtered_MSE=mse, run=record)
            predictions[mode] = output
        changed = noisy.copy(); changed[-1] = 255-changed[-1]
        changed_sigma, _ = first_frame_sigma(changed); assert changed_sigma == sigma
        output, record = filter_video(changed, 20., 'spatial_only', cache/'spatial-support', tool); calls += 1
        delta = float(np.abs(output[2]-predictions['spatial_only'][2]).max()); assert delta <= 1e-4
        write(out/'gate.json', dict(passed=True, constant_checks=constants, noise_checks=noise_checks,
              spatial_only_other_frame_central_max_abs=delta, first_frame_sigma_unchanged=True,
              estimated_sigma_in_artificial_case=sigma, sigma_info=sigma_info, artificial_filter_calls=calls,
              new_real_images=0, new_CNN=0, new_SGD=0, new_GPU=0,
              scope='One fixed artificial fixture, no sigma/precision/search grid or real-quality conclusion.'))
        assert all(digest(BASE/'scripts'/name) == sha for name, sha in frozen.items()); complete = True
    except BaseException:
        (out/'failure.txt').write_text(traceback.format_exc()); raise
    finally:
        write(out/'runtime.json', dict(completed_normally=complete, artificial_filter_calls=calls,
              seconds=time.perf_counter()-start, new_real_images=0, new_CNN=0, new_SGD=0, new_GPU=0,
              ended=datetime.now().astimezone().isoformat()))


if __name__ == '__main__': main()
