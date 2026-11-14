"""Bounded state/control checks; synthetic noise statistics are not DEV evidence."""
import json
from pathlib import Path

import cv2
import numpy as np

from classical_recursion_r36 import METHODS, Recursion
from dynamic_training_r34 import BASE, digest


def main():
    cv2.setNumThreads(1)
    rng = np.random.default_rng(36)
    raw = (100+rng.normal(0, 10, (64, 64, 3))).astype(np.float32)
    zero = np.zeros((64, 64, 2), np.float32)
    valid = np.ones((64, 64), bool)
    state = Recursion()
    first, _ = state.step(raw)
    assert all(np.array_equal(first[name], raw) for name in METHODS)
    next_raw = (180+rng.normal(0, 10, raw.shape)).astype(np.float32)
    rejected, _ = state.step(next_raw, raw, zero, ~valid)
    assert all(np.array_equal(rejected[name], next_raw) for name in METHODS)
    assert np.isfinite(state.variance).all() and (state.variance > 0).all()
    # Same physical surface with independent sensor noise; no synthetic motion
    # oracle enters an actual benchmark. The check detects broken recurrence.
    state, previous, errors = Recursion(), None, {name: [] for name in METHODS}
    for t in range(32):
        current = (100+rng.normal(0, 10, raw.shape)).astype(np.float32)
        values, _ = state.step(current, previous, zero if t else None, valid if t else None)
        if t >= 24:
            for name, value in values.items():
                errors[name].append(float(np.mean((value-100)**2)))
        previous = current
    mse = {name: float(np.mean(values)) for name, values in errors.items()}
    assert mse['recursive_self05'] < .45*mse['raw']
    assert mse['adaptive_recursive'] < .7*mse['raw']
    # A single finite step must exactly match the existing baseline primitive.
    fresh = Recursion()
    fresh.step(raw)
    values, _ = fresh.step(next_raw, raw, zero, valid)
    assert np.array_equal(values['finite_self05'], next_raw+.5*(raw-next_raw))
    result = {'passed': True, 'first_frame_equal': True, 'rejected_history_falls_back_current': True,
              'finite_matches_existing_primitive': True, 'static_iid_noise_mse': mse,
              'online_history_storage_bytes_64x64': state.history_bytes(previous),
              'source_sha256': {name: digest(Path(__file__).parent/name) for name in
                                ['classical_recursion_r36.py', 'check_classical_r36.py']},
              'scope': 'Small synthetic functional checks only; not natural motion, DEV quality, calibrated uncertainty or speed.'}
    out = BASE/'process/exploration-r36/interface-checks.json'
    assert not out.exists()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
