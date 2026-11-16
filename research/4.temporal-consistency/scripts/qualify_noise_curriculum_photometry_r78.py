"""Artificial scalar photometry qualification, without videos, models or training."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / 'research/4.temporal-consistency'


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    out = args.out.resolve()
    assert out.is_relative_to((BASE / 'results').resolve())
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    n, sigma = 65536, .008
    y = np.array([.1, .3, .6, .9], dtype=np.float64)
    dark = .35 * y ** 2.2
    noise = np.random.default_rng(78001).normal(0, sigma, (len(y), n))
    x = dark[:, None] + noise
    rows, checks = [], []
    for g in (16., 1.):
        # Literal residual-scaling arithmetic, applied to a different photometric task.
        naive_neighbor = y[:, None] + (x - y[:, None]) / g
        naive_current = y[:, None] + g * (x - y[:, None])
        dark_neighbor = dark[:, None] + (x - dark[:, None]) / g
        dark_current = dark[:, None] + g * (x - dark[:, None])
        naive_neighbor_mean = y + (dark - y) / g
        naive_current_mean = y + g * (dark - y)
        for i, bright in enumerate(y):
            row = dict(bright_GT=float(bright), dark_clean=float(dark[i]), factor=g,
                noiseless_naive_neighbor=float(naive_neighbor_mean[i]),
                noiseless_naive_current=float(naive_current_mean[i]),
                noiseless_naive_neighbor_shift=float(naive_neighbor_mean[i] - dark[i]),
                noiseless_naive_current_shift=float(naive_current_mean[i] - dark[i]),
                streams={})
            for name, data, expected_mean, expected_sigma in (
                ('bright_GT_neighbor', naive_neighbor[i], naive_neighbor_mean[i], sigma / g),
                ('bright_GT_current', naive_current[i], naive_current_mean[i], sigma * g),
                ('dark_clean_neighbor', dark_neighbor[i], dark[i], sigma / g),
                ('dark_clean_current', dark_current[i], dark[i], sigma * g)):
                sem = expected_sigma / np.sqrt(n)
                mean = float(data.mean())
                variance = float(data.var())
                clipped = np.clip(data, 0, 1)
                row['streams'][name] = dict(mean=mean, expected_mean=float(expected_mean),
                    expected_SEM=float(sem), mean_error_SEM=float(abs(mean - expected_mean) / sem),
                    variance=variance, expected_variance=float(expected_sigma ** 2),
                    variance_relative_error=float(abs(variance / expected_sigma ** 2 - 1)),
                    clipped_mean=float(clipped.mean()),
                    clipped_mean_shift_from_dark=float(clipped.mean() - dark[i]),
                    fraction_clipped=float(np.mean((data < 0) | (data > 1))))
                checks.append(abs(mean - expected_mean) <= 6 * sem)
                checks.append(abs(variance / expected_sigma ** 2 - 1) <= .03)
            # Evaluate the noiseless domain-preserving control, not a target-noise oracle.
            checks.append(abs((dark[i] + (dark[i] - dark[i]) / g) - dark[i]) <= 2e-15)
            checks.append(abs((dark[i] + g * (dark[i] - dark[i])) - dark[i]) <= 2e-15)
            checks.append(abs((y[i] + (dark[i] - y[i]) / g - dark[i]) -
                              (1 - 1 / g) * (y[i] - dark[i])) <= 2e-15)
            rows.append(row)
        if g == 1:
            checks.extend(float(np.max(np.abs(v - x))) <= 2e-15
                          for v in (naive_neighbor, naive_current, dark_neighbor, dark_current))
    assert all(np.isfinite(x).all() for x in (dark, noise))
    shifted = [r['noiseless_naive_neighbor_shift'] for r in rows if r['factor'] == 16]
    protocol = BASE / 'docs/exploration-r78-noise-curriculum-photometric-protocol.md'
    record = dict(at=datetime.now().astimezone().isoformat(), author='Codex  (sqs_A100)',
        classification='Artificial photometric construction qualification, not model performance',
        protocol_sha256=hashlib.sha256(protocol.read_bytes()).hexdigest(),
        seed=78001, draws_per_brightness=n, brightness_values=y.tolist(), sigma=sigma,
        factors=[16, 1], rows=rows, checks=len(checks), checks_passed=sum(bool(c) for c in checks),
        qualification_passed=all(checks),
        noiseless_naive_neighbor_shift_range=[min(shifted), max(shifted)],
        naive_photometry_preserved_at16=False,
        known_dark_mean_control_preserved_before_clip=True,
        clipping_preserves_mean_claim=False,
        real_paired_RGB_noise_identified=False,
        causal_curriculum_effectiveness_established=False,
        new_video_pixels=0, new_CNN=0, new_SGD=0, new_GPU_jobs=0,
        scope='Known artificial dark mean; no real assets, third-party code execution, temporal performance, method novelty or noise-calibration claim. Same draws across endpoints, not independent experiments.')
    write(out / 'result.json', record)
    write(out / 'runtime.json', dict(ended=datetime.now().astimezone().isoformat(),
        completed_normally=all(checks), stop_reason='complete' if all(checks) else 'qualification_failed',
        seconds=time.perf_counter() - started, rows=len(rows), CUDA_initialized=False,
        new_video_pixels=0, new_CNN=0, new_SGD=0))
    print(json.dumps(dict(rows=len(rows), checks_passed=record['checks_passed'], checks=len(checks),
        qualification_passed=all(checks), noiseless_neighbor_shift_range=record['noiseless_naive_neighbor_shift_range'],
        seconds=time.perf_counter() - started)))
    if not all(checks):
        raise SystemExit('Qualification failure retained; do not tune or replay.')


if __name__ == '__main__':
    main()
