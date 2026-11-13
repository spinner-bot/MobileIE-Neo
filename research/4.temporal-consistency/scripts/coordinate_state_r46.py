"""Fixed, oracle-motion resampling screen. No model, flow estimation or GPU."""
import argparse
import csv
import hashlib
import json
import math
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / 'research/4.temporal-consistency'
ALPHAS = [.1, .25, .5, .75, .9]
VELOCITIES = [(0., 0.), (1., 0.), (.375, .25)]
H, W, T, MARGIN = 192, 224, 32, 48
ROI = (slice(MARGIN, -MARGIN), slice(MARGIN, -MARGIN))
Y, X = np.mgrid[:H, :W].astype(np.float32)


def stamp():
    return datetime.now().astimezone().isoformat()


def dump(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')


def truth(velocity):
    vx, vy = velocity
    sequence = []
    for t in range(T):
        value = np.full((H, W, 3), .5, np.float64)
        for amplitude, fx, fy, phase in [(.10, .08, .05, .4),
                                         (.08, .23, -.16, .7),
                                         (.06, .37, .10, 1.1)]:
            # Float64 analytic scene; cast only the completed sampled frame.
            arg = 2 * np.pi * (fx * (X.astype(np.float64) - vx*t)
                               + fy * (Y.astype(np.float64) - vy*t))
            value += amplitude * np.sin(arg[..., None] + np.arange(3)*phase)
        sequence.append(value.astype(np.float32))
    return np.stack(sequence)


def sample(image, dx, dy):
    mx, my = X + dx, Y + dy
    valid = (mx >= 0) & (mx <= W-1) & (my >= 0) & (my <= H-1)
    moved = cv2.remap(image, mx, my, cv2.INTER_LINEAR,
                      borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return moved, valid


def classical(images, velocity, alpha):
    vx, vy = velocity
    state = images[0].copy()
    result = [state]
    for image in images[1:]:
        moved, valid = sample(state, -vx, -vy)
        state = np.where(valid[..., None], image+alpha*(moved-image), image)
        result.append(state)
    return np.stack(result)


def anchored(images, velocity, alpha, period):
    vx, vy = velocity
    state = images[0].copy()
    anchor = 0
    result = [state]
    for t, image in enumerate(images[1:], 1):
        distance = t-anchor
        moved, valid = sample(state, -distance*vx, -distance*vy)
        output = np.where(valid[..., None], image+alpha*(moved-image), image)
        result.append(output)
        if distance == period:
            state, anchor = output, t
        else:
            incoming, accepted = sample(image, distance*vx, distance*vy)
            state = np.where(accepted[..., None], incoming+alpha*(state-incoming), state)
    return np.stack(result)


def write_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def mse_sequence(prediction, target):
    difference = (prediction[:, ROI[0], ROI[1]].astype(np.float64)
                  - target[:, ROI[0], ROI[1]].astype(np.float64))
    return np.mean(difference**2, axis=(1, 2, 3))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    assert out.is_relative_to(BASE/'results'), out
    assert not out.exists(), 'Preserve every previous attempt; use a new directory.'
    out.mkdir(parents=True)
    asset = (ROOT/'.cache/experiments/exploration-r46'/out.name).resolve()
    assert asset.is_relative_to('/data/Users/zrh/data/MobileIE-Neo')
    asset.mkdir(parents=True, exist_ok=False)
    cv2.setNumThreads(2)
    start, started = time.perf_counter(), stamp()
    protocol = BASE/'docs/exploration-r46-coordinate-state-protocol.md'
    dump(out/'config.json', dict(started=started, protocol=str(protocol.relative_to(ROOT)),
        protocol_sha256=hashlib.sha256(protocol.read_bytes()).hexdigest(),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        alphas=ALPHAS, velocities=VELOCITIES, sigma=[0, .03], noisy_seeds=list(range(4)),
        shape=[T, H, W, 3], roi_margin=MARGIN, period=8, late_frames=[16, 31],
        opencv=cv2.__version__, numpy=np.__version__, cv_threads=cv2.getNumThreads(),
        model_calls=0, gpu_use=False, dataset_pixels=0, assets=str(asset),
        formal_outputs=150, invariant_K1_replays=75))
    invariants, rows, frames, paired = [], [], [], []
    try:
        for index, velocity in enumerate(VELOCITIES):
            gt = truth(velocity)
            if index < 2:
                delta = max(float(np.max(np.abs(sample(gt[t-1], -velocity[0],
                          -velocity[1])[0][ROI]-gt[t][ROI]))) for t in range(1, T))
                invariants.append(dict(kind='analytic_integer_alignment', velocity=index,
                                       delta=delta, passed=delta <= 1e-6))
            for sigma in [0., .03]:
                for seed in ([0] if sigma == 0 else range(4)):
                    # Same observations reused for every alpha and representation.
                    rng = np.random.default_rng(np.random.SeedSequence([46, seed, index]))
                    images = gt.copy() if sigma == 0 else gt + rng.normal(
                        0, sigma, gt.shape).astype(np.float32)
                    raw_mse = mse_sequence(images, gt)
                    for alpha in ALPHAS:
                        previous = classical(images, velocity, alpha)
                        one = anchored(images, velocity, alpha, 1)
                        delta = float(np.max(np.abs(one-previous)))
                        invariants.append(dict(kind='K1_full_frame', velocity=index,
                             sigma=sigma, seed=seed, alpha=alpha, delta=delta, passed=delta <= 1e-6))
                        candidate = anchored(images, velocity, alpha, 8)
                        if index < 2:
                            delta = float(np.max(np.abs(candidate[:, ROI[0], ROI[1]]
                                                        - previous[:, ROI[0], ROI[1]])))
                            invariants.append(dict(kind='K8_integer_central', velocity=index,
                                sigma=sigma, seed=seed, alpha=alpha, delta=delta, passed=delta <= 1e-5))
                        late = {}
                        for method, prediction in [('classical', previous), ('anchor8', candidate)]:
                            mse = mse_sequence(prediction, gt)
                            mean, last = float(mse.mean()), float(mse[16:].mean())
                            late[method] = last
                            common = dict(velocity=index, vx=velocity[0], vy=velocity[1],
                                          sigma=sigma, seed=seed, alpha=alpha, method=method)
                            rows.append(dict(**common, mse=mean, late_mse=last,
                                psnr=None if mean == 0 else -10*math.log10(mean),
                                late_psnr=None if last == 0 else -10*math.log10(last),
                                raw_mse=float(raw_mse.mean()), raw_late_mse=float(raw_mse[16:].mean())))
                            frames.extend(dict(**common, frame=t, mse=float(value))
                                          for t, value in enumerate(mse))
                        paired.append(dict(velocity=index, sigma=sigma, seed=seed, alpha=alpha,
                              classical_mse=late['classical'], anchor_mse=late['anchor8'],
                              late_ratio=None if late['classical'] == 0 else late['anchor8']/late['classical']))
                        if index == 2 and alpha == .75 and seed == 0:
                            np.savez_compressed(asset/f'fractional-sigma{sigma:g}.npz',
                                      truth=gt, input=images, classical=previous, anchor8=candidate)
        write_csv(out/'sequence_metrics.csv', rows)
        write_csv(out/'per_frame_mse.csv', frames)
        dump(out/'paired_late_mse.json', paired)
        dump(out/'invariants.json', invariants)
        assert len(rows) == 150 and len(frames) == 4800 and len(paired) == 75
        assert all(item['passed'] for item in invariants), 'Coordinate invariant failed; quality is not accepted.'
        primary = [r for r in paired if r['velocity'] == 2 and r['alpha'] == .75]
        clean = next(r for r in primary if r['sigma'] == 0)
        noisy = [r for r in primary if r['sigma'] != 0]
        assert len(noisy) == 4
        noisy_ratio = float(np.mean([r['anchor_mse'] for r in noisy]) /
                            np.mean([r['classical_mse'] for r in noisy]))
        passed = clean['late_ratio'] < .95 and noisy_ratio < .95
        dump(out/'audit.json', dict(processing_complete=True, invariants_passed=True,
            rows=len(rows), per_frame_rows=len(frames), primary_alpha=.75,
            clean_late_mse_ratio=clean['late_ratio'], noisy_mean_mse_ratio=noisy_ratio,
            primary_noisy_seed_ratios=[r['late_ratio'] for r in noisy],
            mechanism_gate_passed=passed, local_setting_frozen=not passed,
            next_action='register_real_flow_check' if passed else 'stop_K8_bilinear_state',
            novelty_established=False, real_video_evidence=False, phone_cost=False))
        dump(out/'runtime.json', dict(started=started, ended=stamp(),
             seconds=time.perf_counter()-start, returncode=0, model_calls=0,
             formal_sequences=len(rows), invariant_K1_replays=75, gpu_used=False))
        print((out/'audit.json').read_text(), flush=True)
    except BaseException as error:
        dump(out/'runtime.json', dict(started=started, ended=stamp(),
             seconds=time.perf_counter()-start, returncode=1,
             error=type(error).__name__+': '+str(error)))
        raise


if __name__ == '__main__':
    main()
