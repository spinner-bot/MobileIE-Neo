"""Pre-DEV invariants for full-grid flow composition and history semantics."""
import json
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from coordinate_motion_r46 import bidirectional, compose, predict
from coordinate_state_r46 import ROOT, BASE, truth, anchored, classical, ROI, VELOCITIES
from half_motion_r4 import half_forward_backward


def main():
    out = BASE/'results/exploration-r46-motion-gate001'
    out.mkdir(exist_ok=False)
    cv2.setNumThreads(1)
    rows, start = [], time.perf_counter()
    rng = np.random.default_rng(4602)
    for index in range(4):
        previous = rng.uniform(0, 255, (64, 80, 3)).astype(np.float32)
        current = np.roll(previous, index, axis=1)
        back, mask = half_forward_backward(previous, current)
        new_back, new_mask, _, _ = bidirectional(previous, current)
        delta = float(np.max(np.abs(back-new_back)))
        rows.append(dict(kind='unchanged_half_FB', case=index, delta=delta,
                         mask_equal=bool(np.array_equal(mask, new_mask)),
                         passed=delta == 0 and np.array_equal(mask, new_mask)))
    for index, (vx, vy) in enumerate(VELOCITIES):
        gt = truth((vx, vy))
        raw = gt + rng.normal(0, .03, gt.shape).astype(np.float32)
        h, w = raw.shape[1:3]
        back = np.empty((h, w, 2), np.float32)
        back[..., 0], back[..., 1] = -vx, -vy
        mask = np.ones((h, w), bool)
        fields = [(back, mask, -back, mask)]*(len(raw)-1)
        for period in [1, 8]:
            values, _ = predict(raw, fields, .5, period)
            oracle = anchored(raw, (vx, vy), .5, period)
            delta = float(np.max(np.abs(values['anchor8'][:, ROI[0], ROI[1]]
                                       - oracle[:, ROI[0], ROI[1]])))
            rows.append(dict(kind='oracle_central', velocity=index, period=period,
                             delta=delta, passed=delta <= 1e-6))
            if period == 1:
                delta = float(np.max(np.abs(values['anchor8']-values['same_mask'])))
                rows.append(dict(kind='K1_matched_full', velocity=index,
                                 delta=delta, passed=delta <= 1e-6))
            assert all(np.array_equal(v[0], raw[0]) for v in values.values())
        q, c = np.zeros_like(back), np.zeros_like(back)
        va, vc = mask.copy(), mask.copy()
        for step in range(1, 9):
            q, c, va, vc = compose(q, c, va, vc, *fields[0])
            expected = np.array([vx, vy], np.float32)*step
            delta = max(float(np.max(np.abs(q[ROI]-expected))),
                        float(np.max(np.abs(c[ROI]+expected))))
            rows.append(dict(kind='composed_direction', velocity=index, step=step,
                             delta=delta, passed=delta <= 1e-6))
        reject = np.zeros_like(mask)
        values, _ = predict(raw, [(back, reject, -back, reject)]*(len(raw)-1))
        delta = max(float(np.max(np.abs(v-raw))) for v in values.values())
        rows.append(dict(kind='rejected_raw_fallback', velocity=index,
                         delta=delta, passed=delta == 0))
    passed = all(row['passed'] for row in rows)
    record = dict(at=datetime.now().astimezone().isoformat(), passed=bool(passed),
                  cases=rows, seconds=time.perf_counter()-start, gpu=False,
                  model_calls=0, dataset_pixels=0)
    (out/'gate.json').write_text(json.dumps(record, indent=2)+'\n')
    assert passed, 'Preserved failed fixture; do not read DEV.'
    print(json.dumps({'passed': True, 'cases': len(rows), 'seconds': record['seconds']}))


if __name__ == '__main__':
    main()
