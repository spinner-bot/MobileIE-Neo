"""Closed-form fixtures for final evaluation state, units, fallback and alpha counts."""
from datetime import datetime
import json

import numpy as np

from dynamic_training_r34 import BASE, digest
from evaluate_response_recursion_r45 import ALPHAS, predict
from response_cue_r45 import RecursiveControl


def main():
    height, width = 12, 16
    levels = [0., 255., 0., 255.]
    raw = np.stack([np.full((height, width, 3), v, np.float32) for v in levels])
    cues = np.zeros((4, 2, height, width), np.float32)
    controls = {n: RecursiveControl(n, 0).eval() for n in ['scalar', 'rgb', 'response']}
    results = []
    for rejected_frame in [None, 2]:
        fields = [(np.zeros((height, width, 2), np.float32),
                   np.full((height, width), t != rejected_frame)) for t in range(1, 4)]
        for method in [*ALPHAS, *controls]:
            alpha = ALPHAS.get(method, .5)
            expected = [levels[0]]
            for t in range(1, 4):
                expected.append(levels[t] if t == rejected_frame else (1-alpha)*levels[t]+alpha*expected[-1])
            output, stats = predict(method, controls, raw, cues, fields)
            expected = np.asarray(expected, dtype=np.float32)[:, None, None, None]
            delta = float(np.max(np.abs(output-expected)))/255
            assert delta <= 1e-5 and np.array_equal(output[0], raw[0])
            count = (3-int(rejected_frame is not None))*height*width
            assert stats['accepted_pixels'] == count
            assert sum(stats['alpha_histogram_20_bins_0_to_1']) == count
            assert abs(stats['mean_accepted_alpha']-alpha) <= 1e-6
            results.append({'method': method, 'rejected_frame': rejected_frame,
                            'expected_levels': expected[:, 0, 0, 0].tolist(),
                            'max_abs_normalized_delta': delta, 'accepted_pixels': count})
    out = BASE/'process/exploration-r45/evaluation-fixtures.json'
    assert not out.exists()
    out.write_text(json.dumps({'checked_at': datetime.now().astimezone().isoformat(),
        'passed': True, 'fixtures': results,
        'evaluator_sha256': digest(BASE/'scripts/evaluate_response_recursion_r45.py'),
        'scope': 'Closed-form artificial frames only, no CNN inference, training or DEV quality.'}, indent=2)+'\n')
    print(json.dumps({'passed': True, 'fixtures': len(results)}))


if __name__ == '__main__':
    main()
