"""Literal source-expression arithmetic; never imports or runs the upstream Rust project.

Scope: src/main.rs@59bd6746c6cccde4197ccfabb1d60fb509127436.
No media, model, benchmarking, application run or Gate3 experiment.
Values below are transcribed for inspection, not a certified Rust/SIMD emulator.
"""
import json
from pathlib import Path


def nonlinear(d):
    if d == 0:
        return 0
    if d < 3:
        return 32 + (d - 1) * 32
    if d < 6:
        return 128 + (d - 3) * 128 // 3
    if d < 11:
        return 256 + (d - 6) * 256 // 5
    if d < 33:
        return 512 - (d - 11) * 256 // 22
    return max(128, 256 - (d - 33) * 128 // 223)


def simd_selected_nonlinear(d):
    # Only nonnegative arithmetic in the selected segment; masked inactive
    # lane operations and machine integer code generation are not emulated.
    if d == 0:
        return 0
    if d <= 2:
        return 32 * d
    if d <= 5:
        return 128 + (d - 3) * 43
    if d <= 10:
        return min(512, 256 + (d - 6) * 51)
    if d <= 32:
        return 512 - (d - 11) * 12
    return max(128, 256 - (d - 33) * 128 // 223)


def luma(v):
    if v < 32:
        return 256 + (32 - v) * 192 // 32
    if v < 64:
        return 256 + (64 - v) * 192 // 32
    if v < 160:
        return 256
    if v < 192:
        return 256 - (v - 160) * 128 // 32
    return max(128, 128 - (v - 192) * 128 // 64)


def main():
    mismatches = [dict(diff=i, scalar_Q8=nonlinear(i),
                       SIMD_Q8=simd_selected_nonlinear(i))
                  for i in range(256)
                  if nonlinear(i) != simd_selected_nonlinear(i)]
    motion, value = [], 255
    for _ in range(256):
        motion.append(value)
        value = value * 209 // 256
    result = {
        'resource_id': 'MIE-0006',
        'method': __doc__,
        'tone_curve_selection': [
            dict(condition='grad < 16', SIMD='enhanced (no curve)', scalar_tail='curve'),
            dict(condition='grad >= 16', SIMD='curve', scalar_tail='enhanced (no curve)'),
        ],
        'nonlinear_gain_mismatch_count': len(mismatches),
        'nonlinear_gain_mismatches': mismatches,
        'luma_samples': [dict(avg_Y=i, Q8=luma(i))
                         for i in [0, 31, 32, 63, 64, 159, 160, 191, 192, 255]],
        'motion_first_early_return_MAD': next(i for i, x in enumerate(motion) if x < 16),
        'motion_values_first_18': motion[:18],
        'denoise_identity_proof':
            'w=128 only if |y-b|=0, so y=b and rounded weighted blend=y; '
            'else w=0 and blend=y. Only this blend stage, not whole enhancement.',
        'chroma_tick_schedule':
            'tick=0 recomputes activity, then increments modulo8; '
            'saturation applied only if resulting tick=0 and total_adj>0. '
            'No saturation boost on the other seven frames.',
        'not_measured': ['actual visual/flicker effect', 'compiled SIMD equivalence',
                         'runtime initialization behavior', 'latency/memory/energy'],
    }
    target = Path(__file__).with_name('static-arithmetic-audit.json')
    assert not target.exists(), 'Preserve previous audit; append revision instead'
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print('Source arithmetic only:', len(mismatches), 'nonlinear expression differences;',
          'first motion skip MAD:', result['motion_first_early_return_MAD'])


if __name__ == '__main__':
    main()
