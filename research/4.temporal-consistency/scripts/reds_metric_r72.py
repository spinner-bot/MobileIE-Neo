"""Existing fixed ColorVideoVDP, explicit REDS-source 24fps without changing old studies."""
import numpy as np
import torch

from cvvdp_adapter_r66 import artificial_inputs, load_metric


def load24(device):
    metric, source = load_metric(device)
    source.update(playback_fps=24, timing_provenance='REDS author April18 2022 clarification; R71 source-reading001',
                  old_R66_nominal30_conditions_unchanged=True)
    return metric, source


def predict24(metric, test, reference):
    assert test.shape == reference.shape and test.ndim == 4 and test.shape[-1] == 3
    assert test.dtype == reference.dtype == np.float32
    assert np.isfinite(test).all() and np.isfinite(reference).all()
    assert min(test.min(), reference.min()) >= 0 and max(test.max(), reference.max()) <= 1
    with torch.inference_mode():
        score, statistics = metric.predict(test, reference, dim_order='FHWC', frames_per_second=24)
    value = float(score.detach().cpu().reshape(-1)[0]); assert np.isfinite(value)
    assert statistics['N_frames'] == len(test) and statistics['frames_per_second'] == 24
    assert (statistics['height'], statistics['width']) == test.shape[1:3]
    return value, statistics


def qualify24(device, cpu_scores=None):
    metric, source = load24(device); reference, tests = artificial_inputs()
    scores = {name: predict24(metric, value, reference)[0] for name, value in tests.items()}
    assert abs(scores['identity'] - 10) <= 1e-4
    assert scores['large_static'] <= scores['small_static'] < scores['identity']
    differences = None if cpu_scores is None else {k: abs(v - cpu_scores[k]) for k, v in scores.items()}
    assert differences is None or max(differences.values()) <= .01
    return dict(passed=True, scores=scores, CPU_CUDA_absolute_differences=differences,
                source=source, artificial_shape=[16, 64, 64, 3], new_enhancement_CNN=0,
                new_SGD=0, scope='Same interface/layout/finite/identity checks at24fps; no new quality claim.')
