"""Fixed classic controls, with approximate rather than calibrated uncertainty."""
import cv2
import numpy as np

from aggregation_controls_r20 import blend
from aligned_raw_history_r11 import align_history
from video_metrics import warp_previous

METHODS = ('raw', 'finite_self05', 'recursive_self05', 'adaptive_recursive')


def observation_variance(raw):
    h, w = raw.shape[:2]
    crop = raw[:h-h % 2, :w-w % 2]
    hh = (crop[::2, ::2] - crop[::2, 1::2]
          - crop[1::2, ::2] + crop[1::2, 1::2]) * .5
    sigma = float(np.median(np.abs(hh - np.median(hh)))) / .67448975
    return max(sigma*sigma, 1e-4)


class Recursion:
    """The raw motion buffer is counted externally, in addition to our state."""
    def __init__(self):
        self.fixed = self.adaptive = self.variance = None

    def step(self, raw, previous_raw=None, flow=None, mask=None):
        r = observation_variance(raw)
        if self.fixed is None:
            assert previous_raw is None
            self.fixed, self.adaptive = raw.copy(), raw.copy()
            self.variance = np.full(raw.shape[:2], r, dtype=np.float32)
            return {name: raw for name in METHODS}, {
                'r_proxy': r, 'current_weight_mean': 1., 'valid_fraction': 0.}
        assert previous_raw is not None and flow is not None and mask is not None
        finite_history, valid = align_history(previous_raw, raw, flow, mask)
        fixed_history, _ = align_history(self.fixed, raw, flow, mask)
        adaptive_history, _ = align_history(self.adaptive, raw, flow, mask)
        p, _ = warp_previous(self.variance, flow)
        innovation = np.mean((raw-adaptive_history)**2, axis=2)
        local = cv2.boxFilter(innovation, -1, (5, 5), normalize=True,
                              borderType=cv2.BORDER_REFLECT_101)
        process = np.maximum(local-r-p, 0)
        prior = p+process
        weight = prior/(prior+r)
        weight = np.where(valid, weight, 1.).astype(np.float32)
        self.fixed = blend(raw, fixed_history, .5)
        self.adaptive = np.where(valid[..., None],
                                 adaptive_history+weight[..., None]*(raw-adaptive_history), raw).astype(np.float32)
        self.variance = np.where(valid, (1-weight)*prior, r).astype(np.float32)
        return {'raw': raw, 'finite_self05': blend(raw, finite_history, .5),
                'recursive_self05': self.fixed, 'adaptive_recursive': self.adaptive}, {
                    'r_proxy': r, 'current_weight_mean': float(weight.mean()),
                    'valid_fraction': float(valid.mean()), 'posterior_proxy_mean': float(self.variance.mean())}

    def history_bytes(self, previous_raw):
        return {'raw': 0, 'finite_self05': previous_raw.nbytes,
                'recursive_self05': previous_raw.nbytes+self.fixed.nbytes,
                'adaptive_recursive': previous_raw.nbytes+self.adaptive.nbytes+self.variance.nbytes}
