"""Fixed target-contributor flow dispersion; not calibrated uncertainty or novel splatting."""
import cv2
import numpy as np
from fixed_dis_r86 import EXPECTED
from video_metrics import warp_previous


class DISFields:
    def __init__(self):
        assert cv2.__version__ == '4.11.0'
        self.back = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST)
        self.forward = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST)
        for solver in (self.back, self.forward):
            assert {k: getattr(solver, 'get' + k)() for k in EXPECTED} == EXPECTED
        self.pairs = 0

    def __call__(self, previous, current):
        def gray(image):
            return cv2.cvtColor(np.clip(np.rint(image), 0, 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        a, b = gray(previous), gray(current)
        back, forward = self.back.calc(b, a, None), self.forward.calc(a, b, None)
        assert np.isfinite(back).all() and np.isfinite(forward).all()
        sampled, valid = warp_previous(forward, back)
        threshold = 1 + .05 * (np.linalg.norm(back, axis=2) + np.linalg.norm(sampled, axis=2))
        mask = valid & (np.linalg.norm(back + sampled, axis=2) < threshold)
        self.pairs += 1
        return back, forward, mask


class SplatField:
    def __init__(self, forward):
        assert forward.ndim == 3 and forward.shape[-1] == 2 and np.isfinite(forward).all()
        self.h, self.w = forward.shape[:2]
        y, x = np.indices((self.h, self.w), dtype=np.float64)
        target_x, target_y = x + forward[..., 0], y + forward[..., 1]
        bx, by = np.floor(target_x).astype(np.int64), np.floor(target_y).astype(np.int64)
        fx, fy = target_x - bx, target_y - by
        xs = np.stack((bx, bx + 1, bx, bx + 1), axis=-1)
        ys = np.stack((by, by, by + 1, by + 1), axis=-1)
        weights = np.stack(((1 - fx) * (1 - fy), fx * (1 - fy), (1 - fx) * fy, fx * fy), axis=-1)
        valid = (xs >= 0) & (xs < self.w) & (ys >= 0) & (ys < self.h) & (weights > 0)
        self.index = (ys.clip(0, self.h - 1) * self.w + xs.clip(0, self.w - 1)).reshape(-1)
        self.weights = np.where(valid, weights, 0).reshape(-1)
        self.mass = self._sum(np.ones((self.h, self.w), np.float64))
        self.covered = self.mass > 1e-6
        self.mean_flow = np.stack([self._mean(forward[..., c]) for c in range(2)], axis=-1)
        second = self._mean(np.sum(forward.astype(np.float64) ** 2, axis=-1))
        self.variance = np.maximum(second - np.sum(self.mean_flow ** 2, axis=-1), 0)
        self.accepted = self.covered & (np.sqrt(self.variance) < 1 + .1 * np.linalg.norm(self.mean_flow, axis=-1))

    def _sum(self, value):
        weighted = np.repeat(value.astype(np.float64).reshape(-1), 4) * self.weights
        return np.bincount(self.index, weights=weighted, minlength=self.h * self.w).reshape(self.h, self.w)

    def _mean(self, value):
        return self._sum(value) / np.maximum(self.mass, 1e-6)

    def apply(self, previous, current, reject_dispersion):
        moved = np.stack([self._mean(previous[..., c]) for c in range(3)], axis=-1).astype(np.float32)
        accepted = self.accepted if reject_dispersion else self.covered
        moved = np.where(accepted[..., None], moved, current)
        return current + .5 * (moved - current)


def qualification():
    image = np.full((16, 32, 3), 82.25, np.float32)
    zero = SplatField(np.zeros((16, 32, 2), np.float32))
    identity_difference = float(np.max(np.abs(zero.apply(image, image, True) - image)))
    assert identity_difference == 0 and zero.accepted.all()
    translation = np.broadcast_to(np.array([.25, -.5], np.float32), (16, 32, 2)).copy()
    translated = SplatField(translation)
    translation_variance = float(translated.variance[translated.covered].max())
    translation_difference = float(np.max(np.abs(translated.apply(image, image, True) - image)))
    assert translation_variance <= 1e-10 and translation_difference <= 1e-6
    collision = np.zeros((16, 32, 2), np.float32)
    collision[:, :16, 0], collision[:, 16:, 0] = 4, -4
    overlapped = SplatField(collision)
    assert (overlapped.variance[:, 12:20] > 0).all() and not overlapped.accepted[:, 12:20].any()
    return dict(passed=True, identity_constant_max_difference=identity_difference,
                subpixel_translation=[.25, -.5], translation_max_variance=translation_variance,
                translation_constant_max_difference=translation_difference,
                collision_overlap_pixels=16 * 8, collision_all_rejected=True,
                collision_variance=float(overlapped.variance[:, 12:20].mean()),
                new_NN_images=0, new_flow_estimations=0,
                scope='Three fixed pure-array implementation cases, not physical uncertainty calibration')
