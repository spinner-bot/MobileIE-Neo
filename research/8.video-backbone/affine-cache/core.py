"""Fixed A1 color-guided affine fit and keyframe stream; all components known."""
import cv2
import numpy as np


def mean(value):
    shape = value.shape
    flat = value.reshape(*shape[:2], -1)
    result = cv2.boxFilter(flat, -1, (5, 5), normalize=True, borderType=cv2.BORDER_REFLECT_101)
    return result.reshape(shape)


def fit(low, target):
    if low.shape != target.shape or low.ndim != 3 or low.shape[-1] != 3:
        raise ValueError("Expected matching HWC RGB guide and displayed F0 target")
    height, width = low.shape[:2]
    size = ((width + 3) // 4, (height + 3) // 4)
    x = cv2.resize(low.astype(np.float64), size, interpolation=cv2.INTER_AREA)
    y = cv2.resize(target.astype(np.float64), size, interpolation=cv2.INTER_AREA)
    mx, my = mean(x), mean(y)
    cxx = mean(x[..., :, None] * x[..., None, :]) - mx[..., :, None] * mx[..., None, :]
    cxy = mean(x[..., :, None] * y[..., None, :]) - mx[..., :, None] * my[..., None, :]
    a = np.linalg.solve(cxx + 1e-4 * np.eye(3), cxy)
    b = my - np.einsum("...i,...io->...o", mx, a)
    packed = np.concatenate([mean(a).reshape(*a.shape[:2], 9), mean(b)], axis=-1)
    assert np.isfinite(packed).all()
    return np.ascontiguousarray(packed, dtype=np.float32)


def render(low, packed):
    if low.ndim != 3 or low.shape[-1] != 3 or packed.ndim != 3 or packed.shape[-1] != 12:
        raise ValueError("Expected RGB and packed12 coefficient field")
    full = cv2.resize(packed, (low.shape[1], low.shape[0]), interpolation=cv2.INTER_LINEAR)
    a = full[..., :9].reshape(*low.shape[:2], 3, 3)
    result = np.einsum("...i,...io->...o", low, a) + full[..., 9:]
    assert np.isfinite(result).all()
    return result.astype(np.float32)


class AffineStream:
    def __init__(self, mode, provider):
        if mode not in ("current", "cached", "hold"):
            raise ValueError("Unknown registered route")
        self.mode, self.provider = mode, provider
        self.index = 0
        self.field = self.held = None

    def step(self, low):
        key = self.mode == "current" or self.index % 2 == 0
        if key:
            self.field = fit(low, self.provider(self.index))
        if self.mode == "hold":
            if key:
                self.held = render(low, self.field).copy()
                self.field = None
            result = self.held.copy()
        else:
            result = render(low, self.field)
        self.index += 1
        return result

    def state_bytes(self):
        return sum(v.nbytes for v in (self.field, self.held) if v is not None)


def sequence(low, displayed_f0, mode):
    requests = []

    def provider(index):
        if mode != "current" and index % 2:
            raise AssertionError("Non-key F0 forbidden for cached/hold route")
        requests.append(index)
        return displayed_f0[index]

    stream = AffineStream(mode, provider)
    output = np.stack([stream.step(frame) for frame in low])
    return output, requests, stream.state_bytes()
