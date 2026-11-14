"""Exact pixelwise GT oracle for clip(g*x,0,1), one nonnegative RGB gain."""
import numpy as np


def optimum(x, y):
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    assert x.shape == y.shape and x.shape[-1] == 3
    assert np.isfinite(x).all() and np.isfinite(y).all()
    assert (x >= 0).all() and (x <= 1).all() and (y >= 0).all() and (y <= 1).all()
    shape = x.shape[:-1]
    x, y = x.reshape(-1, 3), y.reshape(-1, 3)
    order = np.argsort(-x, axis=1, kind='stable')
    sx, sy = np.take_along_axis(x, order, 1), np.take_along_axis(y, order, 1)
    breaks = np.divide(1, sx, out=np.full_like(sx, np.inf), where=sx > 0)
    best, gain = np.sum(y*y, axis=1), np.zeros(len(x))

    def consider(g):
        nonlocal best, gain
        g = np.where(np.isfinite(g), g, 0)
        error = np.sum((np.clip(g[:, None]*x, 0, 1)-y)**2, axis=1)
        take = error < best
        gain[take], best[take] = g[take], error[take]

    for k in range(3):
        consider(breaks[:, k])
        xx, xy = np.sum(sx[:, k:]**2, axis=1), np.sum(sx[:, k:]*sy[:, k:], axis=1)
        stationary = np.divide(xy, xx, out=np.zeros_like(xx), where=xx > 0)
        lower = breaks[:, k-1] if k else 0
        consider(np.minimum(np.maximum(stationary, lower), breaks[:, k]))
    denominator = np.sum(x*x, axis=1)
    ls_gain = np.divide(np.sum(x*y, axis=1), denominator,
                        out=np.zeros(len(x)), where=denominator > 0)
    ls_error = np.sum((np.clip(ls_gain[:, None]*x, 0, 1)-y)**2, axis=1)
    assert np.all(best <= ls_error+1e-12)
    return dict(gain=gain.reshape(shape), mse=(best/3).reshape(shape),
                least_squares_then_clip_mse=(ls_error/3).reshape(shape))
