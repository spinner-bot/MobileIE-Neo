"""Fixed half-resolution classical motion, confidence recomputed in native units."""
import cv2
import numpy as np

from video_metrics import backward_flow, warp_previous


def half_forward_backward(previous, current):
    h, w = current.shape[:2]
    size = (w//2, h//2)
    if min(size) < 8:
        raise ValueError('motion input too small')
    a = cv2.resize(previous, size, interpolation=cv2.INTER_AREA)
    b = cv2.resize(current, size, interpolation=cv2.INTER_AREA)

    def restore(field):
        result = cv2.resize(field, (w, h), interpolation=cv2.INTER_LINEAR)
        result[..., 0] *= w/size[0]
        result[..., 1] *= h/size[1]
        return result

    back = restore(backward_flow(a, b))
    forward = restore(backward_flow(b, a))
    sampled, valid = warp_previous(forward, back)
    delta = np.linalg.norm(back+sampled, axis=2)
    threshold = 1+.05*(np.linalg.norm(back, axis=2)+np.linalg.norm(sampled, axis=2))
    return back, valid & (delta < threshold)
