"""Native-grid coordinate-state prototype, with explicit rejection-matched control."""
import cv2
import numpy as np

from aligned_raw_history_r11 import align_history
from video_metrics import backward_flow, warp_previous


def bidirectional(previous, current):
    h, w = current.shape[:2]
    size = (w//2, h//2)
    a = cv2.resize(previous, size, interpolation=cv2.INTER_AREA)
    b = cv2.resize(current, size, interpolation=cv2.INTER_AREA)

    def restore(field):
        result = cv2.resize(field, (w, h), interpolation=cv2.INTER_LINEAR)
        result[..., 0] *= w/size[0]
        result[..., 1] *= h/size[1]
        return result

    back = restore(backward_flow(a, b))
    forward = restore(backward_flow(b, a))

    def consistent(field, inverse):
        moved, valid = warp_previous(inverse, field)
        delta = np.linalg.norm(field+moved, axis=2)
        threshold = 1+.05*(np.linalg.norm(field, axis=2)+np.linalg.norm(moved, axis=2))
        return valid & (delta < threshold)

    return back, consistent(back, forward), forward, consistent(forward, back)


def inside(field):
    h, w = field.shape[:2]
    y, x = np.mgrid[:h, :w].astype(np.float32)
    return ((x+field[..., 0] >= 0) & (x+field[..., 0] <= w-1)
            & (y+field[..., 1] >= 0) & (y+field[..., 1] <= h-1))


def compose(q, c, va, vc, back, mb, forward, mf):
    """q: anchor -> previous; c: previous -> anchor; masks on each source grid."""
    f_at_q, q_bounds = warp_previous(forward, q)
    mf_at_q, _ = warp_previous(mf.astype(np.float32), q)
    c_at_b, b_bounds = warp_previous(c, back)
    vc_at_b, _ = warp_previous(vc.astype(np.float32), back)
    q_new, c_new = q+f_at_q, back+c_at_b
    va_new = va & q_bounds & (mf_at_q >= 1-1e-6) & inside(q_new)
    vc_new = mb & b_bounds & (vc_at_b >= 1-1e-6) & inside(c_new)
    return q_new, c_new, va_new, vc_new


def predict(raw, fields, alpha=.5, period=8):
    """All three controls share inputs/flow. Fixed alpha; never fit a parameter."""
    assert len(fields) == len(raw)-1 and period >= 1
    h, w = raw.shape[1:3]
    state = raw[0].copy()
    ordinary = [state]
    matched = [state]
    outputs = [state]
    q = np.zeros((h, w, 2), np.float32)
    c = np.zeros_like(q)
    va, vc = np.ones((h, w), bool), np.ones((h, w), bool)
    age, stats = 0, []
    for t, (back, mb, forward, mf) in enumerate(fields, 1):
        q, c, va, vc = compose(q, c, va, vc, back, mb, forward, mf)
        moved, bounds = warp_previous(state, c)
        accepted_anchor, _ = warp_previous(va.astype(np.float32), c)
        accepted = vc & bounds & (accepted_anchor >= 1-1e-6)
        image = raw[t]
        output = np.where(accepted[..., None], image+alpha*(moved-image), image)
        outputs.append(output)
        for container, mask in [(ordinary, mb), (matched, accepted)]:
            history, _ = align_history(container[-1], image, back, mask)
            container.append(image+alpha*(history-image))
        age += 1
        stats.append({'frame': t, 'age': age, 'ordinary_accepted': int(mb.sum()),
                      'anchor_accepted': int(accepted.sum()),
                      'anchor_grid_accepted': int(va.sum()), 'pixels': h*w})
        if age == period:
            state = output
            q.fill(0); c.fill(0); va.fill(True); vc.fill(True)
            age = 0
        else:
            incoming, bounds = warp_previous(image, q)
            state = np.where((va & bounds)[..., None], incoming+alpha*(state-incoming), state)
    return {'ordinary': np.stack(ordinary), 'same_mask': np.stack(matched),
            'anchor8': np.stack(outputs)}, stats
