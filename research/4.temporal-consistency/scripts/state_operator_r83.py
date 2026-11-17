"""Fixed-field strong-rec state Jacobian, including OpenCV interpolation phases."""
import cv2
import numpy as np


class StateOperator:
    def __init__(self, flow, mask):
        h, w = mask.shape
        assert flow.shape == (h, w, 2)
        self.h, self.w = h, w
        y, x = np.indices((h, w), dtype=np.float32)
        sx, sy = x+flow[..., 0], y+flow[..., 1]
        self.valid = mask.astype(bool) & (sx >= 0) & (sx <= w-1) & (sy >= 0) & (sy <= h-1)
        base, phases = cv2.convertMaps(sx, sy, cv2.CV_16SC2)
        fx = (phases.astype(np.int64) % 32)/32.
        fy = (phases.astype(np.int64) // 32)/32.
        bx, by = base[..., 0].astype(np.int64), base[..., 1].astype(np.int64)
        self.x = np.stack([bx, bx+1, bx, bx+1], -1).clip(0, w-1)
        self.y = np.stack([by, by, by+1, by+1], -1).clip(0, h-1)
        self.index = self.y*w+self.x
        self.weights = .5*self.valid[..., None]*np.stack(
            [(1-fx)*(1-fy), fx*(1-fy), (1-fx)*fy, fx*fy], -1)
        self.local = (np.abs(self.x-x[..., None]) <= 2) & (np.abs(self.y-y[..., None]) <= 2)

    def apply(self, state):
        assert state.shape[:2] == (self.h, self.w)
        values = np.asarray(state, dtype=np.float64).reshape(-1, state.shape[-1])
        return (values[self.index]*self.weights[..., None]).sum(axis=2)

    def dot(self, other, left=None, right=None):
        assert (self.h, self.w) == (other.h, other.w)
        left = self.weights if left is None else left
        right = other.weights if right is None else right
        result = np.zeros((self.h, self.w), dtype=np.float64)
        for a in range(4):
            for b in range(4):
                result += (self.index[..., a] == other.index[..., b])*left[..., a]*right[..., b]
        return result

    def energy(self):
        return self.dot(self)

    def outside_energy(self):
        weights = self.weights*~self.local
        return self.dot(self, weights, weights)

    def difference(self, other):
        denominator = self.energy()+other.energy()
        numerator = np.maximum(denominator-2*self.dot(other), 0.)
        return numerator, denominator


def qualification():
    from aligned_raw_history_r11 import align_history
    from aggregation_controls_r20 import blend
    rng = np.random.default_rng(83001)
    h, w = 16, 19; y, x = np.indices((h, w))
    current = rng.random((h, w, 3), dtype=np.float32)
    masks = [(x+y) % 5 != 0, (x+2*y) % 7 != 0]
    fields = [np.broadcast_to(np.array(v, np.float32), (h, w, 2)).copy()
              for v in ((3.25, -1.5), (-2.25, 1.125))]
    ops = [StateOperator(f, m) for f, m in zip(fields, masks)]
    maximum, empirical = 0., []
    for _ in range(16):
        probe = rng.choice([-1., 1.], (h, w, 3)).astype(np.float32)
        for op, field, mask in zip(ops, fields, masks):
            aligned, valid = align_history(probe, current, field, mask)
            expected = blend(current, aligned, .5)
            actual = (1-.5*op.valid[..., None])*current+op.apply(probe)
            assert np.array_equal(valid, op.valid)
            maximum = max(maximum, float(np.max(np.abs(actual-expected))))
        empirical.append(float(np.mean((ops[0].apply(probe)-ops[1].apply(probe))**2)))
    assert maximum <= 1e-6
    for op in ops:
        assert np.max(np.abs(op.weights.sum(-1)-.5*op.valid)) == 0
        assert op.difference(op)[0].max() == 0
        local_weights = op.weights*op.local
        assert np.max(np.abs(op.energy()-op.outside_energy()-op.dot(op, local_weights, local_weights))) <= 1e-12
    numerator, _ = ops[0].difference(ops[1])
    return dict(passed=True, original_teacher_max_abs_difference=maximum,
        fixed_Rademacher_draws=16, probe_channels=3, analytic_probe_MSE=float(numerator.mean()),
        empirical_probe_MSE=float(np.mean(empirical)), new_backbone_frames=0, optimizer_updates=0,
        scope='Fixed-field interpolation and metric implementation only; random states are not real RGB video.')
