"""One unmodified OpenCV DIS preset, native size, fixed legacy FB acceptance."""
import cv2
import numpy as np

from aligned_raw_history_r11 import align_history
from aggregation_controls_r20 import blend
from video_metrics import warp_previous

EXPECTED = dict(FinestScale=2, GradientDescentIterations=12, PatchSize=8, PatchStride=4,
                UseMeanNormalization=True, UseSpatialPropagation=True, VariationalRefinementIterations=0)


class FixedDIS:
    def __init__(self):
        assert cv2.__version__ == '4.11.0'
        self.backward = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST)
        self.forward = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST)
        self.parameters = {name: getattr(self.backward, 'get'+name)() for name in EXPECTED}
        assert self.parameters == EXPECTED
        assert {name: getattr(self.forward, 'get'+name)() for name in EXPECTED} == EXPECTED
        self.pairs = 0

    def __call__(self, previous, current):
        if previous.shape != current.shape or current.ndim != 3 or current.shape[-1] != 3:
            raise ValueError('Expected matching native HWC RGB arrays')
        def gray(image):
            return cv2.cvtColor(np.clip(np.rint(image), 0, 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        a, b = gray(previous), gray(current)
        back = self.backward.calc(b, a, None)
        forward = self.forward.calc(a, b, None)
        assert back.dtype == np.float32 and np.isfinite(back).all() and np.isfinite(forward).all()
        sampled, valid = warp_previous(forward, back)
        delta = np.linalg.norm(back+sampled, axis=2)
        threshold = 1+.05*(np.linalg.norm(back, axis=2)+np.linalg.norm(sampled, axis=2))
        self.pairs += 1
        return back, valid & (delta < threshold)


class RecursivePipeline:
    def __init__(self, motion):
        self.motion = motion
        self.previous_raw = self.previous_fused = None

    def step(self, raw):
        if self.previous_raw is None:
            output, field = raw, None
        else:
            field = self.motion(self.previous_raw, raw)
            moved, _ = align_history(self.previous_fused, raw, *field)
            output = blend(raw, moved, .5)
        self.previous_raw = raw if raw.flags.owndata else raw.copy()
        self.previous_fused = (self.previous_raw if output is raw else
                               output if output.flags.owndata else output.copy())
        return output, field


def qualification():
    rng = np.random.default_rng(86001)
    image = cv2.GaussianBlur(rng.uniform(0, 255, (128, 160)).astype(np.float32), (0, 0), .7)
    previous = np.repeat(image[..., None], 3, -1)
    target = np.broadcast_to(np.array([3, -2], np.float32), (128, 160, 2)).copy()
    current, _ = warp_previous(previous, target)
    motion = FixedDIS(); flow, mask = motion(previous, current)
    median_epe = float(np.median(np.linalg.norm(flow[16:-16, 16:-16]-target[16:-16, 16:-16], axis=-1)))
    assert median_epe <= 1.
    constant = np.full_like(previous, 64.)
    stream = RecursivePipeline(motion)
    first, field = stream.step(constant)
    second, _ = stream.step(constant)
    assert field is None and np.array_equal(first, constant) and np.isfinite(second).all()
    assert motion.pairs == 2
    return dict(passed=True, OpenCV=cv2.__version__, preset='ULTRAFAST', native=True,
        actual_parameters=motion.parameters, translation=[3, -2], median_translation_EPE=median_epe,
        center_border_excluded=16, translation_accepted_fraction=float(mask.mean()),
        first_frame_max_difference=0., constant_finite=True, DIS_bidirectional_pairs=2,
        DIS_calc_calls=4, enhancement_CNN_calls=0, scope='Fixed clean artificial implementation qualification only.')
