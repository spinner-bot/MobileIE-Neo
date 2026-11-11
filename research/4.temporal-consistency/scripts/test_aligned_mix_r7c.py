"""Direction, confidence, finite history and unaligned replay checks."""
import copy
import unittest

import numpy as np
import torch

from aligned_mix_r7c import AlignedFeatureMix, blend, motion_field, rgb_stream, sample_previous
from feature_mix_r7 import FeatureMix
from train_exploration_r2 import new_model


class AlignedMixTests(unittest.TestCase):
    def test_positive_backward_flow_gathers_right_and_rejects_border(self):
        flow = np.zeros((12, 16, 2), np.float32); flow[..., 0] = 2
        grid, valid = motion_field(flow, np.ones((12, 16), bool), torch.device('cpu'))
        previous = torch.arange(16, dtype=torch.float32)[None, None, None].expand(1, 3, 12, 16)
        warped = sample_previous(previous, grid)
        actual = blend(torch.zeros_like(previous), warped, .5, valid)
        torch.testing.assert_close(actual[..., :-2], previous[..., 2:]*.5, rtol=1e-5, atol=1e-5)
        self.assertFalse(bool(valid[..., -2:].any()))
        self.assertEqual(float(actual[..., -2:].abs().sum()), 0.)

    def test_no_motion_replays_feature_mix_and_empty_mask_falls_back(self):
        torch.set_num_threads(2)
        base = new_model(torch.device('cpu'))
        a = FeatureMix(copy.deepcopy(base), [6, 7, 8], 'post', .5)
        b = AlignedFeatureMix(copy.deepcopy(base), .5)
        with torch.no_grad():
            for _ in range(4):
                x = torch.rand(1, 3, 21, 29)*.1
                torch.testing.assert_close(a(x), b(x), rtol=0, atol=0)
            grid, mask = motion_field(np.zeros((21, 29, 2), np.float32), np.zeros((21, 29), bool), torch.device('cpu'))
            torch.testing.assert_close(b(x, grid, mask), base(x), rtol=0, atol=0)

    def test_rgb_finite_recursive_and_reset(self):
        frames = torch.stack([torch.zeros(3, 4, 6), torch.ones(3, 4, 6), torch.zeros(3, 4, 6)])
        fields = [(None, None)]*2
        finite = rgb_stream(frames, fields, .5)
        recurrent = rgb_stream(frames, fields, .5, True)
        torch.testing.assert_close(finite[:, 0, 0, 0], torch.tensor([0., .5, .5]))
        torch.testing.assert_close(recurrent[:, 0, 0, 0], torch.tensor([0., .5, .25]))
        reset = rgb_stream(frames, fields, .5, True, [False, False, True])
        self.assertEqual(float(reset[2].abs().sum()), 0.)


if __name__ == '__main__':
    unittest.main(verbosity=2)
