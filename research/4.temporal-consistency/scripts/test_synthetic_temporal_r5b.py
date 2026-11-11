"""Direction and gradient tests for the known-motion supervision baseline."""
import unittest

import torch

from synthetic_temporal_r5b import synthetic_residual_temporal


class SyntheticTemporalTests(unittest.TestCase):
    def test_true_translations_reach_floor_with_nonzero_gt_change(self):
        torch.manual_seed(5)
        image = torch.randn(3, 50, 64)
        for dx, dy in [(-4, -3), (0, 0), (4, 3), (2, -4), (-3, 2)]:
            errors = torch.stack([image[:, 8:40, 8:56], image[:, 8+dy:40+dy, 8+dx:56+dx]])
            target = torch.rand_like(errors)
            value = synthetic_residual_temporal(errors+target, target, (dx, dy))
            self.assertAlmostEqual(float(value), .001, places=7)

    def test_only_current_error_receives_direct_gradient(self):
        prediction = torch.randn(2, 3, 32, 40, requires_grad=True)
        loss = synthetic_residual_temporal(prediction, torch.zeros_like(prediction), (3, -2))
        loss.backward()
        self.assertEqual(float(prediction.grad[0].abs().sum()), 0.)
        self.assertGreater(float(prediction.grad[1].abs().sum()), 0.)
        self.assertEqual(float(prediction.grad[1, :, :8].abs().sum()), 0.)

    def test_insufficient_overlap_rejected(self):
        with self.assertRaises(ValueError):
            synthetic_residual_temporal(torch.zeros(2, 3, 20, 20), torch.zeros(2, 3, 20, 20), (5, 0))


if __name__ == '__main__':
    unittest.main(verbosity=2)
