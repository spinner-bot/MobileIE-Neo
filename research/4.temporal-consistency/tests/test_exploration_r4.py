"""Scientific checks for real residual targets and repeatable data streams."""
import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from real_data_r4 import batch, residual_temporal


class ExplorationR4Checks(unittest.TestCase):
    def setUp(self):
        yy, xx = torch.meshgrid(torch.linspace(-1, 1, 12), torch.linspace(-1, 1, 16), indexing='ij')
        self.grid = torch.stack([xx, yy], -1)[None]
        self.mask = torch.ones(1, 1, 12, 16, dtype=torch.bool)

    def test_true_brightness_change_is_not_penalized(self):
        target = torch.zeros(1, 2, 3, 12, 16)
        target[:, 1] = .8
        self.assertAlmostEqual(float(residual_temporal(target, target, self.grid, self.mask)), .001, places=6)
        pred = target+.1
        self.assertAlmostEqual(float(residual_temporal(pred, target, self.grid, self.mask)), .001, places=6)

    def test_only_current_temporal_branch_receives_gradient(self):
        pred = torch.zeros(1, 2, 3, 12, 16, requires_grad=True)
        target = torch.zeros_like(pred)
        target[:, 1] = .5
        residual_temporal(pred, target, self.grid, self.mask).backward()
        torch.testing.assert_close(pred.grad[:, 0], torch.zeros_like(pred.grad[:, 0]), rtol=0, atol=0)
        self.assertGreater(float(pred.grad[:, 1].abs().sum()), 0.)

    def test_no_valid_correspondence_does_not_create_a_supervision_signal(self):
        pred = torch.rand(1, 2, 3, 12, 16, requires_grad=True)
        loss = residual_temporal(pred, torch.zeros_like(pred), self.grid, ~self.mask)
        loss.backward()
        self.assertAlmostEqual(float(loss.detach()), .001, places=6)
        torch.testing.assert_close(pred.grad, torch.zeros_like(pred), rtol=0, atol=0)

    def test_pair_stream_repeats_without_cross_scene_pairs(self):
        data = [{'scene': str(i), 'input': torch.full((16, 3, 12, 16), float(i)),
                 'target': torch.full((16, 3, 12, 16), float(i)),
                 'grid': self.grid.expand(15, -1, -1, -1), 'mask': self.mask.expand(15, -1, -1, -1)} for i in range(4)]
        first = batch(data, 1, 17)
        second = batch(data, 1, 17)
        self.assertEqual(first[-1], second[-1])
        torch.testing.assert_close(first[0], second[0], rtol=0, atol=0)
        for pair, (scene, index) in zip(first[0], first[-1]):
            self.assertTrue(1 <= index <= 15)
            self.assertTrue(bool((pair == float(scene)).all()))


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main(verbosity=2)
