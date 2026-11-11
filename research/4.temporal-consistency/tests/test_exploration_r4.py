"""Scientific checks for real residual targets and repeatable data streams."""
import sys
import unittest
from pathlib import Path

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from real_data_r4 import batch, residual_temporal
from reliability_r4 import census, discrepancy, masks_at_threshold, quality_feasible
from explore_state_r2 import fuse
from calibrate_mapping_r4 import fit_mappings, transform
from train_mixture_r4b import mixture_loss, synthetic_pair
from train_exploration_r2 import temporal_penalty
from calibrate_noise_gate_r4 import haar_noise_scale
from half_motion_r4 import half_forward_backward


class ExplorationR4Checks(unittest.TestCase):
    def test_half_scale_flow_restores_native_translation_units(self):
        import cv2
        image = np.random.default_rng(11).integers(0, 256, (128, 160, 3)).astype(np.float32)
        image = cv2.GaussianBlur(image, (0, 0), .8)
        current = np.roll(image, -4, axis=1)
        flow, mask = half_forward_backward(image, current)
        self.assertTrue(bool(mask[30:-30, 30:-30].any()))
        self.assertAlmostEqual(float(flow[30:-30, 30:-30, 0].mean()), 4., delta=.4)
        self.assertAlmostEqual(float(flow[30:-30, 30:-30, 1].mean()), 0., delta=.3)

    def test_synthetic_stop_keeps_loss_value_and_only_real_gradient(self):
        prediction = torch.full((2, 2, 3, 12, 16), .3, requires_grad=True)
        target = torch.zeros_like(prediction)
        full = mixture_loss(prediction, target)
        stopped = mixture_loss(prediction, target, True)
        torch.testing.assert_close(full, stopped, rtol=0, atol=0)
        stopped.backward()
        self.assertGreater(float(prediction.grad[0].abs().sum()), 0.)
        self.assertEqual(float(prediction.grad[1].abs().sum()), 0.)

    def test_haar_scale_constant_and_known_independent_gaussian(self):
        self.assertEqual(haar_noise_scale(np.full((32, 40, 3), 80, np.float32)), 0.)
        noisy = 100+np.random.default_rng(10).normal(0, 4, (256, 256, 3)).astype(np.float32)
        self.assertAlmostEqual(haar_noise_scale(noisy), 4., delta=.12)

    def test_identity_centered_ridge_preserves_identity(self):
        image = np.random.default_rng(8).uniform(0, 255, (20, 24, 3)).astype(np.float32)
        coefficients = fit_mappings([(image, image)])
        self.assertAlmostEqual(coefficients['scalar_gain'], 1., places=12)
        np.testing.assert_allclose(coefficients['channel_gain'], 1., atol=1e-12)
        np.testing.assert_allclose(coefficients['rgb_affine'], np.vstack([np.eye(3), np.zeros((1, 3))]), atol=1e-10)
        np.testing.assert_allclose(transform(image, 'rgb_affine', coefficients), image, atol=1e-5)

    def test_mapping_output_clips_in_float_domain(self):
        x = np.asarray([[[-10., 300., 128.]]], np.float32)
        result = transform(x, 'scalar_gain', {'scalar_gain': 1.})
        np.testing.assert_array_equal(result, [[[0., 255., 128.]]])
        self.assertEqual(result.dtype, np.float32)

    def test_synthetic_mixture_pairs_repeat_and_preserve_geometry(self):
        images = {'fixed': np.random.default_rng(9).integers(0, 256, (512, 768, 3)).astype(np.float32)}
        low, target, identity = synthetic_pair(images, 0, 3)
        repeated, _, metadata = synthetic_pair(images, 0, 3)
        torch.testing.assert_close(low, repeated, rtol=0, atol=0)
        self.assertEqual(identity, metadata)
        self.assertEqual(tuple(low.shape), (2, 3, 180, 320))
        self.assertAlmostEqual(float(temporal_penalty(target[None], [tuple(identity['offset'])])), .001, places=6)

    def test_census_is_invariant_to_positive_affine_brightness(self):
        image = np.random.default_rng(7).uniform(10, 80, (20, 24, 3)).astype(np.float32)
        np.testing.assert_array_equal(census(image), census(image*2+5))

    def test_zero_threshold_keeps_raw_and_rejects_all_history(self):
        raw = [np.full((20, 24, 3), v, np.float32) for v in [10, 20, 30]]
        flows = [np.zeros((20, 24, 2), np.float32) for _ in range(2)]
        valid = [np.ones((20, 24), bool) for _ in range(2)]
        scores = discrepancy(raw, flows, 'photo')
        masks = masks_at_threshold(scores, valid, 0.)
        self.assertFalse(any(m.any() for m in masks))
        for actual, expected in zip(fuse(raw, flows, masks), raw):
            np.testing.assert_array_equal(actual, expected)

    def test_quality_guard_requires_all_three_quality_constraints(self):
        reference = {'psnr_rgb': 20., 'ssim_rgb': .8, 'edge_mae': 2.}
        self.assertTrue(quality_feasible(reference, reference))
        for key, value in [('psnr_rgb', 19.9), ('ssim_rgb', .79), ('edge_mae', 2.1)]:
            self.assertFalse(quality_feasible({**reference, key: value}, reference))

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
