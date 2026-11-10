"""Analytic checks before using motion/quality metrics for research."""
import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from video_metrics import (aligned_difference, backward_flow, edge_error, psnr,
                           ssim, temporal_error, translation_overlap,
                           translation_warp)


class VideoMetricsChecks(unittest.TestCase):
    def test_known_translation_is_exact_and_legacy_wrap_is_not(self):
        scene = np.random.default_rng(2).integers(0, 256, (40, 96, 3), dtype=np.uint8)
        prev, cur = scene[:, :64], scene[:, 4:68]
        self.assertEqual(aligned_difference(prev, cur, 4), 0)
        legacy = np.abs(np.roll(cur, 4, axis=1)[:, :60].astype(float) - prev[:, :60]).mean()
        self.assertGreater(legacy, 1)

    def test_negative_and_vertical_translation(self):
        scene = np.arange(80 * 80 * 3).reshape(80, 80, 3)
        prev, cur = scene[5:45, 5:45], scene[3:43, 2:42]
        p, c = translation_overlap(prev, cur, -3, -2)
        np.testing.assert_array_equal(p, c)

    def test_zero_motion_and_unsigned_subtraction(self):
        prev = np.full((20, 20, 3), 255, dtype=np.uint8)
        cur = np.zeros_like(prev)
        self.assertEqual(aligned_difference(prev, prev, 0), 0)
        self.assertEqual(aligned_difference(prev, cur, 0), 255)

    def test_warp_mask_and_translation(self):
        scene = np.random.default_rng(3).random((30, 80, 3)).astype(np.float32)
        prev, cur = scene[:, :64], scene[:, 4:68]
        warped, valid = translation_warp(prev, 4)
        self.assertTrue(valid[:, :60].all())
        self.assertFalse(valid[:, 60:].any())
        np.testing.assert_allclose(warped[valid], cur[valid], atol=1e-6)

    def test_backward_flow_recovers_direction(self):
        scene = np.random.default_rng(9).integers(0, 256, (96, 144, 3), dtype=np.uint8)
        scene = cv2.GaussianBlur(scene, (5, 5), .8)
        prev, cur = scene[:, :128], scene[:, 4:132]
        flow = backward_flow(prev, cur)
        interior = flow[16:-16, 16:-16]
        self.assertAlmostEqual(float(np.median(interior[..., 0])), 4.0, delta=.5)
        self.assertAlmostEqual(float(np.median(interior[..., 1])), 0.0, delta=.5)

    def test_quality_and_reference_temporal_change(self):
        prev = np.zeros((20, 20, 3), np.float32)
        cur = np.full_like(prev, 255)
        self.assertEqual(psnr(prev, cur), 0)
        self.assertTrue(np.isinf(psnr(prev, prev)))
        self.assertAlmostEqual(ssim(cur, cur), 1, places=5)
        self.assertEqual(edge_error(cur, cur), 0)
        self.assertEqual(temporal_error(prev, cur, prev, cur, 0), 0)

    def test_no_overlap_is_rejected(self):
        image = np.zeros((20, 20, 3))
        with self.assertRaises(ValueError):
            translation_overlap(image, image, 20)


if __name__ == "__main__":
    unittest.main(verbosity=2)
