"""Scientific invariants for layered motion and matched structural training."""
import copy
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
from bounded_fst import apply_bounded, project_mixtures
from exploration_r2_data import detect_changes, make_sequence
from explore_state_r2 import forward_backward, fuse
from explore_temporal_controls import MobileIELLENetS
from train_exploration_r2 import temporal_penalty, training_batch
from video_metrics import warp_previous


class ExplorationR2Checks(unittest.TestCase):
    def test_change_detector_keeps_constant_input_and_finds_transitions(self):
        dark = np.full((48, 64, 3), 10, np.float32)
        events, _ = detect_changes([dark, dark, dark*3, dark*3])
        self.assertEqual(events, [False, False, True, False])
        pattern = np.zeros_like(dark)
        pattern[:, :32] = 255
        events, _ = detect_changes([pattern, 255-pattern])
        self.assertTrue(events[1])

    def test_confident_static_flow_and_reset_fallback(self):
        image = np.full((48, 64, 3), 20, np.float32)
        flow, confidence = forward_backward(image, image)
        np.testing.assert_allclose(flow, 0, atol=1e-6)
        self.assertTrue(confidence.all())
        outputs = fuse([image*5, image], [flow], [confidence], [False, True])
        np.testing.assert_array_equal(outputs[1], image)

    def test_bounded_starts_with_identical_function(self):
        torch.manual_seed(8)
        original = MobileIELLENetS(12).eval()
        changed = apply_bounded(copy.deepcopy(original), {"head": [1.]*12, "body": [1.]*12}).eval()
        x = torch.rand(2, 3, 24, 24)
        with torch.no_grad():
            torch.testing.assert_close(changed(x), original(x), rtol=0, atol=0)
        self.assertEqual(sum(p.numel() for p in changed.parameters())-sum(p.numel() for p in original.parameters()), 2)

    def test_mixture_projection_and_finite_forward(self):
        model = apply_bounded(MobileIELLENetS(12), {"head": [1.]*12, "body": [1.]*12})
        with torch.no_grad():
            model.head.gamma.fill_(-2)
            model.body.gamma.fill_(2)
        project_mixtures(model)
        self.assertEqual(float(model.head.gamma.detach()), 0)
        self.assertEqual(float(model.body.gamma.detach()), 1)
        self.assertTrue(bool(torch.isfinite(model(torch.rand(1, 3, 24, 24))).all()))

    def test_layered_occlusion_correspondence_matches_ground_truth(self):
        image = np.random.default_rng(1).integers(0, 256, (512, 768, 3)).astype(np.float32)
        gt, _, flows, masks = make_sequence(image, image, "occlusion", 0, frames=3)
        for t in range(1, 3):
            warped, valid = warp_previous(gt[t-1], flows[t-1])
            mask = valid & masks[t-1]
            np.testing.assert_array_equal(warped[mask], gt[t][mask])
            self.assertTrue(mask.any())
            self.assertTrue((~mask).any())

    def test_cut_has_no_correspondence(self):
        image = np.zeros((512, 768, 3), np.float32)
        gt, _, _, masks = make_sequence(image, image+100, "cut", 0, frames=4)
        self.assertFalse(masks[1].any())
        self.assertEqual(float(gt[2].mean()), 100)

    def test_training_stream_is_repeatable_and_preserves_pair_geometry(self):
        image = np.random.default_rng(2).integers(0, 256, (512, 768, 3)).astype(np.float32)
        images = {"a": image}
        a, gt, offsets = training_batch(images, 0, 1, batch_size=2, patch=32)
        b, _, other = training_batch(images, 0, 1, batch_size=2, patch=32)
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        self.assertEqual(offsets, other)
        self.assertAlmostEqual(float(temporal_penalty(gt, offsets)), .001, places=6)


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main(verbosity=2)
