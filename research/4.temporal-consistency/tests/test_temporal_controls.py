"""Check streaming-state partition invariance and preservation of real changes."""
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from explore_temporal_controls import TemporalAttention, generate_sequence
from video_metrics import aligned_difference, temporal_error


class TemporalControlChecks(unittest.TestCase):
    def test_attention_state_does_not_depend_on_inference_batch_partition(self):
        values = torch.rand(9, 12, 1, 1, generator=torch.Generator().manual_seed(1))
        whole = TemporalAttention(torch.nn.Identity())
        streamed = TemporalAttention(torch.nn.Identity())
        with torch.inference_mode():
            expected = whole(values)
            actual = torch.cat([streamed(batch) for batch in values.split(4)])
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_real_lighting_change_is_not_rewarded_for_being_suppressed(self):
        image = np.full((768, 1024, 3), 100, np.float32)
        gt, low = generate_sequence(image, 0, 0, False, 4, 20, 20, target_step=True)
        np.testing.assert_array_equal(image, 100)  # synthesis does not mutate its source
        self.assertAlmostEqual(float(gt[1].mean()), 60, places=4)
        self.assertEqual(float(gt[2].mean()), 100)
        self.assertGreater(float(low[2].mean()), float(low[1].mean()))
        constant = gt[1]
        self.assertEqual(aligned_difference(constant, constant, 0), 0)
        self.assertAlmostEqual(temporal_error(constant, constant, gt[1], gt[2], 0), 40, places=4)


if __name__ == "__main__":
    unittest.main(verbosity=2)
