"""Independent checks of feature moments and history-reset semantics."""
import sys
import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from probe_feature_products_r3 import moments
from explore_resets_r3 import reset_methods
from explore_products_r3 import PairFST, sampling
from explore_temporal_controls import MobileIELLENetS
from paired_fst_r3 import TrainablePairFST
from xiph_motion_r3 import read_y4m


class ExplorationR3Checks(unittest.TestCase):
    def test_pair_training_matches_initial_function_and_learns_mixture(self):
        torch.manual_seed(4)
        original = MobileIELLENetS(12).body
        value = torch.rand(4, 12, 32, 32)
        for mode in ['cross', 'mean_square']:
            module = TrainablePairFST(copy.deepcopy(original), mode)
            module.offsets = [(2, -1), (-3, 2)]
            pred = module(value)
            torch.testing.assert_close(pred, original(value), rtol=0, atol=0)
            pred.square().mean().backward()
            self.assertTrue(bool(torch.isfinite(module.gamma.grad)))
            self.assertGreater(abs(float(module.gamma.grad)), 0.)

    def test_integer_alignment_preserves_equal_features(self):
        original = MobileIELLENetS(12).body
        original.block1 = torch.nn.Identity()
        module = TrainablePairFST(original, 'cross')
        module.gamma.data.fill_(1.)
        module.offsets = [(2, 0)]
        prev = torch.rand(1, 12, 32, 32)
        cur = prev.clone()
        cur[..., :-2] = prev[..., 2:]
        value = torch.cat([prev, cur])
        torch.testing.assert_close(module(value), original(value), rtol=0, atol=0)

    def test_y4m_limited_range_and_truncation(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'sample.y4m'
            header = b'YUV4MPEG2 W4 H4 F30:1 Ip A1:1\nFRAME\n'
            path.write_bytes(header+bytes([16]*16+[128]*8))
            frames, _ = read_y4m(path, 1)
            np.testing.assert_array_equal(frames[0], 0.)
            path.write_bytes(header+bytes([235]*16+[128]*8))
            frames, _ = read_y4m(path, 1)
            np.testing.assert_array_equal(frames[0], 255.)
            path.write_bytes(header+bytes([16]*3))
            with self.assertRaises(ValueError):
                read_y4m(path, 1)

    def test_product_interventions_preserve_first_frame_and_invalid_regions(self):
        torch.manual_seed(3)
        original = MobileIELLENetS(12).eval().body
        current = torch.rand(1, 12, 16, 16)
        following = torch.rand_like(current)
        with torch.inference_mode():
            for mode in ['cross', 'mean_square', 'square_mean']:
                module = PairFST(copy.deepcopy(original), mode)
                torch.testing.assert_close(module(current), original(current), rtol=0, atol=0)
                module.grid, module.mask = sampling(np.zeros((16, 16, 2), np.float32),
                                                     np.zeros((16, 16), bool), 'cpu')
                torch.testing.assert_close(module(following), original(following), rtol=0, atol=0)

    def test_torch_sampling_matches_known_backward_translation(self):
        from torch.nn import functional as F
        source = torch.arange(48., dtype=torch.float32).reshape(1, 1, 6, 8)
        flow = np.zeros((6, 8, 2), np.float32)
        flow[..., 0] = 2
        grid, _ = sampling(flow, np.ones((6, 8), bool), 'cpu')
        result = F.grid_sample(source, grid, align_corners=True)
        torch.testing.assert_close(result[..., :-2], source[..., 2:], rtol=0, atol=1e-5)

    def test_constant_feature_bias_is_not_hidden_in_covariance(self):
        clean = torch.full((1, 2, 2), 2.)
        samples = torch.full((8, 1, 2, 2), 3.)
        row = moments(clean, samples)[0]
        self.assertEqual(row['feature_bias_mse'], 1.)
        self.assertEqual(row['pair_centered_covariance'], 0.)
        self.assertEqual(row['pair_error_product'], 1.)
        self.assertEqual(row['cross_mse'], 25.)

    def test_independent_symmetric_pairs_remove_square_bias(self):
        clean = torch.zeros(1, 1, 1)
        samples = torch.tensor([1., 1., 1., -1., -1., 1., -1., -1.]).reshape(8, 1, 1, 1)
        row = moments(clean, samples)[0]
        self.assertEqual(row['self_bias_mse'], 1.)
        self.assertEqual(row['cross_bias_mse'], 0.)
        self.assertEqual(row['self_mse'], 1.)
        self.assertEqual(row['mean_square_mse'], .5)

    def test_output_reset_does_not_change_gate_selection(self):
        first = np.full((8, 8, 3), 100., np.float32)
        last = first*0
        flow = [np.zeros((8, 8, 2), np.float32)]
        confidence = [np.ones((8, 8), bool)]
        methods = reset_methods({'raw': [first, last], 'ema': [first, last+20],
                                 'reset': [first, last+40]}, flow, confidence, [False, True])
        self.assertEqual(len(methods), 15)
        np.testing.assert_array_equal(methods['raw__fb__reset'][0][1], last)
        np.testing.assert_array_equal(methods['reset__fb__reset'][0][1], last+40)
        np.testing.assert_allclose(methods['raw__fb__keep'][0][1], 70.)


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main(verbosity=2)
