"""Independent checks of feature moments and history-reset semantics."""
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from probe_feature_products_r3 import moments
from explore_resets_r3 import reset_methods


class ExplorationR3Checks(unittest.TestCase):
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
