"""Exact independent Cartesian draws and channel-sum tests for the proxy."""
import unittest

import torch

from channel_noise_r7 import channel_covariance, channel_linear_rgb, predicted_relative_gain, subsets
from noise_moments_r6 import forward_probe
from rgb_noise_components_r6c import body_components
from train_exploration_r2 import new_model


class ChannelNoiseTests(unittest.TestCase):
    def test_independent_cartesian_history_matches_variance_formula(self):
        torch.manual_seed(73)
        z = torch.randn(7, 4, dtype=torch.double)
        y = z[:, 0]-.7*z[:, 1]+torch.randn(7, dtype=torch.double)*.2
        z = z-z.mean(0); y = y-y.mean()
        cov, cross, variance = z.T@z/7, z.T@y/7, y.square().mean()
        for subset in [(0,), (0, 1), (1, 2, 3)]:
            s = z[:, list(subset)].sum(1)
            for gamma in [.1, .5, 1.]:
                new = y[:, None]-gamma*s[:, None]+gamma*s[None, :]
                actual = 1-new.var(unbiased=False)/variance
                expected = predicted_relative_gain(cov, cross, variance, subset, gamma)
                torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)

    def test_channel_sum_matches_exact_mean_gate_linear_rgb(self):
        torch.set_num_threads(2)
        model = new_model(torch.device('cpu')).double()
        with torch.no_grad():
            output, features = forward_probe(model, torch.rand(7, 3, 21, 29, dtype=torch.double)*.1)
            values = channel_linear_rgb(model, features['body'], features['gate'])
            _, components = body_components(model, features['body'], features['gate'], output)
            torch.testing.assert_close(sum(values), components['feature_linear_mean_gate'], rtol=1e-10, atol=1e-12)
            cov, cross, variance = channel_covariance(values, output)
            self.assertEqual(cov.shape, (12, 12))
            self.assertEqual(cross.shape, (12,))
            self.assertGreater(float(variance), 0.)
        self.assertEqual(len(subsets()), 220)
        self.assertEqual(subsets()[0], (0, 1, 2))


if __name__ == '__main__':
    unittest.main(verbosity=2)
