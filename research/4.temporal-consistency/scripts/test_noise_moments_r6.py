"""Falsifiable moment identities and static-forward equivalence checks."""
import unittest

import torch

from causal_fst_r5 import CausalFST
from noise_moments_r6 import central_statistics, forward_probe, quadratic, rgb_statistics
from train_exploration_r2 import new_model


class NoiseMomentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        torch.manual_seed(6)

    def test_empirical_identity_without_gaussian_assumption(self):
        x = torch.distributions.Exponential(1.).sample((37, 3, 4, 5)).double()+2
        r = central_statistics(x)
        torch.testing.assert_close(r['linear']+r['cross']+r['quadratic'], r['total'], rtol=1e-12, atol=1e-12)
        self.assertGreater(float(r['cross'].mean()), 0.)
        a, b = -.002, 3.
        gap = (a*x.square()+b).mean(0)-(a*r['mean'].square()+b)
        torch.testing.assert_close(gap, a*r['variance'], rtol=1e-10, atol=1e-12)

    def test_constant_debias_preserves_local_variance(self):
        net = new_model(torch.device('cpu')).double()
        u = torch.randn(64, 12, 3, 4, dtype=torch.double)+2
        stats = {'variance': torch.rand(12, 3, 4, dtype=torch.double)}
        a, b = quadratic(net.body, u), quadratic(net.body, u, 'debias', stats)
        torch.testing.assert_close(a.var(0, unbiased=False), b.var(0, unbiased=False), rtol=1e-11, atol=1e-13)

    def test_linearization_removes_sample_squared_noise(self):
        net = new_model(torch.device('cpu')).double()
        mu = torch.randn(12, 3, 4, dtype=torch.double)
        u = mu+torch.randn(64, 12, 3, 4, dtype=torch.double)
        result = quadratic(net.body, u, 'linearize', {'mean': mu})
        expected = (net.body.weight1*net.body.weight2)*(2*mu*u-mu.square())+net.body.bias
        torch.testing.assert_close(result, expected, rtol=1e-11, atol=1e-12)

    def test_mse_bias_variance_closure(self):
        x, y = torch.randn(16, 3, 20, 24), torch.randn(3, 20, 24)
        r = rgb_statistics(x, y)
        self.assertLess(r['closure_abs'], 1e-12)

    def test_centered_control_preserves_empirical_feature_mean(self):
        net = new_model(torch.device('cpu')).double()
        u = torch.randn(64, 12, 3, 4, dtype=torch.double)+2
        stats = {'mean': u.mean(0), 'variance': u.var(0, unbiased=False)}
        raw = quadratic(net.body, u)
        centered = quadratic(net.body, u, 'centered', stats)
        torch.testing.assert_close(centered.mean(0), raw.mean(0), rtol=1e-12, atol=1e-12)
        a = net.body.weight1*net.body.weight2
        expected = a*(2*stats['mean']*u-stats['mean'].square()+stats['variance'])+net.body.bias
        torch.testing.assert_close(centered, expected, rtol=1e-11, atol=1e-12)

    def test_static_forward_matches_existing_models(self):
        for mode in ['plain', 'spatial_post']:
            net = CausalFST(new_model(torch.device('cpu')), mode)
            if hasattr(net, 'branch'):
                torch.nn.init.normal_(net.branch.weight, std=.03)
            with torch.no_grad():
                for batch in [1, 8]:
                    x = torch.rand(batch, 3, 21, 29)*.1
                    net.reset()
                    actual, features = forward_probe(net, x)
                    torch.testing.assert_close(actual, net(x), rtol=0, atol=0)
                    self.assertEqual(features['head'].shape, (batch, 12, 21, 29))


if __name__ == '__main__':
    unittest.main(verbosity=2)
