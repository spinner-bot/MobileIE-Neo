"""Check full covariance, small roundoff and constant-gate limiting case."""
import unittest

import torch

from causal_fst_r5 import CausalFST
from noise_moments_r6 import forward_probe, quadratic
from rgb_noise_components_r6c import body_components, covariance_summary
from train_exploration_r2 import new_model


class ComponentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        torch.manual_seed(62)

    def test_reconstruction_and_covariance_including_clipping(self):
        for mode in ['plain', 'spatial_post']:
            model = CausalFST(new_model(torch.device('cpu')), mode).double()
            if hasattr(model, 'branch'):
                torch.nn.init.normal_(model.branch.weight, std=.01)
            with torch.no_grad():
                x = torch.rand(9, 3, 21, 29, dtype=torch.double)*.2
                output, features = forward_probe(model, x)
                for split in [False, True]:
                    baseline, components = body_components(model, features['body'], features['gate'], output, split_interaction=split)
                    self.assertLess(float(components['rounding_residual'].abs().max()), 1e-10)
                    torch.testing.assert_close(baseline+sum(components.values()), output, rtol=1e-12, atol=1e-12)
                    for clipped in [False, True]:
                        row = covariance_summary(components, output, clipped)
                        self.assertLess(row['variance_closure_abs'], 1e-12)
                        self.assertLess(row['centered_reconstruction_max_abs'], 1e-12)
                        self.assertAlmostEqual(sum(row['covariance_with_total_over_total_variance']), 1., places=10)

    def test_constant_gates_zero_gate_variation_components(self):
        model = new_model(torch.device('cpu')).double()
        u = torch.randn(9, 12, 21, 29, dtype=torch.double)+2
        gates = torch.full_like(u, .2)
        output = model.tail(gates*quadratic(model.body, u))
        _, components = body_components(model, u, gates, output)
        self.assertLess(float(components['gate_mean_feature'].abs().max()), 1e-12)
        self.assertLess(float(components['gate_feature_interaction'].abs().max()), 1e-12)
        self.assertLess(float(components['rounding_residual'].abs().max()), 1e-12)


if __name__ == '__main__':
    unittest.main(verbosity=2)
