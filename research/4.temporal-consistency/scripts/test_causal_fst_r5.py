"""Mechanism tests: arithmetic, initial gradients, causality and state storage."""
import copy
import unittest

import torch
from torch.nn import functional as F

from causal_fst_r5 import CausalFST, MODES, coarse_pool, coarse_up
from train_exploration_r2 import new_model


class CausalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        torch.manual_seed(7)
        cls.base = new_model(torch.device('cpu'))

    def model(self, mode):
        return CausalFST(copy.deepcopy(self.base), mode)

    def test_pool_and_interpolation_forward_backward(self):
        for h, w in [(1, 1), (3, 5), (8, 12), (17, 22)]:
            x = torch.randn(2, 3, h, w, dtype=torch.double, requires_grad=True)
            ref_pool = F.avg_pool2d(F.pad(x, (0, (-w)%4, 0, (-h)%4), mode='replicate'), 4)
            actual = coarse_up(coarse_pool(x), (h, w))
            expected = F.interpolate(ref_pool, scale_factor=4, mode='bilinear', align_corners=False)[..., :h, :w]
            torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
            weights = torch.randn_like(actual)
            ga = torch.autograd.grad((actual*weights).sum(), x, retain_graph=True)[0]
            ge = torch.autograd.grad((expected*weights).sum(), x)[0]
            torch.testing.assert_close(ga, ge, rtol=1e-12, atol=1e-12)

    def test_parameter_count(self):
        expected = [4075, 4183, 4183, 4183, 4076, 4075, 4183]
        for mode, count in zip(MODES, expected):
            self.assertEqual(sum(p.numel() for p in self.model(mode).parameters()), count)

    def test_initial_output_and_backbone_gradient(self):
        # Low-light range avoids the pretrained attention saturating on white noise.
        x = torch.rand(2, 2, 3, 20, 28)*.1
        reference = self.model('plain')
        y = reference.forward_pair(x)
        y.square().mean().backward()
        for mode in [m for m in MODES if m != 'tsm_fixed']:
            net = self.model(mode)
            pred = net.forward_pair(x)
            torch.testing.assert_close(pred, y, rtol=0, atol=0)
            pred.square().mean().backward()
            for p, q in zip(net.backbone.parameters(), reference.backbone.parameters()):
                torch.testing.assert_close(p.grad, q.grad, rtol=0, atol=0)
            if hasattr(net, 'branch'):
                self.assertGreater(float(net.branch.weight.grad.abs().sum()), 0.)
            if mode == 'tsm_learned':
                self.assertGreater(float(net.gamma.grad.abs()), 0.)

    def test_pairs_equal_streams_and_future_cannot_affect_first(self):
        x = torch.rand(2, 2, 3, 21, 29)
        for mode in MODES:
            net = self.model(mode)
            if hasattr(net, 'branch'):
                torch.nn.init.normal_(net.branch.weight, std=.02)
            if hasattr(net, 'gamma'):
                net.gamma.data.fill_(.4)
            for history in ('causal', 'current', 'off'):
                net.reset(history)
                paired = net.forward_pair(x)
                net.reset(history)
                streamed = torch.stack([net(x[:, i]) for i in range(2)], 1)
                torch.testing.assert_close(paired, streamed, rtol=2e-5, atol=2e-6)
                changed = x.clone(); changed[:, 1] = torch.rand_like(changed[:, 1])*4
                torch.testing.assert_close(net.forward_pair(changed)[:, 0], paired[:, 0], rtol=0, atol=0)

    def test_state_storage_reset_and_shape_change(self):
        for mode in MODES:
            net = self.model(mode)
            with torch.no_grad():
                net(torch.rand(1, 3, 21, 29))
                if mode == 'plain' or mode.startswith('spatial_'):
                    self.assertEqual(net.state_storage_bytes(), 0)
                else:
                    elements = 3*21*29 if mode.startswith('tsm_') else 12*6*8
                    self.assertEqual(net.state_storage_bytes(), elements*4)
                    self.assertFalse(net.state.requires_grad)
                    self.assertIsNone(net.state._base)
                net(torch.rand(1, 3, 17, 19))
                net.reset()
                self.assertEqual(net.state_storage_bytes(), 0)

    def test_bad_inputs(self):
        with self.assertRaises(ValueError):
            self.model('plain').forward_pair(torch.rand(2, 3, 4, 4))
        with self.assertRaises(ValueError):
            self.model('plain').reset('future')
        with self.assertRaises(ValueError):
            coarse_pool(torch.empty(1, 3, 0, 4))


if __name__ == '__main__':
    unittest.main(verbosity=2)
