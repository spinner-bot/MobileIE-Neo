"""Function-preserving limits, mean identity and finite actual state checks."""
import copy
import unittest

import torch

from feature_mix_r7 import FeatureMix
from train_exploration_r2 import new_model


class FeatureMixTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        torch.manual_seed(75)

    def test_zero_mix_and_first_frame_equal_original(self):
        for placement in ['pre', 'post']:
            for channels in [[0, 1, 2], [6, 7, 8], [0, 4, 8]]:
                net = new_model(torch.device('cpu'))
                x = torch.rand(1, 3, 21, 29)*.1
                with torch.no_grad():
                    reference = net(x)
                    for gamma in [0., .5, 1.]:
                        model = FeatureMix(copy.deepcopy(net), channels, placement, gamma)
                        torch.testing.assert_close(model(x), reference, rtol=0, atol=0)
                        if gamma == 0:
                            model(torch.rand_like(x))
                            torch.testing.assert_close(model(x), reference, rtol=0, atol=0)

    def test_only_one_past_raw_frame_and_exact_state_storage(self):
        for placement in ['pre', 'post']:
            for channels in [[6, 7, 8], [0, 4, 8]]:
                a = FeatureMix(new_model(torch.device('cpu')), channels, placement, .5)
                b = copy.deepcopy(a)
                with torch.no_grad():
                    a(torch.rand(1, 3, 21, 29)*.1)
                    b(torch.rand(1, 3, 21, 29)*.1)
                    x1, x2 = torch.rand(1, 3, 21, 29)*.1, torch.rand(1, 3, 21, 29)*.1
                    a(x1); b(x1)
                    torch.testing.assert_close(a(x2), b(x2), rtol=0, atol=0)
                    self.assertEqual(a.state_storage_bytes(), 3*21*29*4)
                    self.assertIsNone(a.state._base)
                    self.assertFalse(a.state.requires_grad)
                    a(torch.rand(1, 3, 17, 19)*.1)
                    a.reset()
                    self.assertEqual(a.state_storage_bytes(), 0)

    def test_pre_post_difference_identity(self):
        u, p = torch.randn(11, dtype=torch.double), torch.randn(11, dtype=torch.double)
        a, bias, gamma = -.002, 3., .5
        pre = a*((1-gamma)*u+gamma*p).square()+bias
        post = (1-gamma)*(a*u.square()+bias)+gamma*(a*p.square()+bias)
        torch.testing.assert_close(post-pre, a*gamma*(1-gamma)*(u-p).square(), rtol=1e-10, atol=1e-14)


if __name__ == '__main__':
    unittest.main(verbosity=2)
