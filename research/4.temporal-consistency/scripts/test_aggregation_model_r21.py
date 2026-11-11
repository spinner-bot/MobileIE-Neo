"""Static limit, parameter isolation and live gradients at the alpha boundary."""
import copy
import unittest

import torch
from torch import nn

from aggregation_model_r21 import AggregationFST,MODES
from train_exploration_r2 import new_model


class Square(nn.Module):
    def forward(self,x):return x*x


class AggregationModelTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(2);torch.manual_seed(21)

    def test_static_limit_and_parameter_counts(self):
        base=new_model(torch.device('cpu'));x=torch.rand(2,3,31,39)
        with torch.no_grad():expected=base(x)
        for mode in MODES:
            net=AggregationFST(copy.deepcopy(base),mode)
            with torch.no_grad():actual=net.transform(x,x)
            self.assertLessEqual(float((actual-expected).abs().max()),2e-5)
            self.assertEqual(sum(p.numel() for p in net.parameters()),4076 if mode.startswith('pre_learn') else 4075)

    def test_zero_initialization_and_parameter_isolation(self):
        base=new_model(torch.device('cpu'));models=[AggregationFST(copy.deepcopy(base),mode) for mode in MODES];pointers=set()
        for net in models:
            own={p.data_ptr() for p in net.parameters()};self.assertFalse(own&pointers);pointers|=own
        x=torch.rand(2,3,31,39);previous=torch.rand_like(x)
        with torch.no_grad():self.assertTrue(torch.equal(models[1].transform(x,previous),base(x)))
        with torch.no_grad():self.assertTrue(torch.equal(models[0].transform(x,previous),models[2].transform(x,previous)))

    def test_alpha_zero_has_gradient_and_projection_is_explicit(self):
        net=AggregationFST(nn.Identity(),'pre_learnzero');x=torch.full((1,3,8,8),.2);p=torch.full_like(x,.8)
        net.transform(x,p).sum().backward();self.assertGreater(float(net.alpha.grad),0.)
        with torch.no_grad():net.alpha.fill_(-.3)
        net.project();self.assertEqual(net.coefficient(),0.)
        with torch.no_grad():net.alpha.fill_(1.3)
        net.project();self.assertEqual(net.coefficient(),1.)

    def test_post_averages_raw_network_outputs(self):
        net=AggregationFST(Square(),'post_fixedhalf');x=torch.ones(1,3,8,8)*2;p=torch.zeros_like(x)
        self.assertTrue(torch.equal(net.transform(x,p),torch.ones_like(x)*2))


if __name__=='__main__':unittest.main()
