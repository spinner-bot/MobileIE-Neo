"""Initial function/gradient identity, pair causality and foldable current limits."""
import copy
import unittest

import torch

from raw_pair_head_r8 import MODES, RawPairHead, augmented_history
from train_exploration_r2 import new_model


class RawPairHeadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        torch.manual_seed(82)

    def test_initial_output_gradient_and_learnable_auxiliary(self):
        base = new_model(torch.device('cpu'))
        low = torch.rand(2,2,3,21,29)*.1
        y = base(low.flatten(0,1)).reshape_as(low)
        y.square().mean().backward()
        for mode in MODES:
            net = RawPairHead(copy.deepcopy(base),mode)
            net.zero_grad(set_to_none=True)
            pred = net.forward_pair(low)
            torch.testing.assert_close(pred,y,rtol=0,atol=0)
            pred.square().mean().backward()
            for p,q in zip(net.backbone.parameters(),base.parameters()):
                torch.testing.assert_close(p.grad,q.grad,rtol=0,atol=0)
            self.assertGreater(float(net.auxiliary.weight.grad.abs().sum()),0.)
            self.assertEqual(sum(p.numel() for p in net.parameters()),4975)

    def test_pair_matches_stream_and_state_has_only_raw_rgb(self):
        low = torch.rand(2,2,3,21,29)*.1
        for mode in MODES:
            net = RawPairHead(new_model(torch.device('cpu')),mode)
            torch.nn.init.normal_(net.auxiliary.weight,std=.001)
            with torch.no_grad():
                paired = net.forward_pair(low)
                net.reset()
                streamed = torch.stack([net(low[:,i]) for i in range(2)],1)
                torch.testing.assert_close(paired,streamed,rtol=2e-5,atol=2e-6)
                self.assertEqual(net.state_storage_bytes(),0 if mode=='spatial_detail' else 2*3*21*29*4)
                net.reset()
                self.assertEqual(net.state_storage_bytes(),0)

    def test_current_history_folds_concat_and_disables_delta(self):
        x = torch.rand(2,3,21,29,dtype=torch.double)*.1
        for mode in ['temporal_concat','temporal_delta']:
            net = RawPairHead(new_model(torch.device('cpu')),mode).double()
            torch.nn.init.normal_(net.auxiliary.weight,std=.001)
            folded = copy.deepcopy(net.backbone)
            with torch.no_grad():
                if mode=='temporal_concat':
                    folded.head.block1[0].weight.add_(net.auxiliary.weight)
                net.reset('current')
                torch.testing.assert_close(net(x),folded(x),rtol=1e-10,atol=1e-11)

    def test_history_augmentation_preserves_current_and_is_reproducible(self):
        low = torch.rand(2,2,3,21,29)
        original = low.clone()
        h,action = augmented_history(low,0,8)
        repeated,again = augmented_history(low,0,8)
        torch.testing.assert_close(h,repeated,rtol=0,atol=0)
        torch.testing.assert_close(low,original,rtol=0,atol=0)
        self.assertEqual(action,again)


if __name__=='__main__':
    unittest.main(verbosity=2)
