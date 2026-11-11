"""Initial-model parity and operational temporal-path controls, without evaluation data."""
import copy
import unittest

import torch

from multiscale_fst_r16 import MODES,MultiscaleFST
from train_exploration_r2 import new_model


class MultiscaleControls(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2);torch.manual_seed(16)
        self.base=new_model(torch.device('cpu'));self.current=torch.rand(2,3,37,53);self.past=torch.rand_like(self.current)

    def test_official_initial_output_and_shapes(self):
        net=MultiscaleFST(copy.deepcopy(self.base),'raw_history')
        with torch.no_grad():a=net.transform(self.current,self.past);b=self.base(self.current)
        self.assertEqual(a.shape,self.current.shape);self.assertTrue(bool(torch.isfinite(a).all()))
        self.assertLessEqual(float((a-b).abs().max()),2e-5)
        self.assertTrue(torch.equal(net.backbone.body.block1.current.weight,self.base.body.block1.weight))
        self.assertEqual(float(net.backbone.body.block1.context.weight.detach().abs().max()),0.)

    def test_identical_seed_and_parameter_capacity(self):
        models=[]
        for mode in MODES:
            torch.manual_seed(16);models.append(MultiscaleFST(copy.deepcopy(self.base),mode))
        for net in models[1:]:
            self.assertEqual(sum(p.numel() for p in net.parameters()),sum(p.numel() for p in models[0].parameters()))
            for key,value in net.state_dict().items():self.assertTrue(torch.equal(value,models[0].state_dict()[key]))

    def test_operational_history_and_spatial_independence(self):
        net=MultiscaleFST(copy.deepcopy(self.base),'raw_history')
        # Nonzero decoder context fixture tests actual routing rather than relying on initially zero columns.
        with torch.no_grad():net.backbone.body.block1.context.weight.fill_(.002)
        a=net.transform(self.current,self.past);b=net.transform(self.current,self.current)
        self.assertGreater(float((a-b).detach().abs().max()),1e-9)
        a.mean().backward();self.assertGreater(float(net.encoder[0].weight.grad.abs().sum()),0.)
        net.mode='spatial'
        self.assertTrue(torch.equal(net.transform(self.current,self.past),net.transform(self.current,self.current)))

    def test_input_contract(self):
        net=MultiscaleFST(copy.deepcopy(self.base),'spatial')
        with self.assertRaises(ValueError):net.transform(self.current,self.past[:1])


if __name__=='__main__':unittest.main()
