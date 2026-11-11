"""Actual reset/prefix behavior, guarding against reuse of spatial-only R14 aliases."""
import unittest
import numpy as np
import torch

from evaluate_aligned_raw_r11 import infer
from evaluation_aliases_r16 import cache_keys


class PairFixture:
    def __init__(self,spatial=False):self.spatial=spatial
    def transform(self,current,previous):return current if self.spatial else (current+previous)/2


class ResetAndCausality(unittest.TestCase):
    def test_raw_reset_has_actual_effect_and_distinct_key(self):
        x=torch.arange(4,dtype=torch.float32)[:,None,None,None].expand(4,3,8,8)
        events=np.array([False,False,True,False]);net=PairFixture()
        a=infer(net,x,{},'concat_high','causal',None);b=infer(net,x,{},'concat_high','causal',events)
        self.assertFalse(torch.equal(a[2],b[2]));self.assertTrue(torch.equal(a[[0,1,3]],b[[0,1,3]]))
        self.assertNotEqual(cache_keys('raw_history','causal',False,True,'raw'),cache_keys('raw_history','causal',True,True,'raw'))

    def test_current_or_spatial_raw_is_reset_invariant(self):
        x=torch.rand(4,3,8,8);events=np.array([False,False,True,False])
        for variant,history,net in [('raw_history','current',PairFixture()),('spatial','causal',PairFixture(True))]:
            a=infer(net,x,{},'concat_high',history,None);b=infer(net,x,{},'concat_high',history,events)
            self.assertTrue(torch.equal(a,b));self.assertEqual(cache_keys(variant,history,False,True,'raw'),cache_keys(variant,history,True,True,'raw'))
            self.assertNotEqual(cache_keys(variant,history,False,True,'half01'),cache_keys(variant,history,True,True,'half01'))

    def test_future_perturbation_cannot_change_prefix(self):
        x=torch.rand(5,3,8,8);changed=x.clone();changed[3:]=torch.rand_like(changed[3:])
        a=infer(PairFixture(),x,{},'concat_high','causal',None);b=infer(PairFixture(),changed,{},'concat_high','causal',None)
        self.assertTrue(torch.equal(a[:3],b[:3]))


if __name__=='__main__':unittest.main()
