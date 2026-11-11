"""Affine commutation, nonlinear Jensen difference, and exact finite reset boundaries."""
import unittest
import numpy as np

from aggregation_controls_r20 import blend,event_output,finite_controls


class AggregationTests(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(20);self.low=rng.random((4,8,9,3),dtype=np.float32)
        self.history=np.concatenate([self.low[:1],self.low[:-1]])

    def test_affine_point_mapping_commutes(self):
        f=lambda x:.5*x+.2;raw=f(self.low);prior=f(self.history)
        def forward(x,first):
            y=f(x);y[0]=first;return y
        out=finite_controls(self.low,raw,self.history,prior,prior,forward)
        for suffix in ['01','05']:
            np.testing.assert_allclose(out['pre'+suffix],out['postwarp'+suffix],atol=1e-7,rtol=0)
            np.testing.assert_array_equal(out['postaligned'+suffix],out['postwarp'+suffix])

    def test_quadratic_jensen_gap(self):
        for alpha in [.1,.5]:
            pre=blend(self.low,self.history,alpha)**2;post=blend(self.low**2,self.history**2,alpha)
            np.testing.assert_allclose(post-pre,alpha*(1-alpha)*(self.low-self.history)**2,atol=2e-7,rtol=0)

    def test_identical_history_and_event_reset_are_exact(self):
        for alpha in [.1,.5]:np.testing.assert_array_equal(blend(self.low,self.low,alpha),self.low)
        raw=self.low**2;output=blend(raw,self.history**2,.5);events=[False,False,True,False]
        reset=event_output(output,raw,events);np.testing.assert_array_equal(reset[2],raw[2])
        np.testing.assert_array_equal(reset[[0,1,3]],output[[0,1,3]])


if __name__=='__main__':unittest.main()
