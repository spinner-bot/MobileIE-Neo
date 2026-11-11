"""Correspondence direction, rejection fallback, and unchanged source checks."""
import unittest

import numpy as np

from aligned_raw_history_r11 import align_history


class AlignedRawHistoryTests(unittest.TestCase):
    def test_integer_motion_direction_and_mask_fallback(self):
        y,x=np.mgrid[:12,:20].astype(np.float32)
        previous=np.stack([x,y,x+y],-1);current=np.full_like(previous,99)
        before=previous.copy();now=current.copy()
        flow=np.zeros((12,20,2),np.float32);flow[...,0]=-2
        mask=np.ones((12,20),bool);mask[3,8]=False
        output,valid=align_history(previous,current,flow,mask)
        expected=current.copy();expected[:,2:]=previous[:,:-2];expected[3,8]=current[3,8]
        np.testing.assert_array_equal(output,expected)
        self.assertFalse(valid[:,:2].any());self.assertFalse(valid[3,8]);self.assertTrue(valid[5,10])
        np.testing.assert_array_equal(previous,before);np.testing.assert_array_equal(current,now)

    def test_zero_field_and_reject_all(self):
        rng=np.random.default_rng(11);p=rng.random((12,20,3),dtype=np.float32);c=rng.random(p.shape,dtype=np.float32)
        f=np.zeros((12,20,2),np.float32)
        out,mask=align_history(p,c,f,np.ones((12,20),bool))
        np.testing.assert_array_equal(out,p);self.assertTrue(mask.all())
        out,mask=align_history(p,c,f,np.zeros((12,20),bool))
        np.testing.assert_array_equal(out,c);self.assertFalse(mask.any())

    def test_invalid_grid_and_nonfinite_rejected(self):
        p=np.zeros((12,20,3),np.float32);f=np.zeros((12,20,2),np.float32);m=np.ones((12,20),bool)
        with self.assertRaises(ValueError):align_history(p,p,f[:,:-1],m)
        f[2,3,0]=np.nan
        with self.assertRaises(ValueError):align_history(p,p,f,m)


if __name__=='__main__':unittest.main()
