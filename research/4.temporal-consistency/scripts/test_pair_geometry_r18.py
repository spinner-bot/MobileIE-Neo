"""Known shift/gain fixtures and degenerate texture protection."""
import unittest
import cv2
import numpy as np

from pair_geometry_r18 import diagnose,oracle_psnr


class PairGeometryTests(unittest.TestCase):
    def setUp(self):
        cv2.setNumThreads(1);rng=np.random.default_rng(18);self.target=cv2.GaussianBlur(rng.random((128,160,3),dtype=np.float32),(0,0),1.)

    def test_known_translation_sign_with_affine_brightness(self):
        shifted=cv2.warpAffine(self.target,np.float32([[1,0,5],[0,1,-3]]),(160,128),borderMode=cv2.BORDER_CONSTANT)
        r,_,_=diagnose(.35*shifted+.1,self.target)
        self.assertEqual(r['best_dxdy'],[5,-3]);self.assertGreater(r['best_ncc'],.999)
        oracle=oracle_psnr(shifted,self.target,r['best_dxdy']);self.assertGreater(oracle['oracle_delta_dB'],50)

    def test_same_image_zero_shift(self):
        r,_,_=diagnose(self.target,self.target);self.assertEqual(r['best_dxdy'],[0,0]);self.assertGreater(r['best_ncc'],.999)

    def test_constant_texture_is_not_spurious_perfect_registration(self):
        r,score,_=diagnose(self.target,np.ones_like(self.target)*.5)
        self.assertTrue(r['degenerate']);self.assertIsNone(r['best_dxdy']);self.assertIsNone(score)
        self.assertFalse(r['diagnostic_peak_flag'])


if __name__=='__main__':unittest.main()
