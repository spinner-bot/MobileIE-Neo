"""Fixed-noise counterfactuals, marginal moments and covariance identity."""
import unittest
import cv2
import numpy as np

from correlated_observations_r19 import encode,pair_statistics,sensor_pair


class ObservationTests(unittest.TestCase):
    def setUp(self):cv2.setNumThreads(1)

    def test_current_is_identical_across_correlations(self):
        clean=np.full((16,16,3),.04)
        a,_=sensor_pair(clean,20,0,19,20,8);b,_=sensor_pair(clean,20,.9,19,21,8)
        self.assertTrue(np.array_equal(a,b))

    def test_rho_one_remains_identical_after_quantization_and_jpeg(self):
        clean=np.random.default_rng(19).random((16,16,3))*.2;a,b=sensor_pair(clean,80,1,19,20,8)
        for jpeg in [False,True]:self.assertTrue(np.array_equal(encode(a,jpeg),encode(b,jpeg)))

    def test_preclip_marginals_and_covariance(self):
        clean=np.full((32,32,3),.04);variance=.04/20+.008**2
        for rho in [0,.9]:
            a,b=sensor_pair(clean,20,rho,19,20,500)
            for x in [a,b]:
                self.assertLessEqual(abs(float(x.mean())-.04),.0002)
                self.assertLessEqual(abs(float(x.var())/variance-1),.02)
            self.assertLessEqual(abs(float(np.mean((a-a.mean())*(b-b.mean())))/variance-rho),.02)

    def test_encoded_variance_identity(self):
        a,b=sensor_pair(np.full((16,16,3),.04),20,.9,19,20,64)
        r=pair_statistics(encode(a,False),encode(b,False));self.assertLessEqual(r['variance_identity_abs'],1e-12)


if __name__=='__main__':unittest.main()
