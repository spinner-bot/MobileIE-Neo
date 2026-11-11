"""Independent frame-index fixtures for future context and event segmentation."""
import unittest

import numpy as np

from fastdvdnet_adapter_r12 import window_indices


class FastDVDnetWindowTests(unittest.TestCase):
    def test_original_reflection_and_two_future_frames(self):
        actual=window_indices(8,'window')
        np.testing.assert_array_equal(actual[0],[2,1,0,1,2])
        np.testing.assert_array_equal(actual[3],[1,2,3,4,5])
        np.testing.assert_array_equal(actual[7],[5,6,7,6,5])
        self.assertEqual(int(np.max(actual-np.arange(8)[:,None])),2)

    def test_segment_boundary_does_not_cross_scene_and_handles_singletons(self):
        events=[False,False,False,True,False,False,False,False]
        a=window_indices(8,'window_segmented',events)
        np.testing.assert_array_equal(a[2],[0,1,2,1,0])
        np.testing.assert_array_equal(a[3],[5,4,3,4,5])
        self.assertTrue((a[:3]<3).all());self.assertTrue((a[3:]>=3).all())
        np.testing.assert_array_equal(window_indices(3,'window_segmented',[False,True,True]),np.repeat(np.arange(3)[:,None],5,axis=1))

    def test_five_current_has_no_future_and_invalid_shapes_fail(self):
        np.testing.assert_array_equal(window_indices(8,'five_current'),np.repeat(np.arange(8)[:,None],5,axis=1))
        np.testing.assert_array_equal(window_indices(1,'window'),np.zeros((1,5),int))
        with self.assertRaises(ValueError):window_indices(0,'window')
        with self.assertRaises(ValueError):window_indices(8,'window',[False])


if __name__=='__main__':unittest.main()
