"""Task-stream invariance and physical crop history, independent of model behavior."""
import unittest

import numpy as np
import torch

from train_mixture_r4b import synthetic_pair
from training_context_r13 import previous_indices, synthetic_pair_with_context


class TrainingContextTests(unittest.TestCase):
    def test_existing_task_pixels_labels_and_identity_are_exact(self):
        image=np.random.default_rng(13).uniform(0,255,(512,768,3)).astype(np.float32)
        images={'fixture':image}
        for seed,step in [(0,0),(1,999),(0,317),(1,40)]:
            old=synthetic_pair(images,seed,step)
            low,target,meta,past,info=synthetic_pair_with_context(images,seed,step)
            torch.testing.assert_close(low,old[0],rtol=0,atol=0);torch.testing.assert_close(target,old[1],rtol=0,atol=0)
            self.assertEqual(meta,old[2]);self.assertEqual(tuple(past.shape),(3,180,320))
            again=synthetic_pair_with_context(images,seed,step)
            torch.testing.assert_close(past,again[3],rtol=0,atol=0)
            self.assertEqual(info,again[4]);self.assertEqual(info['offset_from_first'],[-x for x in meta['offset']])
            self.assertTrue(bool(torch.isfinite(past).all()));self.assertFalse(info['self_fallback'])

    def test_real_boundary_and_nonboundary_are_causal(self):
        self.assertEqual(previous_indices(1),(0,0))
        self.assertEqual(previous_indices(2),(0,1))
        self.assertEqual(previous_indices(15),(13,14))
        with self.assertRaises(ValueError):previous_indices(0)


if __name__=='__main__':unittest.main()
