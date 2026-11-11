"""Reset aliases cannot collapse real causal transitions or output-fusion history."""
import unittest

from evaluation_aliases_r14 import cache_keys


class EvaluationAliasTests(unittest.TestCase):
    def test_no_events_is_an_exact_invariant(self):
        for variant in ['plain','copy','aligned_concat']:
            for history in ['causal','current']:
                for fusion in ['raw','half01']:
                    self.assertEqual(cache_keys(variant,history,False,False,fusion),cache_keys(variant,history,True,False,fusion))

    def test_true_causal_reset_stays_distinct(self):
        a=cache_keys('aligned_concat','causal',False,True,'raw')
        b=cache_keys('aligned_concat','causal',True,True,'raw')
        self.assertEqual(a,(False,(False,'raw',False)));self.assertEqual(b,(True,(True,'raw',False)))
        self.assertNotEqual(a,b)

    def test_spatial_or_current_reset_does_not_erase_output_reset(self):
        for variant,history in [('plain','causal'),('copy','causal'),('aligned_concat','current')]:
            self.assertEqual(cache_keys(variant,history,False,True,'raw'),cache_keys(variant,history,True,True,'raw'))
            a=cache_keys(variant,history,False,True,'half01');b=cache_keys(variant,history,True,True,'half01')
            self.assertEqual(a[0],b[0]);self.assertNotEqual(a[1],b[1]);self.assertEqual(b[1],(False,'half01',True))


if __name__=='__main__':unittest.main()
