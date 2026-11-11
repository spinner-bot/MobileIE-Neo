"""Independent fixtures keep current-history and event resets scientifically valid."""
import unittest

import numpy as np
import torch

from evaluate_aligned_raw_r11 import infer


class SumHead:
    def transform(self,current,previous):return current+previous


class EvaluationHistoryTests(unittest.TestCase):
    def fixture(self):
        inputs=torch.stack([torch.full((3,12,20),float(t)) for t in [1,2,3]])
        low=inputs.permute(0,2,3,1).numpy()*255
        return inputs,{'low':low,'half_flow':np.zeros((2,12,20,2),np.float32),'half_mask':np.ones((2,12,20),bool)}

    def test_same_weights_current_history_survives_event_reset(self):
        x,data=self.fixture();events=[False,True,False]
        raw=infer(SumHead(),x,data,'concat_high','causal',None)
        reset=infer(SumHead(),x,data,'concat_high','causal',events)
        current=infer(SumHead(),x,data,'concat_high','current',events)
        torch.testing.assert_close(raw[:,0,0,0],torch.tensor([2.,3.,5.]),rtol=0,atol=0)
        torch.testing.assert_close(reset[:,0,0,0],torch.tensor([2.,4.,5.]),rtol=0,atol=0)
        torch.testing.assert_close(current[:,0,0,0],torch.tensor([2.,4.,6.]),rtol=0,atol=0)

    def test_aligned_rejection_falls_back_and_current_never_uses_motion(self):
        x,data=self.fixture();data['half_mask'][:]=False
        aligned=infer(SumHead(),x,data,'concat_aligned_high','causal',None)
        torch.testing.assert_close(aligned,2*x,rtol=0,atol=0)
        data['half_flow'][:]=np.nan
        current=infer(SumHead(),x,data,'concat_aligned_high','current',[False,True,False])
        torch.testing.assert_close(current,2*x,rtol=0,atol=0)

    def test_zero_motion_matches_unaligned_causal_path(self):
        x,data=self.fixture()
        a=infer(SumHead(),x,data,'concat_high','causal',None)
        b=infer(SumHead(),x,data,'concat_aligned_high','causal',None)
        torch.testing.assert_close(a,b,rtol=0,atol=0)


if __name__=='__main__':unittest.main()
