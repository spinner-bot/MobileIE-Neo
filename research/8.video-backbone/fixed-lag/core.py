"""Fixed-gain one-step innovation reference; not calibrated RTS for RGB video."""
from fractions import Fraction
import cv2
import numpy as np
from fixed_dis_r86 import EXPECTED,FixedDIS
from video_metrics import warp_previous
from aligned_raw_history_r11 import align_history
from aggregation_controls_r20 import blend


def consistent(back,forward):
    sampled,valid=warp_previous(forward,back)
    delta=np.linalg.norm(back+sampled,axis=2)
    limit=1+.05*(np.linalg.norm(back,axis=2)+np.linalg.norm(sampled,axis=2))
    return valid & (delta<limit)


class BidirectionalDIS:
    def __init__(self):
        assert cv2.__version__=='4.11.0'
        self.backward=cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST)
        self.forward=cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST)
        assert all({n:getattr(obj,'get'+n)() for n in EXPECTED}==EXPECTED for obj in (self.backward,self.forward))
        self.pairs=0

    def __call__(self,previous,current):
        assert previous.shape==current.shape and previous.ndim==3 and previous.shape[-1]==3
        def gray(image):return cv2.cvtColor(np.clip(np.rint(image),0,255).astype(np.uint8),cv2.COLOR_RGB2GRAY)
        a,b=gray(previous),gray(current)
        back=self.backward.calc(b,a,None)
        forward=self.forward.calc(a,b,None)
        assert back.dtype==forward.dtype==np.float32 and np.isfinite(back).all() and np.isfinite(forward).all()
        mask_back=consistent(back,forward)
        mask_forward=consistent(forward,back)
        self.pairs+=1
        return back,mask_back,forward,mask_forward


def smooth(filtered,future_raw,future_filtered,forward,mask_forward,mask_back):
    residual,_=warp_previous(future_raw-future_filtered,forward)
    future,inside=warp_previous(future_raw,forward)
    reliability,bounds=warp_previous(mask_back.astype(np.float32),forward)
    mask=mask_forward & inside & bounds & (reliability>=1-1e-6)
    m=mask[...,None]
    innovation=filtered+.5*np.where(m,residual,0.)
    naive=filtered+.25*np.where(m,future-filtered,0.)
    assert np.isfinite(innovation).all() and np.isfinite(naive).all()
    return innovation,naive,mask


def qualification():
    R,Q,P=Fraction(1),Fraction(1,2),Fraction(1,2)
    Ppred=P+Q;K=Ppred/(Ppred+R);G=P/Ppred
    smoothed=P+G*G*(P-Ppred)
    assert Ppred==1 and K==G==Fraction(1,2) and (1-K)*Ppred==P and smoothed==Fraction(3,8)
    rng=np.random.default_rng(86001)
    gray=cv2.GaussianBlur(rng.uniform(0,255,(128,160)).astype(np.float32),(0,0),.7)
    previous=np.repeat(gray[...,None],3,-1)
    field=np.broadcast_to(np.array([3,-2],np.float32),(128,160,2)).copy()
    current,_=warp_previous(previous,field)
    old,old_mask=FixedDIS()(previous,current)
    new=BidirectionalDIS();back,mb,_,_=new(previous,current)
    assert float(np.max(np.abs(old-back)))==0 and np.array_equal(old_mask,mb)
    rng=np.random.default_rng(8400)
    filtered=rng.random((64,80,3),dtype=np.float32)
    raw=rng.random((64,80,3),dtype=np.float32)
    zero=np.zeros((64,80,2),np.float32);yes=np.ones((64,80),bool);no=~yes
    future=blend(raw,filtered,.5)
    result,naive,_=smooth(filtered,raw,future,zero,yes,yes)
    identity_error=float(np.max(np.abs(result-naive)))
    assert identity_error<=1e-6
    rejected,_,_=smooth(filtered,raw,future,zero,no,yes)
    empty,_,_=smooth(filtered,raw,raw,zero,yes,yes)
    assert np.array_equal(rejected,filtered) and np.array_equal(empty,filtered)
    cycles=[]
    for dx in (-1.25,1.25):
        b=np.broadcast_to(np.array([dx,-.5],np.float32),(64,80,2)).copy();f=-b
        predicted_image,valid=align_history(filtered,raw,b,yes)
        future=blend(raw,predicted_image,.5)
        result,naive,mask=smooth(filtered,raw,future,f,yes,valid)
        first,_=warp_previous(filtered,b);cycle,_=warp_previous(first,f)
        expected=.25*(filtered-cycle)
        error=float(np.max(np.abs((result-naive-expected)[mask])))
        assert mask.any() and error<=1e-6
        cycles.append(dict(dx=dx,dy=-.5,raw_cycle_identity_max_difference=error))
    return dict(passed=True,scalar_stationary={k:str(v) for k,v in dict(R=R,Q=Q,P=P,Ppred=Ppred,K=K,G=G,Psmooth=smoothed).items()},
        scalar_model_not_real_noise_calibration=True,legacy_back_max_difference=0.,legacy_mask_exact=True,
        normalized_zero_motion_identity_max_difference=identity_error,rejected_and_zero_innovation_exact=True,
        signed_known_field_checks=cycles,qualification_DIS_bidirectional_pairs=2,qualification_DIS_calc_calls=4,
        new_CNN_calls=0,optimizer_updates=0)
