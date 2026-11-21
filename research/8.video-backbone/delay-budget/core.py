"""Fixed known three-observation means with explicit temporal access budgets."""
import numpy as np
from aligned_raw_history_r11 import align_history
from video_metrics import warp_previous


def indices(mode,t):
    if mode=='past3':
        return [t,t-1,t-2]
    if mode=='center3':
        return [t,t-1,t+1]
    if mode=='center_self':
        return [t,t-1]
    raise ValueError('Unregistered route')


def synthesize(provider,mode,t):
    requested=indices(mode,t)
    current=provider(requested[0])
    past=provider(requested[1])
    other=current if len(requested)==2 else provider(requested[2])
    result=(current+past+other)/3
    assert result.dtype==np.float32 and np.isfinite(result).all()
    return result


def qualification():
    rng=np.random.default_rng(8300)
    fixtures=[np.full((128,128,3),.25,np.float32),rng.random((128,128,3),dtype=np.float32)]
    routes={};maximum=0.
    for value in fixtures:
        zero=np.zeros((*value.shape[:2],2),np.float32)
        for accept in (True,False):
            mask=np.full(value.shape[:2],accept,bool)
            aligned,_=align_history(value,value,zero,mask)
            for mode in ('past3','center3','center_self'):
                calls=[]
                def provider(i):
                    assert i in indices(mode,2)
                    calls.append(i)
                    return value if i==2 else aligned
                result=synthesize(provider,mode,2)
                maximum=max(maximum,float(np.max(np.abs(result-value))))
                assert calls==indices(mode,2)
                routes[mode]=calls
    assert maximum<=1e-6
    # Known signed geometric fields validate gather direction and bounds/fallback only.
    import cv2
    previous=cv2.GaussianBlur(rng.random((64,80,3),dtype=np.float32),(0,0),.7)
    current=previous*.7
    geometric=[]
    for dx in (-1.25,1.25):
        field=np.broadcast_to(np.array([dx,-.5],np.float32),(64,80,2)).copy()
        moved,bounds=warp_previous(previous,field)
        got,accepted=align_history(previous,current,field,np.ones((64,80),bool))
        expected=np.where(bounds[...,None],moved,current)
        error=float(np.max(np.abs(got-expected)))
        rejected,_=align_history(previous,current,field,np.zeros((64,80),bool))
        assert error==0 and np.array_equal(accepted,bounds) and np.array_equal(rejected,current)
        geometric.append(dict(dx=dx,dy=-.5,max_difference=error,all_rejected_exact=True))
    return dict(passed=True,artificial_seed=8300,normalized_identity_max_difference=maximum,
        routes=routes,known_gathers=geometric,scope='Finite implementation checks, not motion-estimation accuracy',
        CNN_calls=0,optimization_steps=0,new_flow_estimates=0)
