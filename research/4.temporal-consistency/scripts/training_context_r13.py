"""An earlier independently degraded frame, preserving the original task stream."""
import numpy as np
import torch

from exploration_r2_data import degrade
from train_mixture_r4b import synthetic_pair


def synthetic_pair_with_context(images,seed,step):
    low,target,meta=synthetic_pair(images,seed,step)
    rng=np.random.default_rng(910000000+seed*1000000+step)
    names=list(images);name=names[int(rng.integers(len(names)))];image=images[name]
    x=int(rng.integers(4,image.shape[1]-320-3));y=int(rng.integers(4,image.shape[0]-180-3))
    dx,dy=[int(v) for v in rng.integers(-4,5,size=2)]
    assert name==meta['scene'] and [dx,dy]==meta['offset']
    past=image[y-dy:y-dy+180,x-dx:x-dx+320]*meta['brightness']
    if past.shape!=(180,320,3):raise ValueError('synthetic context outside source bounds')
    context_seed=1401300000+seed*1000000+step
    noisy=degrade(past,np.random.default_rng(context_seed),meta['photons'],meta['jpeg'])
    context=torch.from_numpy(noisy.transpose(2,0,1).copy()/255.)
    return low,target,meta,context,{'source':'synthetic_previous','crop_xy':[x-dx,y-dy],
        'offset_from_first':[-dx,-dy],'noise_rng_seed':context_seed,'self_fallback':False}


def previous_indices(t):
    if not isinstance(t,int) or t<1:raise ValueError('pair ends at integer frame t>=1')
    return max(0,t-2),t-1
