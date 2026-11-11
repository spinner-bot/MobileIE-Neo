"""Pinned official network/inference with explicit five-frame information controls."""
import importlib.util
import json

import numpy as np
import torch

from exploration_r2_data import ROOT
from explore_temporal_controls import sha256

MANIFEST=ROOT/'research/4.temporal-consistency/data/exploration-r12-fastdvdnet/manifest.json'
MODES=('window','window_segmented','five_current')


def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    loaded=importlib.util.module_from_spec(spec);spec.loader.exec_module(loaded)
    return loaded


def load_model(device):
    manifest=json.loads(MANIFEST.read_text())
    for row in manifest['sources']:
        if sha256(ROOT/row['path'])!=row['sha256']:raise ValueError('vendored source changed')
    path=ROOT/manifest['path']
    if sha256(path)!=manifest['sha256']:raise ValueError('official weights changed')
    models=module(ROOT/'third_party/fastdvdnet/models.py','mobileie_reference_fastdvd_models')
    model=models.FastDVDnet(num_input_frames=5).to(device)
    state=torch.load(path,map_location=device,weights_only=True)
    model.load_state_dict({k.removeprefix('module.'):v for k,v in state.items()},strict=True)
    return model.eval()


def official_inference():
    return module(ROOT/'third_party/fastdvdnet/fastdvdnet.py','mobileie_reference_fastdvd_inference')


def window_indices(length,mode,events=None):
    if length<1 or mode not in MODES:raise ValueError('nonempty sequence and registered mode required')
    if events is None:events=np.zeros(length,bool)
    events=np.asarray(events,bool)
    if events.shape!=(length,):raise ValueError('one event decision per frame required')
    starts=[0]+[int(x) for x in np.flatnonzero(events) if x>0]
    result=[]
    for t in range(length):
        if mode=='five_current':result.append([t]*5);continue
        start,end=0,length
        if mode=='window_segmented':
            i=int(np.searchsorted(starts,t,side='right'))-1
            start=starts[i];end=starts[i+1] if i+1<len(starts) else length
        n=end-start
        def reflect(index):
            if n==1:return start
            q=(index-start)%(2*n-2)
            return start+(q if q<n else 2*n-2-q)
        result.append([reflect(t+d) for d in [-2,-1,0,1,2]])
    return np.asarray(result,dtype=np.int64)


@torch.inference_mode()
def denoise(model,sequence,sigma,mode='window',events=None):
    if sequence.ndim!=4 or sequence.shape[1]!=3 or not bool(torch.isfinite(sequence).all()):
        raise ValueError('finite TCHW RGB required')
    if not np.isfinite(sigma) or sigma<0:raise ValueError('finite nonnegative sigma in 0-255 units required')
    indices=window_indices(len(sequence),mode,events)
    n,c,h,w=sequence.shape
    noise_map=torch.full((1,1,h,w),float(sigma)/255.,dtype=sequence.dtype,device=sequence.device)
    routine=official_inference();output=[]
    for row in indices:
        patch=torch.stack([sequence[int(t)] for t in row]).contiguous().view(1,5*c,h,w)
        output.append(routine.temp_denoise(model,patch,noise_map)[0])
    return torch.stack(output)
