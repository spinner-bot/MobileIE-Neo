"""Quadratic commutator using the same OpenCV bilinear gather as RGB controls."""
import cv2
import numpy as np
import torch


def gather(value,flow):
    if value.ndim!=4 or value.shape[0]!=1:raise ValueError('one NCHW CPU feature required')
    array=value[0].detach().numpy().transpose(1,2,0).copy()
    h,w=array.shape[:2];x,y=np.meshgrid(np.arange(w,dtype=np.float32),np.arange(h,dtype=np.float32))
    sx,sy=x+flow[...,0],y+flow[...,1]
    result=cv2.remap(array,sx,sy,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
    if result.ndim==2:result=result[...,None]
    valid=(sx>=0)&(sx<=w-1)&(sy>=0)&(sy<=h-1)
    return torch.from_numpy(result.transpose(2,0,1).copy())[None],torch.from_numpy(valid)[None,None]


def commutator(previous,current,flow,mask,a,b):
    if previous.dtype!=torch.float64 or current.dtype!=torch.float64:raise ValueError('FP64 identity required')
    mean,bounds=gather(previous,flow);second,_=gather(previous.square(),flow)
    post,_=gather(a*previous.square()+b,flow)
    valid=bounds&torch.from_numpy(mask.astype(bool))[None,None]
    variance=second-mean.square();pre=a*mean.square()+b;correction=a*variance
    error=(post-pre-correction)[valid.expand_as(post)]
    scale=max(float(post.abs().max()),float(pre.abs().max()),float(correction.abs().max()),1e-30)
    residual=float(error.abs().max()) if error.numel() else 0.
    if residual/scale>1e-10:raise AssertionError('FP64 quadratic commutator identity failed')
    current_q=a*current.square()+b
    return torch.where(valid,pre,current_q),torch.where(valid,post,current_q),{
        'valid_fraction':float(valid.double().mean()),'a':float(a),'variance_mean_valid':float(variance[valid.expand_as(variance)].mean()),
        'variance_min_valid':float(variance[valid.expand_as(variance)].min()),
        'commutator_rms_valid':float(correction[valid.expand_as(correction)].square().mean().sqrt()),
        'identity_max_abs':residual,'identity_scaled_max_abs':residual/scale}


def continue_after(net,stage,value):
    body=net.body(value) if stage=='head' else value
    gate=net.att(body);maximum=(gate*body).max(1,keepdim=True).values
    return net.tail((net.att1(maximum)*gate)*body)


def check_fixtures():
    generator=torch.Generator().manual_seed(220011)
    u=torch.randn(1,3,13,17,generator=generator,dtype=torch.float64);current=u+1
    mask=np.ones((13,17),bool);fractional=np.zeros((13,17,2),np.float32);fractional[:]=[.375,-.625]
    pre,post,audit=commutator(u,current,fractional,mask,torch.tensor(-.3,dtype=torch.float64),torch.tensor(.7,dtype=torch.float64))
    assert audit['commutator_rms_valid']>0
    integer=fractional.copy();integer[:]=[1,-1]
    _,_,exact=commutator(u,current,integer,mask,torch.tensor(-.3,dtype=torch.float64),torch.tensor(.7,dtype=torch.float64))
    assert exact['commutator_rms_valid']==0
    return {'fractional':audit,'integer':exact,'fixtures_passed':2}
