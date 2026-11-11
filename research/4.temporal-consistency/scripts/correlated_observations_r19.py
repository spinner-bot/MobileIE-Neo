"""Poisson thinning and conditional Gaussian coupling with a fixed current observation."""
import cv2
import numpy as np


def sensor_pair(clean,photons,rho,current_seed,history_seed,draws):
    if not 0<=rho<=1 or photons<=0 or draws<1:raise ValueError('valid correlation, exposure and draws required')
    clean=np.asarray(clean,np.float64)
    if not np.isfinite(clean).all() or clean.min()<0:raise ValueError('nonnegative finite clean intensities required')
    rate=np.broadcast_to(clean*photons,(draws,*clean.shape));current_rng=np.random.default_rng(current_seed)
    counts=current_rng.poisson(rate);read=current_rng.normal(0,.008,rate.shape)
    history_rng=np.random.default_rng(history_seed)
    shared=history_rng.binomial(counts,rho);independent=history_rng.poisson((1-rho)*rate)
    previous_read=rho*read+np.sqrt(1-rho*rho)*history_rng.normal(0,.008,rate.shape)
    return counts/photons+read,(shared+independent)/photons+previous_read


def encode(samples,jpeg):
    pixels=(np.clip(samples,0,1)*255).astype(np.uint8)
    if not jpeg:return pixels.astype(np.float32)/255.
    result=[]
    for rgb in pixels:
        ok,encoded=cv2.imencode('.jpg',cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR),[cv2.IMWRITE_JPEG_QUALITY,70])
        if not ok:raise RuntimeError('JPEG encoding failed')
        result.append(cv2.cvtColor(cv2.imdecode(encoded,cv2.IMREAD_COLOR),cv2.COLOR_BGR2RGB))
    return np.stack(result).astype(np.float32)/255.


def pair_statistics(current,previous):
    a,b=np.asarray(current,np.float64),np.asarray(previous,np.float64)
    if a.shape!=b.shape or a.shape[0]<2:raise ValueError('matching repeated observations required')
    ea,eb=a-a.mean(0),b-b.mean(0)
    va,vb=float(np.mean(ea*ea)),float(np.mean(eb*eb));cov=float(np.mean(ea*eb))
    avg=(a+b)/2;direct=float(np.mean((avg-avg.mean(0))**2));formula=(va+vb+2*cov)/4
    return {'variance_current':va,'variance_previous':vb,'covariance':cov,'empirical_correlation':cov/max(np.sqrt(va*vb),1e-30),
        'variance_average':direct,'variance_average_formula':formula,'variance_identity_abs':abs(direct-formula),
        'average_variance_over_current':direct/max(va,1e-30)}
