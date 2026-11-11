"""GT-informed integer-translation diagnostic, never a deployment/evaluation correction."""
import cv2
import numpy as np


def gradients(rgb):
    rgb=np.ascontiguousarray(rgb,dtype=np.float32)
    if rgb.ndim!=3 or rgb.shape[2]!=3 or not np.isfinite(rgb).all():raise ValueError('finite HWC RGB required')
    gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY);gray=cv2.GaussianBlur(gray,(0,0),1.,borderType=cv2.BORDER_REFLECT_101)
    return np.ascontiguousarray(np.stack([cv2.Sobel(gray,cv2.CV_32F,1,0,ksize=3,scale=1/8),
        cv2.Sobel(gray,cv2.CV_32F,0,1,ksize=3,scale=1/8)],axis=-1))


def diagnose(moving,target):
    if moving.shape!=target.shape or min(moving.shape[:2])<=96:raise ValueError('matched RGB images larger than96 required')
    a,b=gradients(moving),gradients(target);h,w=a.shape[:2];template=b[48:-48,48:-48];search=a[16:-16,16:-16]
    energy={'input_gradient_std':float(search.std()),'GT_gradient_std':float(template.std())}
    profiles={'input_y':np.abs(a).mean((1,2)),'GT_y':np.abs(b).mean((1,2))}
    if min(energy.values())<1e-7:
        return {**energy,'degenerate':True,'best_dxdy':None,'best_ncc':None,'zero_ncc':None,
            'peak_gap_outside_radius3':None,'search_boundary':None,'diagnostic_peak_flag':False},None,profiles
    score=cv2.matchTemplate(search,template,cv2.TM_CCOEFF_NORMED)
    if score.shape!=(65,65) or not np.isfinite(score).all():raise FloatingPointError('invalid NCC surface')
    y,x=np.unravel_index(np.argmax(score),score.shape);best=float(score[y,x]);yy,xx=np.indices(score.shape)
    second=float(score[np.maximum(np.abs(yy-y),np.abs(xx-x))>3].max());gap=best-second
    dx,dy=int(x)-32,int(y)-32;boundary=max(abs(dx),abs(dy))==32
    return {**energy,'degenerate':False,'best_dxdy':[dx,dy],'best_ncc':best,'zero_ncc':float(score[32,32]),
        'peak_gap_outside_radius3':gap,'search_boundary':boundary,
        'diagnostic_peak_flag':bool(best>=.3 and gap>=.01 and not boundary)},score,profiles


def oracle_psnr(moving,target,dxdy):
    """Identical target support; fitted shift uses GT, not a model performance metric."""
    dx,dy=dxdy;h,w=moving.shape[:2];ref=target[48:h-48,48:w-48].astype(np.float64)
    raw=moving[48:h-48,48:w-48].astype(np.float64)
    shifted=moving[48+dy:h-48+dy,48+dx:w-48+dx].astype(np.float64)
    def value(image):return float(-10*np.log10(max(float(np.mean((image-ref)**2)),1e-15)))
    a,b=value(raw),value(shifted)
    return {'common_support_PSNR':a,'GT_fitted_translation_PSNR':b,'oracle_delta_dB':b-a,
        'scope':'GT-informed diagnostic, not deployable or replacement evaluation'}
