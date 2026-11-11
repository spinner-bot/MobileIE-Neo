"""Falsifiable official-weight, normalization and inference equivalence smoke."""
import argparse
import json
from pathlib import Path
import subprocess
import time

import cv2
import numpy as np
import torch

from exploration_r2_data import ROOT, load_images
from explore_temporal_controls import save_json, sha256
from fastdvdnet_adapter_r12 import MANIFEST, denoise, load_model, official_inference
from video_metrics import psnr


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',required=True)
    out=Path(p.parse_args().out);out.mkdir(parents=True,exist_ok=False)
    cv2.setNumThreads(1);torch.set_num_threads(2)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    device=torch.device('cuda');start=time.perf_counter()
    model=load_model(device);image=load_images('train')['kodim01']
    gt=np.stack([image[160:352,240+2*t:496+2*t] for t in range(8)])
    rng=np.random.default_rng(12);low=np.clip(gt+rng.normal(0,25,gt.shape),0,255).astype(np.float32)
    sequence=torch.from_numpy((low/255.).transpose(0,3,1,2).copy()).to(device)
    adapted=denoise(model,sequence,25,'window')
    with torch.inference_mode():
        original=official_inference().denoise_seq_fastdvdnet(sequence,torch.tensor([25/255.],device=device),5,model)
    delta=float((adapted-original).abs().max())
    output=adapted.permute(0,2,3,1).cpu().numpy()*255.
    noisy_quality=float(np.mean([psnr(x,y) for x,y in zip(low,gt)]))
    denoised_quality=float(np.mean([psnr(x,y) for x,y in zip(output,gt)]))
    result={'input_sigma':25,'rng_seed':12,'source':'Kodak train01, fitted768512, fixed crop y160:352 x240+2t:496+2t',
        'frames':8,'size':[256,192],'input_psnr':noisy_quality,'denoised_psnr':denoised_quality,
        'psnr_gain':denoised_quality-noisy_quality,'official_normalized_max_abs_difference':delta,
        'preset_max_abs_tolerance':1e-6,'preset_minimum_gain_db':1.,'parameters':sum(p.numel() for p in model.parameters()),
        'finite':bool(torch.isfinite(adapted).all()),'wall_seconds':time.perf_counter()-start,
        'manifest_sha256':sha256(MANIFEST),'protocol_sha256':sha256(ROOT/'research/4.temporal-consistency/docs/exploration-r12-protocol.md'),
        'git_revision':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'scope':'clipped AWGN functionality check only; not LLVE transfer evidence'}
    save_json(out/'smoke.json',result);print(json.dumps(result),flush=True)
    if not result['finite'] or delta>1e-6 or result['psnr_gain']<1:raise AssertionError('predeclared denoiser smoke failed')


if __name__=='__main__':main()
