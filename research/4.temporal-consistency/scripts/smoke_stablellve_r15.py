"""Strict state loading and original BGR mathematical path on four spatial shapes."""
import argparse
import json
from pathlib import Path
import time

import cv2
import numpy as np
import torch

from exploration_r2_data import ROOT
from explore_temporal_controls import save_json, sha256
from stablellve_adapter_r15 import MANIFEST, infer, input_tensor, load_model


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',required=True)
    out=Path(p.parse_args().out);out.mkdir(parents=True,exist_ok=False)
    cv2.setNumThreads(1);torch.set_num_threads(2)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    device=torch.device('cuda');net=load_model(device);rng=np.random.default_rng(15);rows=[];start=time.perf_counter()
    for h,w in [(32,48),(37,53),(64,65),(47,80)]:
        rgb=rng.integers(0,256,(h,w,3),dtype=np.uint8);bgr=rgb[...,::-1]
        official=np.expand_dims((bgr/255.).transpose(2,0,1),0)
        reference_input=torch.from_numpy(official.copy()).to(device=device,dtype=torch.float32)
        with torch.inference_mode():reference=net(reference_input)
        adapted=infer(net,input_tensor(rgb[None],device))
        delta=float((reference-adapted).abs().max())
        record={'HW':[h,w],'shape_preserved':list(adapted.shape)==[1,3,h,w],
            'finite':bool(torch.isfinite(adapted).all()),'official_max_abs_normalized':delta}
        rows.append(record)
    result={'rows':rows,'parameters':sum(p.numel() for p in net.parameters()),'preset_max_abs_tolerance':1e-6,
        'all_finite':all(r['finite'] for r in rows),'all_shapes_preserved':all(r['shape_preserved'] for r in rows),
        'max_abs_normalized':max(r['official_max_abs_normalized'] for r in rows),'wall_seconds':time.perf_counter()-start,
        'manifest_sha256':sha256(MANIFEST),'protocol_sha256':sha256(ROOT/'research/4.temporal-consistency/docs/exploration-r15-protocol.md'),
        'scope':'official clean checkpoint, BGR convention and shape functionality; no task-quality claim'}
    save_json(out/'smoke.json',result);print(json.dumps(result),flush=True)
    if not result['all_finite'] or not result['all_shapes_preserved'] or result['max_abs_normalized']>1e-6:
        raise AssertionError('predeclared StableLLVE parity failed')


if __name__=='__main__':main()
