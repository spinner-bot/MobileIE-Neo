"""Official BGR/255 inference; no external resize/pad, added noise or temporal state."""
import importlib.util
import json

import numpy as np
import torch

from exploration_r2_data import ROOT
from explore_temporal_controls import sha256

MANIFEST=ROOT/'research/4.temporal-consistency/data/exploration-r15-stablellve/manifest.json'


def load_model(device):
    manifest=json.loads(MANIFEST.read_text())
    for row in manifest['sources']:
        if sha256(ROOT/row['path'])!=row['sha256']:raise ValueError('official source changed')
    path=ROOT/manifest['path']
    if sha256(path)!=manifest['sha256']:raise ValueError('official checkpoint changed')
    source=ROOT/'third_party/stablellve/model.py'
    spec=importlib.util.spec_from_file_location('mobileie_stablellve_reference',source)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    net=module.UNet(n_channels=3,bilinear=True).to(device)
    state=torch.load(path,map_location=device,weights_only=True)
    net.load_state_dict(state,strict=True)
    return net.eval()


def input_tensor(rgb_frames,device):
    values=np.asarray(rgb_frames)
    if values.ndim!=4 or values.shape[-1]!=3 or min(values.shape[1:3])<16 or not np.isfinite(values).all():
        raise ValueError('finite THWC RGB, minimum16x16, required')
    bgr=values[...,::-1].astype(np.float64)/255.
    return torch.from_numpy(bgr.transpose(0,3,1,2).copy()).to(device=device,dtype=torch.float32)


def output_rgb(bgr_tensor):
    return np.ascontiguousarray((bgr_tensor.clamp(0,1).permute(0,2,3,1).cpu().numpy()*255.)[...,::-1])


@torch.inference_mode()
def infer(net,inputs):
    return torch.stack([net(x[None])[0] for x in inputs])
