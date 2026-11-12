"""Synthetic integration check; no new real images, SGD or CUDA."""
from datetime import datetime
import json

import cv2
import numpy as np
import torch

from drv_references_r59 import NETWORKS, PRIMARY, load_network, infer, postprocess, metrics
from dynamic_training_r34 import BASE, digest
from fastdvdnet_adapter_r12 import window_indices


def main():
    torch.set_num_threads(2); cv2.setNumThreads(1)
    torch.use_deterministic_algorithms(True)
    assert not torch.cuda.is_initialized()
    out = BASE/'process/exploration-r59/cpu-gate001'; out.mkdir(parents=True,exist_ok=False)
    constant = np.full((8,16,20,3),5,np.float32); zero = np.zeros_like(constant)
    result,_ = metrics(constant,zero)
    assert result['same_coordinate_time_std_RMS'] == result['same_coordinate_adjacent_RGB_MAE'] == 0
    assert abs(result['psnr_rgb']-20*np.log10(255/5)) < 1e-12
    ramp = np.broadcast_to(np.arange(8,dtype=np.float32)[:,None,None,None],constant.shape)
    result,_ = metrics(ramp,zero)
    assert result['same_coordinate_adjacent_RGB_MAE'] == 1
    assert abs(result['same_coordinate_time_std_RMS']-np.sqrt(5.25)) < 1e-12
    indices = window_indices(8,'window')
    assert all(len(set(indices[t])) == 5 for t in PRIMARY)
    low = np.random.default_rng(59001).integers(0,256,(8,16,20,3),dtype=np.uint8)
    records = []
    for name in NETWORKS:
        model,control,source = load_network(name)
        raw = infer(name,model,low); outputs,stats = postprocess(raw,control)
        assert raw.dtype == np.float32 and 0 <= raw.min() <= raw.max() <= 255
        assert all(np.array_equal(v[0],raw[0]) for v in outputs.values())
        records.append(dict(name=name, parameters=source['parameters'], outputs=list(outputs), finite=True))
    assert not torch.cuda.is_initialized()
    record = dict(at=datetime.now().astimezone().isoformat(),passed=True, networks=records,
                  synthetic_center_outputs=64, unique_synthetic_frames=8, source_window_indices=indices.tolist(),
                  primary_frames=PRIMARY, real_images_opened=0, SGD=0, cuda_initialized=False,
                  source_sha256={n:digest(BASE/'scripts'/n) for n in ['drv_references_r59.py','check_drv_references_r59.py']},
                  scope='One bounded synthetic integration check, not external quality or CPU bitwise replay.')
    (out/'gate.json').write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
    print(json.dumps(record))


if __name__ == '__main__': main()
