"""Explicit eager model-stage cost; neither full-pipeline nor mobile benchmarking."""
import argparse
import json
from pathlib import Path
import time

import cv2
import numpy as np
import torch

from exploration_r2_data import ROOT
from explore_temporal_controls import save_json,sha256
from train_exploration_r2 import new_model
from train_multiscale_r16 import create_model


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',required=True)
    out=Path(p.parse_args().out);out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2);cv2.setNumThreads(1);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    device=torch.device('cuda');torch.manual_seed(16)
    source=ROOT/'research/4.temporal-consistency/results/exploration-r16-spatial-raw/spatial-seed0/checkpoint-step4000.json'
    item=json.loads(source.read_text());checkpoint=ROOT/item['path'];assert sha256(checkpoint)==item['sha256']
    saved=torch.load(checkpoint,map_location=device,weights_only=False);assert saved['step']==4000
    results=[]
    for hw in [(270,480),(288,512),(1080,1920)]:
        current=torch.rand(1,3,*hw,device=device);previous=torch.rand_like(current)
        for variant in ['official','spatial','raw_history']:
            net=new_model(device) if variant=='official' else create_model(variant,device)
            if variant!='official':net.load_state_dict(saved['model'],strict=True)
            net.eval();macs=[];handles=[]
            def hook(module,inputs,output):macs.append(output.numel()*(module.in_channels//module.groups)*int(np.prod(module.kernel_size)))
            for module in net.modules():
                if isinstance(module,torch.nn.Conv2d):handles.append(module.register_forward_hook(hook))
            def forward():return net(current) if variant=='official' else net.transform(current,previous)
            with torch.inference_mode():
                output=forward();assert bool(torch.isfinite(output).all()) and output.shape==current.shape
                for handle in handles:handle.remove()
                for _ in range(10):forward()
                torch.cuda.synchronize();baseline=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats();times=[]
                for _ in range(50):
                    torch.cuda.synchronize();start=time.perf_counter();prediction=forward();torch.cuda.synchronize()
                    times.append(1000*(time.perf_counter()-start));del prediction
                peak=torch.cuda.max_memory_allocated()
            row={'HW':list(hw),'variant':variant,'parameters':sum(p.numel() for p in net.parameters()),'conv_MACs_per_frame':sum(macs),
                'conv_calls':len(macs),'warmups':10,'timed_forwards':50,'p10_median_p90_ms':np.percentile(times,[10,50,90]).tolist(),
                'allocation_baseline_bytes':baseline,'peak_allocated_bytes':peak,'incremental_peak_bytes':peak-baseline,
                'additional_raw_previous_bytes':0 if variant in ['official','spatial'] else previous.numel()*previous.element_size(),
                'spatial_duplicates_encoder':variant=='spatial','weights':'official' if variant=='official' else 'spatial seed0 step4000 for topology cost only'}
            results.append(row);print(json.dumps(row),flush=True);del output,net
        del current,previous
    save_json(out/'cost.json',{'rows':results,'checkpoint':str(checkpoint.relative_to(ROOT)),'checkpoint_sha256':item['sha256'],
        'protocol_sha256':sha256(ROOT/'research/4.temporal-consistency/docs/exploration-r16-protocol.md'),
        'cudnn_tf32_allowed':torch.backends.cudnn.allow_tf32,'cudnn_deterministic':True,'dtype':'float32','batch':1,
        'scope':'synchronized eager GPU-stage wall including Python/allocation; excludes CPU/loading/transfers/teacher/flow/metrics; shared A100, not phone',
        'MAC_scope':'Conv2d only, multiply-add counts as1; FST, gates, interpolation, activations and GAP excluded',
        'memory_scope':'PyTorch allocated memory, not reserved or total device; baseline includes allocated inputs/models; raw state listed separately'})


if __name__=='__main__':main()
