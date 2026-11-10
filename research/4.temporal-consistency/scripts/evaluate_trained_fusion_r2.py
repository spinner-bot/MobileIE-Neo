"""Strengthen controls by pairing trained models with identical flow fusion."""
import argparse
import json
import time
from pathlib import Path

import cv2
import torch

from bounded_fst import apply_bounded
from exploration_r2_data import ROOT, evaluate, load_images, make_sequence
from explore_state_r2 import forward_backward, fuse
from explore_temporal_controls import MobileIELLENetS, inference, save_json, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    cv2.setNumThreads(1)
    torch.set_num_threads(2)
    device = torch.device('cuda')
    models, provenance = {}, {}
    choices = [("task", "exploration-r2", "task"),
               ("task_T1", "exploration-r2-lambda10", "task_temporal"),
               ("bounded_T0.1", "exploration-r2", "bounded_temporal")]
    for label, cache, variant in choices:
        for seed in [0,1]:
            path = ROOT/'.cache/checkpoints'/cache/f'{variant}-seed{seed}'/'last.pth'
            saved = torch.load(path,map_location=device,weights_only=False)
            net = MobileIELLENetS(12).eval().to(device)
            if variant.startswith('bounded'):
                apply_bounded(net,saved['config']['limits'])
            net.load_state_dict(saved['model'])
            name=f'{label}-seed{seed}'
            models[name]=net
            provenance[name]={'path':str(path.relative_to(ROOT)),'sha256':sha256(path),'step':saved['step']}
    save_json(out/'config.json',{'experiment_id':'exploration-r2-trained-fusion','models':provenance,
                               'fusion_alpha':.7,'note':'selected exploratory configurations, not formal test-set model selection'})
    images=load_images('exploratory_validation')
    names=list(images)
    start=time.perf_counter()
    flow_seconds=0.
    rows=[]
    for i,scene in enumerate(names):
        for case in ['pan','light_step','occlusion','cut']:
            for input_seed in [0,1]:
                gt,low,true_flow,mask=make_sequence(images[scene],images[names[(i+1)%len(names)]],case,input_seed)
                flow_start=time.perf_counter()
                pairs=[forward_backward(low[t-1],low[t]) for t in range(1,len(low))]
                flows,confidence=[p[0] for p in pairs],[p[1] for p in pairs]
                flow_seconds+=time.perf_counter()-flow_start
                for name,net in models.items():
                    raw,_=inference(net,low,device,4)
                    for fusion,output in [('raw',raw),('flow',fuse(raw,flows)),('fb',fuse(raw,flows,confidence))]:
                        metrics,_=evaluate(output,gt,true_flow,mask)
                        row={'model':name,'fusion':fusion,'scene':scene,'case':case,'input_seed':input_seed,**metrics}
                        rows.append(row)
                        with (out/'per_sequence.jsonl').open('a') as stream:
                            stream.write(json.dumps(row,allow_nan=False)+'\n')
                print(json.dumps({'scene':scene,'case':case,'input_seed':input_seed,'rows':len(rows)}),flush=True)
    save_json(out/'runtime.json',{'wall_seconds':time.perf_counter()-start,'flow_cpu_seconds':flow_seconds,
                                'flow_pairs':6*4*2*23,'timing_note':'CPU flow diagnostic, not end-to-end latency benchmark'})


if __name__=='__main__':
    main()
