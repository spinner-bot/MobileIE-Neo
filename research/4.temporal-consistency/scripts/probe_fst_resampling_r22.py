"""CPU-only, fixed-frame frozen-operator diagnostic; no quality ranking or training."""
import argparse
import json
from pathlib import Path
import resource
import subprocess
import time

import cv2
import numpy as np
import torch

from exploration_r2_data import ROOT
from explore_temporal_controls import save_json,sha256
from fst_resampling_r22 import check_fixtures,commutator,continue_after
from prepare_cache_r5 import load_one
from prepare_native_r10 import INDEX
from train_exploration_r2 import new_model


@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',required=True)
    out=Path(parser.parse_args().out);out.mkdir(parents=True,exist_ok=False)
    cv2.setNumThreads(1);torch.set_num_threads(2);start=time.perf_counter()
    save_json(out/'fixtures.json',check_fixtures())
    weights=[{'method':'Official','path':'MobileIE/pretrain/lolv1_best_slim.pkl'}]
    for seed in [0,1]:
        folder=ROOT/'research/4.temporal-consistency/results/exploration-r14-spatial'/f'native_crop-plain-seed{seed}'
        item=json.loads((folder/'checkpoint-step4000.json').read_text());assert sha256(ROOT/item['path'])==item['sha256']
        weights.append({'method':f'R14_seed{seed}',**item})
    models=[]
    for item in weights:
        path=ROOT/item['path'];item['sha256']=sha256(path);net=new_model(torch.device('cpu')).eval()
        if item['method']!='Official':
            saved=torch.load(path,map_location='cpu',weights_only=False);assert saved['step']==4000
            net.load_state_dict({k.removeprefix('backbone.'):v for k,v in saved['model'].items()},strict=True)
        models.append((item['method'],net))
    results=[];replays=[];caches=[]
    for meta in json.loads(INDEX.read_text())['scenes']:
        if meta['domain']!='native_crop':continue
        data=load_one(meta);caches.append(meta)
        inputs=torch.from_numpy((data['low'][7:9]/255.).transpose(0,3,1,2).copy())
        for method,net in models:
            captured={};handles=[]
            for key in ['head.block1','head','body.block1','body']:
                handles.append(net.get_submodule(key).register_forward_hook(
                    lambda module,args,value,name=key:captured.__setitem__(name,value.detach().clone())))
            net(inputs[:1]);previous=dict(captured);current_output=net(inputs[1:]);current=dict(captured)
            for handle in handles:handle.remove()
            for stage in ['head','body']:
                replay=continue_after(net,stage,current[stage]);delta=float((replay-current_output).abs().max())
                assert delta<=1e-6;replays.append({'method':method,'scene':meta['scene'],'stage':stage,'max_abs':delta})
                module=net.get_submodule(stage);a=module.weight1.double()*module.weight2.double();b=module.bias.double()
                pre,post,audit=commutator(previous[stage+'.block1'].double(),current[stage+'.block1'].double(),
                    data['half_flow'][7],data['half_mask'][7],a,b)
                rgb_pre=continue_after(net,stage,pre.float());rgb_post=continue_after(net,stage,post.float())
                results.append({'method':method,'scene':meta['scene'],'frame_pair':[7,8],'stage':stage,**audit,
                    'RGB_raw_MAE255':float((rgb_post-rgb_pre).abs().mean()*255),
                    'RGB_clipped_MAE255':float((rgb_post.clamp(0,1)-rgb_pre.clamp(0,1)).abs().mean()*255),
                    'RGB_raw_max_abs255':float((rgb_post-rgb_pre).abs().max()*255)})
            print(json.dumps({'method':method,'scene':meta['scene'],'rows':len(results)}),flush=True)
    assert len(results)==len(replays)==24 and not torch.cuda.is_initialized()
    save_json(out/'commutator.json',results);save_json(out/'continuation_replays.json',replays)
    save_json(out/'config.json',{'device':'cpu','frame_pair':[7,8],'weights':weights,'cache_entries':caches,
        'cache_index_sha256':sha256(INDEX),'protocol_sha256':sha256(ROOT/'research/4.temporal-consistency/docs/exploration-r22-protocol.md'),
        'git_revision':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'source_sha256':{str(p.relative_to(ROOT)):sha256(p) for p in Path(__file__).parent.glob('*.py')},
        'scope':'Frozen operator differences; no GT comparison, temporal noise attribution, training, or novelty claim.'})
    save_json(out/'runtime.json',{'rows':24,'wall_seconds':time.perf_counter()-start,'cuda_initialized':False,
        'peak_RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024})


if __name__=='__main__':main()
