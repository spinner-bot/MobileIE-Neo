"""Zero-future published clean-checkpoint enhancement transfer, float RGB metrics."""
import argparse
import json
from pathlib import Path
import subprocess
import time

import cv2
import numpy as np
import torch

from exploration_r2_data import ROOT, detect_changes, evaluate
from explore_state_r2 import fuse
from explore_temporal_controls import save_json, sha256
from prepare_cache_r5 import CACHE_INDEX, load_one
from prepare_development_r9 import DEV_INDEX
from prepare_native_r10 import INDEX as NATIVE_INDEX
from stablellve_adapter_r15 import MANIFEST, infer, input_tensor, load_model, output_rgb


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',required=True)
    out=Path(p.parse_args().out);out.mkdir(parents=True,exist_ok=False)
    smoke=ROOT/'research/4.temporal-consistency/results/exploration-r15-smoke/smoke.json';check=json.loads(smoke.read_text())
    if not check['all_finite'] or not check['all_shapes_preserved'] or check['max_abs_normalized']>1e-6:raise ValueError('official parity prerequisite failed')
    cv2.setNumThreads(1);torch.set_num_threads(2)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    device=torch.device('cuda');revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    sources={str(f.relative_to(ROOT)):sha256(f) for f in Path(__file__).parent.glob('*.py')};net=load_model(device)
    save_json(out/'config.json',{'method':'StableLLVE official synthetic-clean checkpoint','training_updates':0,'future_wait_frames':0,
        'parameters':sum(p.numel() for p in net.parameters()),'manifest_sha256':sha256(MANIFEST),'smoke_sha256':sha256(smoke),
        'protocol_sha256':sha256(ROOT/'research/4.temporal-consistency/docs/exploration-r15-protocol.md'),
        'cache_indices':{str(f.relative_to(ROOT)):sha256(f) for f in [CACHE_INDEX,DEV_INDEX,NATIVE_INDEX]},
        'git_revision_at_startup':revision,'source_sha256_at_startup':sources,
        'scope':'clean-pretrained model transfer, not complete noisy training reproduction; no resize/pad/extra noise/PNG quantization',
        'postprocessing':'optional shared original-MobileIE-teacher half FB alpha.1 adds teacher/history/motion cost',
        'stage_timing_scope':'synchronized GPU-stage wall incl Python/allocation; excludes transfers/loading/metrics/full CPU pipeline; no phone benchmark'})
    metas=json.loads(CACHE_INDEX.read_text())['scenes']+json.loads(DEV_INDEX.read_text())['scenes']
    metas += [{**r,'domain':'dev_native'} for r in json.loads(NATIVE_INDEX.read_text())['scenes'] if r['domain']=='native_crop']
    assert len(metas)==46
    start=time.perf_counter();count=0;computations=0;aliases=0;cost=[];events_audit=[]
    for meta in metas:
        data=load_one(meta);events,_=detect_changes(data['low']);has_events=bool(events.any())
        events_audit.append({'domain':meta['domain'],'scene':meta['scene'],'reset_frames':np.flatnonzero(events).tolist()})
        inputs=input_tensor(data['low'],device);torch.cuda.synchronize();begin=time.perf_counter()
        prediction=infer(net,inputs);torch.cuda.synchronize();seconds=time.perf_counter()-begin
        if not bool(torch.isfinite(prediction).all()):raise FloatingPointError('nonfinite official checkpoint transfer')
        raw=output_rgb(prediction);cache={}
        cost.append({'domain':meta['domain'],'scene':meta['scene'],'shape':list(inputs.shape),'seconds':seconds,'ms_per_frame':1000*seconds/len(inputs)})
        for reset in [False,True]:
            for fusion in ['raw','half01']:
                key=(fusion,bool(reset and has_events and fusion=='half01'))
                identity={'domain':meta['domain'],'scene':meta['scene'],'variant':'stablellve_official_clean','input_reset':reset,'fusion':fusion}
                label='|'.join(str(identity[k]) for k in identity)
                if key not in cache:
                    output=raw if fusion=='raw' else fuse(raw,data['half_flow'],data['half_mask'],resets=events if key[1] else None,alpha=.1)
                    metrics,frames=evaluate(output,data['gt'],data['reference_flow'],data['reference_mask'])
                    cache[key]=(metrics,frames,label);alias=None;computations+=1
                else:metrics,frames,alias=cache[key];aliases+=1
                with (out/'per_sequence.jsonl').open('a') as stream:stream.write(json.dumps({**identity,'metric_alias_of':alias,**metrics},allow_nan=False)+'\n')
                with (out/'per_frame.jsonl').open('a') as stream:
                    for frame in frames:stream.write(json.dumps({**identity,'metric_alias_of':alias,**frame},allow_nan=False)+'\n')
                count+=1
        print(json.dumps({'domain':meta['domain'],'scene':meta['scene'],'rows':count,'metric_computations':computations}),flush=True)
    assert count==184 and computations+aliases==184
    save_json(out/'stage_cost.json',cost);save_json(out/'input_events.json',events_audit)
    save_json(out/'runtime.json',{'rows':count,'actual_inference_sequences':46,'metric_computations':computations,
        'metric_aliases':aliases,'wall_seconds':time.perf_counter()-start})


if __name__=='__main__':main()
