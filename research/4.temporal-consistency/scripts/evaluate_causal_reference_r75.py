"""Final-only original-size CPU evaluation of causal and same-weight current routes."""
import argparse
import json
import os
import time
import traceback

import cv2
import numpy as np
import torch

from causal_reference_io_r75 import ROOT, BASE, PROTOCOL, write, append, digest, stamp
from causal_reference_evaluation_r75 import metas,final_markers,load_final
from rebotnet_reference import native_pair,tile_layout
from prepare_cache_r5 import load_one
from exploration_r2_data import evaluate


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed',type=int,choices=[0,1],required=True)
    args=parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':
        raise SystemExit('CPU evaluator requires an initially empty CUDA mask.')
    torch.set_num_threads(2);cv2.setNumThreads(1)
    torch.use_deterministic_algorithms(True)
    markers=final_markers();model=load_final(args.seed,markers)
    scenes,sources=metas()
    out=BASE/f'results/exploration-r75-eval-seed{args.seed}'
    cache=ROOT/f'.cache/experiments/exploration-r75/eval001/seed{args.seed}'
    assert cache.resolve().is_relative_to('/data/Users/zrh/data/MobileIE-Neo')
    out.mkdir(exist_ok=False);cache.mkdir(parents=True,exist_ok=False)
    write(out/'config.json',dict(at=stamp(),seed=args.seed,checkpoint=markers[args.seed],
        cache_indices=sources,protocol_sha256=digest(PROTOCOL),device='cpu',torch_threads=2,
        expected_rows=20,expected_frames=896,expected_tile_CNN_calls=8192,
        source_sha256={name:digest(BASE/'scripts'/name) for name in
            ['evaluate_causal_reference_r75.py','causal_reference_evaluation_r75.py','rebotnet_reference.py',
             'exploration_r2_data.py','video_metrics.py','prepare_cache_r5.py']},
        scope='Released XS final64K/self-start native-tile adaptation, reused DEV; no new post/GT state/future/flow.'))
    started,complete,count,frames,calls= time.perf_counter(),False,0,0,0
    try:
        for meta in scenes:
            data=load_one(meta)
            expected=64 if meta['domain']=='davis_train_pilot' else 16
            assert data['low'].shape==data['gt'].shape and len(data['low'])==expected
            causal_first=None
            for route in ('causal','current'):
                identity=dict(domain=meta['domain'],scene=meta['scene'],source_video=meta['source_video'],
                    seed=args.seed,train_seed=args.seed,model='ReBotNet_XS',history=route,method='raw',
                    label=f'ReBotNet_{route}',parameters=890275,project_updates=64000,future_wait_max=0)
                path=cache/f'{meta["scene"]}--{route}.npy'
                assert not path.exists()
                output=np.lib.format.open_memmap(path,mode='w+',dtype=np.float32,shape=data['gt'].shape)
                previous=None;began=time.perf_counter()
                with torch.inference_mode():
                    for t in range(expected):
                        current=torch.from_numpy(data['low'][t].transpose(2,0,1).copy()[None]).float()/255
                        history=current if route=='current' or previous is None else previous
                        value=native_pair(model,history,current)
                        assert torch.isfinite(value).all()
                        append(out/'state_range.jsonl',dict(**identity,frame=t,
                            internal_min=float(value.min()),internal_max=float(value.max()),
                            persistent_RGB_state_bytes=value.numel()*value.element_size() if route=='causal' else 0))
                        output[t]=value[0].clamp(0,1).permute(1,2,0).contiguous().numpy()*255
                        if route=='causal':
                            previous=value
                        calls+=len(tile_layout(*current.shape[-2:]))
                output.flush()
                if route=='causal':
                    causal_first=np.asarray(output[0]).copy()
                else:
                    delta=float(np.abs(causal_first-output[0]).max())
                    append(out/'first_frame.jsonl',dict(**identity,max_abs_delta_0to255=delta))
                    assert delta<=1e-4
                append(out/'prediction_manifest.jsonl',dict(**identity,path=str(path.relative_to(ROOT)),
                    shape=list(output.shape),dtype='float32',sha256=digest(path),checkpoint_sha256=markers[args.seed]['sha256']))
                append(out/'stage_cost.jsonl',dict(**identity,frames=expected,seconds=time.perf_counter()-began,
                    scope='Unwarmed CPU evaluation with checks/cache writes; not model/phone benchmark.'))
                metrics,details=evaluate(output,data['gt'],data['reference_flow'],data['reference_mask'])
                append(out/'per_sequence.jsonl',dict(**identity,**metrics))
                for row in details:
                    append(out/'per_frame.jsonl',dict(**identity,**row))
                count+=1;frames+=len(details)
                print(json.dumps(dict(**identity,completed_rows=count)),flush=True)
                del output
            del data
        assert (count,frames,calls)==(20,896,8192) and not torch.cuda.is_initialized()
        complete=True
    except BaseException:
        (out/'failure.txt').write_text(traceback.format_exc());raise
    finally:
        write(out/'runtime.json',dict(ended=stamp(),completed_normally=complete,seed=args.seed,
            quality_rows=count,frame_rows=frames,neural_frame_outputs=frames,tile_CNN_calls=calls,
            training_updates=0,CUDA_initialized=torch.cuda.is_initialized(),seconds=time.perf_counter()-started))


if __name__=='__main__':
    main()
