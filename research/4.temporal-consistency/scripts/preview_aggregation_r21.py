"""All fixed native clips and candidates, with exact common-CPU metric replays."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from aggregation_controls_r20 import blend
from aggregation_model_r21 import MODES
from aligned_raw_history_r11 import align_history
from continuation_r14 import create_model as plain_model
from evaluate_aligned_raw_r11 import infer
from evaluate_aggregation_r20 import metric_delta
from exploration_r2_data import ROOT,evaluate
from explore_temporal_controls import save_json,sha256
from prepare_cache_r5 import load_one
from prepare_native_r10 import INDEX
from summarize_causal_r5 import jsonl
from train_aggregation_r21 import create_model

BASE=ROOT/'research/4.temporal-consistency'


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',required=True)
    out=Path(parser.parse_args().out);out.mkdir(parents=True,exist_ok=False)
    assert json.loads((BASE/'report/exploration-r21/audit.json').read_text())['new_rows']==5888
    cache=ROOT/'.cache/qualitative-r21';cache.mkdir(parents=True,exist_ok=False)
    cv2.setNumThreads(1);torch.set_num_threads(2);device=torch.device('cpu');models={};weights=[]
    for mode in ['R14_plain',*MODES]:
        folder=BASE/'results'/('exploration-r14-spatial/native_crop-plain-seed0' if mode=='R14_plain' else f'exploration-r21-seed0/{mode}')
        item=json.loads((folder/'checkpoint-step4000.json').read_text());assert sha256(ROOT/item['path'])==item['sha256']
        saved=torch.load(ROOT/item['path'],map_location='cpu',weights_only=False);net=plain_model('plain',device) if mode=='R14_plain' else create_model(mode,device)
        net.load_state_dict(saved['model'],strict=True);net=net.backbone if mode=='R14_plain' else net
        models[mode]=net.eval();weights.append({'method':mode,**item})
    candidates=[];candidate_frames=[]
    for tag in ['pre-zero','half-post']:
        folder=BASE/'results'/('exploration-r21-eval-'+tag)
        candidates+=jsonl(folder/'per_sequence.jsonl');candidate_frames+=jsonl(folder/'per_frame.jsonl')
    def selected(r):return r['train_seed']==0 and r['step']==4000 and not r['input_reset'] and r['fusion']=='raw'
    references={(r['scene'],r['variant'],r['history']):r for r in candidates if selected(r)}
    frame_refs={(r['scene'],r['variant'],r['history'],r['frame']):r for r in candidate_frames if selected(r)}
    folder=BASE/'results/exploration-r21-reference-post-seed0-cpu'
    post={(r['scene'],r['method']):r for r in jsonl(folder/'per_sequence.jsonl') if not r['input_reset']}
    post_frames={(r['scene'],r['method'],r['frame']):r for r in jsonl(folder/'per_frame.jsonl') if not r['input_reset']}
    arrays=[];replays=[];movies=[]
    for meta in json.loads(INDEX.read_text())['scenes']:
        if meta['domain']!='native_crop':continue
        data=load_one(meta);inputs=torch.from_numpy((data['low']/255.).transpose(0,3,1,2).copy())
        def prediction(net,variant,history):return np.ascontiguousarray(infer(net,inputs,data,variant,history,None).clamp(0,1).permute(0,2,3,1).numpy()*255.)
        raw=prediction(models['R14_plain'],'plain_high','causal');warped=[raw[0]]
        for t in range(1,len(raw)):warped.append(align_history(raw[t-1],raw[t],data['half_flow'][t-1],data['half_mask'][t-1])[0])
        warped=np.stack(warped);panels={'Input':data['low'],'GT':data['gt'],'R14 single':raw,
            'R20 postwarp .1':blend(raw,warped,.1),'R20 postwarp .5':blend(raw,warped,.5)}
        identities={'R14 single':('post','raw'),'R20 postwarp .1':('post','postwarp01'),'R20 postwarp .5':('post','postwarp05')}
        for mode in MODES:
            for history in ['current','causal']:
                name=mode+' / '+history;panels[name]=prediction(models[mode],'concat_aligned_high',history);identities[name]=(mode,history)
        assert len(panels)==13 and len(raw)==16
        for name,value in panels.items():
            p=cache/(meta['scene']+'__'+name.replace(' / ','__').replace(' ','_')+'.npy');np.save(p,value)
            arrays.append({'scene':meta['scene'],'panel':name,'path':str(p.relative_to(ROOT)),'sha256':sha256(p),'shape':list(value.shape)})
            if name not in identities:continue
            mode,history=identities[name];metrics,frames=evaluate(value,data['gt'],data['reference_flow'],data['reference_mask'])
            ref=post[(meta['scene'],history)] if mode=='post' else references[(meta['scene'],mode,history)]
            delta=metric_delta(metrics,ref)
            for frame in frames:
                ref=post_frames[(meta['scene'],history,frame['frame'])] if mode=='post' else frame_refs[(meta['scene'],mode,history,frame['frame'])]
                delta=max(delta,metric_delta(frame,ref))
            assert delta<=1e-10;replays.append({'scene':meta['scene'],'panel':name,'sequence_and_frame_max_abs_metric':delta})
        path=out/(meta['scene']+'.mp4');writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'mp4v'),8,(1024,704));assert writer.isOpened()
        for t in range(16):
            board=np.zeros((704,1024,3),np.uint8)
            cv2.putText(board,f"{meta['scene']} frame {t:02d}/15; display8fps; seed0 step4000; no quality from encoded preview",(8,22),cv2.FONT_HERSHEY_SIMPLEX,.43,(255,255,255),1,cv2.LINE_AA)
            for i,(name,value) in enumerate(panels.items()):
                y=32+(i//4)*168;x=(i%4)*256
                image=cv2.resize(np.clip(np.rint(value[t]),0,255).astype(np.uint8),(256,144),interpolation=cv2.INTER_AREA)
                board[y+24:y+168,x:x+256]=cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
                cv2.putText(board,name,(x+5,y+17),cv2.FONT_HERSHEY_SIMPLEX,.38,(255,255,255),1,cv2.LINE_AA)
            writer.write(board)
            if t==8:assert cv2.imwrite(str(out/(meta['scene']+'-frame8.png')),board)
        writer.release();cap=cv2.VideoCapture(str(path));decoded=0
        while True:
            ok,frame=cap.read()
            if not ok:break
            assert frame.shape==(704,1024,3);decoded+=1
        cap.release();assert decoded==16
        movies.append({'scene':meta['scene'],'path':str(path.relative_to(ROOT)),'sha256':sha256(path),'decoded_frames':decoded,'display_fps':8})
        print(json.dumps({'scene':meta['scene'],'panels':13,'decoded_frames':decoded}),flush=True)
    assert len(replays)==44 and len(arrays)==52 and not torch.cuda.is_initialized()
    save_json(out/'manifest.json',{'arrays':arrays,'weights':weights,'movies':movies,'metric_replays':replays,
        'scope':'All four fixed exploratory clips, seed0, all16 frames; display8fps is not sourcefps. Lossy video only preview; fixed-frame PNGs inspected separately; no independent subjective/video test.'})


if __name__=='__main__':main()
