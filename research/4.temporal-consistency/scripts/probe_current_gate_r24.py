"""Frozen global-channel readout intervention, with unchanged body features and exact replays."""
import argparse
import hashlib
import json
from pathlib import Path
import resource
import subprocess
import time

import cv2
import numpy as np
import torch

from aggregation_model_r21 import MODES
from aligned_raw_history_r11 import align_history
from evaluate_aggregation_r20 import metric_delta
from exploration_r2_data import ROOT,evaluate
from explore_temporal_controls import save_json,sha256
from prepare_cache_r5 import load_one
from prepare_native_r10 import INDEX
from train_aggregation_r21 import create_model

BASE=ROOT/'research/4.temporal-consistency'


def tensor_sha(tensor):return hashlib.sha256(tensor.detach().contiguous().numpy().tobytes()).hexdigest()


def select(row):return row['domain']=='dev_native' and row['step']==4000 and not row['input_reset'] and row['fusion']=='raw'


@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',required=True)
    out=Path(parser.parse_args().out);out.mkdir(parents=True,exist_ok=False)
    cv2.setNumThreads(1);torch.set_num_threads(2);start=time.perf_counter();sources={};references={};frame_references={}
    for tag in ['pre-zero','half-post']:
        folder=BASE/'results'/('exploration-r21-eval-'+tag)
        for name,destination in [('per_sequence.jsonl',references),('per_frame.jsonl',frame_references)]:
            path=folder/name;sources[str(path.relative_to(ROOT))]=sha256(path)
            with path.open() as stream:
                for line in stream:
                    r=json.loads(line)
                    if select(r):destination[(r['scene'],r['variant'],r['train_seed'],r['history'],*((r['frame'],) if name=='per_frame.jsonl' else ()))]=r
    assert len(references)==64 and len(frame_references)==1024
    models=[];weights=[]
    for mode in MODES:
        for seed in [0,1]:
            folder=BASE/'results'/f'exploration-r21-seed{seed}'/mode;item=json.loads((folder/'checkpoint-step4000.json').read_text())
            assert sha256(ROOT/item['path'])==item['sha256'];saved=torch.load(ROOT/item['path'],map_location='cpu',weights_only=False)
            assert saved['step']==4000;net=create_model(mode,torch.device('cpu'));net.load_state_dict(saved['model'],strict=True)
            models.append((mode,seed,net.eval()));weights.append({'mode':mode,'seed':seed,**item})
    save_json(out/'config.json',{'device':'cpu','batch':1,'Torch_threads':2,'OpenCV_threads':1,'weights':weights,
        'reference_sources':sources,'native_index_sha256':sha256(INDEX),'protocol_sha256':sha256(BASE/'docs/exploration-r24-protocol.md'),
        'revision':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'source_sha256':{str(p.relative_to(ROOT)):sha256(p) for p in Path(__file__).parent.glob('*.py')},
        'scope':'Frozen att-node intervention; spatial att1 formula unchanged, its values can change downstream; no deployment cost claim.'})
    rows=0;replays=[];body_audit=[];gate_rows=[];first=[];PSNR_delta_max=0.;closure_max=0.
    for meta in json.loads(INDEX.read_text())['scenes']:
        if meta['domain']!='native_crop':continue
        data=load_one(meta);inputs=torch.from_numpy((data['low']/255.).transpose(0,3,1,2).copy())
        for mode,seed,net in models:
            state={};predictions={name:[] for name in ['current','causal','current_gate']}
            def gate_hook(module,args,value):
                if state['phase']=='current_gate':return state['gate_current']
                state['gates'].append(value.detach().clone());return value
            def body_hook(module,args,value):
                if state['phase']=='causal':state['bodies'].append(value.detach().clone())
                elif state['phase']=='current_gate':
                    j=state['body_index'];original=state['bodies'][j];assert torch.equal(value,original)
                    body_audit.append({'scene':meta['scene'],'mode':mode,'seed':seed,'frame':state['frame'],'branch':j,
                        'original_sha256':tensor_sha(original),'intervened_sha256':tensor_sha(value),'pixel_equal':True})
                    state['body_index']+=1
            handles=[net.backbone.att.register_forward_hook(gate_hook),net.backbone.body.register_forward_hook(body_hook)]
            for t,current in enumerate(inputs):
                previous=current
                if t:
                    warped,_=align_history(data['low'][t-1],data['low'][t],data['half_flow'][t-1],data['half_mask'][t-1])
                    previous=torch.from_numpy((warped/255.).transpose(2,0,1).copy())
                state.update(phase='current',gates=[],frame=t)
                predictions['current'].append(net.transform(current[None],current[None])[0])
                assert all(torch.equal(state['gates'][0],v) for v in state['gates'])
                g_cur=state['gates'][0];state.update(phase='causal',gates=[],bodies=[])
                predictions['causal'].append(net.transform(current[None],previous[None])[0])
                g_actual=state['gates'];state.update(phase='current_gate',gate_current=g_cur,body_index=0)
                predictions['current_gate'].append(net.transform(current[None],previous[None])[0])
                assert state['body_index']==len(state['bodies'])==(2 if mode=='post_fixedhalf' else 1)
                gate_rows.append({'scene':meta['scene'],'mode':mode,'seed':seed,'frame':t,'current_global_gate':g_cur.flatten().tolist(),
                    'original_causal_global_gates':[g.flatten().tolist() for g in g_actual],
                    'max_abs_gate_change':max(float((g-g_cur).abs().max()) for g in g_actual)})
            for handle in handles:handle.remove()
            first_reference=None
            for path,values in predictions.items():
                raw=torch.stack(values);assert bool(torch.isfinite(raw).all())
                output=np.ascontiguousarray(raw.clamp(0,1).permute(0,2,3,1).numpy()*255.)
                if first_reference is None:first_reference=output[0].copy()
                first_delta=float(np.max(np.abs(output[0]-first_reference)));assert first_delta<=1e-4
                first.append({'scene':meta['scene'],'mode':mode,'seed':seed,'path':path,'max_abs_pixel':first_delta})
                metrics,frames=evaluate(output,data['gt'],data['reference_flow'],data['reference_mask'])
                identity={'domain':'dev_native','scene':meta['scene'],'variant':mode,'train_seed':seed,'path':path}
                if path!='current_gate':
                    delta=metric_delta(metrics,references[(meta['scene'],mode,seed,path)])
                    for frame in frames:delta=max(delta,metric_delta(frame,frame_references[(meta['scene'],mode,seed,path,frame['frame'])]))
                    assert delta<=1e-10;replays.append({**identity,'sequence_and_frame_max_abs_metric':delta})
                global_bias=[];centered=[]
                for frame in frames:
                    t=frame['frame'];error=(output[t].astype(np.float64)-data['gt'][t].astype(np.float64))/255.
                    mu=error.mean((0,1));mse=float(np.mean(error**2));bias=float(np.mean(mu**2));spatial=float(np.mean((error-mu)**2))
                    closure=abs(mse-bias-spatial);assert closure<=1e-12;closure_max=max(closure_max,closure)
                    delta=abs(float(-10*np.log10(mse))-frame['psnr_rgb']);assert delta<=1e-8;PSNR_delta_max=max(PSNR_delta_max,delta)
                    global_bias.append(bias);centered.append(spatial)
                    with (out/'per_frame.jsonl').open('a') as stream:stream.write(json.dumps({**identity,**frame,
                        'mse_normalized':mse,'global_RGB_bias_mse':bias,'centered_error_mse':spatial,'global_RGB_mean_error_255':(mu*255).tolist()},allow_nan=False)+'\n')
                with (out/'per_sequence.jsonl').open('a') as stream:stream.write(json.dumps({**identity,**metrics,
                    'global_RGB_bias_mse':float(np.mean(global_bias)),'centered_error_mse':float(np.mean(centered)),
                    'mse_normalized':float(np.mean(global_bias)+np.mean(centered))},allow_nan=False)+'\n')
                rows+=1
            print(json.dumps({'scene':meta['scene'],'mode':mode,'seed':seed,'rows':rows,'body_checks':len(body_audit)}),flush=True)
    assert rows==96 and len(replays)==64 and len(body_audit)==640 and len(gate_rows)==512 and not torch.cuda.is_initialized()
    for name,values in [('baseline_replays.json',replays),('body_invariance.json',body_audit),('gates.json',gate_rows),('first_frame_audit.json',first)]:save_json(out/name,values)
    save_json(out/'runtime.json',{'rows':96,'frames':1536,'new_intervention_rows':32,'replayed_sequence_rows':64,'replayed_frame_rows':1024,
        'body_pixel_equal_checks':640,'MSE_closure_max_abs':closure_max,'PSNR_identity_max_abs_dB':PSNR_delta_max,
        'device':'cpu','cuda_initialized':False,'wall_seconds':time.perf_counter()-start,
        'peak_RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024})


if __name__=='__main__':main()
