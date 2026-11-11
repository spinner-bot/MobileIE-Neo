"""Conditional three-channel-state feature/RGB controls with matched supplied motion."""
import argparse
import json
from pathlib import Path
import time

import cv2
import numpy as np
import torch

from aligned_mix_r7c import AlignedFeatureMix, motion_field, rgb_stream
from exploration_r2_data import ROOT, detect_changes, evaluate
from explore_temporal_controls import save_json, sha256
from frozen_mix_utils_r7 import provenance
from prepare_cache_r5 import CACHE_INDEX, load_one
from train_exploration_r2 import new_model


def read_rows(path):
    return [json.loads(s) for s in path.read_text().splitlines()]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scope', choices=['features', 'outputs'], required=True)
    p.add_argument('--out', required=True)
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    prior_feature = read_rows(ROOT/'research/4.temporal-consistency/results/exploration-r7b-video-post/per_sequence.jsonl')
    prior_rgb = read_rows(ROOT/'research/4.temporal-consistency/results/exploration-r7b-video-pre/per_sequence.jsonl')
    feature_replay = {(r['scene'], r['input_reset']): r for r in prior_feature if r['method']=='body_post_selected_g05' and r['fusion']=='raw'}
    rgb_replay = {(r['scene'], r['input_reset']): r for r in prior_rgb if r['method']=='pretrained' and r['fusion']=='half01'}
    cv2.setNumThreads(1)
    torch.set_num_threads(2)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device('cuda')
    methods = ['feature_selected_post'] if args.scope=='features' else ['output_finite', 'output_recursive']
    models = {g: AlignedFeatureMix(new_model(device), g).to(device).eval() for g in [.1,.5]} if args.scope=='features' else {}
    source = provenance()
    source['base_protocol_sha256'] = source.pop('protocol_sha256')
    save_json(out/'config.json', {**source, 'scope': args.scope, 'methods': methods, 'gamma': [.1,.5],
        'protocol_sha256': sha256(ROOT/'research/4.temporal-consistency/docs/exploration-r7c-protocol.md'),
        'cache_index_sha256': sha256(CACHE_INDEX), 'motion': ['zero','teacher_half_fb','reference_proxy'],
        'sampling': 'common PyTorch bilinear align_corners=True; bounds intersected with confidence; zero bypasses interpolation',
        'state_scope': 'conditional fusion history3 channels; supplied motion/reference not free or universally deployable'})
    start, count, first, replay, sampler = time.perf_counter(), 0, [], [], []
    for meta in json.loads(CACHE_INDEX.read_text())['scenes']:
        data = load_one(meta)
        events, _ = detect_changes(data['low'])
        fields = {'zero':[(None,None)]*(len(data['low'])-1),
                  'teacher_half_fb':[motion_field(f,m,device) for f,m in zip(data['half_flow'],data['half_mask'])],
                  'reference_proxy':[motion_field(f,m,device) for f,m in zip(data['reference_flow'],data['reference_mask'])]}
        teacher = torch.from_numpy((data['teacher']/255.).transpose(0,3,1,2).copy()).to(device)
        inputs = torch.from_numpy((data['low']/255.).transpose(0,3,1,2).copy()).to(device) if args.scope=='features' else None
        entries = [('pretrained',None,'none',reset) for reset in [False,True]] if args.scope=='features' else []
        entries += [(method,g,motion,reset) for method in methods for g in [.1,.5] for motion in fields for reset in [False,True]]
        for method,gamma,motion,reset in entries:
            if method=='pretrained':
                output = data['teacher']
            else:
                if method=='feature_selected_post':
                    model = models[gamma]
                    model.reset()
                    frames = []
                    with torch.inference_mode():
                        for t,x in enumerate(inputs):
                            grid,mask = fields[motion][t-1] if t else (None,None)
                            frames.append(model(x[None],grid,mask,reset=reset and events[t])[0])
                    prediction = torch.stack(frames)
                else:
                    prediction = rgb_stream(teacher,fields[motion],gamma,recursive=method=='output_recursive',resets=events if reset else None)
                output = np.ascontiguousarray(prediction.clamp(0,1).permute(0,2,3,1).cpu().numpy()*255.)
            difference = float(np.abs(output[0]-data['teacher'][0]).max()/255.)
            if difference>2e-6:
                raise AssertionError('common first frame changed')
            identity = {'domain':meta['domain'],'scene':meta['scene'],'method':method,'gamma':gamma,'alignment':motion,'input_reset':reset}
            metrics,frames = evaluate(output,data['gt'],data['reference_flow'],data['reference_mask'])
            for filename,values in [('per_sequence.jsonl',[metrics]),('per_frame.jsonl',frames)]:
                with (out/filename).open('a') as stream:
                    for value in values:
                        stream.write(json.dumps({**identity,**value},allow_nan=False)+'\n')
            if method=='feature_selected_post' and motion=='zero' and gamma==.5:
                ref = feature_replay[(meta['scene'],reset)]
                delta = {k:abs(metrics[k]-ref[k]) for k in ['psnr_rgb','ssim_rgb','temporal_error','edge_mae']}
                if max(delta.values())>1e-8:
                    raise AssertionError('R7b zero-motion replay changed')
                replay.append({**identity,'max_abs_by_metric':delta})
            if method=='output_recursive' and motion=='teacher_half_fb' and gamma==.1:
                ref = rgb_replay[(meta['scene'],reset)]
                sampler.append({**identity,'torch_minus_cv2':{k:metrics[k]-ref[k] for k in ['psnr_rgb','ssim_rgb','temporal_error','edge_mae']}})
            first.append(difference)
            count+=1
        print(json.dumps({'scene':meta['scene'],'rows':count}),flush=True)
    if count!=(532 if args.scope=='features' else 912):
        raise ValueError('matched-state budget mismatch')
    save_json(out/'zero_motion_replay.json',replay)
    save_json(out/'sampler_differences.json',sampler)
    save_json(out/'runtime.json',{'wall_seconds':time.perf_counter()-start,'rows':count,'max_first_frame_difference':max(first)})


if __name__=='__main__':
    main()
