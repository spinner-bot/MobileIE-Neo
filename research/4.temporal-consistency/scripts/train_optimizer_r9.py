"""Fixed-stream single-frame optimizer controls; no extra model capacity."""
import argparse
import json
from pathlib import Path
import subprocess
import time

import cv2
import torch

from causal_fst_r5 import CausalFST
from exploration_r2_data import ROOT, load_images
from explore_temporal_controls import save_json, sha256
from real_data_r4 import batch, load_sequences, prepare_training
from train_exploration_r2 import new_model
from train_mixture_r4b import EXPANDED, mixture_loss, synthetic_pair

MODES = ('full_lr1e4','full_lr1e3','head_lr1e3')


def optimizer_groups(net, mode):
    if mode not in MODES:raise ValueError(mode)
    if mode.startswith('full_'):
        return [{'params':list(net.backbone.parameters()),'lr':1e-4 if mode=='full_lr1e4' else 1e-3}]
    head = list(net.backbone.head.block1[0].parameters())
    identities = {id(p) for p in head}
    rest = [p for p in net.backbone.parameters() if id(p) not in identities]
    assert sum(p.numel() for p in head)==912
    return [{'params':rest,'lr':1e-5},{'params':head,'lr':1e-3}]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--variants',required=True)
    p.add_argument('--out',required=True)
    args = p.parse_args()
    variants = args.variants.split(',')
    if len(set(variants))!=len(variants) or any(v not in MODES for v in variants):p.error('unique registered modes required')
    out = Path(args.out);out.mkdir(parents=True,exist_ok=False)
    cv2.setNumThreads(1);torch.set_num_threads(2)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    device=torch.device('cuda')
    revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    sources={str(f.relative_to(ROOT)):sha256(f) for f in Path(__file__).parent.glob('*.py')}
    data=prepare_training(load_sequences('train',(320,180),EXPANDED),device)
    images=load_images('train')
    for variant in variants:
        for seed in [0,1]:
            folder=out/f'{variant}-seed{seed}';folder.mkdir()
            checkpoint=ROOT/'.cache/checkpoints/exploration-r9'/folder.name
            checkpoint.mkdir(parents=True,exist_ok=False)
            prior_path=ROOT/'research/4.temporal-consistency/results/exploration-r5-spatial'/f'plain-seed{seed}'/'training.jsonl'
            prior=[json.loads(s) for s in prior_path.read_text().splitlines()]
            torch.manual_seed(seed)
            net=CausalFST(new_model(device),'plain').to(device).train()
            groups=optimizer_groups(net,variant)
            optimizer=torch.optim.Adam(groups,weight_decay=0)
            parameters=list(net.parameters())
            config={'variant':variant,'seed':seed,'steps':1000,'parameters':sum(p.numel() for p in parameters),
                'optimizer_groups':[{'lr':g['lr'],'parameter_count':sum(p.numel() for p in g['params'])} for g in groups],
                'clip':'all backbone parameters jointly at 1','size':[320,180],'loss':'Charbonnier .001',
                'git_revision_at_startup':revision,'source_sha256_at_startup':sources,
                'protocol_sha256':sha256(ROOT/'research/4.temporal-consistency/docs/exploration-r9-protocol.md'),
                'expanded_manifest_sha256':sha256(EXPANDED),'plain_trace_sha256':sha256(prior_path),
                'pretrained_sha256':sha256(ROOT/'MobileIE/pretrain/lolv1_best_slim.pkl'),
                'resume_supported':False,'cudnn_deterministic':True,'cudnn_tf32_allowed':torch.backends.cudnn.allow_tf32}
            assert config['parameters']==4075
            save_json(folder/'config.json',config)
            start=time.perf_counter()
            for step in range(1000):
                low,target,_,_,identity=batch(data,seed,step)
                sx,sy,meta=synthetic_pair(images,seed,step)
                low[1],target[1]=sx.to(device),sy.to(device)
                samples=[{'source':'real','scene':identity[0][0],'frame_pair':identity[0][1]},meta]
                if samples!=prior[step]['sample_identity']:raise AssertionError('current/GT stream changed')
                loss=mixture_loss(net.forward_pair(low),target)
                if not bool(torch.isfinite(loss)):raise FloatingPointError('nonfinite task')
                optimizer.zero_grad(set_to_none=True);loss.backward()
                norm=float(torch.nn.utils.clip_grad_norm_(parameters,1.))
                if not torch.isfinite(torch.tensor(norm)):raise FloatingPointError('nonfinite gradients')
                if step==0:
                    audit={'initial_task_delta':float(loss.detach())-prior[0]['task'],
                           'initial_gradient_norm_delta':norm-prior[0]['gradient_norm_backbone']}
                    if max(abs(v) for v in audit.values())>2e-6:raise AssertionError('initial function/gradient changed')
                    save_json(folder/'initial_plain_audit.json',audit)
                optimizer.step()
                row={'step':step+1,'task':float(loss.detach()),'gradient_norm_backbone':norm,'sample_identity':samples}
                with (folder/'training.jsonl').open('a') as stream:stream.write(json.dumps(row,allow_nan=False)+'\n')
                if (step+1)%100==0:
                    path=checkpoint/f'step{step+1:04d}.pth'
                    torch.save({'model':net.state_dict(),'optimizer':optimizer.state_dict(),'step':step+1,'config':config},path)
                    save_json(folder/f'checkpoint-step{step+1:04d}.json',{'path':str(path.relative_to(ROOT)),'sha256':sha256(path)})
                    print(json.dumps({'variant':variant,'seed':seed,**row}),flush=True)
            save_json(folder/'runtime.json',{'wall_seconds':time.perf_counter()-start,'all_current_samples_match_plain':True,
                'checkpoint':str(path.relative_to(ROOT)),'checkpoint_sha256':sha256(path)})
            del optimizer,net


if __name__=='__main__':main()
