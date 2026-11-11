"""Matched raw-pair versus spatial auxiliary-head task fine-tuning."""
import argparse
import json
from pathlib import Path
import subprocess
import time

import cv2
import torch

from exploration_r2_data import ROOT, load_images
from explore_temporal_controls import save_json, sha256
from raw_pair_head_r8 import MODES, RawPairHead, augmented_history
from real_data_r4 import batch, load_sequences, prepare_training
from train_exploration_r2 import new_model
from train_mixture_r4b import EXPANDED, mixture_loss, synthetic_pair


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--variants',required=True)
    p.add_argument('--out',required=True)
    args = p.parse_args()
    variants = args.variants.split(',')
    if len(set(variants))!=len(variants) or any(v not in MODES for v in variants):
        p.error('unique known variants required')
    out = Path(args.out)
    out.mkdir(parents=True,exist_ok=False)
    cv2.setNumThreads(1)
    torch.set_num_threads(2)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device('cuda')
    revision = subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    sources = {str(path.relative_to(ROOT)):sha256(path) for path in Path(__file__).parent.glob('*.py')}
    data = prepare_training(load_sequences('train',(320,180),EXPANDED),device)
    synthetic = load_images('train')
    for variant in variants:
        for seed in [0,1]:
            folder = out/f'{variant}-seed{seed}'
            folder.mkdir()
            checkdir = ROOT/'.cache/checkpoints/exploration-r8'/folder.name
            checkdir.mkdir(parents=True,exist_ok=False)
            prior_path = ROOT/'research/4.temporal-consistency/results/exploration-r5-spatial'/f'plain-seed{seed}'/'training.jsonl'
            prior = [json.loads(s) for s in prior_path.read_text().splitlines()]
            torch.manual_seed(seed)
            net = RawPairHead(new_model(device),variant).to(device).train()
            main_parameters, auxiliary = list(net.backbone.parameters()), list(net.auxiliary.parameters())
            optimizer = torch.optim.Adam([{'params':main_parameters,'lr':1e-5},{'params':auxiliary,'lr':1e-3}],weight_decay=0)
            config = {'variant':variant,'seed':seed,'steps':1000,'batch_pairs':2,'size':[320,180],
                'parameters':sum(p.numel() for p in net.parameters()),'lr_backbone':1e-5,'lr_auxiliary':1e-3,
                'clip_backbone':1.,'clip_auxiliary':1.,'loss':'task Charbonnier epsilon .001',
                'real_scenes':[r['scene'] for r in data],'expanded_manifest_sha256':sha256(EXPANDED),
                'plain_trace_sha256':sha256(prior_path),'pretrained_sha256':sha256(ROOT/'MobileIE/pretrain/lolv1_best_slim.pkl'),
                'protocol_sha256':sha256(ROOT/'research/4.temporal-consistency/docs/exploration-r8-protocol.md'),
                'git_revision_at_startup':revision,'source_sha256_at_startup':sources,
                'cudnn_deterministic':True,'cudnn_tf32_allowed':torch.backends.cudnn.allow_tf32,'resume_supported':False}
            save_json(folder/'config.json',config)
            start = time.perf_counter()
            for step in range(1000):
                low,target,_,_,identity = batch(data,seed,step)
                sx,sy,meta = synthetic_pair(synthetic,seed,step)
                low[1],target[1] = sx.to(device),sy.to(device)
                samples = [{'source':'real','scene':identity[0][0],'frame_pair':identity[0][1]},meta]
                if samples!=prior[step]['sample_identity']:
                    raise AssertionError('current data stream differs from R5 plain')
                history,actions = augmented_history(low,seed,step) if variant=='temporal_delta_aug' else (None,[])
                pred = net.forward_pair(low,history)
                loss = mixture_loss(pred,target)
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError('nonfinite task loss')
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                main_norm = float(torch.nn.utils.clip_grad_norm_(main_parameters,1.))
                aux_norm = float(torch.nn.utils.clip_grad_norm_(auxiliary,1.))
                if not bool(torch.isfinite(torch.tensor([main_norm,aux_norm])).all()):
                    raise FloatingPointError('nonfinite gradients')
                optimizer.step()
                row = {'step':step+1,'task':float(loss.detach()),'gradient_norm_backbone':main_norm,'gradient_norm_auxiliary':aux_norm,
                    'auxiliary_l2':float(net.auxiliary.weight.detach().norm()),'sample_identity':samples,'history_augmentation':actions}
                if step==0:
                    audit = {'initial_task_delta_from_plain':row['task']-prior[0]['task'],
                        'initial_gradient_norm_delta_from_plain':main_norm-prior[0]['gradient_norm_backbone']}
                    if abs(audit['initial_task_delta_from_plain'])>2e-6 or abs(audit['initial_gradient_norm_delta_from_plain'])>2e-5:
                        raise AssertionError('initial backbone function/gradient changed')
                    save_json(folder/'initial_plain_audit.json',audit)
                with (folder/'training.jsonl').open('a') as stream:
                    stream.write(json.dumps(row,allow_nan=False)+'\n')
                if (step+1)%100==0:
                    path = checkdir/f'step{step+1:04d}.pth'
                    torch.save({'model':net.state_dict(),'optimizer':optimizer.state_dict(),'step':step+1,'config':config},path)
                    save_json(folder/f'checkpoint-step{step+1:04d}.json',{'path':str(path.relative_to(ROOT)),'sha256':sha256(path)})
                    print(json.dumps({'variant':variant,'seed':seed,**row}),flush=True)
            save_json(folder/'runtime.json',{'wall_seconds':time.perf_counter()-start,
                'checkpoint':str((checkdir/'step1000.pth').relative_to(ROOT)),
                'checkpoint_sha256':sha256(checkdir/'step1000.pth'),'all_current_samples_match_plain':True})
            del optimizer,net


if __name__=='__main__':
    main()
