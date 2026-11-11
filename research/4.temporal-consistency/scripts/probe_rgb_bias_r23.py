"""Exact RGB-space error decomposition of existing fixed native predictions."""
import argparse
import json
from pathlib import Path
import resource
import subprocess
import time

import numpy as np
import torch

from exploration_r2_data import ROOT
from explore_temporal_controls import save_json,sha256

BASE=ROOT/'research/4.temporal-consistency'
COMPONENTS=['mse_normalized','global_RGB_bias_mse','common_channel_bias_mse','opponent_channel_bias_mse','centered_error_mse']


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',required=True)
    out=Path(parser.parse_args().out);out.mkdir(parents=True,exist_ok=False);start=time.perf_counter()
    manifest_path=BASE/'report/exploration-r21/qualitative/manifest.json';manifest=json.loads(manifest_path.read_text())
    assert len(manifest['arrays'])==52 and len(manifest['metric_replays'])==44
    arrays={};source_arrays=[]
    for item in manifest['arrays']:
        path=ROOT/item['path'];assert sha256(path)==item['sha256']
        value=np.load(path,mmap_mode='r');assert list(value.shape)==item['shape']==[16,288,512,3]
        arrays[(item['scene'],item['panel'])]=path;source_arrays.append(item)
    references={}
    for tag in ['pre-zero','half-post']:
        path=BASE/'results'/('exploration-r21-eval-'+tag)/'per_frame.jsonl'
        with path.open() as stream:
            for line in stream:
                r=json.loads(line)
                if r['domain']=='dev_native' and r['train_seed']==0 and r['step']==4000 and not r['input_reset'] and r['fusion']=='raw':
                    references[(r['scene'],r['variant']+' / '+r['history'],r['frame'])]=r['psnr_rgb']
    path=BASE/'results/exploration-r21-reference-post-seed0-cpu/per_frame.jsonl'
    names={'raw':'R14 single','postwarp01':'R20 postwarp .1','postwarp05':'R20 postwarp .5'}
    with path.open() as stream:
        for line in stream:
            r=json.loads(line)
            if r['domain']=='dev_native' and not r['input_reset'] and r['method'] in names:
                references[(r['scene'],names[r['method']],r['frame'])]=r['psnr_rgb']
    assert len(references)==704
    frame_rows=[];sequence_rows=[];psnr_max=0.;closure_max=0.
    for scene in sorted({k[0] for k in arrays}):
        gt=np.load(arrays[(scene,'GT')],mmap_mode='r')
        for (name,panel),path in arrays.items():
            if name!=scene or panel in ['Input','GT']:continue
            predicted=np.load(path,mmap_mode='r');local=[]
            for t in range(16):
                error=(predicted[t].astype(np.float64)-gt[t].astype(np.float64))/255.
                mu=error.mean((0,1));mse=float(np.mean(error**2));global_bias=float(np.mean(mu**2))
                common=float(mu.mean()**2);opponent=float(np.mean((mu-mu.mean())**2));centered=float(np.mean((error-mu)**2))
                closure=max(abs(mse-global_bias-centered),abs(global_bias-common-opponent));assert closure<=1e-12;closure_max=max(closure_max,closure)
                psnr=float(-10*np.log10(mse));delta=abs(psnr-references[(scene,panel,t)]);assert delta<=1e-8;psnr_max=max(psnr_max,delta)
                row={'scene':scene,'panel':panel,'frame':t,'mse_normalized':mse,'global_RGB_bias_mse':global_bias,
                    'common_channel_bias_mse':common,'opponent_channel_bias_mse':opponent,'centered_error_mse':centered,
                    'global_RGB_mean_error_255':(mu*255).tolist(),'psnr_rgb':psnr,
                    'centered_error_oracle_psnr':float(-10*np.log10(centered)),'closure_abs':closure}
                frame_rows.append(row);local.append(row)
            mu=np.array([r['global_RGB_mean_error_255'] for r in local])/255.
            persistent=float(np.mean(mu.mean(0)**2));fluctuation=float(np.mean((mu-mu.mean(0))**2))
            avg={k:float(np.mean([r[k] for r in local])) for k in COMPONENTS}
            closure=abs(avg['global_RGB_bias_mse']-persistent-fluctuation);assert closure<=1e-12;closure_max=max(closure_max,closure)
            sequence_rows.append({'scene':scene,'panel':panel,**avg,'mean_frame_psnr_rgb':float(np.mean([r['psnr_rgb'] for r in local])),
                'mean_centered_error_oracle_psnr':float(np.mean([r['centered_error_oracle_psnr'] for r in local])),
                'persistent_global_RGB_bias_mse':persistent,'temporal_global_RGB_error_fluctuation_mse':fluctuation,
                'global_RGB_mean_error_255':(mu.mean(0)*255).tolist(),'global_time_closure_abs':closure})
        print(json.dumps({'scene':scene,'sequence_rows':len(sequence_rows),'frame_rows':len(frame_rows)}),flush=True)
    assert len(sequence_rows)==44 and len(frame_rows)==704 and not torch.cuda.is_initialized()
    lookup={(r['scene'],r['panel']):r for r in sequence_rows};deltas=[]
    for r in sequence_rows:
        if ' / ' not in r['panel']:continue
        refs=[('vs_single','R14 single')]
        if r['panel'].endswith(' / causal'):refs+=[('actual_history',r['panel'].replace(' / causal',' / current')),('vs_strong_post','R20 postwarp .5')]
        for contrast,panel in refs:
            ref=lookup[(r['scene'],panel)];diff={k:r[k]-ref[k] for k in COMPONENTS}
            closure=abs(diff['mse_normalized']-diff['common_channel_bias_mse']-diff['opponent_channel_bias_mse']-diff['centered_error_mse'])
            assert closure<=1e-12
            deltas.append({'scene':r['scene'],'panel':r['panel'],'reference':panel,'contrast':contrast,**diff,
                'global_bias_excess_over_total_excess':diff['global_RGB_bias_mse']/diff['mse_normalized'] if abs(diff['mse_normalized'])>1e-15 else None,
                'closure_abs':closure})
    assert len(deltas)==64
    for name,rows in [('per_frame.jsonl',frame_rows),('per_sequence.jsonl',sequence_rows),('paired_deltas.jsonl',deltas)]:
        with (out/name).open('w') as stream:
            for row in rows:stream.write(json.dumps(row,allow_nan=False)+'\n')
    save_json(out/'config.json',{'spatial_scope':'full-frame, matching formal PSNR; temporal metric border is independent','manifest_sha256':sha256(manifest_path),'source_arrays':source_arrays,
        'protocol_sha256':sha256(BASE/'docs/exploration-r23-protocol.md'),
        'revision':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'source_sha256':{str(p.relative_to(ROOT)):sha256(p) for p in Path(__file__).parent.glob('*.py')},
        'scope':'Same fixed native seed0 arrays; GT-derived oracle decomposition, not an enhancement algorithm or perceptual color space.'})
    save_json(out/'runtime.json',{'sequences':44,'frames':704,'contrasts':64,'source_arrays_verified':52,
        'MSE_closure_max_abs':closure_max,'formal_frame_PSNR_replay_max_abs_dB':psnr_max,'device':'cpu','cuda_initialized':False,
        'wall_seconds':time.perf_counter()-start,'peak_RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024})


if __name__=='__main__':main()
