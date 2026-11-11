"""Fixed training/dev label-pair audit; all GT-fitted shifts remain diagnostic only."""
import collections
import json
from pathlib import Path
import resource
import subprocess
import time

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from exploration_r2_data import ROOT
from explore_temporal_controls import save_json,sha256
from native_training_r14 import load_training
from pair_geometry_r18 import diagnose,oracle_psnr
from prepare_cache_r5 import load_one
from prepare_native_r10 import INDEX


def main():
    base=ROOT/'research/4.temporal-consistency';out=base/'report/exploration-r18';out.mkdir(parents=True,exist_ok=False)
    cache=ROOT/'.cache/pair-geometry-r18';cache.mkdir(parents=True,exist_ok=False)
    cv2.setNumThreads(1);torch.set_num_threads(2);start=time.perf_counter();rows=[];examples={}
    revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    save_json(out/'config.json',{'protocol_sha256':sha256(base/'docs/exploration-r18-protocol.md'),
        'revision':revision,'source_sha256':{str(p.relative_to(ROOT)):sha256(p) for p in Path(__file__).parent.glob('*.py')},
        'opencv_version':cv2.__version__,'scope':'GT-informed diagnostic only; no new training, no changes to old metrics'})
    data,integrity=load_training();save_json(out/'training_source_integrity.json',integrity)
    def one(split,scene,frame,kind,moving,target,sources):
        result,surface,profiles=diagnose(moving,target);label=f'{split}-{scene}-{frame}-{kind}'
        path=cache/(label+'.npz');np.savez_compressed(path,surface=np.empty(0) if surface is None else surface,**profiles)
        row={'split':split,'scene':scene,'frame':frame,'kind':kind,'sources':sources,**result,
            'map_path':str(path.relative_to(ROOT)),'map_sha256':sha256(path)}
        if kind=='R14_native_plain' and result['best_dxdy'] is not None:row['GT_oracle_diagnostic']=oracle_psnr(moving,target,result['best_dxdy'])
        rows.append(row)
        if split=='dev' and frame==8 and kind=='input':examples[scene]=(row,surface,profiles)
        with (out/'per_pair.jsonl').open('a') as stream:stream.write(json.dumps(row,allow_nan=False)+'\n')
    for scene in data:
        for frame in [0,8,15]:
            moving=scene['native_input'][frame,396:684,704:1216].astype(np.float32)/255.
            target=scene['native_gt'][frame,396:684,704:1216].astype(np.float32)/255.
            one('train',scene['scene'],frame,'input',moving,target,
                {side:scene['source_files'][side][frame] for side in ['input','gt']})
    del data
    preview=base/'results/exploration-r16-preview/manifest.json';manifest=json.loads(preview.read_text())
    for meta in json.loads(INDEX.read_text())['scenes']:
        if meta['domain']!='native_crop':continue
        sequence=load_one(meta);assert sequence['low'].shape==(16,288,512,3)
        for frame in range(16):one('dev',meta['scene'],frame,'input',sequence['low'][frame]/255.,sequence['gt'][frame]/255.,{'cache_metadata':meta})
        record=next(r for r in manifest['panels'] if r['scene']==meta['scene'] and r['method']=='R14 native plain')
        path=ROOT/record['display_array_path'];assert sha256(path)==record['sha256']
        prediction=np.load(path,allow_pickle=False)
        one('dev',meta['scene'],8,'R14_native_plain',prediction,sequence['gt'][8]/255.,
            {'preview_manifest_sha256':sha256(preview),'array_sha256':record['sha256'],'cache_metadata':meta})
    assert len(rows)==104 and sum(r['kind']=='input' for r in rows)==100 and len(examples)==4
    groups=collections.defaultdict(list)
    for r in rows:groups[(r['split'],r['scene'],r['kind'])].append(r)
    summary=[]
    for (split,scene,kind),rr in sorted(groups.items()):
        valid=[r for r in rr if not r['degenerate']]
        summary.append({'split':split,'scene':scene,'kind':kind,'rows':len(rr),'nondegenerate':len(valid),
            'diagnostic_peak_flags':sum(r['diagnostic_peak_flag'] for r in rr),'search_boundary_hits':sum(bool(r['search_boundary']) for r in rr),
            'median_abs_best_dxdy_all_nondegenerate':np.median(np.abs([r['best_dxdy'] for r in valid]),axis=0).tolist() if valid else None,
            'mean_zero_ncc':float(np.mean([r['zero_ncc'] for r in valid])) if valid else None,
            'mean_best_ncc':float(np.mean([r['best_ncc'] for r in valid])) if valid else None,
            'scope':'all maxima retained, including weak/ambiguous; not ground-truth registration accuracy'})
    save_json(out/'by_scene.json',summary)
    fig,axes=plt.subplots(4,2,figsize=(11,12),layout='constrained')
    for (scene,(r,surface,profiles)),(a,b) in zip(sorted(examples.items()),axes):
        if surface is not None:
            artist=a.imshow(surface,extent=[-32,32,32,-32],cmap='viridis');a.plot(0,0,'wx');a.plot(*r['best_dxdy'],'r+')
            fig.colorbar(artist,ax=a,fraction=.04)
        a.set(title=scene+' frame8 / NCC '+str(r['best_dxdy']),xlabel='Input sample dx',ylabel='Input sample dy')
        for key in ['input_y','GT_y']:
            v=profiles[key];b.plot(v/max(float(v.max()),1e-12),label=key)
        b.set(title='Normalized signed-gradient magnitude row profile',xlabel='Native ROI row',ylabel='Relative magnitude');b.legend()
    fig.suptitle('R18: GT-informed label-pair translation diagnostic; all fixed scenes\nSearch maxima are not true flow, deployable correction or replacement quality metrics.',fontsize=12)
    for extension in ['png','pdf']:fig.savefig(out/('pair_geometry.'+extension),dpi=160)
    plt.close(fig)
    save_json(out/'runtime.json',{'wall_seconds':time.perf_counter()-start,'rows':104,'raw_pairs':100,'model_output_pairs':4,
        'peak_process_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'cuda_initialized':torch.cuda.is_initialized(),
        'native_index_sha256':sha256(INDEX),'scope':'CPU audit on known train/dev only; no GPU job or new test pixels'})


if __name__=='__main__':main()
