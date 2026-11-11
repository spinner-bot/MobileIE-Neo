"""Audit fixed-current causal contrasts and conditional bias/variance identities."""
import collections
import json

import numpy as np

from correlated_observations_r19 import pair_statistics
from exploration_r2_data import ROOT
from explore_temporal_controls import save_json,sha256
from summarize_causal_r5 import jsonl

BASE=ROOT/'research/4.temporal-consistency'
ID=['scene','photons','rho','jpeg','kind','seed','path']
METRICS=['mean_draw_psnr','mse_normalized','bias_mse_normalized','variance_normalized','noise_sd_255','raw_clip_fraction']


def main():
    source=BASE/'results/exploration-r19';out=BASE/'report/exploration-r19';out.mkdir(parents=True,exist_ok=False)
    rows=jsonl(source/'rgb.jsonl');assert len(rows)==352==len({tuple(r[k] for k in ID) for r in rows})
    runtime=json.loads((source/'runtime.json').read_text());assert runtime['rows']==352 and runtime['all_current_pixels_equal_across_rho']
    replays=json.loads((source/'current_replays.json').read_text());assert len(replays)==72 and all(r['pixel_sha_equal'] for r in replays)
    config=json.loads((source/'config.json').read_text());assert len(config['models'])==9
    for r in config['models']:assert sha256(ROOT/r['path'])==r['sha256']
    caches=json.loads((source/'cache_index.json').read_text())['conditions'];assert len(caches)==16
    currents={}
    for r in caches:
        path=ROOT/r['path'];assert sha256(path)==r['sha256']
        with np.load(path,allow_pickle=False) as data:
            assert data['current'].shape==data['previous'].shape==(64,128,128,3)
            actual=pair_statistics(data['current'],data['previous'])
        assert actual==r['input_statistics'] and actual['variance_identity_abs']<=1e-12
        key=(r['scene'],r['photons'],r['jpeg'])
        if key in currents:assert currents[key]==r['current_array_sha256']
        currents[key]=r['current_array_sha256']
    lookup={tuple(r[k] for k in ID):r for r in rows};paired=[];groups=collections.defaultdict(list)
    for r in rows:
        assert len(r['per_draw_mse_normalized'])==64 and abs(float(np.mean(r['per_draw_mse_normalized']))-r['mse_normalized'])<=1e-12
        assert abs(r['mse_normalized']-r['bias_mse_normalized']-r['variance_normalized'])<=1e-12
        groups[(r['kind'],r['path'],r['rho'],r['jpeg'])].append(r)
        if r['path']=='current':continue
        references=['current']+(['learned_history'] if r['path']=='preaverage_current' and r['kind'] in ['raw_history','aligned_history'] else [])
        for path in references:
            key=tuple(r[k] if k!='path' else path for k in ID);ref=lookup[key]
            paired.append({**{k:r[k] for k in ID if k!='path'},'first_path':r['path'],'second_path':path,
                **{k:r[k]-ref[k] for k in METRICS},
                'variance_ratio':r['variance_normalized']/max(ref['variance_normalized'],1e-30),
                'mse_ratio':r['mse_normalized']/max(ref['mse_normalized'],1e-30)})
    assert len(paired)==272
    summary=[{'kind':kind,'path':path,'rho':rho,'jpeg':jpeg,'rows':len(rr),**{k:float(np.mean([r[k] for r in rr])) for k in METRICS}}
        for (kind,path,rho,jpeg),rr in sorted(groups.items())]
    save_json(out/'domain_summary.json',summary);save_json(out/'paired_per_condition_seed.json',paired)
    groups=collections.defaultdict(list)
    for r in paired:groups[(r['kind'],r['rho'],r['jpeg'],r['first_path'],r['second_path'])].append(r)
    contrasts=[{'kind':kind,'rho':rho,'jpeg':jpeg,'first_path':a,'second_path':b,'rows':len(rr),
        **{k:float(np.mean([r[k] for r in rr])) for k in METRICS+['variance_ratio','mse_ratio']}}
        for (kind,rho,jpeg,a,b),rr in sorted(groups.items())]
    save_json(out/'contrast_summary.json',contrasts)
    save_json(out/'input_statistics.json',[{k:v for k,v in r.items() if k in ['scene','photons','rho','jpeg','input_statistics','preclip_statistics']} for r in caches])
    save_json(out/'audit.json',{'rows':352,'conditions':16,'independent_pairs_per_condition':64,'weights':9,
        'current_pixel_replays_exact':72,'input_cache_hashes_verified':16,'checkpoint_hashes_verified':9,
        'all_bias_variance_and_input_covariance_identities_pass':True,'paired_contrasts':272,
        'source_sha256':{str(p.relative_to(ROOT)):sha256(p) for p in source.iterdir() if p.is_file()},
        'scope':'two reused training images, static known correspondence; not real video, independent scenes or a new method'})
    lines=['# Known static temporal information','','All fixed conditions; means across the two photon counts/images and model seeds. Noise correlation after JPEG is measured separately.','',
        '| Model | rho | JPEG | First − second | PSNR delta | Bias MSE delta | Variance ratio | Noise SD delta |',
        '|---|---:|---|---|---:|---:|---:|---:|']
    for r in contrasts:lines.append(f"| {r['kind']} | {r['rho']} | {r['jpeg']} | {r['first_path']} − {r['second_path']} | {r['mean_draw_psnr']:.6f} | {r['bias_mse_normalized']:.8f} | {r['variance_ratio']:.6f} | {r['noise_sd_255']:.6f} |")
    (out/'tables.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':main()
