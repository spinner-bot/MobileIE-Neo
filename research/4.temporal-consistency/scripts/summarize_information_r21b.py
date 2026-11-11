"""Complete static aggregation-information results; separate bias and draw variance."""
import collections
import json

import numpy as np

from aggregation_model_r21 import MODES
from exploration_r2_data import ROOT
from explore_temporal_controls import save_json,sha256
from summarize_causal_r5 import jsonl

BASE=ROOT/'research/4.temporal-consistency'
KEYS=['mean_draw_psnr','mse_normalized','bias_mse_normalized','variance_normalized','raw_clip_fraction']


def main():
    folder=BASE/'results/exploration-r21b';out=BASE/'report/exploration-r21b';out.mkdir(parents=True,exist_ok=False)
    rows=jsonl(folder/'rgb.jsonl');runtime=json.loads((folder/'runtime.json').read_text())
    assert len(rows)==runtime['rows']==384 and runtime['current_output_replays']==80 and not runtime['cuda_initialized']
    assert all(r['pixel_sha_equal'] for r in json.loads((folder/'current_replays.json').read_text()))
    assert max(r['closure_abs'] for r in rows)<=1e-12
    for item in json.loads((folder/'config.json').read_text())['weights']:assert sha256(ROOT/item['path'])==item['sha256']
    for item in json.loads((folder/'cache_audit.json').read_text()):assert sha256(ROOT/item['path'])==item['sha256']
    identity=['scene','photons','rho','jpeg','seed']
    lookup={tuple(r[k] for k in identity)+(r['kind'],r['path']):r for r in rows};assert len(lookup)==384
    groups=collections.defaultdict(list)
    for r in rows:groups[(r['kind'],r['path'],r['rho'],r['jpeg'])].append(r)
    summary=[{'kind':kind,'path':path,'rho':rho,'jpeg':jpeg,'rows':len(rr),
        **{k:float(np.mean([r[k] for r in rr])) for k in KEYS}} for (kind,path,rho,jpeg),rr in sorted(groups.items())]
    deltas=[]
    for r in rows:
        if r['kind'] not in MODES or r['path']!='learned_history':continue
        for kind,path,label in [(r['kind'],'current','actual_history'),('R14_plain','current','total_vs_single'),
                ('R14_plain','preaverage_current','vs_unadapted_preaverage'),('R14_plain','postaverage_thenclip','vs_unadapted_postaverage')]:
            ref=lookup[tuple(r[k] for k in identity)+(kind,path)]
            deltas.append({**{k:r[k] for k in identity},'kind':r['kind'],'contrast':label,
                **{k:r[k]-ref[k] for k in KEYS},'variance_ratio':r['variance_normalized']/ref['variance_normalized']})
    save_json(out/'domain_summary.json',summary);save_json(out/'paired_condition_deltas.json',deltas)
    groups=collections.defaultdict(list)
    for r in deltas:groups[(r['kind'],r['contrast'],r['rho'],r['jpeg'])].append(r)
    save_json(out/'contrast_summary.json',[{'kind':kind,'contrast':contrast,'rho':rho,'jpeg':jpeg,'rows':len(rr),
        **{k:float(np.mean([r[k] for r in rr])) for k in [*KEYS,'variance_ratio']}}
        for (kind,contrast,rho,jpeg),rr in sorted(groups.items())])
    save_json(out/'audit.json',{'rows':384,'candidate_rows':256,'new_same_CPU_reference_rows':128,'fixed_conditions':16,
        'weights':10,'current_output_pixel_replays':80,'all_current_replays_equal':True,
        'bias_variance_closure_max_abs':max(r['closure_abs'] for r in rows),
        'source_sha256':{str(p.relative_to(ROOT)):sha256(p) for p in folder.iterdir()},
        'scope':'Static synthetic TRAIN images, two scenes, fixed endpoints; not video performance or independent test.'})
    lines=['# Aggregation-trained static information controls','','All fixed endpoints; mean draw PSNR is averaged per draw, not PSNR of the mean output.','',
        '| Method | Path | rho | JPEG | Mean draw PSNR | Bias MSE | Draw variance |','|---|---|---:|---|---:|---:|---:|']
    for r in summary:lines.append(f"|{r['kind']}|{r['path']}|{r['rho']}|{r['jpeg']}|{r['mean_draw_psnr']:.6f}|{r['bias_mse_normalized']:.8f}|{r['variance_normalized']:.8f}|")
    (out/'tables.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':main()
