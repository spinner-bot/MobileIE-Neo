"""Identical-pair pre/post enhancement aggregation with full source replays."""
import collections
import json

import numpy as np

from exploration_r2_data import ROOT
from explore_temporal_controls import save_json,sha256
from summarize_causal_r5 import jsonl
from summarize_information_r19 import ID,METRICS

BASE=ROOT/'research/4.temporal-consistency'


def main():
    source=BASE/'results/exploration-r19-aggregation';old=BASE/'results/exploration-r19';out=BASE/'report/exploration-r19-aggregation'
    out.mkdir(parents=True,exist_ok=False);rows=jsonl(source/'rgb.jsonl');assert len(rows)==288
    runtime=json.loads((source/'runtime.json').read_text());assert runtime['new_rows']==288
    config=json.loads((source/'config.json').read_text());assert sha256(old/'config.json')==config['source_config_sha256']
    assert sha256(old/'cache_index.json')==config['source_cache_index_sha256'] and sha256(old/'rgb.jsonl')==config['source_rgb_sha256']
    replays=json.loads((source/'replays.json').read_text());assert len(replays)==288 and all(r['pixel_sha_equal'] for r in replays)
    gaps=jsonl(source/'aggregation_gap.jsonl');assert len(gaps)==144 and max(r['closure_abs'] for r in gaps)<=1e-12
    for r in rows:
        assert abs(float(np.mean(r['per_draw_mse_normalized']))-r['mse_normalized'])<=1e-12
        assert abs(r['mse_normalized']-r['bias_mse_normalized']-r['variance_normalized'])<=1e-12
    combined=jsonl(old/'rgb.jsonl')+rows;assert len(combined)==640
    lookup={tuple(r[k] for k in ID):r for r in combined};assert len(lookup)==640
    deltas=[]
    for r in rows:
        for other in ['current','preaverage_current']:
            ref=lookup[tuple(r[k] if k!='path' else other for k in ID)]
            deltas.append({**{k:r[k] for k in ID if k!='path'},'first_path':r['path'],'second_path':other,
                **{k:r[k]-ref[k] for k in METRICS},'variance_ratio':r['variance_normalized']/max(ref['variance_normalized'],1e-30)})
    assert len(deltas)==576
    save_json(out/'paired_per_condition_seed.json',deltas)
    groups=collections.defaultdict(list)
    for r in deltas:groups[(r['kind'],r['rho'],r['jpeg'],r['first_path'],r['second_path'])].append(r)
    contrasts=[{'kind':kind,'rho':rho,'jpeg':jpeg,'first_path':a,'second_path':b,'rows':len(rr),
        **{k:float(np.mean([r[k] for r in rr])) for k in METRICS+['variance_ratio']}}
        for (kind,rho,jpeg,a,b),rr in sorted(groups.items())]
    save_json(out/'contrast_summary.json',contrasts)
    groups=collections.defaultdict(list)
    for r in gaps:groups[(r['kind'],r['rho'],r['jpeg'])].append(r)
    save_json(out/'aggregation_gap_summary.json',[{'kind':kind,'rho':rho,'jpeg':jpeg,'rows':len(rr),
        **{k:float(np.mean([r[k] for r in rr])) for k in ['raw_network_gap_mse','raw_network_gap_mean_squared','raw_network_gap_variance','raw_network_gap_signed_mean_255']}}
        for (kind,rho,jpeg),rr in sorted(groups.items())])
    save_json(out/'audit.json',{'new_RGB_rows':288,'reused_RGB_rows':352,'total_RGB_rows':640,'identical_observation_conditions':16,
        'exact_current_and_preaverage_pixel_replays':288,'raw_network_gap_rows':144,'paired_contrasts':576,
        'all_scalar_moment_identities_pass':True,'source_sha256':{str(p.relative_to(ROOT)):sha256(p) for p in source.iterdir() if p.is_file()},
        'scope':'standard static two-observation references; gap reflects complete network nonlinearity, not FST alone; no real-video claim'})
    lines=['# Same-pair aggregation-order controls','','Post average minus current / pre-average. All conditions retained; no new sampling or training.','',
        '| Model | rho | JPEG | First − second | PSNR delta | Bias MSE delta | Variance ratio |',
        '|---|---:|---|---|---:|---:|---:|']
    for r in contrasts:lines.append(f"| {r['kind']} | {r['rho']} | {r['jpeg']} | {r['first_path']} − {r['second_path']} | {r['mean_draw_psnr']:.6f} | {r['bias_mse_normalized']:.8f} | {r['variance_ratio']:.6f} |")
    (out/'tables.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':main()
