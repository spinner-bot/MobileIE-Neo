"""Matched motion/state comparisons without treating reference motion as deployable."""
import argparse
import collections
import json
from pathlib import Path

import numpy as np

from exploration_r2_data import ROOT
from explore_temporal_controls import save_json,sha256

METRICS=['psnr_rgb','ssim_rgb','temporal_error','edge_mae']


def jsonl(path):
    return [json.loads(s) for s in path.read_text().splitlines()]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',required=True)
    args=p.parse_args()
    out=Path(args.out); out.mkdir(parents=True,exist_ok=False)
    base=ROOT/'research/4.temporal-consistency/results'
    rows,frames,hashes=[],[],{}
    for scope,expected in [('features',532),('outputs',912)]:
        folder=base/('exploration-r7c-'+scope)
        r=jsonl(folder/'per_sequence.jsonl')
        if len(r)!=expected or json.loads((folder/'runtime.json').read_text())['rows']!=expected:
            raise ValueError('incomplete motion comparison')
        rows+=r; frames+=jsonl(folder/'per_frame.jsonl')
        hashes.update({str(path.relative_to(ROOT)):sha256(path) for path in folder.iterdir() if path.is_file()})
    if len({(r['scene'],r['method'],r['gamma'],r['alignment'],r['input_reset']) for r in rows})!=1444:
        raise ValueError('duplicate evaluation')
    groups=collections.defaultdict(list)
    for r in rows:
        groups[(r['domain'],r['method'],r['gamma'],r['alignment'],r['input_reset'])].append(r)
    summaries=[{'domain':d,'method':m,'gamma':g,'alignment':a,'input_reset':reset,'conditions':len(rr),
        **{k:float(np.mean([r[k] for r in rr])) for k in METRICS}}
        for (d,m,g,a,reset),rr in sorted(groups.items())]
    save_json(out/'domain_summary.json',summaries)
    rgb={(r['scene'],r['gamma'],r['alignment'],r['input_reset']):r for r in rows if r['method']=='output_finite'}
    deltas=[]
    for r in rows:
        if r['method']!='feature_selected_post':continue
        ref=rgb[(r['scene'],r['gamma'],r['alignment'],r['input_reset'])]
        deltas.append({k:r[k] for k in ['domain','scene','gamma','alignment','input_reset']}|{k:r[k]-ref[k] for k in METRICS})
    save_json(out/'feature_minus_same_state_rgb.json',deltas)
    grouped=collections.defaultdict(list)
    for r in deltas:grouped[(r['domain'],r['gamma'],r['alignment'],r['input_reset'])].append(r)
    save_json(out/'paired_summary.json',[{'domain':d,'gamma':g,'alignment':a,'input_reset':reset,
        **{k:{'mean':float(np.mean([r[k] for r in rr])),'median':float(np.median([r[k] for r in rr])),
              'positive_conditions':sum(r[k]>0 for r in rr),'conditions':len(rr)} for k in METRICS}}
        for (d,g,a,reset),rr in sorted(grouped.items())])
    reference={(r['scene'],r['input_reset']):r for r in rows if r['method']=='pretrained'}
    feasible=[]
    for r in rows:
        if r['method']=='pretrained':continue
        ref=reference[(r['scene'],r['input_reset'])]
        violations={'psnr':r['psnr_rgb']<ref['psnr_rgb']-.05,'ssim':r['ssim_rgb']<ref['ssim_rgb']-.005,
                    'edge':r['edge_mae']>ref['edge_mae']+.05}
        feasible.append({k:r[k] for k in ['domain','scene','method','gamma','alignment','input_reset']}|
                        {'violations':violations,'all_within_r4_tolerance':not any(violations.values())})
    save_json(out/'quality_tolerance_diagnostic.json',feasible)
    save_json(out/'event_frame12.json',[r for r in frames if r['domain'].startswith('event_') and r['frame']==12])
    feature_folder=base/'exploration-r7c-features'
    output_folder=base/'exploration-r7c-outputs'
    replay=json.loads((feature_folder/'zero_motion_replay.json').read_text())
    sampler=json.loads((output_folder/'sampler_differences.json').read_text())
    save_json(out/'audit.json',{'sequence_evaluations':1444,'video_input_conditions':38,
        'zero_motion_replay_max_abs':max(v for r in replay for v in r['max_abs_by_metric'].values()),
        'cv2_torch_sampler_max_abs':{k:max(abs(r['torch_minus_cv2'][k]) for r in sampler) for k in METRICS},
        'source_hashes':hashes,
        'scope':'fixed official weights; supplied motion is not free; reference_proxy is not deployable or true real motion; recurrent RGB has longer history'})
    lines=['# Matched conditional 3-channel-state/motion controls','',
           'Same supplied motion and sampler. Reference proxy is not deployable. Recurrent RGB has longer temporal context.','']
    for domain in ['test','legacy_synthetic','event_light_step','event_cut']:
        lines+=['## '+domain,'','| Method | Gamma | Motion | Reset | PSNR | SSIM | Temporal |','|---|---:|---|---|---:|---:|---:|']
        for r in summaries:
            if r['domain']==domain:
                lines.append(f"| {r['method']} | {r['gamma']} | {r['alignment']} | {r['input_reset']} | {r['psnr_rgb']:.5f} | {r['ssim_rgb']:.5f} | {r['temporal_error']:.5f} |")
        lines.append('')
    (out/'tables.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':main()
