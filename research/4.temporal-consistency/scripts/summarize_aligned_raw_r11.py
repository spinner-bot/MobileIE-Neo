"""Audit stronger history controls without conflating spatial or training gains."""
import argparse
import collections
import json
from pathlib import Path

import numpy as np

from exploration_r2_data import ROOT
from explore_temporal_controls import save_json, sha256
from summarize_causal_r5 import KEYS, means, method, paired

BASE=ROOT/'research/4.temporal-consistency'
DOMAINS=['test','dev','dev_native','legacy_synthetic','event_cut','event_light_step']


def jsonl(path):return [json.loads(s) for s in path.read_text().splitlines()]


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',required=True)
    out=Path(p.parse_args().out);out.mkdir(parents=True,exist_ok=False)
    rows=[];frames=[];sources={};repeats=[];first=[]
    for scope,expected in [('raw',1472),('aligned',736)]:
        folder=BASE/'results'/('exploration-r11-eval-'+scope)
        values=jsonl(folder/'per_sequence.jsonl');assert len(values)==expected==json.loads((folder/'runtime.json').read_text())['rows']
        rows+=values;frames+=jsonl(folder/'per_frame.jsonl')
        repeats+=json.loads((folder/'plain_replay.json').read_text());first+=json.loads((folder/'first_frame_audit.json').read_text())
        for r in json.loads((folder/'checkpoints.json').read_text()):assert sha256(ROOT/r['path'])==r['sha256']
        events=json.loads((folder/'input_events.json').read_text());assert len(events)==46
        for e in events:
            if e['domain'].startswith('event_'):assert e['reset_frames']==[12]
        sources.update({str(f.relative_to(ROOT)):sha256(f) for f in folder.iterdir() if f.is_file()})
    assert len(rows)==2208==len({(r['domain'],r['scene'],r['variant'],r['train_seed'],r['history'],r['input_reset'],r['fusion']) for r in rows})
    groups=collections.defaultdict(list);clips=collections.defaultdict(list)
    for r in rows:
        key=(r['domain'],r['variant'],r['history'],r['input_reset'],r['fusion'])
        groups[key].append(r);clips[(r['scene'],*key)].append(r)
    summary=[{'domain':d,'variant':v,'history':h,'input_reset':reset,'fusion':f,'rows':len(rr),**means(rr)} for (d,v,h,reset,f),rr in sorted(groups.items())]
    save_json(out/'domain_summary.json',summary)
    save_json(out/'by_scene.json',[{'scene':s,'domain':d,'variant':v,'history':h,'input_reset':reset,'fusion':f,**means(rr)} for (s,d,v,h,reset,f),rr in sorted(clips.items())])
    contrasts=[('copy_high','causal','plain_high','causal'),('concat_high','causal','copy_high','causal'),
        ('concat_aligned_high','causal','copy_high','causal'),('concat_aligned_high','causal','concat_high','causal'),
        ('concat_high','causal','concat_high','current'),('concat_aligned_high','causal','concat_aligned_high','current'),
        ('concat_aligned_high','current','concat_high','current')]
    deltas=[]
    for reset in [False,True]:
        rr=[{**r,'method':method(r['variant'],r['history'],r['fusion'])} for r in rows if r['input_reset']==reset]
        for a,ah,b,bh in contrasts:
            for fusion in ['raw','half01']:
                for domain in DOMAINS:deltas.append({**paired(rr,method(a,ah,fusion),method(b,bh,fusion),domain),'input_reset':reset})
    save_json(out/'paired_deltas.json',deltas)
    training=[];alignment=[];real_hashes={}
    for scope in ['raw','aligned']:
        for folder in sorted((BASE/'results'/('exploration-r11-'+scope)).iterdir()):
            config=json.loads((folder/'config.json').read_text());trace=jsonl(folder/'training.jsonl')
            prior=jsonl(BASE/'results/exploration-r5-spatial'/f"plain-seed{config['seed']}"/'training.jsonl')
            assert [r['step'] for r in trace]==list(range(1,1001))
            assert [r['sample_identity'] for r in trace]==[r['sample_identity'] for r in prior]
            checkpoints=list(folder.glob('checkpoint-step*.json'));assert len(checkpoints)==10
            for path in checkpoints:
                r=json.loads(path.read_text());assert sha256(ROOT/r['path'])==r['sha256']
            for r in trace:
                assert len(r['alignment'])==(2 if config['variant']=='concat_aligned_high' else 0)
                for a in r['alignment']:
                    assert 0<=a['valid_fraction']<=1
                    if a['source']=='real':
                        identity=r['sample_identity'][0];key=(identity['scene'],identity['frame_pair'])
                        if key in real_hashes:assert real_hashes[key]==a['history_array_sha256']
                        real_hashes[key]=a['history_array_sha256']
                    alignment.append(a)
            training.append({'variant':config['variant'],'seed':config['seed'],'parameters':config['parameters'],'steps':1000,
                'initial_plain_audit':json.loads((folder/'initial_plain_audit.json').read_text()),
                'last100_task':float(np.mean([r['task'] for r in trace[-100:]])),
                'steps_clipped':sum(r['gradient_norm_all']>1 for r in trace),
                'alignment_wall_seconds':sum(r['alignment_wall_seconds'] for r in trace),'auxiliary_l2':trace[-1]['auxiliary_l2']})
    assert len(training)==6 and len(alignment)==4000
    save_json(out/'training_audit.json',{'runs':training,'updates':6000,'checkpoints_verified':60,'current_GT_stream_matches_plain':True})
    save_json(out/'alignment_audit.json',{'pairs':4000,'unique_real_pairs':len(real_hashes),'repeated_real_history_hashes_match':True,
        'by_source':{source:{'pairs':len(rr),'valid_fraction_mean':float(np.mean([r['valid_fraction'] for r in rr])),
            'raw_pair_mae_mean':float(np.mean([r['raw_pair_mae'] for r in rr])),
            'aligned_pair_mae_mean':float(np.mean([r['aligned_pair_mae'] for r in rr]))}
            for source in ['real','synthetic'] for rr in [[r for r in alignment if r['source']==source]]},
        'scope':'photometric difference is diagnostic, not GT correspondence accuracy; fixed teacher/CPU flow cost included in training wall time'})
    save_json(out/'event_frame12.json',[r for r in frames if r['domain'].startswith('event_') and r['frame']==12])
    save_json(out/'audit.json',{'evaluations':2208,'input_conditions':46,'training_updates':6000,'new_training_runs':6,
        'plain_replay_max_abs':max(v for r in repeats for v in r['max_abs_by_metric'].values()),
        'first_frame_history_reset_max_abs_pixel':max(r['max_abs_pixel'] for r in first),'sources':sources,
        'scope':'native and resized dev share four physical clips; adaptive exploratory strong-optimizer setting; no final-test or novelty claim'})
    lines=['# Strong-optimizer history and correspondence controls','','Final1000, two seeds, all global Adam1e-3/clip1; shared teacher half-flow is not free.','']
    for domain in DOMAINS:
        lines+=['## '+domain,'','| Variant | History | Reset | Fusion | PSNR | SSIM | Temporal | Edge |','|---|---|---|---|---:|---:|---:|---:|']
        for r in summary:
            if r['domain']==domain:lines.append(f"| {r['variant']} | {r['history']} | {r['input_reset']} | {r['fusion']} | {r['psnr_rgb']:.5f} | {r['ssim_rgb']:.5f} | {r['temporal_error']:.5f} | {r['edge_mae']:.5f} |")
        lines.append('')
    (out/'tables.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':main()
