"""Audit matched task supervision, exact spatial replay, and context-only changes."""
import argparse
import collections
import json
from pathlib import Path

import numpy as np

from exploration_r2_data import ROOT
from explore_temporal_controls import save_json, sha256
from summarize_causal_r5 import means, method, paired

BASE=ROOT/'research/4.temporal-consistency'
DOMAINS=['test','dev','dev_native','legacy_synthetic','event_cut','event_light_step']


def jsonl(path):return [json.loads(s) for s in path.read_text().splitlines()]


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',required=True)
    out=Path(p.parse_args().out);out.mkdir(parents=True,exist_ok=False)
    rows=[];frames=[];sources={};first=[]
    for scope in ['raw','aligned']:
        folder=BASE/'results'/('exploration-r13-eval-'+scope)
        values=jsonl(folder/'per_sequence.jsonl');assert len(values)==736==json.loads((folder/'runtime.json').read_text())['rows']
        rows+=values;frames+=jsonl(folder/'per_frame.jsonl');first+=json.loads((folder/'first_frame_audit.json').read_text())
        for r in json.loads((folder/'checkpoints.json').read_text()):assert sha256(ROOT/r['path'])==r['sha256']
        sources.update({str(f.relative_to(ROOT)):sha256(f) for f in folder.iterdir() if f.is_file()})
    assert len(rows)==1472==len({(r['domain'],r['scene'],r['variant'],r['train_seed'],r['history'],r['input_reset'],r['fusion']) for r in rows})
    for scope in ['raw','aligned']:
        path=BASE/'results'/('exploration-r11-eval-'+scope)/'per_sequence.jsonl'
        rows+=jsonl(path);sources[str(path.relative_to(ROOT))]=sha256(path)
    assert len(rows)==3680
    grouped=collections.defaultdict(list);clips=collections.defaultdict(list)
    for r in rows:
        key=(r['domain'],r['variant'],r['history'],r['input_reset'],r['fusion'])
        grouped[key].append(r);clips[(r['scene'],*key)].append(r)
    summary=[{'domain':d,'variant':v,'history':h,'input_reset':reset,'fusion':f,'rows':len(rr),**means(rr)} for (d,v,h,reset,f),rr in sorted(grouped.items())]
    save_json(out/'domain_summary.json',summary)
    save_json(out/'by_scene.json',[{'scene':s,'domain':d,'variant':v,'history':h,'input_reset':reset,'fusion':f,**means(rr)} for (s,d,v,h,reset,f),rr in sorted(clips.items())])
    contrasts=[('concat_context','causal','concat_high','causal'),('concat_aligned_context','causal','concat_aligned_high','causal'),
        ('concat_context','current','concat_high','current'),('concat_aligned_context','current','concat_aligned_high','current'),
        ('concat_context','causal','concat_context','current'),('concat_aligned_context','causal','concat_aligned_context','current'),
        ('concat_context','causal','copy_high','causal'),('concat_aligned_context','causal','copy_high','causal')]
    deltas=[]
    for reset in [False,True]:
        rr=[{**r,'method':method(r['variant'],r['history'],r['fusion'])} for r in rows if r['input_reset']==reset]
        for a,ah,b,bh in contrasts:
            for f in ['raw','half01']:
                for d in DOMAINS:deltas.append({**paired(rr,method(a,ah,f),method(b,bh,f),d),'input_reset':reset})
    save_json(out/'paired_deltas.json',deltas)
    training=[];self_counts={};alignment_count=0
    for scope in ['raw','aligned']:
        for folder in sorted((BASE/'results'/('exploration-r13-'+scope)).iterdir()):
            cfg=json.loads((folder/'config.json').read_text());trace=jsonl(folder/'training.jsonl')
            prior=jsonl(BASE/'results/exploration-r5-spatial'/f"plain-seed{cfg['seed']}"/'training.jsonl')
            assert [r['step'] for r in trace]==list(range(1,1001))
            assert [r['sample_identity'] for r in trace]==[r['sample_identity'] for r in prior]
            count=0
            for r in trace:
                real,synth=r['first_context'];identity=r['sample_identity'][0];t=identity['frame_pair']
                assert real['frame_index']==max(0,t-2) and real['task_first_index']==t-1
                assert real['self_fallback']==(t==1) and not synth['self_fallback']
                assert synth['offset_from_first']==[-x for x in r['sample_identity'][1]['offset']]
                assert synth['noise_rng_seed']==1401300000+cfg['seed']*1000000+r['step']-1
                count+=int(real['self_fallback']);alignment_count+=len(r['alignment'])
                assert len(r['alignment'])==(4 if scope=='aligned' else 0)
            self_counts[folder.name]={'self_history_tasks':count,'all_supervised_tasks':4000,'fraction':count/4000}
            checkpoints=list(folder.glob('checkpoint-step*.json'));assert len(checkpoints)==10
            for path in checkpoints:
                item=json.loads(path.read_text());assert sha256(ROOT/item['path'])==item['sha256']
            training.append({'variant':cfg['variant'],'seed':cfg['seed'],'steps':1000,'parameters':cfg['parameters'],
                'initial_plain_audit':json.loads((folder/'initial_plain_audit.json').read_text()),
                'last100_task':float(np.mean([r['task'] for r in trace[-100:]])),
                'steps_clipped':sum(r['gradient_norm_all']>1 for r in trace)})
    assert len(training)==5 and alignment_count==8000
    replay_path=BASE/'results/exploration-r13-raw/copy_context_replay-seed0/spatial_replay_audit.json'
    replay=json.loads(replay_path.read_text());assert max(replay['max_abs_trace_delta'],replay['max_abs_model_state_delta'])<=1e-7
    save_json(out/'training_audit.json',{'runs':training,'updates':5000,'temporal_updates':4000,'checkpoints_verified':50,
        'current_GT_identity_matches_R5_R11':True,'self_history_tasks':self_counts,'spatial_replay':replay,'aligned_pairs':alignment_count})
    save_json(out/'event_frame12.json',[r for r in frames if r['domain'].startswith('event_') and r['frame']==12])
    save_json(out/'audit.json',{'new_evaluations':1472,'reused_R11_evaluations':2208,'input_conditions':46,
        'new_updates':5000,'new_temporal_updates':4000,'first_frame_history_reset_max_abs_pixel':max(r['max_abs_pixel'] for r in first),
        'spatial_replay_max_abs':max(replay['max_abs_trace_delta'],replay['max_abs_model_state_delta']),'sources':sources,
        'scope':'same task stream and inference as R11, changed first-task training history; ND exploratory data; no final test or novelty claim'})
    lines=['# Earlier training context controls','','Current task supervision/optimizer/inference unchanged; spatial full1000 replay passed; fixed final1000/two seeds.','']
    for d in DOMAINS:
        lines+=['## '+d,'','| Variant | History | Reset | Fusion | PSNR | SSIM | Temporal | Edge |','|---|---|---|---|---:|---:|---:|---:|']
        for r in summary:
            if r['domain']==d:lines.append(f"| {r['variant']} | {r['history']} | {r['input_reset']} | {r['fusion']} | {r['psnr_rgb']:.5f} | {r['ssim_rgb']:.5f} | {r['temporal_error']:.5f} | {r['edge_mae']:.5f} |")
        lines.append('')
    (out/'tables.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':main()
