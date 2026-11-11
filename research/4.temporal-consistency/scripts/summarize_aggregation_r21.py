"""Separate history contribution, mapping adaptation, initialization, and strong post controls."""
import collections
import json

from aggregation_model_r21 import MODES
from evaluation_aliases_r16 import cache_keys
from exploration_r2_data import ROOT
from explore_temporal_controls import save_json,sha256
from summarize_causal_r5 import KEYS,jsonl,means,paired

BASE=ROOT/'research/4.temporal-consistency'
DOMAINS=['test','dev','dev_native','legacy_synthetic','event_cut','event_light_step']
IDENTITY=['domain','scene','variant','train_seed','step','history','input_reset','fusion']


def label(r):return '|'.join(str(r[k]) for k in IDENTITY)
def method(v,step,history,fusion):return '__'.join([v,str(step),history,fusion])


def main():
    out=BASE/'report/exploration-r21';out.mkdir(parents=True,exist_ok=False)
    values=[];first=[];runtimes=[];sources={}
    for part in ['pre-zero','half-post']:
        folder=BASE/'results'/('exploration-r21-eval-'+part);rows=jsonl(folder/'per_sequence.jsonl')
        runtime=json.loads((folder/'runtime.json').read_text());assert runtime['device']=='cpu' and not runtime['cuda_initialized'];assert len(rows)==runtime['rows']==2944
        events=json.loads((folder/'input_events.json').read_text());assert len(events)==46
        event_map={(r['domain'],r['scene']):bool(r['reset_frames']) for r in events};seen={}
        for r in rows:
            _,key=cache_keys('aligned_history',r['history'],r['input_reset'],event_map[(r['domain'],r['scene'])],r['fusion'])
            assert (r['effective_model_reset'],r['effective_output_reset'])==(key[0],key[2])
            if r['metric_alias_of'] is not None:
                ref=seen[r['metric_alias_of']];assert all(r[k]==ref[k] for k in IDENTITY if k!='input_reset')
                assert all(r[k]==ref[k] for k in KEYS)
            assert label(r) not in seen;seen[label(r)]=r
        assert sum(r['metric_alias_of'] is None for r in rows)==runtime['metric_computations']
        seen={}
        for r in jsonl(folder/'per_frame.jsonl'):
            if r['metric_alias_of'] is not None:assert all(r[k]==seen[(r['metric_alias_of'],r['frame'])][k] for k in KEYS)
            seen[(label(r),r['frame'])]=r
        for r in json.loads((folder/'checkpoints.json').read_text()):assert sha256(ROOT/r['path'])==r['sha256']
        values+=rows;runtimes.append(runtime);first+=json.loads((folder/'first_frame_audit.json').read_text())
        sources.update({str(p.relative_to(ROOT)):sha256(p) for p in folder.iterdir() if p.is_file()})
    assert len(values)==len({label(r) for r in values})==5888
    assert sum(r['actual_inference_sequences'] for r in runtimes)==1856
    assert sum(r['metric_computations'] for r in runtimes)==4096
    assert sum(r['metric_aliases'] for r in runtimes)==1792
    training_path=BASE/'process/exploration-r21/training-audit.json';training=json.loads(training_path.read_text())
    assert training['new_updates']==32000 and len(training['checkpoints'])==32
    sources[str(training_path.relative_to(ROOT))]=sha256(training_path)
    refs=[]
    path=BASE/'results/exploration-r21-reference-plain-cpu/per_sequence.jsonl'
    for r in jsonl(path):
        if r['train_preparation']=='native_crop' and r['variant']=='plain':refs.append({**r,'variant':'R14_plain'})
    assert len(refs)==736;sources[str(path.relative_to(ROOT))]=sha256(path)
    for seed in [0,1]:
        path=BASE/'results'/f'exploration-r21-reference-post-seed{seed}-cpu'/'per_sequence.jsonl';sources[str(path.relative_to(ROOT))]=sha256(path)
        for r in jsonl(path):refs.append({**r,'variant':'R20_'+r['method'],'step':4000,'history':'causal','fusion':'raw'})
    assert len(refs)==2208
    combined=values+refs;groups=collections.defaultdict(list);clips=collections.defaultdict(list)
    for r in combined:
        key=(r['domain'],r['variant'],r['step'],r['history'],r['input_reset'],r['fusion'])
        groups[key].append(r);clips[(r['scene'],*key)].append(r)
    summary=[{'domain':d,'variant':v,'step':s,'history':h,'input_reset':reset,'fusion':f,'rows':len(rr),**means(rr)}
        for (d,v,s,h,reset,f),rr in sorted(groups.items())]
    save_json(out/'domain_summary.json',summary)
    save_json(out/'by_scene.json',[{'scene':c,'domain':d,'variant':v,'step':s,'history':h,'input_reset':reset,'fusion':f,**means(rr)}
        for (c,d,v,s,h,reset,f),rr in sorted(clips.items())])
    contrasts=[]
    for step in [1000,4000]:
        for v in MODES:
            contrasts += [('actual_history',(v,step,'causal'),(v,step,'current')),
                ('mapping_adaptation',(v,step,'current'),('R14_plain',step,'causal')),
                ('total_vs_single',(v,step,'causal'),('R14_plain',step,'causal'))]
        contrasts += [('initialization',('pre_learnzero',step,'causal'),('pre_learnhalf',step,'causal')),
            ('trained_order',('pre_fixedhalf',step,'causal'),('post_fixedhalf',step,'causal'))]
    for v in MODES:
        for history in ['causal','current']:contrasts.append(('training_depth',(v,4000,history),(v,1000,history)))
    deltas=[]
    for reset in [False,True]:
        rows=[{**r,'method':method(r['variant'],r['step'],r['history'],r['fusion'])} for r in combined if r['input_reset']==reset]
        for kind,a,b in contrasts:
            for fusion in ['raw','half01']:
                for domain in DOMAINS:deltas.append({**paired(rows,method(*a,fusion),method(*b,fusion),domain),'contrast':kind,'input_reset':reset})
        for v in MODES:
            for post in ['postwarp01','postwarp05','postaligned01','postaligned05','recursive_half01']:
                for domain in DOMAINS:deltas.append({**paired(rows,method(v,4000,'causal','raw'),method('R20_'+post,4000,'causal','raw'),domain),
                    'contrast':'strong_post_control','input_reset':reset})
    save_json(out/'paired_deltas.json',deltas)
    save_json(out/'audit.json',{'new_rows':5888,'new_CPU_reference_rows':len(refs),'actual_inference_sequences':1856,
        'metric_computations':4096,'metric_aliases':1792,'sequence_and_frame_alias_exact':True,
        'first_frame_max_abs_pixel':max(r['max_abs_pixel'] for r in first),'source_sha256':sources,
        'evaluation_device':'cpu','scope':'Repeated exploratory conditions, two seeds; no novelty/independent test/mobile latency claim. R20 post clamps branches before aggregation; R21 post averages raw branches.'})
    lines=['# Aggregation-aware training controls','','All fixed checkpoints and controls retained; current replacement separates mapping adaptation from actual history.','',
        '| Domain | Variant | Step | History | Reset | Fusion | PSNR | SSIM | Temporal | Edge |',
        '|---|---|---:|---|---|---|---:|---:|---:|---:|']
    for r in summary:lines.append(f"| {r['domain']} | {r['variant']} | {r['step']} | {r['history']} | {r['input_reset']} | {r['fusion']} | {r['psnr_rgb']:.5f} | {r['ssim_rgb']:.5f} | {r['temporal_error']:.5f} | {r['edge_mae']:.5f} |")
    (out/'tables.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':main()
