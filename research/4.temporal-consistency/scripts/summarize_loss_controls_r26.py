"""Full paired objective controls, including new-objective spatial-plus-post references."""
import collections
import json

from evaluate_loss_controls_r26 import keys
from exploration_r2_data import ROOT
from explore_temporal_controls import save_json,sha256
from loss_controls_r26 import ARCHITECTURES,OBJECTIVES
from summarize_causal_r5 import KEYS,jsonl,means,paired

BASE=ROOT/'research/4.temporal-consistency'
DOMAINS=['test','dev','dev_native','legacy_synthetic','event_cut','event_light_step']
ID=['domain','scene','architecture','objective','train_seed','step','history','input_reset','fusion']


def label(r):return '|'.join(str(r[k]) for k in ID)
def method(a,o,s,h,f):return '__'.join([a,o,str(s),h,f])


def main():
    out=BASE/'report/exploration-r26';out.mkdir(parents=True,exist_ok=False)
    rows=[];first=[];runtimes=[];sources={};devices=set()
    for architecture in ARCHITECTURES:
        folder=BASE/'results'/('exploration-r26-eval-'+architecture);values=jsonl(folder/'per_sequence.jsonl');run=json.loads((folder/'runtime.json').read_text())
        assert len(values)==run['rows']==2944 and run['metric_computations']==2048 and run['metric_aliases']==896
        devices.add(run['device']);runtimes.append(run)
        events=json.loads((folder/'input_events.json').read_text());assert len(events)==46
        event_map={(r['domain'],r['scene']):bool(r['reset_frames']) for r in events};seen={}
        for r in values:
            assert r['architecture']==architecture
            _,key=keys(architecture,r['history'],r['input_reset'],event_map[(r['domain'],r['scene'])],r['fusion'])
            assert (r['effective_model_reset'],r['effective_output_reset'])==(key[0],key[2])
            if r['metric_alias_of'] is not None:
                ref=seen[r['metric_alias_of']];assert all(r[k]==ref[k] for k in ID if k!='input_reset') and all(r[k]==ref[k] for k in [*KEYS,'flicker'])
            assert label(r) not in seen;seen[label(r)]=r
        seen={}
        for r in jsonl(folder/'per_frame.jsonl'):
            if r['metric_alias_of'] is not None:
                ref=seen[(r['metric_alias_of'],r['frame'])];assert all(r[k]==ref[k] for k in [*KEYS,'flicker','output_mean','target_mean'])
            seen[(label(r),r['frame'])]=r
        markers=json.loads((folder/'checkpoints.json').read_text());assert len(markers)==8
        for r in markers:assert sha256(ROOT/r['path'])==r['sha256']
        rows+=values;first+=json.loads((folder/'first_frame_audit.json').read_text())
        sources.update({str(p.relative_to(ROOT)):sha256(p) for p in folder.iterdir() if p.is_file()})
    assert len(devices)==1 and len(rows)==len({label(r) for r in rows})==8832
    assert sum(r['actual_inference_sequences'] for r in runtimes)==2224
    assert sum(r['metric_computations'] for r in runtimes)==6144 and sum(r['metric_aliases'] for r in runtimes)==2688
    training=BASE/'process/exploration-r26/training-audit.json';ta=json.loads(training.read_text());assert ta['new_updates']==48000 and len(ta['checkpoints'])==48
    sources[str(training.relative_to(ROOT))]=sha256(training)
    groups=collections.defaultdict(list);clips=collections.defaultdict(list)
    for r in rows:
        key=tuple(r[k] for k in ['domain','architecture','objective','step','history','input_reset','fusion'])
        groups[key].append(r);clips[(r['scene'],*key)].append(r)
    summary=[{'domain':d,'architecture':a,'objective':o,'step':s,'history':h,'input_reset':reset,'fusion':f,'rows':len(rr),**means(rr)}
        for (d,a,o,s,h,reset,f),rr in sorted(groups.items())]
    save_json(out/'domain_summary.json',summary)
    save_json(out/'by_scene.json',[{'scene':c,'domain':d,'architecture':a,'objective':o,'step':s,'history':h,'input_reset':reset,'fusion':f,**means(rr)}
        for (c,d,a,o,s,h,reset,f),rr in sorted(clips.items())])
    contrasts=[]
    for a in ARCHITECTURES:
        histories=['causal'] if a=='plain' else ['causal','current'];fusions=['raw','half01','postwarp01','postwarp05'] if a=='plain' else ['raw','half01']
        for h in histories:
            for f in fusions:
                for s in [1000,4000]:contrasts.append(('objective_setup',method(a,'mse',s,h,f),method(a,'charb',s,h,f)))
                for o in OBJECTIVES:contrasts.append(('training_depth',method(a,o,4000,h,f),method(a,o,1000,h,f)))
        if a!='plain':
            for o in OBJECTIVES:
                for s in [1000,4000]:
                    for f in fusions:
                        contrasts += [('actual_history',method(a,o,s,'causal',f),method(a,o,s,'current',f)),
                            ('same_objective_spatial',method(a,o,s,'causal',f),method('plain',o,s,'causal',f)),
                            ('same_objective_strong_post',method(a,o,s,'causal',f),method('plain',o,s,'causal','postwarp05'))]
    for o in OBJECTIVES:
        for s in [1000,4000]:
            for h in ['causal','current']:
                for f in ['raw','half01']:contrasts.append(('trained_order',method('pre_fixedhalf',o,s,h,f),method('post_fixedhalf',o,s,h,f)))
            for a,b in [('postwarp01','half01'),('postwarp05','postwarp01')]:contrasts.append(('classical_post_control',method('plain',o,s,'causal',a),method('plain',o,s,'causal',b)))
    deltas=[]
    for reset in [False,True]:
        rr=[{**r,'method':method(r['architecture'],r['objective'],r['step'],r['history'],r['fusion'])} for r in rows if r['input_reset']==reset]
        for kind,a,b in contrasts:
            for domain in DOMAINS:deltas.append({**paired(rr,a,b,domain),'contrast':kind,'input_reset':reset})
    save_json(out/'paired_deltas.json',deltas)
    save_json(out/'audit.json',{'rows':8832,'input_conditions':46,'checkpoint_weights':24,'new_training_updates':48000,
        'actual_inference_sequences':2224,'metric_computations':6144,'metric_aliases':2688,'sequence_and_frame_alias_exact':True,
        'first_frame_max_abs_pixel':max(r['max_abs_pixel'] for r in first),'common_evaluation_device':next(iter(devices)),
        'old_Charb_training_replay_exact':ta['all_old_Charb_models_and_Adam_exact'],'source_sha256':sources,
        'scope':'Exploratory fixed objective/setup controls, no new structure/novelty or independent-test/mobile timing claim.'})
    lines=['# Paired training-objective controls','','All fixed checkpoints; compare temporal candidates to the same-objective spatial-plus-post references.','',
        '| Domain | Architecture | Objective | Step | History | Reset | Fusion | PSNR | SSIM | Temporal | Edge |',
        '|---|---|---|---:|---|---|---|---:|---:|---:|---:|']
    for r in summary:lines.append(f"|{r['domain']}|{r['architecture']}|{r['objective']}|{r['step']}|{r['history']}|{r['input_reset']}|{r['fusion']}|{r['psnr_rgb']:.6f}|{r['ssim_rgb']:.6f}|{r['temporal_error']:.6f}|{r['edge_mae']:.6f}|")
    (out/'tables.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':main()
