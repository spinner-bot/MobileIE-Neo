"""Published denoiser transfer, actual temporal contribution, and stage-cost scope."""
import argparse
import collections
import json
from pathlib import Path

import numpy as np

from exploration_r2_data import ROOT
from explore_temporal_controls import save_json, sha256
from summarize_causal_r5 import means, paired

BASE=ROOT/'research/4.temporal-consistency'
DOMAINS=['test','dev','dev_native','legacy_synthetic','event_cut','event_light_step']


def jsonl(path):return [json.loads(s) for s in path.read_text().splitlines()]


def method(mode,sigma):return mode if sigma is None else mode+'_sigma'+str(sigma)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',required=True)
    out=Path(p.parse_args().out);out.mkdir(parents=True,exist_ok=False)
    rows=[];frames=[];cost=[];repeats=[];sources={};context=None
    for seed in [0,1]:
        folder=BASE/'results'/f'exploration-r12-eval-seed{seed}'
        values=jsonl(folder/'per_sequence.jsonl');assert len(values)==460==json.loads((folder/'runtime.json').read_text())['rows']
        rows+=values;frames+=jsonl(folder/'per_frame.jsonl');cost+=jsonl(folder/'stage_cost.jsonl')
        repeats+=json.loads((folder/'base_replay.json').read_text());cfg=json.loads((folder/'config.json').read_text())
        assert sha256(ROOT/cfg['checkpoint'])==cfg['checkpoint_sha256']
        contexts=json.loads((folder/'context_index.json').read_text());assert len(contexts)==138
        if context is not None:assert context==contexts
        context=contexts
        for row in context:
            assert row['future_wait_frames_max']<=2
            if row['mode']=='five_current':assert row['future_wait_frames_max']==0
            if row['domain'].startswith('event_'):assert row['detected_events']==[12]
            if row['mode']=='window_segmented':
                for t,indices in enumerate(row['indices']):
                    for event in row['detected_events']:
                        assert all((j<event)==(t<event) for j in indices)
        sources.update({str(f.relative_to(ROOT)):sha256(f) for f in folder.iterdir() if f.is_file()})
    assert len(rows)==920==len({(r['domain'],r['scene'],r['train_seed'],r['mode'],r['sigma']) for r in rows})
    groups=collections.defaultdict(list);clips=collections.defaultdict(list)
    for r in rows:
        groups[(r['domain'],r['mode'],r['sigma'])].append(r)
        clips[(r['domain'],r['scene'],r['mode'],r['sigma'])].append(r)
    summary=[{'domain':d,'mode':m,'sigma':s,'rows':len(rr),**means(rr)} for (d,m,s),rr in sorted(groups.items())]
    save_json(out/'domain_summary.json',summary)
    save_json(out/'by_scene.json',[{'domain':d,'scene':c,'mode':m,'sigma':s,**means(rr)} for (d,c,m,s),rr in sorted(clips.items())])
    rr=[{**r,'method':method(r['mode'],r['sigma'])} for r in rows];contrasts=[]
    for sigma in [5,15,30]:
        pairs=[(method(m,sigma),'raw') for m in ['window','window_segmented','five_current']]
        pairs += [(method('window',sigma),method('five_current',sigma)),(method('window_segmented',sigma),method('five_current',sigma)),
                  (method('window_segmented',sigma),method('window',sigma))]
        for a,b in pairs:
            for domain in DOMAINS:contrasts.append(paired(rr,a,b,domain))
    save_json(out/'paired_deltas.json',contrasts)
    save_json(out/'event_frames.json',[r for r in frames if r['domain'].startswith('event_') and 10<=r['frame']<=14])
    groups=collections.defaultdict(list)
    for r in cost:groups[(tuple(r['shape'][2:]),r['stage'],r['mode'],r['sigma'])].append(r['ms_per_frame'])
    save_json(out/'stage_cost_summary.json',{'rows':[{'HW':list(hw),'stage':stage,'mode':m,'sigma':s,'sequences':len(values),
        'median_ms_per_frame':float(np.median(values)),'p10_p90_ms':np.percentile(values,[10,90]).tolist()}
        for (hw,stage,m,s),values in sorted(groups.items())],
        'scope':'synchronized GPU-stage wall time includes Python/allocation, excludes loading/transfers/metrics/CPU pipeline; shared A100, no phone or strict benchmark claim'})
    save_json(out/'audit.json',{'evaluations':920,'input_conditions':46,'base_replays':len(repeats),
        'base_replay_max_abs':max(v for r in repeats for v in r['max_abs_by_metric'].values()),
        'context_indices_match_between_seeds':True,'max_future_wait_frames':2,'five_current_future_frames':0,
        'denoiser_parameters':2479096,'training_updates':0,'sources':sources,
        'scope':'clipped-AWGN pretrained transfer; native/resized dev share source; no final test or original benchmark reproduction'})
    lines=['# Published five-frame denoiser transfer','','Fixed R9full1e-3 base; all sigma/call modes retained. Window needs two future frames; five_current repeats current five times.','']
    for d in DOMAINS:
        lines+=['## '+d,'','| Mode | Sigma | PSNR | SSIM | Temporal | Edge |','|---|---|---:|---:|---:|---:|']
        for r in summary:
            if r['domain']==d:lines.append(f"| {r['mode']} | {r['sigma']} | {r['psnr_rgb']:.5f} | {r['ssim_rgb']:.5f} | {r['temporal_error']:.5f} | {r['edge_mae']:.5f} |")
        lines.append('')
    (out/'tables.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':main()
