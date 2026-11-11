"""Matched-field diagnostics with explicit content/context confounds."""
import argparse
import collections
import json
from pathlib import Path

from exploration_r2_data import ROOT
from explore_temporal_controls import save_json, sha256
from prepare_cache_r5 import load_one
from summarize_causal_r5 import means, paired

BASE=ROOT/'research/4.temporal-consistency'


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',required=True)
    out=Path(p.parse_args().out);out.mkdir(parents=True,exist_ok=False)
    source=BASE/'results/exploration-r10-eval';cache=BASE/'results/exploration-r10-cache'
    rows=[json.loads(s) for s in (source/'per_sequence.jsonl').read_text().splitlines()]
    assert len(rows)==264==json.loads((source/'runtime.json').read_text())['rows']
    assert len({(r['scene'],r['preparation'],r['variant'],r['train_seed'],r['fusion']) for r in rows})==264
    for r in json.loads((source/'checkpoints.json').read_text()):assert sha256(ROOT/r['path'])==r['sha256']
    for r in json.loads((source/'preview_index.json').read_text()):assert sha256(ROOT/r['path'])==r['sha256']
    groups=collections.defaultdict(list);scenes=collections.defaultdict(list)
    for r in rows:
        groups[(r['preparation'],r['variant'],r['fusion'])].append(r)
        scenes[(r['preparation'],r['scene'],r['variant'],r['fusion'])].append(r)
    summary=[{'preparation':p,'variant':v,'fusion':f,'rows':len(rr),**means(rr)} for (p,v,f),rr in sorted(groups.items())]
    save_json(out/'domain_summary.json',summary)
    save_json(out/'by_scene.json',[{'preparation':p,'scene':s,'variant':v,'fusion':f,**means(rr)} for (p,s,v,f),rr in sorted(scenes.items())])
    deltas=[]
    variants=sorted({r['variant'] for r in rows})
    for variant in variants:
        for fusion in ['raw','half01']:
            rr=[{**r,'domain':'dev','method':r['preparation']} for r in rows if r['variant']==variant and r['fusion']==fusion]
            deltas.append({**paired(rr,'smooth_crop','native_crop','dev'),'variant':variant,'fusion':fusion,
                           'scope':'same GT/FOV/grid; input smoothing alters texture and noise, not pure noise intervention'})
    save_json(out/'matched_smoothing_deltas.json',deltas)
    index=json.loads((cache/'index.json').read_text());checks=[]
    for scene in sorted({r['scene'] for r in index['scenes']}):
        a,b=[load_one(next(r for r in index['scenes'] if r['scene']==scene and r['domain']==v)) for v in ['native_crop','smooth_crop']]
        import numpy as np
        for key in ['gt','reference_flow','reference_mask']:
            if not np.array_equal(a[key],b[key]):raise AssertionError('matched reference changed')
        checks.append({'scene':scene,'GT_and_reference_exactly_equal':True})
    replay=json.loads((source/'teacher_replay.json').read_text())
    save_json(out/'audit.json',{'evaluations':264,'clips':4,'preparations':3,'weights':11,'new_training_updates':0,
        'matched_reference_checks':checks,'teacher_max_abs_pixel_repeat':max(r['max_abs_pixel'] for r in replay),
        'sources':{str(f.relative_to(ROOT)):sha256(f) for folder in [source,cache] for f in folder.iterdir() if f.is_file()},
        'scope':'same-field smoothing changes detail and noise; full-area changes content/context; ND dev, not final test'})
    lines=['# Native-scale and matched-field input smoothing','','Two-seed means; all frozen candidates, all four project-dev clips.','',
           'Full-area and crop have different field of view/content/context; do not interpret their metric gap as pure noise.','']
    for prep in ['native_crop','smooth_crop','full_area']:
        lines+=['## '+prep,'','| Variant | Fusion | PSNR | SSIM | Temporal | Edge |','|---|---|---:|---:|---:|---:|']
        for r in summary:
            if r['preparation']==prep:lines.append(f"| {r['variant']} | {r['fusion']} | {r['psnr_rgb']:.5f} | {r['ssim_rgb']:.5f} | {r['temporal_error']:.5f} | {r['edge_mae']:.5f} |")
        lines.append('')
    (out/'tables.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':main()
