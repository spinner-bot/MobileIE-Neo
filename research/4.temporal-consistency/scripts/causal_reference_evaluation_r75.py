"""Common final-only inputs, published-reference provenance and R75 checkpoint readers."""
import json

from causal_reference_io_r75 import ROOT, BASE, seed_root, digest, lines, verify_training_sources


def key(row):
    return row['seed'],row['scene'],row['label']


def metas():
    paths=[BASE/f'results/{name}/index.json' for name in ['exploration-r30-cache','exploration-r10-cache']]
    values=json.loads(paths[0].read_text())['scenes']
    values += [dict(r,domain='paired_real_ND_DEV',source_video=r['scene'])
               for r in json.loads(paths[1].read_text())['scenes'] if r['domain']=='native_crop']
    assert len(values)==10 and len({r['scene'] for r in values})==10
    return values,{str(p.relative_to(ROOT)):digest(p) for p in paths}


def reference_quality():
    path=BASE/'report/exploration-r70-classical-joint/all_conditions.jsonl'
    assert json.loads((path.parent/'audit.json').read_text())['completed']
    rows=[r for r in lines(path) if r['label'] in ('MobileIE_raw','MobileIE_recursive_self05')]
    assert len(rows)==40 and len({key(r) for r in rows})==40 and all('JOD' in r for r in rows)
    _,sources=metas()
    for seed in (0,1):
        old=BASE/f'results/exploration-r44-stage001-eval-seed{seed}/config.json'
        assert json.loads(old.read_text())['cache_indices']==sources
    return rows,{str(path.relative_to(ROOT)):digest(path)}


def reference_predictions():
    path=BASE/'process/exploration-r73/references001/prediction_manifest.jsonl'
    assert json.loads((path.parent/'runtime.json').read_text())['completed_normally']
    rows=[dict(r,label='MobileIE_raw' if r['method']=='raw' else 'MobileIE_recursive_self05')
          for r in lines(path) if r['model']=='R44_plain']
    assert len(rows)==40 and len({key(r) for r in rows})==40
    return rows


def final_markers():
    verify_training_sources()
    markers={}
    for seed in (0,1):
        root=seed_root(seed)
        marker=json.loads((root/'final.json').read_text())
        runtime=json.loads((ROOT/marker['runtime']).read_text())
        assert marker['completed_normally'] and marker['formal_updates']==marker['completed_steps']==64000
        assert runtime['completed_normally'] and runtime['completed_steps']==64000 and runtime['stop_reason']=='complete'
        cumulative=0
        for path in sorted(root.glob('attempt*/runtime.json')):
            old=json.loads(path.read_text())
            assert old['seed']==seed and old['inherited_steps']==cumulative
            assert old['new_steps']==old['completed_steps']-cumulative
            assert old['stop_reason'] in ('resource','complete')
            cumulative=old['completed_steps']
        assert cumulative==64000
        path=ROOT/marker['path']
        assert path.resolve().is_relative_to('/data/Users/zrh/data/MobileIE-Neo') and digest(path)==marker['sha256']
        markers[seed]=marker
    return markers


def load_final(seed,markers):
    import torch
    from rebotnet_reference import new_model
    saved=torch.load(ROOT/markers[seed]['path'],map_location='cpu',weights_only=True)
    assert saved['experiment']=='R75' and saved['seed']==seed
    assert saved['completed_steps']==saved['next_data_step']==saved['registered_steps']==64000
    model=new_model(seed).eval().requires_grad_(False)
    model.load_state_dict(saved['state_dict'],strict=True)
    assert sum(p.numel() for p in model.parameters())==890275
    assert all(torch.isfinite(p).all() for p in model.parameters())
    return model
