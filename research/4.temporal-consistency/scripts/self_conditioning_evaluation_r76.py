"""Strict primary-result prerequisites and complete self4 output metadata."""
import json

from causal_reference_io_r75 import ROOT,BASE,lines,digest
from causal_reference_evaluation_r75 import final_markers,metas,key


def require_primary_pipeline():
    markers=final_markers()
    joint=BASE/'report/exploration-r75-joint'
    audit=json.loads((joint/'audit.json').read_text())
    rows=json.loads((joint/'all_conditions.json').read_text())
    gates=json.loads((joint/'gates.json').read_text())
    assert audit['completed'] and audit['all_quality_rows']==80 and audit['full_gates_evaluated']
    assert len(rows)==len({key(r) for r in rows})==80 and len(gates)==8
    assert all(r['JOD'] is not None for r in rows)
    visual=BASE/'report/exploration-r75-contacts/runtime.json'
    assert json.loads(visual.read_text())['completed_normally']
    assert json.loads(visual.read_text())['contacts']==20
    return markers,rows,gates


def self4_quality():
    _,indices=metas()
    rows,entries,sources=[],[],{}
    for seed in (0,1):
        run=BASE/f'results/exploration-r76-eval-seed{seed}'
        terminal=json.loads((run/'runtime.json').read_text())
        assert terminal['completed_normally'] and terminal['quality_rows']==10 and terminal['frame_rows']==448
        assert terminal['tile_CNN_calls']==9728 and terminal['training_updates']==0
        config=json.loads((run/'config.json').read_text())
        assert config['cache_indices']==indices and config['passes']==4
        part=lines(run/'per_sequence.jsonl');predictions=lines(run/'prediction_manifest.jsonl')
        assert len(part)==len(predictions)==10 and {key(r) for r in part}=={key(r) for r in predictions}
        assert all(r['seed']==seed and r['label']=='ReBotNet_self4' and r['passes']==4
                   and r['distinct_observations_per_output']==1 for r in part)
        rows+=part;entries+=predictions
        for name in ('runtime.json','config.json','per_sequence.jsonl','per_frame.jsonl','prediction_manifest.jsonl'):
            path=run/name;sources[str(path.relative_to(ROOT))]=digest(path)
    assert len(rows)==len({key(r) for r in rows})==20
    return rows,entries,sources
