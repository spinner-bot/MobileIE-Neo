"""Summarize both completed R34 seeds and separate mapping from history gains."""
from collections import defaultdict
import json
from statistics import mean

from dynamic_training_r34 import BASE, ROOT, digest

METRICS = ['psnr_rgb', 'ssim_rgb', 'temporal_error', 'edge_mae']
FIELDS = ['model', 'history', 'fusion']


def main():
    rows, sources = [], {}
    for seed in [0, 1]:
        folder = BASE/f'results/exploration-r34-eval-seed{seed}'
        runtime = json.loads((folder/'runtime.json').read_text())
        assert runtime['completed_normally'] and runtime['rows'] == 42 and runtime['neural_sequences'] == 24
        assert not runtime['cuda_initialized']
        values = [json.loads(line) for line in (folder/'per_sequence.jsonl').read_text().splitlines()]
        assert len(values) == 42 and {r['train_seed'] for r in values} == {seed}
        assert max(r['first_frame_max_abs_0to255'] for r in json.loads((folder/'first_frame.json').read_text())) <= 1e-4
        rows.extend(values)
        for name in ['runtime.json', 'config.json', 'per_sequence.jsonl', 'first_frame.json']:
            path = folder/name
            sources[str(path.relative_to(ROOT))] = digest(path)
    key = lambda row: (row['train_seed'], row['scene'], *(row[f] for f in FIELDS))
    lookup = {key(row): row for row in rows}
    assert len(rows) == len(lookup) == 84
    groups = defaultdict(list)
    for row in rows:
        assert not row['input_reset']
        groups[(row['train_seed'], *(row[f] for f in FIELDS))].append(row)
    seeds, domains = [], []
    for group, values in sorted(groups.items()):
        assert len(values) == 6 and len({r['source_video'] for r in values}) == 3
        seeds.append({'seed': group[0], **dict(zip(FIELDS, group[1:])),
                      **{m: mean(r[m] for r in values) for m in METRICS}})
    for group in sorted({tuple(r[f] for f in FIELDS) for r in seeds}):
        values = [r for r in seeds if tuple(r[f] for f in FIELDS) == group]
        assert len(values) == 2 and {r['seed'] for r in values} == {0, 1}
        domains.append({**dict(zip(FIELDS, group)), **{m: mean(r[m] for r in values) for m in METRICS}})
    assert len(domains) == 7
    paired = []
    for row in rows:
        if (row['model'], row['history'], row['fusion']) != ('causal', 'causal', 'raw'):
            continue
        for model in ['original', 'spatial', 'causal']:
            for fusion in ['raw', 'postwarp05']:
                other = lookup[key({**row, 'model': model, 'history': 'current', 'fusion': fusion})]
                paired.append({'seed': row['train_seed'], 'scene': row['scene'], 'source_video': row['source_video'],
                               'reference_model': model, 'reference_fusion': fusion,
                               **{'delta_'+m: row[m]-other[m] for m in METRICS},
                               'candidate_temporal': row['temporal_error'], 'reference_temporal': other['temporal_error']})
    paired_groups = defaultdict(list)
    for row in paired:
        paired_groups[(row['seed'], row['reference_model'], row['reference_fusion'])].append(row)
    paired_seeds = []
    for (seed, model, fusion), values in sorted(paired_groups.items()):
        assert len(values) == 6
        delta = {m: mean(r['delta_'+m] for r in values) for m in METRICS}
        ratio = mean(r['candidate_temporal'] for r in values)/mean(r['reference_temporal'] for r in values)
        paired_seeds.append({'seed': seed, 'reference_model': model, 'reference_fusion': fusion,
                             **{'delta_'+m: delta[m] for m in METRICS}, 'temporal_ratio_of_condition_means': ratio,
                             'descriptive_quality_temporal_gate': delta['psnr_rgb'] >= -.05 and ratio <= .9,
                             'registered_primary_gate': (delta['psnr_rgb'] >= -.05 and ratio <= .9)
                                                        if model == 'spatial' and fusion == 'postwarp05' else None})
    primary = [r for r in paired_seeds if r['registered_primary_gate'] is not None]
    assert len(primary) == 2
    out = BASE/'report/exploration-r34'
    out.mkdir(parents=True, exist_ok=False)
    audit = {'passed': True, 'rows': 84, 'source_videos': 3, 'noise_conditions_per_video': 2, 'training_seeds': 2,
             'source_sha256': sources, 'registered_spatial_post_gate_passed_both_seeds': all(r['registered_primary_gate'] for r in primary),
             'scope': 'Common CPU DEV, final4000 weights; passing a threshold is not proof of novelty, phone efficiency or independent test generalization.'}
    for filename, value in [('domain_seed_summary.json', seeds), ('domain_summary.json', domains),
                            ('paired_by_condition.json', paired), ('paired_domain_seed.json', paired_seeds), ('audit.json', audit)]:
        (out/filename).write_text(json.dumps(value, indent=2)+'\n')
    lines = ['# R34 final-weight external development comparison', '',
             'Both training seeds, equal means of six conditions from three source videos. Common CPU, no event reset, no independent final-test or mobile-latency claim.', '',
             '|Model|History|Output|PSNR|SSIM|Temporal↓|Edge↓|', '|---|---|---|---:|---:|---:|---:|']
    for row in domains:
        lines.append('|'+ '|'.join(row[f] for f in FIELDS)+'|'+'|'.join(f'{row[m]:.6f}' for m in METRICS)+'|')
    lines += ['', 'Causal/current includes the trained low-resolution spatial path; it is not the bare modified backbone. Actual history benefits must survive this same-weight control and the independently trained equal-capacity spatial control. Post .5 uses the common cached motion for quality; a deployed post can estimate flow from its own enhancement.', '',
              '|Reference|Seed|ΔPSNR|Temporal ratio|Registered spatial-post gate|', '|---|---:|---:|---:|---|']
    for row in paired_seeds:
        lines.append(f"|{row['reference_model']}/{row['reference_fusion']}|{row['seed']}|{row['delta_psnr_rgb']:.6f}|{row['temporal_ratio_of_condition_means']:.6f}|{row['registered_primary_gate']}|")
    lines += ['', 'A gate against other references is descriptive only; the registered primary comparator is independent spatial + post .5. Report per-seed results and condition pairs rather than selecting the best seed.']
    (out/'tables.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'rows': 84, 'primary_gate_passed_both_seeds': audit['registered_spatial_post_gate_passed_both_seeds']}))


if __name__ == '__main__':
    main()
