"""Summarize explicitly completed R30 phases; never infer a missing seed's results."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
from statistics import mean

from explore_temporal_controls import ROOT, save_json, sha256

BASE = ROOT/'research/4.temporal-consistency'
METRICS = ['psnr_rgb', 'ssim_rgb', 'temporal_error', 'edge_mae']
FIELDS = ['scene', 'mode', 'objective', 'train_seed', 'history', 'input_reset', 'fusion']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phases', required=True, help='references,post-seed0,post-seed1; all included phases must have terminal runtime')
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    phases = args.phases.split(',')
    assert phases and phases[0] == 'references' and len(set(phases)) == len(phases)
    assert set(phases) <= {'references', 'post-seed0', 'post-seed1'}
    sources, rows, runs = {}, [], []
    for phase in phases:
        folder = BASE/'results'/('exploration-r30-eval-'+phase)
        runtime = json.loads((folder/'runtime.json').read_text())
        assert runtime['completed_normally'] and runtime['device'] == 'cpu' and not runtime['cuda_initialized']
        assert runtime['rows'] == (168 if phase == 'references' else 72)
        values = [json.loads(line) for line in (folder/'per_sequence.jsonl').read_text().splitlines()]
        assert len(values) == runtime['rows']
        assert max(r['max_abs_first_frame'] for r in json.loads((folder/'first_frame_audit.json').read_text())) <= 1e-4
        rows.extend(values)
        runs.append({'phase': phase, **runtime})
        for name in ['per_sequence.jsonl', 'runtime.json', 'checkpoints.json', 'config.json']:
            path = folder/name;sources[str(path.relative_to(ROOT))] = sha256(path)
    key = lambda r: tuple(r[f] for f in FIELDS)
    lookup = {key(r): r for r in rows}
    assert len(lookup) == len(rows)
    fields = ['mode', 'objective', 'train_seed', 'history', 'input_reset', 'fusion']
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[f] for f in fields)].append(row)
    summaries = []
    for group, values in sorted(groups.items()):
        assert len(values) == 6 and len({r['source_video'] for r in values}) == 3
        summaries.append({**dict(zip(fields, group)), 'input_conditions': 6, 'source_videos': 3,
                          **{m: mean(r[m] for r in values) for m in METRICS}})
    pairs = []
    for row in rows:
        if row['mode'] != 'post_transport' or row['fusion'] != 'raw':
            continue
        for label, mode, objective in [('same_weight_current_post05', 'post_transport', row['objective']),
                                       ('plain_same_objective_post05', 'copy', row['objective']),
                                       ('fixed_R26_pre_MSE_current_post05', 'r26_pre_mse_spatial', 'mse')]:
            reference = {**row, 'mode': mode, 'objective': objective, 'history': 'current', 'fusion': 'postwarp05'}
            other = lookup[key(reference)]
            pairs.append({**{k: row[k] for k in FIELDS}, 'reference': label,
                          **{'delta_'+m: row[m]-other[m] for m in METRICS},
                          'candidate_temporal': row['temporal_error'], 'reference_temporal': other['temporal_error'],
                          'temporal_ratio': row['temporal_error']/other['temporal_error']})
    paired_groups = defaultdict(list)
    fields_pair = ['objective', 'train_seed', 'history', 'input_reset', 'reference']
    for row in pairs:
        paired_groups[tuple(row[k] for k in fields_pair)].append(row)
    paired_summary = []
    for group, values in sorted(paired_groups.items()):
        assert len(values) == 6
        ratio = mean(r['candidate_temporal'] for r in values)/mean(r['reference_temporal'] for r in values)
        deltas = {'delta_'+m: mean(r['delta_'+m] for r in values) for m in METRICS}
        paired_summary.append({**dict(zip(fields_pair, group)), **deltas,
                               'temporal_ratio_of_condition_means': ratio,
                               'registered_quality_temporal_gate': deltas['delta_psnr_rgb'] >= -.05 and ratio <= .9,
                               'source_videos': 3, 'input_conditions': 6})
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    save_json(out/'domain_seed_summary.json', summaries)
    save_json(out/'paired_by_condition.json', pairs)
    save_json(out/'paired_domain_seed.json', paired_summary)
    save_json(out/'audit.json', {'passed': True, 'included_phases': phases, 'rows': len(rows), 'runtimes': runs,
              'source_sha256': sources, 'complete_registered_model_set': set(phases) == {'references', 'post-seed0', 'post-seed1'},
              'scope': 'Three external TRAIN video sources, six custom degraded conditions; repeated frames/conditions are not independent samples. No standard-test or phone-latency claim.'})
    lines = ['# R30 external development results: completed phases only', '',
             'Included phases: '+', '.join(phases)+'. All rows use the common CPU backend. Each seed row averages six conditions from three source videos; these are not six independent videos.', '',
             'Missing phases have no inferred values. Official pretrained MobileIE has no project training seed. Full registered model set is '+('complete.' if len(phases) == 3 else '**incomplete**.'), '',
             '|Mode|Loss|Seed|History|Reset|Output|PSNR|SSIM|Temporal|Edge|',
             '|---|---|---:|---|---|---|---:|---:|---:|---:|']
    for r in summaries:
        lines.append('|'+ '|'.join(str(r[k]) for k in fields)+'|'+ '|'.join(f'{r[m]:.6f}' for m in METRICS)+'|')
    lines += ['', 'Current/raw is single-frame. Current/post .5 and causal mappings need flow/history. This custom low-light DEV task is not an official DAVIS LLVE benchmark.']
    (out/'tables.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'phases': phases, 'rows': len(rows), 'groups': len(summaries), 'out': str(out)}))


if __name__ == '__main__':
    main()
