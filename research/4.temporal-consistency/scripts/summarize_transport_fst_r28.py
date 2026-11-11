"""Summarize the registered R28 endpoints; no checkpoint or output selection."""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT/'research/4.temporal-consistency'
METRICS = ['psnr_rgb', 'ssim_rgb', 'temporal_error', 'edge_mae']
IDENTITY = ['domain', 'scene', 'mode', 'objective', 'train_seed', 'step', 'history', 'input_reset', 'fusion']
EXTERNAL = 'r26_pre_mse_spatial'


def read_rows(path):
    with path.open() as stream:
        return [json.loads(line) for line in stream]


def key(row, fields=IDENTITY):
    return tuple(row[field] for field in fields)


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', default=str(BASE/'report/exploration-r28'))
    args = parser.parse_args()
    rows, sources, runtimes = [], {}, []
    for mode, counts in [('copy', (2208, 276, 1536, 672)), ('post_transport', (2208, 464, 1584, 624))]:
        folder = BASE/'results'/('exploration-r28-eval-'+mode)
        runtime = json.loads((folder/'runtime.json').read_text())
        assert tuple(runtime[k] for k in ['rows', 'actual_inference_sequences', 'metric_computations', 'metric_aliases']) == counts
        assert runtime['device'] == 'cuda:0'
        runtimes.append(runtime)
        values = read_rows(folder/'per_sequence.jsonl')
        assert len(values) == counts[0]
        for path in folder.iterdir():
            if path.is_file():
                sources[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
        first = json.loads((folder/'first_frame_audit.json').read_text())
        assert first and max(r['max_abs_pixel'] for r in first) <= 1e-4
        for r in values:
            assert r['step'] == 4000 and all(math.isfinite(r[m]) for m in METRICS)
        rows.extend(values)
    lookup = {key(r): r for r in rows}
    assert len(rows) == len(lookup) == 4416
    groups = defaultdict(list)
    for r in rows:
        groups[key(r, ['domain', 'mode', 'objective', 'history', 'input_reset', 'fusion', 'train_seed'])].append(r)
    summary = []
    for group, values in sorted(groups.items()):
        record = dict(zip(['domain', 'mode', 'objective', 'history', 'input_reset', 'fusion', 'train_seed'], group))
        record.update(conditions=len(values), **{m: mean(r[m] for r in values) for m in METRICS})
        summary.append(record)
    # Every registered output remains visible. The screening gate uses raw only.
    comparisons = []
    for r in rows:
        if r['mode'] != 'post_transport':
            continue
        for name, mode, objective, history in [
            ('same_weight_current_post05', 'post_transport', r['objective'], 'current'),
            ('plain_same_objective_post05', 'copy', r['objective'], 'causal'),
            ('fixed_R26_pre_MSE_current_post05', EXTERNAL, 'mse', 'causal'),
        ]:
            reference = {**r, 'mode': mode, 'objective': objective, 'history': history, 'fusion': 'postwarp05'}
            other = lookup[key(reference)]
            comparisons.append({**dict(zip(IDENTITY, key(r))), 'reference': name,
                                **{'delta_'+m: r[m]-other[m] for m in METRICS},
                                'candidate_temporal': r['temporal_error'], 'reference_temporal': other['temporal_error']})
    paired_groups = defaultdict(list)
    fields = ['domain', 'objective', 'history', 'input_reset', 'fusion', 'train_seed', 'reference']
    for row in comparisons:
        paired_groups[key(row, fields)].append(row)
    paired = []
    for group, values in sorted(paired_groups.items()):
        record = dict(zip(fields, group))
        record.update(conditions=len(values), **{'delta_'+m: mean(r['delta_'+m] for r in values) for m in METRICS})
        reference_temporal = mean(r['reference_temporal'] for r in values)
        record['temporal_ratio_of_condition_means'] = mean(r['candidate_temporal'] for r in values)/reference_temporal if reference_temporal > 0 else None
        record['registered_quality_temporal_gate'] = (
            record['delta_psnr_rgb'] >= -.05 and record['temporal_ratio_of_condition_means'] is not None
            and record['temporal_ratio_of_condition_means'] <= .9) if record['fusion'] == 'raw' else None
        paired.append(record)
    # Required historical controls: same final weights, CUDA and condition/reset.
    replays = []
    for mode, directory, architecture, history in [
        ('copy', 'exploration-r26-eval-plain', 'plain', 'causal'),
        (EXTERNAL, 'exploration-r26b-eval-pre_fixedhalf', 'pre_fixedhalf', 'current'),
    ]:
        path = BASE/'results'/directory/'per_sequence.jsonl'
        old = {key({**r, 'mode': mode, 'history': 'causal'}): r for r in read_rows(path)
               if r['architecture'] == architecture and r['step'] == 4000 and r['history'] == history}
        actual = [r for r in rows if r['mode'] == mode]
        assert len(actual) == len(old)
        maximum = max(abs(r[m]-old[key(r)][m]) for r in actual for m in METRICS)
        assert maximum <= 1e-10, (mode, maximum)
        replays.append({'mode': mode, 'sequence_rows': len(actual), 'max_abs_metric': maximum})
        sources[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    write(out/'domain_seed_summary.json', summary)
    write(out/'paired_by_condition.json', comparisons)
    write(out/'paired_domain_seed.json', paired)
    write(out/'audit.json', {'rows': len(rows), 'runtimes': runtimes, 'historical_control_replays': replays,
                            'source_sha256': sources, 'passed': True,
                            'scope': 'Exploratory condition means, not independent-video sample sizes; per-seed gates are screening, not method novelty or final-test success. Current is single-frame; causal requires flow/history.'})
    lines = ['# R28 fixed4000 exploratory endpoints', '',
             'No checkpoint/output selection. Each row is a condition mean for one seed; repeated scenes are not independent videos. Current is single-frame; causal uses history/flow. Mobile latency remains unmeasured.', '',
             '|Domain|Mode|Loss|History|Reset|Output|Seed|Conditions|PSNR|SSIM|Temporal|Edge|',
             '|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|']
    for r in summary:
        lines.append('|'+ '|'.join(str(r[k]) for k in ['domain', 'mode', 'objective', 'history', 'input_reset', 'fusion', 'train_seed', 'conditions'])+'|'+ '|'.join(f'{r[m]:.6f}' for m in METRICS)+'|')
    lines.extend(['', 'Registered raw-output screening against every fixed reference is in [paired_domain_seed.json](paired_domain_seed.json). SSIM/edge, domains, seeds and external validation remain necessary even when the PSNR/temporal gate passes.'])
    (out/'tables.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'rows': len(rows), 'groups': len(summary), 'out': str(out)}))


if __name__ == '__main__':
    main()
