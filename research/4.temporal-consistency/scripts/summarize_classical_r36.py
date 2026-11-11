"""Close complete R36 results without selecting a condition or training seed."""
from collections import defaultdict
import json
from statistics import mean

from classical_recursion_r36 import METHODS
from dynamic_training_r34 import BASE, ROOT, digest

METRICS = ('psnr_rgb', 'ssim_rgb', 'temporal_error', 'edge_mae')


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def main():
    source = BASE/'results/exploration-r36-classical'
    runtime = json.loads((source/'runtime.json').read_text())
    assert runtime['completed_normally'] and runtime['rows'] == 48
    assert runtime['neural_sequences'] == 12 and not runtime['cuda_initialized']
    rows = list(map(json.loads, (source/'per_sequence.jsonl').read_text().splitlines()))
    checks = list(map(json.loads, (source/'reference_checks.jsonl').read_text().splitlines()))
    assert len(rows) == 48 and len(checks) == 24
    lookup = {(r['method'], r['scene'], r['train_seed']): r for r in rows}
    assert len(lookup) == 48
    assert all(max(r['metric_abs_delta'].values()) <= 1e-6 for r in checks)
    groups, domain, seeds, pairs, gates = defaultdict(list), [], [], [], []
    for row in rows:
        groups[(row['method'], row['train_seed'])].append(row)
    for (method, seed), values in sorted(groups.items()):
        assert len(values) == 6 and len({r['scene'] for r in values}) == 6
        assert {r['source_video'] for r in values} == {'bear', 'lucia', 'walking'}
        assert len({r['online_history_storage_bytes'] for r in values}) == 1
        seeds.append({'method': method, 'seed': seed, 'source_videos': 3,
                      'online_history_storage_bytes': values[0]['online_history_storage_bytes'],
                      **{m: mean(r[m] for r in values) for m in METRICS}})
    for method in METHODS:
        values = [r for r in seeds if r['method'] == method]
        assert {r['seed'] for r in values} == {0, 1}
        domain.append({'method': method, 'source_videos': 3, 'noise_conditions_per_video': 2, 'seeds': 2,
                       'online_history_storage_bytes': values[0]['online_history_storage_bytes'],
                       **{m: mean(r[m] for r in values) for m in METRICS}})
    for method in ['recursive_self05', 'adaptive_recursive']:
        for seed in [0, 1]:
            value = next(r for r in seeds if r['method'] == method and r['seed'] == seed)
            ref = next(r for r in seeds if r['method'] == 'finite_self05' and r['seed'] == seed)
            delta, ratio = value['psnr_rgb']-ref['psnr_rgb'], value['temporal_error']/ref['temporal_error']
            gates.append({'method': method, 'seed': seed, 'reference': 'finite_self05',
                          'delta_psnr': delta, 'temporal_ratio': ratio,
                          'passed': delta >= -.05 and ratio <= .9})
        for row in rows:
            if row['method'] != method:
                continue
            for reference in ['raw', 'finite_self05']:
                ref = lookup[(reference, row['scene'], row['train_seed'])]
                pairs.append({'method': method, 'reference': reference, 'scene': row['scene'],
                              'source_video': row['source_video'], 'seed': row['train_seed'],
                              **{'delta_'+m: row[m]-ref[m] for m in METRICS},
                              'temporal_ratio': row['temporal_error']/ref['temporal_error']})
    out = BASE/'report/exploration-r36-classical'
    out.mkdir(parents=True, exist_ok=False)
    write(out/'domain_summary.json', domain)
    write(out/'domain_seed_summary.json', seeds)
    write(out/'paired_by_condition.json', pairs)
    write(out/'gates.json', gates)
    write(out/'audit.json', {'passed': True, 'rows': 48, 'reference_checks': 24,
          'gate_passed_both_seeds': {method: all(r['passed'] for r in gates if r['method'] == method)
                                   for method in ['recursive_self05', 'adaptive_recursive']},
          'source_sha256': {str((source/name).relative_to(ROOT)): digest(source/name) for name in
                           ['runtime.json', 'config.json', 'per_sequence.jsonl', 'reference_checks.jsonl']},
          'scope': 'Three DEV source videos, fixed R34 original weights. No phone/independent test, novel method or calibrated posterior claim.'})
    lines = ['# R36: classic recursive controls on the strong R34 original backbone', '',
             'Common CPU, fixed two final4000 seeds, three DEV source videos × two noise conditions, 64 frames. No new training or parameter selection. Metrics are equally averaged by video/condition/seed; conditions are not six independent videos.', '',
             '|Method|PSNR|SSIM|Temporal↓|Edge↓|History MiB|', '|---|---:|---:|---:|---:|---:|']
    for row in domain:
        lines.append('|'+row['method']+'|'+'|'.join(f'{row[m]:.6f}' for m in METRICS)
                     +f"|{row['online_history_storage_bytes']/2**20:.3f}|")
    lines += ['', '|Method|Seed|ΔPSNR vs finite|Temporal ratio|Joint gate|',
              '|---|---:|---:|---:|---|']
    for row in gates:
        lines.append(f"|{row['method']}|{row['seed']}|{row['delta_psnr']:.6f}|{row['temporal_ratio']:.6f}|{row['passed']}|")
    lines += ['', 'All post controls use the same own-raw half-FB motion and invalid-current fallback. Recursive controls use past fused outputs in addition to the past raw used for motion. Adaptive variance is a Haar-MAD/innovation heuristic; it is not a calibrated covariance or a BNLK/EMVD reproduction. History includes every required raw/fused/variance buffer and excludes offline outputs and transient activations.', '',
              'No standalone latency is inferred from the shared quality-evaluation stages. Server CPU is not phone latency. Raw/finite reference checks passed for all 12 input/weight pairs within 1e-6 for all four reported metrics. Passing a development gate does not establish novelty or independent generalization.']
    (out/'tables.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'rows': len(rows), 'gates': gates}))


if __name__ == '__main__':
    main()
