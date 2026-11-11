"""Both-seed closure of R37, with finite and recursive primary gates retained."""
from collections import defaultdict
import json
from statistics import mean

from dynamic_training_r34 import BASE, ROOT, digest

METRICS = ['psnr_rgb', 'ssim_rgb', 'temporal_error', 'edge_mae']
METHODS = ['raw', 'finite_self05', 'recursive_self05', 'adaptive_recursive',
           'zero_alignment', 'learned_alignment']


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def main():
    rows, sources = [], {}
    for seed in [0, 1]:
        folder = BASE/f'results/exploration-r37-eval-seed{seed}'
        runtime = json.loads((folder/'runtime.json').read_text())
        assert runtime['completed_normally'] and runtime['rows'] == 36
        assert runtime['neural_sequences'] == 12 and not runtime['cuda_initialized']
        values = list(map(json.loads, (folder/'per_sequence.jsonl').read_text().splitlines()))
        assert len(values) == 36 and {r['train_seed'] for r in values} == {seed}
        rows += values
        for name in ['runtime.json', 'config.json', 'per_sequence.jsonl', 'first_frame.jsonl']:
            path = folder/name
            sources[str(path.relative_to(ROOT))] = digest(path)
    lookup = {(r['method'], r['scene'], r['train_seed']): r for r in rows}
    assert len(lookup) == 72
    groups, seeds, domains, pairs, gates = defaultdict(list), [], [], [], []
    for row in rows:
        groups[(row['method'], row['train_seed'])].append(row)
    for (method, seed), values in sorted(groups.items()):
        assert len(values) == 6 and len({r['scene'] for r in values}) == 6
        assert {r['source_video'] for r in values} == {'bear', 'lucia', 'walking'}
        assert len({r['online_history_storage_bytes'] for r in values}) == 1
        seeds.append({'method': method, 'seed': seed,
                      'online_history_storage_bytes': values[0]['online_history_storage_bytes'],
                      **{m: mean(r[m] for r in values) for m in METRICS}})
    for method in METHODS:
        values = [r for r in seeds if r['method'] == method]
        assert {r['seed'] for r in values} == {0, 1}
        domains.append({'method': method, 'source_videos': 3, 'noise_conditions_per_video': 2, 'seeds': 2,
                        'parameters': 4587 if method.endswith('alignment') else 4075,
                        'online_history_storage_bytes': values[0]['online_history_storage_bytes'],
                        **{m: mean(r[m] for r in values) for m in METRICS}})
    for method in ['learned_alignment', 'zero_alignment']:
        for seed in [0, 1]:
            value = next(r for r in seeds if r['method'] == method and r['seed'] == seed)
            for reference in ['finite_self05', 'recursive_self05']:
                ref = next(r for r in seeds if r['method'] == reference and r['seed'] == seed)
                delta, ratio = value['psnr_rgb']-ref['psnr_rgb'], value['temporal_error']/ref['temporal_error']
                gates.append({'method': method, 'reference': reference, 'seed': seed,
                              'delta_psnr': delta, 'temporal_ratio': ratio,
                              'passed': delta >= -.05 and ratio <= .9})
        for row in rows:
            if row['method'] != method:
                continue
            for reference in ['raw', 'finite_self05', 'recursive_self05', 'adaptive_recursive']+(
                              ['zero_alignment'] if method == 'learned_alignment' else []):
                ref = lookup[(reference, row['scene'], row['train_seed'])]
                pairs.append({'method': method, 'reference': reference, 'scene': row['scene'],
                              'source_video': row['source_video'], 'seed': row['train_seed'],
                              **{'delta_'+m: row[m]-ref[m] for m in METRICS},
                              'temporal_ratio': row['temporal_error']/ref['temporal_error']})
    out = BASE/'report/exploration-r37'
    out.mkdir(parents=True, exist_ok=False)
    for name, value in [('domain_summary.json', domains), ('domain_seed_summary.json', seeds),
                        ('paired_by_condition.json', pairs), ('gates.json', gates)]:
        write(out/name, value)
    primary = {method: all(r['passed'] for r in gates if r['method'] == method)
               for method in ['learned_alignment', 'zero_alignment']}
    write(out/'audit.json', {'passed': True, 'rows': 72, 'new_neural_sequences': 24,
          'reused_reference_rows': 48, 'primary_passed_both_seeds_both_references': primary,
          'source_sha256': sources,
          'scope': 'Fixed final4000 CPU DEV; raw is the immutable-anchor/current mapping. Screening alone is not novelty, independent generalization or mobile performance.'})
    lines = ['# R37: preserved spatial anchor and full-resolution projected state', '',
             'Final4000 only, both seeds, common CPU, three DEV videos × two noise conditions. Four classical rows reuse the exact R36 immutable-anchor weights/input/CPU reference; current explicitly returns this anchor. Two actual candidate routes require their explicit FP16 state and sampling.', '',
             '|Method|PSNR|SSIM|Temporal↓|Edge↓|Params|History MiB@480×854|',
             '|---|---:|---:|---:|---:|---:|---:|']
    for row in domains:
        lines.append('|'+row['method']+'|'+'|'.join(f'{row[m]:.6f}' for m in METRICS)
                     +f"|{row['parameters']}|{row['online_history_storage_bytes']/2**20:.3f}|")
    lines += ['', '|Candidate|Reference|Seed|ΔPSNR|Temporal ratio|Joint gate|',
              '|---|---|---:|---:|---:|---|']
    for row in gates:
        lines.append(f"|{row['method']}|{row['reference']}|{row['seed']}|{row['delta_psnr']:.6f}|{row['temporal_ratio']:.6f}|{row['passed']}|")
    lines += ['', 'Both-seed, both-reference primary screening: '+json.dumps(primary)+'. No CPU pipeline, phone latency, energy, peak memory or independent-test claim follows from these quality-stage times.', '',
              'All conditions and both seeds remain in the paired table. Full-resolution state does not preserve every feature channel; its FP16 storage is three times the R34 low-resolution state and half one RGB FP32 buffer. Zero alignment retains the offset output parameters but they do not influence its prediction, so effective capacity is not asserted identical. Classical adaptive variance is a heuristic. None of projection, residual initialization, gating or warp alone establishes novelty.']
    (out/'tables.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'rows': 72, 'primary': primary}))


if __name__ == '__main__':
    main()
