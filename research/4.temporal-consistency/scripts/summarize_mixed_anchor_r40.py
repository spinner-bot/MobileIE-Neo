"""Keep both domains separate and test the preregistered mixed-anchor diagnosis."""
from collections import defaultdict
import json
from statistics import mean

from dynamic_training_r34 import BASE, ROOT, digest

METRICS = ['psnr_rgb', 'ssim_rgb', 'temporal_error', 'edge_mae']
METHODS = ['raw', 'finite_self05', 'recursive_self05']
SYNTHETIC, REAL = 'davis_train_pilot', 'paired_real_ND_DEV'


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def main():
    rows, sources = [], {}
    for seed in [0, 1]:
        folder = BASE/f'results/exploration-r40-eval-seed{seed}'
        runtime = json.loads((folder/'runtime.json').read_text())
        assert runtime['completed_normally'] and runtime['rows'] == 30
        assert runtime['neural_sequences'] == 10 and not runtime['cuda_initialized']
        values = list(map(json.loads, (folder/'per_sequence.jsonl').read_text().splitlines()))
        assert len(values) == 30 and {r['train_seed'] for r in values} == {seed}
        rows += values
        for name in ['config.json', 'runtime.json', 'per_sequence.jsonl']:
            path = folder/name
            sources[str(path.relative_to(ROOT))] = digest(path)
    for stage, expected in [('exploration-r36-classical', 48), ('exploration-r39-anchor-transfer', 60)]:
        folder = BASE/'results'/stage
        runtime = json.loads((folder/'runtime.json').read_text())
        assert runtime['completed_normally'] and runtime['rows'] == expected and not runtime['cuda_initialized']
        values = list(map(json.loads, (folder/'per_sequence.jsonl').read_text().splitlines()))
        for row in values:
            if stage == 'exploration-r36-classical':
                if row['method'] not in METHODS:
                    continue
                row = {**row, 'anchor_family': 'R34_original', 'anchor': f"R34_seed{row['train_seed']}", 'domain': SYNTHETIC}
            rows.append({**row, 'reference_source': str((folder/'per_sequence.jsonl').relative_to(ROOT))})
        for name in ['runtime.json', 'config.json', 'per_sequence.jsonl']:
            path = folder/name
            sources[str(path.relative_to(ROOT))] = digest(path)
    assert len(rows) == 156
    lookup = {(r['domain'], r['anchor'], r['scene'], r['method']): r for r in rows}
    assert len(lookup) == 156
    groups, seeds, domains = defaultdict(list), [], []
    for row in rows:
        groups[(row['domain'], row['anchor_family'], row['train_seed'], row['method'])].append(row)
    for (domain, family, seed, method), values in sorted(groups.items(), key=lambda r: str(r[0])):
        assert len(values) == (6 if domain == SYNTHETIC else 4)
        seeds.append({'domain': domain, 'anchor_family': family, 'seed': seed, 'method': method,
                      **{m: mean(r[m] for r in values) for m in METRICS}})
    groups = defaultdict(list)
    for row in seeds:
        groups[(row['domain'], row['anchor_family'], row['method'])].append(row)
    for (domain, family, method), values in sorted(groups.items()):
        assert {r['seed'] for r in values} == ({None} if family == 'official_LOL' else {0, 1})
        domains.append({'domain': domain, 'anchor_family': family, 'method': method,
                        'source_videos': 3 if domain == SYNTHETIC else 4,
                        'input_conditions': 6 if domain == SYNTHETIC else 4,
                        'weight_instances': len(values), **{m: mean(r[m] for r in values) for m in METRICS}})
    gates = []
    for seed in [0, 1]:
        def select(domain, family):
            return next(r for r in seeds if r['domain'] == domain and r['anchor_family'] == family
                        and r['seed'] == seed and r['method'] == 'raw')
        synthetic_delta = select(SYNTHETIC, 'R40_mixed')['psnr_rgb']-select(SYNTHETIC, 'R34_original')['psnr_rgb']
        real_delta = select(REAL, 'R40_mixed')['psnr_rgb']-select(REAL, 'R34_original')['psnr_rgb']
        stronger_delta = select(REAL, 'R40_mixed')['psnr_rgb']-select(REAL, 'R26_plain_MSE')['psnr_rgb']
        gates.append({'seed': seed, 'synthetic_delta_vs_R34': synthetic_delta, 'real_delta_vs_R34': real_delta,
                      'real_delta_vs_stronger_R26': stronger_delta, 'real_recovery_gate': real_delta >= 2,
                      'synthetic_retention_gate': synthetic_delta >= -.2,
                      'registered_diagnostic_passed': real_delta >= 2 and synthetic_delta >= -.2})
    pairs = []
    for row in rows:
        if row['anchor_family'] != 'R40_mixed':
            continue
        refs = [f"R34_seed{row['train_seed']}"]
        if row['domain'] == REAL:
            refs += [f"R26_seed{row['train_seed']}", 'official_LOL']
        for anchor in refs:
            ref = lookup[(row['domain'], anchor, row['scene'], row['method'])]
            pairs.append({'domain': row['domain'], 'scene': row['scene'], 'seed': row['train_seed'],
                          'method': row['method'], 'reference': anchor, 'reference_seed': ref['train_seed'],
                          **{'delta_'+m: row[m]-ref[m] for m in METRICS}})
    out = BASE/'report/exploration-r40-mixed-anchor'
    out.mkdir(parents=True, exist_ok=False)
    for name, value in [('domain_summary.json', domains), ('domain_seed_summary.json', seeds),
                        ('paired_by_condition.json', pairs), ('gates.json', gates)]:
        write(out/name, value)
    write(out/'audit.json', {'passed': True, 'new_rows': 60, 'new_neural_sequences': 20,
          'reference_rows': 96, 'total_rows': 156, 'domains_separate': True,
          'registered_diagnostic_both_seeds': all(r['registered_diagnostic_passed'] for r in gates),
          'source_sha256': sources,
          'scope': 'Fixed data-mixing baseline, not temporal innovation or independent final validation. Official is one weight instance.'})
    lines = ['# R40 fixed mixed-data anchor', '',
             'Final4000 only, both seeds, common CPU. 60 new quality rows/20 new CNN sequences plus 96 complete same-CPU references. Domain means stay separate; no pooled score can hide a transfer loss. Original 4075-parameter backbone and optimization are retained. This is a strong-data baseline, not a new temporal method.', '']
    for domain in [SYNTHETIC, REAL]:
        lines += ['## '+domain, '', '|Anchor family|Method|PSNR|SSIM|Temporal↓|Edge↓|',
                  '|---|---|---:|---:|---:|---:|']
        for row in domains:
            if row['domain'] == domain:
                lines.append('|'+row['anchor_family']+'|'+row['method']+'|'+'|'.join(f'{row[m]:.6f}' for m in METRICS)+'|')
        lines.append('')
    lines += ['|Seed|Synthetic ΔPSNR vs R34|Real ΔPSNR vs R34|Real ΔPSNR vs R26|Diagnostic gate|',
              '|---|---:|---:|---:|---|']
    for row in gates:
        lines.append(f"|{row['seed']}|{row['synthetic_delta_vs_R34']:.6f}|{row['real_delta_vs_R34']:.6f}|{row['real_delta_vs_stronger_R26']:.6f}|{row['registered_diagnostic_passed']}|")
    lines += ['', 'Raw diagnostic requires at least +2dB real recovery vs R34 and synthetic loss no greater than .2dB, both seeds. The stronger R26 real reference is retained explicitly. Passing only restores a baseline; it does not prove novel video modeling, actual phone quality/latency or independent generalization. Post uses own adjacent raw half-FB; recursive also retains previous fused output. No extra teacher or GT reset. Real sources are four reused daytime ND crops, not natural-night test videos.']
    (out/'tables.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'new_rows': 60, 'reference_rows': 96, 'gates': gates}))


if __name__ == '__main__':
    main()
