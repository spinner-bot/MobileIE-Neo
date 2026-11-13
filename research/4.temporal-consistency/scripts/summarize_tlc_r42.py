"""Close the fixed TLC control against reused same-weight global evaluations."""
from collections import defaultdict
import json
from statistics import mean

from dynamic_training_r34 import BASE, ROOT, digest

METRICS = ['psnr_rgb', 'ssim_rgb', 'temporal_error', 'edge_mae']
DOMAINS = ['davis_train_pilot', 'paired_real_ND_DEV']


def main():
    rows, pairs, sources = [], [], {}
    for seed in [0, 1]:
        folder = BASE/f'results/exploration-r42-seed{seed}'
        runtime = json.loads((folder/'runtime.json').read_text())
        assert runtime['completed_normally'] and runtime['rows'] == 30
        assert runtime['neural_sequences'] == 10 and not runtime['cuda_initialized']
        config = json.loads((folder/'config.json').read_text())
        reference = ROOT/config['reference_rows_path']
        assert digest(reference) == config['reference_rows_sha256']
        local = list(map(json.loads, (folder/'per_sequence.jsonl').read_text().splitlines()))
        global_rows = [r for r in map(json.loads, reference.read_text().splitlines())
                       if r['anchor_family'] == 'R41_expanded_fp32']
        assert len(local) == len(global_rows) == 30
        lookup = {(r['domain'], r['scene'], r['method']): r for r in global_rows}
        assert len(lookup) == 30
        for row in local:
            assert row['train_seed'] == seed and row['anchor_family'] == 'R42_TLC192'
            ref = lookup[(row['domain'], row['scene'], row['method'])]
            pairs.append({'domain': row['domain'], 'scene': row['scene'], 'seed': seed,
                          'method': row['method'], **{'delta_'+m: row[m]-ref[m] for m in METRICS},
                          'temporal_ratio': row['temporal_error']/ref['temporal_error']})
        rows += [{**r, 'evidence_source': 'new_R42'} for r in local]
        rows += [{**r, 'evidence_source': 'reused_R41_global'} for r in global_rows]
        for path in [folder/'config.json', folder/'runtime.json', folder/'per_sequence.jsonl', reference]:
            sources[str(path.relative_to(ROOT))] = digest(path)
    assert len(rows) == 120 and len(pairs) == 60
    assert len({(r['domain'], r['scene'], r['anchor_family'], r['train_seed'], r['method']) for r in rows}) == 120
    groups, seeds, domains = defaultdict(list), [], []
    for row in rows:
        groups[(row['domain'], row['anchor_family'], row['train_seed'], row['method'])].append(row)
    for (domain, family, seed, method), values in sorted(groups.items()):
        assert len(values) == (6 if domain == DOMAINS[0] else 4)
        seeds.append({'domain': domain, 'anchor_family': family, 'seed': seed, 'method': method,
                      **{m: mean(r[m] for r in values) for m in METRICS}})
    assert len(seeds) == 24
    lookup = {(r['domain'], r['anchor_family'], r['seed'], r['method']): r for r in seeds}
    effects = []
    for domain in DOMAINS:
        for seed in [0, 1]:
            local, ref = [lookup[(domain, family, seed, 'raw')]
                          for family in ['R42_TLC192', 'R41_expanded_fp32']]
            effects.append({'domain': domain, 'seed': seed,
                            **{'delta_'+m: local[m]-ref[m] for m in METRICS},
                            'temporal_ratio': local['temporal_error']/ref['temporal_error']})
    groups = defaultdict(list)
    for row in seeds:
        groups[(row['domain'], row['anchor_family'], row['method'])].append(row)
    for (domain, family, method), values in sorted(groups.items()):
        assert {r['seed'] for r in values} == {0, 1}
        domains.append({'domain': domain, 'anchor_family': family, 'method': method,
                        **{m: mean(r[m] for r in values) for m in METRICS}})
    winning = []
    time_passed = all(r['temporal_ratio'] <= 1.02 for r in effects)
    for domain in DOMAINS:
        gains = [r['delta_psnr_rgb'] for r in effects if r['domain'] == domain]
        other = [r['delta_psnr_rgb'] for r in effects if r['domain'] != domain]
        if all(v >= .1 for v in gains) and all(v >= -.05 for v in other) and time_passed:
            winning.append(domain)
    out = BASE/'report/exploration-r42-tlc'
    out.mkdir(parents=True, exist_ok=False)
    def write(name, value):
        (out/name).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    for name, value in [('domain_summary.json', domains), ('domain_seed_summary.json', seeds),
                        ('paired_by_condition.json', pairs), ('raw_effects.json', effects)]:
        write(name, value)
    (out/'all_quality_rows.jsonl').write_text(''.join(json.dumps(r, allow_nan=False)+'\n' for r in rows))
    raw_pairs = [r for r in pairs if r['method'] == 'raw']
    write('audit.json', {'passed': True, 'new_rows': 60, 'reused_reference_rows': 60, 'total_rows': 120,
          'new_neural_sequences': 20, 'new_training_updates': 0, 'fixed_window': 192,
          'registered_screening_passed': bool(winning), 'winning_domains': winning,
          'all_raw_time_ratios_le_1_02': time_passed,
          'worst_raw_condition_delta_psnr': min(r['delta_psnr_rgb'] for r in raw_pairs),
          'source_sha256': sources,
          'scope': 'Known spatial TLC control on existing DEV; same weights and parameters, extra operations/memory. No phone or final-test result.'})
    lines = ['# R42 fixed TLC spatial control', '',
             'Both R41 expanded final4000 seeds, window192 only. 60 new rows / 20 CNN sequences, plus 60 reused same-weight global CPU references. No new training, future input, or raw temporal state. Finite/recursive retain their explicit RGB history and own-raw half-FB.', '',
             '|Domain|Spatial pooling|Output|PSNR|SSIM|Temporal↓|Edge↓|', '|---|---|---|---:|---:|---:|---:|']
    for row in domains:
        lines.append('|'+row['domain']+'|'+row['anchor_family']+'|'+row['method']+'|'
                     +'|'.join(f'{row[m]:.6f}' for m in METRICS)+'|')
    lines += ['', '|Domain|Seed|TLC−global raw PSNR|Raw temporal ratio|', '|---|---:|---:|---:|']
    for row in effects:
        lines.append(f"|{row['domain']}|{row['seed']}|{row['delta_psnr_rgb']:.6f}|{row['temporal_ratio']:.6f}|")
    lines += ['', 'Registered fixed-control screen: '+str(bool(winning))+'. Both seeds must gain .1dB in the same domain, lose no more than .05dB in the other, and have raw temporal ratios ≤1.02 in all domain/seed pairs.', '',
              'Worst raw per-condition ΔPSNR: '+str(min(r['delta_psnr_rgb'] for r in raw_pairs))+'. All condition pairs and four metrics are retained. Three synthetic source videos are six conditions, not six independent videos; real sources are reused daytime ND DEV. Known TLC is not a new video method. Extra convolution/prefix-sum memory is not zero inference overhead; phone speed remains unmeasured.']
    (out/'tables.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'new_rows': 60, 'screening_passed': bool(winning)}), flush=True)


if __name__ == '__main__':
    main()
